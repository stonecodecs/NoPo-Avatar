import os.path
from copy import deepcopy
from dataclasses import dataclass
from typing import Literal, Optional
import numpy as np

import torch
import torch.nn.functional as F
import torchvision.ops
from einops import rearrange, repeat
from jaxtyping import Float
from torch import Tensor, nn

from .backbone.croco.misc import transpose_to_landscape
from .heads import head_factory
from ...dataset.shims.bounds_shim import apply_bounds_shim
from ...dataset.shims.normalize_shim import apply_normalize_shim
from ...dataset.shims.patch_shim import apply_patch_shim
from ...dataset.types import BatchedExample, DataShim
from ...geometry.projection import sample_image_grid
from ..types import Gaussians
from .backbone import Backbone, BackboneCfg, get_backbone
from .backbone.croco.blocks import CrossAttention
from .common.gaussian_lbs_adapter import GaussianLBSAdapter, GaussianLBSAdapterCfg, UnifiedGaussianLBSAdapter
from .encoder import Encoder
from .visualization.encoder_visualizer_epipolar_cfg import EncoderVisualizerEpipolarCfg
from ...misc.utils import inverse_normalize
from ...misc.body_utils import SMPLX_N_BONES, SMPL_N_BONES, bone_lbs_weights_to_joint_lbs_weights
from .encoder_head_fpn import EncoderHeadDINOv2FPN, EncoderHeadDINOv2FPNCfg


inf = float('inf')


@dataclass
class OpacityMappingCfg:
    initial: float
    final: float
    warm_up: int


@dataclass
class EncoderLBSNoPoSplatFaceCfg:
    name: Literal["template_uv_concat_bone", "template_uv_face"]
    d_feature: int
    num_monocular_samples: int
    backbone: BackboneCfg
    visualizer: EncoderVisualizerEpipolarCfg
    gaussian_adapter: GaussianLBSAdapterCfg
    apply_bounds_shim: bool
    opacity_mapping: OpacityMappingCfg
    gaussians_per_pixel: int
    num_surfaces: int
    pts3d_head_type: str
    pts3d_head_skip: bool
    gs_params_head_type: str
    input_mean: list[float] = (0.5, 0.5, 0.5)
    input_std: list[float] = (0.5, 0.5, 0.5)
    pretrained_weights: Optional[str] = ""
    pretrained_template_reinit: bool = False
    pose_free: bool = True
    apply_mask: str = "none"
    pts3d_for_lbs_weights: bool = False
    highres_uv: bool = False

    has_conf: bool | None = None

    separate_xyz_head: bool = False
    n_hooks: int = 4

    debug: bool = False
    face_encoder_cfg: Optional[EncoderHeadDINOv2FPNCfg] = None


class _BackboneDimOverride:
    """Wrapper so image-branch heads see combined_dim (CroCo + DINO) instead of backbone.dec_embed_dim."""
    def __init__(self, backbone: nn.Module, dec_embed_dim_override: int):
        self._backbone = backbone
        self._dec_embed_dim = dec_embed_dim_override

    @property
    def dec_embed_dim(self) -> int:
        return self._dec_embed_dim

    def __getattr__(self, name: str):
        return getattr(self._backbone, name)


class EncoderTemplateUVFace(Encoder[EncoderLBSNoPoSplatFaceCfg]):
    backbone: nn.Module
    gaussian_adapter: GaussianLBSAdapter

    def __init__(self, cfg: EncoderLBSNoPoSplatFaceCfg) -> None:
        super().__init__(cfg)

        self.backbone = get_backbone(cfg.backbone, 3)

        self.face_encoder = None
        dino_dim = 0
        if getattr(cfg, "face_encoder_cfg", None) is not None:
            self.face_encoder = EncoderHeadDINOv2FPN(cfg.face_encoder_cfg)
            dino_dim = cfg.face_encoder_cfg.out_dim

        # Face features fused via cross-attention (query=CroCo, key/value=face); image branch stays 768-d
        self._image_branch_backbone = self.backbone
        self._dec_to_combined_proj = None
        if dino_dim > 0:
            self.face_kv_proj = nn.Linear(dino_dim, self.backbone.dec_embed_dim)
            self.face_cross_attn = CrossAttention(
                self.backbone.dec_embed_dim,
                rope=None,
                num_heads=8,
                qkv_bias=False,
                attn_drop=0.0,
                proj_drop=0.0,
            )
            with torch.no_grad():
                self.face_cross_attn.proj.weight.zero_()
                self.face_cross_attn.proj.bias.zero_()

        self.pose_free = cfg.pose_free
        if self.pose_free:
            self.gaussian_adapter = UnifiedGaussianLBSAdapter(cfg.gaussian_adapter)
        else:
            self.gaussian_adapter = GaussianLBSAdapter(cfg.gaussian_adapter)
        self.apply_mask = cfg.apply_mask

        self.patch_size = self.backbone.patch_embed.patch_size[0]
        self.raw_gs_dim = 1 + self.gaussian_adapter.d_in - 55 + SMPLX_N_BONES  # 1 for opacity

        self.pretrained_template_reinit = cfg.pretrained_template_reinit

        self.pts3d_head_type = cfg.pts3d_head_type
        self.gs_params_head_type = cfg.gs_params_head_type

        self.separate_xyz_head = cfg.separate_xyz_head
        self.n_hooks = cfg.n_hooks
        self.set_mean_head(
            output_mode='pts3d',
            head_type=cfg.pts3d_head_type,
            landscape_only=False,
            depth_mode=('exp', -inf, inf),
            conf_mode=("exp", 1, inf) if self.cfg.has_conf else None,
            skip=cfg.pts3d_head_skip,
            image_branch_backbone=self._image_branch_backbone,
        )
        self.set_gs_params_head(cfg, cfg.gs_params_head_type, image_branch_backbone=self._image_branch_backbone)

        pts3d_mean = torch.tensor([0., 0., 0.], dtype=torch.float32)
        pts3d_std = torch.tensor([1., 1., 1.], dtype=torch.float32)
        self.register_buffer('pts3d_mean', pts3d_mean)
        self.register_buffer('pts3d_std', pts3d_std)

        self.debug = cfg.debug

    
    def get_face_features(self, images, face_bboxes, target_res=224):
        """
        Get features from the face encoder.
        NOTE: face_bboxes can be empty list, where we return fully zero feature maps.
        """
        device = images.device
        B, V, C, H, W = images.shape
        
        images_flat = rearrange(images, "b v c h w -> (b v) c h w")
        if len(face_bboxes) > 0:
            bboxes_flat = rearrange(face_bboxes, "b v c -> (b v) c")
        else:
            bboxes_flat = torch.zeros((B*V, 4), device=device)
            # this forces is_valid to be all false
            # later, from valid_mask, it will return the 0-tensor
            # this should make DDP happier rather than returning a zero tensor conditionally
        
        # 1. get valid detections (bboxes not 0-vector or out of image)
        widths = bboxes_flat[:, 2] - bboxes_flat[:, 0]
        heights = bboxes_flat[:, 3] - bboxes_flat[:, 1]
        is_valid = (widths > 1.0) & (heights > 1.0)
        
        # 2. Create "Safe" Boxes (Dummy Box Strategy
        safe_bboxes = bboxes_flat.clone()
        # Replace invalid boxes with a 1x1 box at (0,0)
        # This guarantees RoIAlign produces valid output tensors
        safe_bboxes[~is_valid] = torch.tensor([0.0, 0.0, 1.0, 1.0], device=device)
        
        # 3. Square the Boxes
        safe_w = safe_bboxes[:, 2] - safe_bboxes[:, 0]
        safe_h = safe_bboxes[:, 3] - safe_bboxes[:, 1]
        max_side = torch.maximum(safe_w, safe_h)
        
        cx = (safe_bboxes[:, 0] + safe_bboxes[:, 2]) / 2
        cy = (safe_bboxes[:, 1] + safe_bboxes[:, 3]) / 2
        
        sq_x1 = cx - max_side / 2
        sq_y1 = cy - max_side / 2
        sq_x2 = cx + max_side / 2
        sq_y2 = cy + max_side / 2
        
        squared_rois = torch.stack([sq_x1, sq_y1, sq_x2, sq_y2], dim=1)
        
        # 4. RoIAlign
        batch_idxs = torch.arange(B*V, device=device).float().unsqueeze(1)
        rois = torch.cat([batch_idxs, squared_rois], dim=1)
        
        crops = torchvision.ops.roi_align(
            images_flat,
            rois,
            output_size=(target_res, target_res),
            spatial_scale=1.0,
            sampling_ratio=-1,
            aligned=True
        )
        
        # 5. Run Encoder (ALWAYS RUNS, keeping DDP happy)
        with torch.no_grad():
            dino_out = self.face_encoder(crops) # returns (BV, 256, 256)?
        dino_out = rearrange(dino_out, "(b v) l c -> b v l c", b=B, v=V)
            
        # 6. Masking
        # Zero out the features from the dummy boxes so they act like "Empty Signals"
        valid_mask = is_valid.float().view(B,V,1,1) 
        dino_out = dino_out * valid_mask
        return dino_out # (B,V,256,256)

    def set_mean_head(self, output_mode, head_type, landscape_only, depth_mode, conf_mode, skip,
                      image_branch_backbone=None):
        net_image = image_branch_backbone if image_branch_backbone is not None else self.backbone
        self.backbone.depth_mode = depth_mode
        self.backbone.conf_mode = conf_mode
        if net_image is not self.backbone:
            net_image.depth_mode = depth_mode
            net_image.conf_mode = conf_mode
        # allocate heads (template branch uses backbone; image branch uses net_image for combined CroCo+DINO dim)
        if self.pts3d_head_type == 'dpt':
            self.downstream_head1_template = head_factory(head_type, output_mode, self.backbone,
                                                          has_conf=bool(conf_mode),
                                                          out_nchan=3 + 3 * self.separate_xyz_head,
                                                          n_hooks=self.n_hooks)
            self.downstream_head2 = head_factory(head_type, output_mode, net_image, has_conf=bool(conf_mode),
                                                 skip=skip, out_nchan=3 + 3 * self.separate_xyz_head)

            # magic wrapper
            self.head1_template = transpose_to_landscape(self.downstream_head1_template, activate=landscape_only)
            self.head2 = transpose_to_landscape(self.downstream_head2, activate=landscape_only)
        else:
            self.downstream_head1_template = head_factory(head_type, output_mode, self.backbone,
                                                          has_conf=bool(conf_mode), img_nchan=55 + 3,
                                                          n_hooks=self.n_hooks)
            self.downstream_head2_rgb = head_factory(head_type, output_mode, net_image, has_conf=bool(conf_mode))

            # magic wrapper
            self.head1_template = transpose_to_landscape(self.downstream_head1_template, activate=landscape_only)
            self.head2 = transpose_to_landscape(self.downstream_head2_rgb, activate=landscape_only)

        if self.pretrained_template_reinit:
            nn.init.uniform_(self.downstream_head1_template.dpt.head[-1].weight, -1e-5, 1e-5)
            nn.init.zeros_(self.downstream_head1_template.dpt.head[-1].bias)

    def set_gs_params_head(self, cfg, head_type, image_branch_backbone=None):
        net_image = image_branch_backbone if image_branch_backbone is not None else self.backbone
        if head_type == 'linear':
            self.gaussian_param_head = nn.Sequential(
                nn.ReLU(),
                nn.Linear(
                    self.backbone.dec_embed_dim,
                    cfg.num_surfaces * self.patch_size ** 2 * self.raw_gs_dim,
                ),
            )
            self.gaussian_param_head2 = nn.Sequential(
                nn.ReLU(),
                nn.Linear(
                    net_image.dec_embed_dim,
                    cfg.num_surfaces * self.patch_size ** 2 * self.raw_gs_dim,
                ),
            )
        elif head_type == 'dpt':
            self.gaussian_param_head = head_factory(head_type, 'gs_params', self.backbone, has_conf=False,
                                                    out_nchan=self.raw_gs_dim)  # for view1 3DGS
            self.gaussian_param_head2 = head_factory(head_type, 'gs_params', net_image, has_conf=False,
                                                     out_nchan=self.raw_gs_dim)  # for view2 3DGS
        elif head_type == 'dpt_gs' or head_type == 'dpt_gs_debug':
            self.gaussian_param_head_template = head_factory(head_type, 'gs_params', self.backbone, has_conf=False,
                                                             out_nchan=self.raw_gs_dim + SMPL_N_BONES, img_nchan=55 + 3,
                                                             n_hooks=self.n_hooks)
            self.gaussian_param_head2 = head_factory(head_type, 'gs_params', net_image, has_conf=False,
                                                     out_nchan=self.raw_gs_dim + SMPL_N_BONES)
        else:
            raise NotImplementedError(f"unexpected {head_type=}")

    def map_pdf_to_opacity(
            self,
            pdf: Float[Tensor, " *batch"],
            global_step: int,
    ) -> Float[Tensor, " *batch"]:
        # https://www.desmos.com/calculator/opvwti3ba9

        # Figure out the exponent.
        cfg = self.cfg.opacity_mapping
        x = cfg.initial + min(global_step / cfg.warm_up, 1) * (cfg.final - cfg.initial)
        exponent = 2 ** x

        # Map the probability density to an opacity.
        return 0.5 * (1 - (1 - pdf) ** exponent + pdf ** (1 / exponent))

    def _downstream_head(self, head_num, decout, img_shape, ray_embedding=None, pos=None):
        if head_num == 1:
            head = self.head1_template
        else:
            head = self.head2
        return head(decout, img_shape, ray_embedding=ray_embedding, pos=pos)

    def filter_by_mask(self, gaussians_template, masks_template, gaussians, masks):

        b, v, r, srf, spp, xyz = gaussians.means.shape

        masks_template = repeat(masks_template, "b v r srf c -> b v r srf spp c", spp=spp)
        masks_template = rearrange(masks_template.squeeze(-1), "b v r srf spp -> b (v r srf spp)").contiguous()

        masks = repeat(masks, "b v r srf c -> b v r srf spp c", spp=spp)
        masks = rearrange(masks.squeeze(-1), "b v r srf spp -> b (v r srf spp)").contiguous()

        if self.debug:
            means_ori = torch.cat([
                rearrange(
                    gaussians.means,
                    "b v r srf spp xyz -> b (v r srf spp) xyz",
                ).contiguous()
            ], dim=1)
            covariance_ori = torch.cat([
                rearrange(
                    gaussians.covariances,
                    "b v r srf spp i j -> b (v r srf spp) i j",
                ).contiguous()
            ], dim=1)
            harmonics_ori = torch.cat([
                rearrange(
                    gaussians.harmonics,
                    "b v r srf spp c d_sh -> b (v r srf spp) c d_sh",
                ).contiguous(),
            ], dim=1)
            opacities_ori = torch.cat([
                rearrange(
                    gaussians.opacities,
                    "b v r srf spp -> b (v r srf spp)",
                ).contiguous(),
            ], dim=1)
            lbs_weights_ori = torch.cat([
                rearrange(
                    gaussians.lbs_weights,
                    "b v r srf spp w -> b (v r srf spp) w",
                ).contiguous()
            ], dim=1)
            idx_ori = torch.cat([
                rearrange(
                    gaussians.idx,
                    "b v r srf spp c -> b (v r srf spp) c",
                ).contiguous()
            ], dim=1)
            if gaussians.conf is not None:
                conf_ori = torch.cat([
                    rearrange(
                        gaussians.conf,
                        "b v r srf spp -> b (v r srf spp)",
                    ).contiguous()
                ], dim=1)
                conf = []
        else:
            masks = torch.cat([masks_template, masks], dim=1)
            means_ori = torch.cat([
                rearrange(
                    gaussians_template.means,
                    "b v r srf spp xyz -> b (v r srf spp) xyz",
                ).contiguous(),
                rearrange(
                    gaussians.means,
                    "b v r srf spp xyz -> b (v r srf spp) xyz",
                ).contiguous()
            ], dim=1)
            covariance_ori = torch.cat([
                rearrange(
                    gaussians_template.covariances,
                    "b v r srf spp i j -> b (v r srf spp) i j",
                ).contiguous(),
                rearrange(
                    gaussians.covariances,
                    "b v r srf spp i j -> b (v r srf spp) i j",
                ).contiguous()
            ], dim=1)
            harmonics_ori = torch.cat([
                rearrange(
                    gaussians_template.harmonics,
                    "b v r srf spp c d_sh -> b (v r srf spp) c d_sh",
                ).contiguous(),
                rearrange(
                    gaussians.harmonics,
                    "b v r srf spp c d_sh -> b (v r srf spp) c d_sh",
                ).contiguous(),
            ], dim=1)
            opacities_ori = torch.cat([
                rearrange(
                    gaussians_template.opacities,
                    "b v r srf spp -> b (v r srf spp)",
                ).contiguous(),
                rearrange(
                    gaussians.opacities,
                    "b v r srf spp -> b (v r srf spp)",
                ).contiguous(),
            ], dim=1)
            lbs_weights_ori = torch.cat([
                rearrange(
                    gaussians_template.lbs_weights,
                    "b v r srf spp w -> b (v r srf spp) w",
                ).contiguous(),
                rearrange(
                    gaussians.lbs_weights,
                    "b v r srf spp w -> b (v r srf spp) w",
                ).contiguous()
            ], dim=1)
            lbs_weights_bone_ori = torch.cat([
                rearrange(
                    gaussians_template.lbs_weights_bones,
                    "b v r srf spp w -> b (v r srf spp) w",
                ).contiguous(),
                rearrange(
                    gaussians.lbs_weights_bones,
                    "b v r srf spp w -> b (v r srf spp) w",
                ).contiguous()
            ], dim=1)
            idx_ori = torch.cat([
                rearrange(
                    gaussians_template.idx,
                    "b v r srf spp c -> b (v r srf spp) c",
                ).contiguous(),
                rearrange(
                    gaussians.idx,
                    "b v r srf spp c -> b (v r srf spp) c",
                ).contiguous()
            ], dim=1)
            if gaussians.conf is not None:
                conf_ori = torch.cat([
                    rearrange(
                        gaussians_template.conf,
                        "b v r srf spp -> b (v r srf spp)",
                    ).contiguous(),
                    rearrange(
                        gaussians.conf,
                        "b v r srf spp -> b (v r srf spp)",
                    ).contiguous()
                ], dim=1)
                conf = []

        means, covariances, harmonics, opacities, lbs_weights, lbs_weights_bone, idx = [], [], [], [], [], [], []
        nums = []
        for bi in range(b):
            valid = masks[bi] > 1e-3
            means.append(means_ori[bi][valid])
            covariances.append(covariance_ori[bi][valid])
            harmonics.append(harmonics_ori[bi][valid])
            opacities.append(opacities_ori[bi][valid])
            lbs_weights.append(lbs_weights_ori[bi][valid])
            lbs_weights_bone.append(lbs_weights_bone_ori[bi][valid])
            idx.append(idx_ori[bi][valid])
            nums.append(means[-1].shape[0])
            if gaussians.conf is not None:
                conf.append(conf_ori[bi][valid])

        max_num = max(nums)
        means_new = means_ori.new_zeros(b, max_num, *means_ori.shape[2:])
        means_new.fill_(1e8)
        covariances_new = covariance_ori.new_zeros(b, max_num, *covariance_ori.shape[2:])
        harmonics_new = harmonics_ori.new_zeros(b, max_num, *harmonics_ori.shape[2:])
        opacities_new = opacities_ori.new_zeros(b, max_num, *opacities_ori.shape[2:])
        lbs_weights_new = lbs_weights_ori.new_zeros(b, max_num, *lbs_weights_ori.shape[2:])
        lbs_weights_bone_new = lbs_weights_bone_ori.new_zeros(b, max_num, *lbs_weights_bone_ori.shape[2:])
        idx_new = idx_ori.new_full((b, max_num, *idx_ori.shape[2:]), -1)
        if gaussians.conf is not None:
            conf_new = conf_ori.new_ones(b, max_num, *conf_ori.shape[2:])
        else:
            conf_new = None
        for bi in range(b):
            means_new[bi][:nums[bi]] = means[bi]
            covariances_new[bi][:nums[bi]] = covariances[bi]
            harmonics_new[bi][:nums[bi]] = harmonics[bi]
            opacities_new[bi][:nums[bi]] = opacities[bi]
            lbs_weights_new[bi][:nums[bi]] = lbs_weights[bi]
            lbs_weights_bone_new[bi][:nums[bi]] = lbs_weights_bone[bi]
            idx_new[bi][:nums[bi]] = idx[bi]
            if gaussians.conf is not None:
                conf_new[bi][:nums[bi]] = conf[bi]

        return Gaussians(means_new, covariances_new, harmonics_new, opacities_new, lbs_weights_new, idx=idx_new,
                         nums=nums, conf=conf_new, lbs_weights_bones=lbs_weights_bone_new, )

    def forward(
            self,
            context: dict,
            global_step: int = 0,
            visualization_dump: Optional[dict] = None,
            return_complete_gaussians_rgb: bool = False,
            return_complete_gaussians: bool = False,
    ):
        use_smplx = context["use_smplx"].any()
        device = context["image"].device
        b, v, _, h, w = context["image"].shape

        rgbs = rearrange(inverse_normalize(context["image"]), "b v c h w -> b v h w c").contiguous()
        rgbs = repeat(rgbs, "b v h w c -> b v h w srf c", srf=self.cfg.num_surfaces)
        rgbs = rearrange(rgbs, "b v h w srf c -> b v (h w) srf () c").contiguous()

        # Encode the context images.
        [dec_feat_template, dec_feat], [shape_template, shape], [template, images], pose_template, pose = self.backbone(context)

        croco_feats = dec_feat[-1]  # (B, V, L, C_croco)
        dec_feat_for_image = list(dec_feat)  # default; replaced when fusing DINO
        # this is [B,3,(HW)_patch, C_croco=768]

        # Face encoder: fuse via cross-attention (query=full-image CroCo, key/value=face crop features)
        if self.face_encoder is not None:
            face_bboxes = context["face_bbox"]
            images = context["image"]  # (B,V,C=3,H,W)
            if images.max() > 1.0 or images.min() < 0.0:
                images = (images + 1.0) / 2.0
            images = images.clamp(0.0, 1.0)
            with torch.no_grad():
                dino_out = self.get_face_features(images, face_bboxes)  # (B, V, N_dino, C_dino)
            face_kv = self.face_kv_proj(dino_out.float())  # (B, V, N_dino, dec_embed_dim)
            face_kv = rearrange(face_kv, "b v n c -> (b v) n c", b=b, v=v)  # (B*V, N_dino, 768)
            croco_flat = rearrange(croco_feats.float(), "b v l c -> (b v) l c")
            cross_out = self.face_cross_attn(
                croco_flat, face_kv, face_kv, None, None
            )
            cross_out = croco_flat + cross_out  # residual
            cross_out = rearrange(cross_out, "(b v) l c -> b v l c", b=b, v=v)
            dec_feat_for_image = list(dec_feat)
            dec_feat_for_image[-1] = cross_out

        with torch.amp.autocast('cuda', enabled=False):
            if self.pts3d_head_type == 'dpt':
                res1, _ = self._downstream_head(1, [tok.float() for tok in dec_feat_template] + [template],
                                                shape_template, pos=pose_template)
                if use_smplx or not self.separate_xyz_head:
                    res1['pts3d'] = res1['pts3d'][..., :3] + context["template_3d"]
                else:
                    res1['pts3d'] = res1['pts3d'][..., 3:] + context["template_3d"]
                all_mean_res = []
                for i in range(v):
                    res2, _ = self._downstream_head(2, [tok[:, i].float() for tok in dec_feat_for_image] + [images[:, i]],
                                                    shape[:, i], pos=pose[:, i])
                    if use_smplx or not self.separate_xyz_head:
                        res2['pts3d'] = res2['pts3d'][..., :3]
                    else:
                        res2['pts3d'] = res2['pts3d'][..., 3:]
                    all_mean_res.append(res2)
            else:
                res1 = self.downstream_head1_template([tok.float() for tok in dec_feat_template], None, template,
                                                      shape_template[0].cpu().tolist())
                res1['pts3d'] += context["template_3d"]
                all_mean_res = []
                for i in range(v):
                    res2 = self.downstream_head2_rgb(
                        [tok[:, i].float() for tok in dec_feat_for_image], None, images[:, i, :3],
                        shape[0, i].cpu().tolist(),
                    )
                    all_mean_res.append(res2)

            # for the 3DGS heads
            if self.gs_params_head_type == 'dpt_gs' or self.gs_params_head_type == 'dpt_gs_debug':
                GS_res1 = self.gaussian_param_head_template([tok.float() for tok in dec_feat_template],
                                                            all_mean_res[0]['pts3d'].permute(0, 3, 1, 2), template,
                                                            shape_template[0].cpu().tolist(), pos=pose_template)
                GS_res1 = rearrange(GS_res1, "b d h w -> b (h w) d").contiguous()
                all_other_params = []
                for i in range(v):
                    GS_res2 = self.gaussian_param_head2(
                        [tok[:, i].float() for tok in dec_feat_for_image],
                        all_mean_res[i]['pts3d'].permute(0, 3, 1, 2), images[:, i, :3],
                        shape[0, i].cpu().tolist(),
                        pos=pose[:, i],
                    )
                    GS_res2 = rearrange(GS_res2, "b d h w -> b (h w) d").contiguous()
                    all_other_params.append(GS_res2)
            else:
                raise NotImplementedError(f"unexpected {self.gs_params_head_type=}")

        # first wrap up the prediction for template branch
        pts_template = res1['pts3d']
        pts_template = rearrange(pts_template, "b h w xyz -> b () (h w) xyz").contiguous()
        pts_template = pts_template.unsqueeze(-2)  # for cfg.num_surfaces

        depths_template = pts_template[..., -1].unsqueeze(-1)

        gaussians_template = GS_res1.unsqueeze(1)
        gaussians_template = rearrange(gaussians_template, "... (srf c) -> ... srf c",
                                       srf=self.cfg.num_surfaces).contiguous()
        densities_template = gaussians_template[..., 0].sigmoid().unsqueeze(-1)

        if use_smplx:
            lbs_weights_bones_template = gaussians_template[..., -(SMPL_N_BONES + SMPLX_N_BONES):-SMPL_N_BONES]
        else:
            lbs_weights_bones_template = gaussians_template[..., -SMPL_N_BONES:]
        # print('bone', lbs_weights_bones_template.isnan().any(), lbs_weights_bones_template.isinf().any())
        lbs_weights_joints_template = bone_lbs_weights_to_joint_lbs_weights(lbs_weights_bones_template,
                                                                            use_smplx=use_smplx)
        # print('joint', lbs_weights_joints_template.isnan().any(), lbs_weights_joints_template.isinf().any())
        gaussians_template = torch.concatenate(
            [gaussians_template[..., :-(SMPL_N_BONES + SMPLX_N_BONES)], lbs_weights_joints_template], dim=-1)

        opacities_template = self.map_pdf_to_opacity(densities_template, global_step)
        # insert template uv as canonical view
        mask_template = repeat(context["template_mask"], "b h w -> b () h w srf c", srf=self.cfg.num_surfaces, c=1)
        mask_template = rearrange(mask_template, "b v h w srf c -> b v (h w) srf c").contiguous()
        if self.apply_mask != 'none':
            if self.apply_mask == 'soft':
                opacities_template *= mask_template
            else:
                opacities_template = mask_template

        # Convert the features and depths into Gaussians.
        gaussians_template = self.gaussian_adapter.forward(
            pts_template.unsqueeze(-2),
            depths_template,
            opacities_template,
            rearrange(gaussians_template[..., 1:], "b v r srf c -> b v r srf () c").contiguous(),
            use_smplx=use_smplx
        )
        h_template, w_template = context["template_mask"].shape[-2:]
        idx_template = torch.stack(
            torch.meshgrid(torch.arange(1, device=device), torch.arange(h_template, device=device),
                           torch.arange(w_template, device=device)), dim=-1)
        idx_template = repeat(idx_template, "v h w c -> b v h w srf spp c", b=b, srf=self.cfg.num_surfaces,
                              spp=1).contiguous()
        gaussians_template.idx = rearrange(idx_template,
                                           "b v h w srf spp c -> b v (h w) srf spp c").contiguous()  # view_id, h, w
        gaussians_template.lbs_weights_bones = rearrange(lbs_weights_bones_template,
                                                         "b v h w c -> b v (h w) () () c").contiguous()

        if 'conf' in res1:
            conf_template = res1['conf']
            conf_template = repeat(conf_template, "b h w -> b v (h w) srf spp", v=1, srf=self.cfg.num_surfaces, spp=1)
            gaussians_template.conf = conf_template
        else:
            gaussians_template.conf = None

        # now handle rgb branches
        pts_all = [all_mean_res_i['pts3d'] for all_mean_res_i in all_mean_res]
        pts_all = torch.stack(pts_all, dim=1)
        pts_all = rearrange(pts_all, "b v h w xyz -> b v (h w) xyz").contiguous()
        pts_all = pts_all.unsqueeze(-2)  # for cfg.num_surfaces

        depths = pts_all[..., -1].unsqueeze(-1)

        gaussians = torch.stack(all_other_params, dim=1)
        gaussians = rearrange(gaussians, "... (srf c) -> ... srf c", srf=self.cfg.num_surfaces).contiguous()
        densities = gaussians[..., 0].sigmoid().unsqueeze(-1)

        if use_smplx:
            lbs_weights_bones = gaussians[..., -(SMPL_N_BONES + SMPLX_N_BONES):-SMPL_N_BONES]
        else:
            lbs_weights_bones = gaussians[..., -SMPL_N_BONES:]
        # print('bone', lbs_weights_bones.isnan().any(), lbs_weights_bones.isinf().any())
        lbs_weights_joints = bone_lbs_weights_to_joint_lbs_weights(lbs_weights_bones, use_smplx=use_smplx)
        # print('joint', lbs_weights_joints.isnan().any(), lbs_weights_joints.isinf().any())
        gaussians = torch.concatenate(
            [gaussians[..., :-(SMPL_N_BONES + SMPLX_N_BONES)], lbs_weights_joints], dim=-1)

        opacities = self.map_pdf_to_opacity(densities, global_step)
        # insert template uv as canonical view
        mask = repeat(context["mask"], "b v h w -> b v h w srf c", srf=self.cfg.num_surfaces, c=1)
        mask = rearrange(mask, "b v h w srf c -> b v (h w) srf c").contiguous()
        if self.apply_mask != 'none':
            if self.apply_mask == 'soft':
                opacities *= mask
            else:
                opacities = mask

        # Convert the features and depths into Gaussians.
        gaussians = self.gaussian_adapter.forward(
            pts_all.unsqueeze(-2),
            depths,
            opacities,
            rearrange(gaussians[..., 1:], "b v r srf c -> b v r srf () c").contiguous(),
            use_smplx=use_smplx
        )
        idx = torch.stack(torch.meshgrid(torch.arange(1, v + 1, device=device), torch.arange(h, device=device),
                                         torch.arange(w, device=device)), dim=-1)
        idx = repeat(idx, "v h w c -> b v h w srf spp c", b=b, srf=self.cfg.num_surfaces, spp=1).contiguous()
        gaussians.idx = rearrange(idx, "b v h w srf spp c -> b v (h w) srf spp c").contiguous()  # view_id, h, w
        gaussians.lbs_weights_bones = rearrange(lbs_weights_bones, "b v h w c -> b v (h w) () () c")

        if "conf" in all_mean_res[0].keys():
            conf_all = [all_mean_res_i['conf'] for all_mean_res_i in all_mean_res]
            conf_all = torch.stack(conf_all, dim=1)
            conf_all = repeat(conf_all, "b v h w -> b v (h w) srf spp", srf=self.cfg.num_surfaces, spp=1).contiguous()
            gaussians.conf = conf_all
        else:
            gaussians.conf = None

        # Dump visualizations if needed.
        if visualization_dump is not None:
            visualization_dump["depth"] = rearrange(
                depths, "b v (h w) srf s -> b v h w srf s", h=h, w=w
            ).contiguous()
            visualization_dump["scales"] = rearrange(
                gaussians.scales, "b v r srf spp xyz -> b (v r srf spp) xyz"
            ).contiguous()
            visualization_dump["rotations"] = rearrange(
                gaussians.rotations, "b v r srf spp xyzw -> b (v r srf spp) xyzw"
            ).contiguous()
            visualization_dump["means"] = rearrange(
                gaussians.means, "b v (h w) srf spp xyz -> b v h w (srf spp) xyz", h=h, w=w
            ).contiguous()
            visualization_dump['opacities'] = rearrange(
                gaussians.opacities, "b v (h w) srf s -> b v h w srf s", h=h, w=w
            ).contiguous()

        if return_complete_gaussians_rgb:
            if "lbs_weights" in context:
                context["lbs_weights"] = torch.cat(
                    [context["template_lbs_weights"].unsqueeze(1), context["lbs_weights"]], dim=1)
                context["mask"] = torch.cat([context["template_mask"].unsqueeze(1), context["mask"]], dim=1)
            return self.filter_by_mask(gaussians_template, mask_template, gaussians, mask), Gaussians(
                rearrange(
                    torch.cat([gaussians_template.means, gaussians.means], dim=1),
                    "b v r srf spp xyz -> b (v r srf spp) xyz",
                ),
                rearrange(
                    torch.cat([gaussians_template.covariances, gaussians.covariances], dim=1),
                    "b v r srf spp i j -> b (v r srf spp) i j",
                ),
                rearrange(
                    torch.cat([gaussians_template.harmonics, gaussians.harmonics], dim=1),
                    "b v r srf spp c d_sh -> b (v r srf spp) c d_sh",
                ),
                rearrange(
                    torch.cat([gaussians_template.opacities, gaussians.opacities], dim=1),
                    "b v r srf spp -> b (v r srf spp)",
                ),
                rearrange(
                    torch.cat([gaussians_template.lbs_weights, gaussians.lbs_weights], dim=1),
                    "b v r srf spp d -> b (v r srf spp) d",
                ),
                lbs_weights_bones=rearrange(
                    torch.cat([gaussians_template.lbs_weights_bones, gaussians.lbs_weights_bones], dim=1),
                    "b v r srf spp d -> b (v r srf spp) d",
                ),
            )

        if return_complete_gaussians:
            gall = Gaussians(
                rearrange(
                    gaussians.means,
                    "b v r srf spp xyz -> b (v r srf spp) xyz",
                ),
                rearrange(
                    gaussians.covariances,
                    "b v r srf spp i j -> b (v r srf spp) i j",
                ),
                rearrange(
                    gaussians.harmonics,
                    "b v r srf spp c d_sh -> b (v r srf spp) c d_sh",
                ),
                rearrange(
                    gaussians.opacities,
                    "b v r srf spp -> b (v r srf spp)",
                ),
                rearrange(
                    gaussians.lbs_weights,
                    "b v r srf spp d -> b (v r srf spp) d",
                ),
                lbs_weights_bones=rearrange(
                    gaussians.lbs_weights_bones,
                    "b v r srf spp d -> b (v r srf spp) d",
                ),
                idx=rearrange(
                    gaussians.idx,
                    "b v r srf spp d -> b (v r srf spp) d"
                )
            )
            gall_template = Gaussians(
                rearrange(
                    gaussians_template.means,
                    "b v r srf spp xyz -> b (v r srf spp) xyz",
                ),
                rearrange(
                    gaussians_template.covariances,
                    "b v r srf spp i j -> b (v r srf spp) i j",
                ),
                rearrange(
                    gaussians_template.harmonics,
                    "b v r srf spp c d_sh -> b (v r srf spp) c d_sh",
                ),
                rearrange(
                    gaussians_template.opacities,
                    "b v r srf spp -> b (v r srf spp)",
                ),
                rearrange(
                    gaussians_template.lbs_weights,
                    "b v r srf spp d -> b (v r srf spp) d",
                ),
                lbs_weights_bones=rearrange(
                    gaussians_template.lbs_weights_bones,
                    "b v r srf spp d -> b (v r srf spp) d",
                ),
            )
            return self.filter_by_mask(gaussians_template, mask_template, gaussians, mask), gall_template, gall

        return self.filter_by_mask(gaussians_template, mask_template, gaussians, mask)

    def get_data_shim(self) -> DataShim:
        # Load constant mesh once for UV projection (computed on GPU in the shim, not in the dataloader).
        _uv_mesh_cache = [None]  # mutable so inner function can assign

        def data_shim(batch: BatchedExample) -> BatchedExample:
            batch = apply_normalize_shim(
                batch,
                self.cfg.input_mean,
                self.cfg.input_std,
            )
            # Compute uv_map / uv_valid on GPU when the backbone uses UV PE and the batch has poses/cameras.
            if (
                getattr(self.backbone, "use_uv_pe", False)
                and "uv_map" not in batch["context"]
                and "Rs" in batch["context"]
                and "Ts" in batch["context"]
                and "extrinsics" in batch["context"]
                and "intrinsics" in batch["context"]
            ):
                try:
                    import sys
                    from pathlib import Path
                    from einops import rearrange
                    repo_root = Path(__file__).resolve().parents[3]  # repo root (file is in src/model/encoder/)
                    if str(repo_root) not in sys.path:
                        sys.path.insert(0, str(repo_root))
                    from smplx_uv_projection import (
                        get_batched_image_space_uv,
                        build_pytorch3d_cameras_from_w2c_k,
                        load_smplx_uv_mesh_constants,
                    )
                    from ...misc.body_utils import apply_lbs_to_means
                    if _uv_mesh_cache[0] is None:
                        obj_path = repo_root / "assets" / "templates" / "smplx_uv" / "smplx_uv.obj"
                        smplx_path = str(repo_root / "datasets" / "smplx" / "SMPLX_MALE.npz")
                        _uv_mesh_cache[0] = load_smplx_uv_mesh_constants(obj_path, smplx_path)
                    mesh = _uv_mesh_cache[0]
                    if mesh is not None:
                        vertex_np, faces_np, verts_uv_np, lbs_weights_np = mesh
                        device = batch["context"]["image"].device
                        B, V = batch["context"]["image"].shape[:2]
                        H_img = batch["context"]["image"].shape[-2]
                        W_img = batch["context"]["image"].shape[-1]

                        faces = torch.tensor(faces_np, dtype=torch.int64, device=device)
                        verts_uv = torch.tensor(verts_uv_np, dtype=torch.float32, device=device)

                        # Prefer per-subject identity-fitted canonical mesh from the batch.
                        # "canonical_vertex" is the rest-pose mesh fitted to this person (betas-shaped).
                        # Fall back to the constant OBJ template if not available.
                        ctx_vertex = batch["context"].get("canonical_vertex", None)
                        ctx_weights = batch["context"].get("canonical_lbs_weights", None)
                        if ctx_vertex is not None and ctx_weights is not None:
                            vertex = ctx_vertex.to(device).float()    # (B, N, 3)
                            weights = ctx_weights.to(device).float()  # (B, N, J)
                        else:
                            vertex = torch.tensor(vertex_np, dtype=torch.float32, device=device).unsqueeze(0).expand(B, -1, -1)
                            weights = torch.tensor(lbs_weights_np, dtype=torch.float32, device=device)

                        Rs = batch["context"]["Rs"]   # (B, V, 55, 3, 3)
                        Ts = batch["context"]["Ts"]   # (B, V, 55, 3)
                        B_flat = B * V
                        vertex_batch = repeat(vertex, "b n c -> (b v) n c", v=V)
                        if weights.dim() == 2:  # (N, J) constant template
                            weights_batch = weights.unsqueeze(0).expand(B_flat, -1, -1)
                        else:  # (B, N, J) per-subject
                            weights_batch = repeat(weights, "b n j -> (b v) n j", v=V)
                        Rs_flat = rearrange(Rs, "b v j r c -> (b v) j r c")
                        Ts_flat = rearrange(Ts, "b v j d -> (b v) j d")
                        posed_flat = apply_lbs_to_means(vertex_batch, Rs_flat, Ts_flat, weights_batch)
                        posed_vertices = rearrange(posed_flat, "(b v) n c -> b v n c", b=B, v=V)
                        w2c = batch["context"]["extrinsics"].inverse()
                        K = batch["context"]["intrinsics"].clone()
                        K[:, :, 0, 0] *= W_img
                        K[:, :, 1, 1] *= H_img
                        K[:, :, 0, 2] *= W_img
                        K[:, :, 1, 2] *= H_img
                        w2c_flat = rearrange(w2c, "b v r c -> (b v) r c")
                        K_flat = rearrange(K, "b v r c -> (b v) r c")
                        cameras = build_pytorch3d_cameras_from_w2c_k(
                            w2c_flat, K_flat, (H_img, W_img), device
                        )
                        uv_map, uv_valid = get_batched_image_space_uv(
                            posed_vertices,
                            faces,
                            verts_uv,
                            cameras,
                            (H_img, W_img),
                            device=device,
                        )
                        batch["context"]["uv_map"] = uv_map
                        batch["context"]["uv_valid"] = uv_valid
                except Exception as e:
                    print(f"Error computing uv_map/uv_valid: {e}")
                    pass
            return batch

        return data_shim