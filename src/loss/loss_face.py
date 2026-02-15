"""
Face losses: MSE, LPIPS, and ArcFace identity loss computed inside face bounding boxes.
When a view has no valid face bbox (e.g. zero-area or sentinel), that view is skipped.
"""
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from jaxtyping import Float, Bool
from typing import Literal
from lpips import LPIPS
from torch import Tensor
import torchvision

from ..dataset.types import BatchedExample
from ..misc.nn_module_tools import convert_to_buffer
from ..model.decoder.decoder import DecoderOutput
from ..model.types import Gaussians
from .loss import Loss


# ---------------------------------------------------------------------------
# Configs (nested so LossFace composes MSE, LPIPS, ArcFace)
# ---------------------------------------------------------------------------

@dataclass
class LossMseCfg:
    weight: float = 1.0
    use_conf: bool = False
    alpha: float = 0.0
    apply_mask: bool = False


@dataclass
class LossLpipsCfg:
    weight: float = 0.1
    apply_after_step: int = 0


@dataclass
class LossArcFaceCfg:
    weight: float = 0.1


@dataclass
class LossFaceCfg:
    mse: LossMseCfg = field(default_factory=LossMseCfg)
    lpips: LossLpipsCfg = field(default_factory=LossLpipsCfg)
    arcface: LossArcFaceCfg = field(default_factory=LossArcFaceCfg)


@dataclass
class LossFaceCfgWrapper:
    faceloss: LossFaceCfg


# ---------------------------------------------------------------------------
# Helpers: face crop extraction and valid mask
# ---------------------------------------------------------------------------

def _get_face_bbox_tensor(batch: BatchedExample, key: str = "target") -> Tensor | None:
    """Return face_bbox as (B, V, 4) or None if missing/empty."""
    views = batch.get(key)
    if views is None:
        return None
    bboxes = views.get("face_bbox")
    if bboxes is None or (isinstance(bboxes, (list, tuple)) and len(bboxes) == 0):
        return None
    if isinstance(bboxes, (list, tuple)):
        # Batch: list of lists of tensors (B, V) or list of tensors (B, V, 4)
        first = bboxes[0]
        if isinstance(first, (list, tuple)):
            bboxes = torch.stack([torch.stack([t for t in b]) for b in bboxes], dim=0)
        else:
            bboxes = torch.stack(bboxes, dim=0)
    return bboxes


def _valid_face_mask(bboxes: Float[Tensor, "b v 4"]) -> Bool[Tensor, "b v"]:
    """True where face bbox has area > 0 (width and height > 1 pixel)."""
    x1, y1, x2, y2 = bboxes[..., 0], bboxes[..., 1], bboxes[..., 2], bboxes[..., 3]
    w = x2 - x1
    h = y2 - y1
    return (w > 1.0) & (h > 1.0)


def _squared_rois_xyxy(bboxes: Float[Tensor, "bv 4"], device: torch.device) -> Float[Tensor, "bv 4"]:
    """Square the boxes (center, max side). Invalid boxes are replaced with (0,0,1,1)."""
    w = bboxes[:, 2] - bboxes[:, 0]
    h = bboxes[:, 3] - bboxes[:, 1]
    valid = (w > 1.0) & (h > 1.0)
    cx = (bboxes[:, 0] + bboxes[:, 2]) / 2
    cy = (bboxes[:, 1] + bboxes[:, 3]) / 2
    max_side = torch.maximum(w.clamp(min=1.0), h.clamp(min=1.0))
    x1 = cx - max_side / 2
    y1 = cy - max_side / 2
    x2 = cx + max_side / 2
    y2 = cy + max_side / 2
    rois = torch.stack([x1, y1, x2, y2], dim=1)
    rois[~valid] = torch.tensor([0.0, 0.0, 1.0, 1.0], device=device)
    return rois


def _crop_faces_roi(
    images: Float[Tensor, "b v c h w"],
    bboxes: Float[Tensor, "b v 4"],
    output_size: int = 224,
) -> tuple[Float[Tensor, "bv c oh ow"], Float[Tensor, "b v"]]:
    """
    Crop each view by its face bbox using RoI align.
    images: (B, V, C, H, W), bboxes: (B, V, 4) x1,y1,x2,y2 in pixel coords.
    Returns: crops (B*V, C, output_size, output_size), valid (B, V).
    """
    B, V, C, H, W = images.shape
    device = images.device
    images_flat = rearrange(images, "b v c h w -> (b v) c h w")
    bboxes_flat = rearrange(bboxes, "b v c -> (b v) c")
    rois_xyxy = _squared_rois_xyxy(bboxes_flat, device)
    batch_idx = torch.arange(B * V, device=device, dtype=torch.float32).unsqueeze(1)
    rois = torch.cat([batch_idx, rois_xyxy], dim=1)  # (BV, 5) for roi_align
    crops = torchvision.ops.roi_align( # square crop around the face coordinates
        images_flat,
        rois,
        output_size=(output_size, output_size),
        spatial_scale=1.0,
        sampling_ratio=-1,
        aligned=True,
    )
    valid = _valid_face_mask(bboxes).float()
    return crops, valid


# ---------------------------------------------------------------------------
# LossFace: MSE + LPIPS + ArcFace on face crops; skip invalid views
# ---------------------------------------------------------------------------

class LossFace(Loss[LossFaceCfg, LossFaceCfgWrapper]):
    lpips_net: LPIPS
    face_crop_size: int = 224

    def __init__(self, cfg: LossFaceCfgWrapper) -> None:
        super().__init__(cfg)
        self.lpips_net = LPIPS(net="vgg")
        convert_to_buffer(self.lpips_net, persistent=False)
        # Optional: set from outside to enable ArcFace loss (e.g. insightface embedder)
        self.arcface_embedder: nn.Module | None = None

    def forward(
        self,
        prediction: DecoderOutput,
        batch: BatchedExample,
        gaussians: Gaussians,
        global_step: int,
        is_target: bool = True
    ) -> tuple[Float[Tensor, ""], dict[str, Float[Tensor, ""]]]:
        imgtype = "target" if is_target else "context"
        context_or_target_imgs = batch.get(imgtype)
        if context_or_target_imgs is None:
            return torch.tensor(0.0, device=prediction.color.device), {}

        face_bbox = _get_face_bbox_tensor(batch, imgtype)
        if face_bbox is None:
            return torch.tensor(0.0, device=prediction.color.device), {}
        else:
            face_bbox = face_bbox.squeeze(1)

        pred_color = prediction.color
        gt_image = context_or_target_imgs["image"] if is_target else context_or_target_imgs["image_gt"]
        if pred_color.shape != gt_image.shape:
            return torch.tensor(0.0, device=prediction.color.device), {}

        B, V = pred_color.shape[0], pred_color.shape[1]
        device = pred_color.device
        zero = torch.tensor(0.0, device=device)

        # Crop to face regions
        pred_crops, valid = _crop_faces_roi(pred_color, face_bbox, output_size=self.face_crop_size)
        gt_crops, _ = _crop_faces_roi(gt_image, face_bbox, output_size=self.face_crop_size)
        n_valid = valid.sum().item()
        if n_valid == 0:
            return zero, {}

        # MSE on valid face crops only
        sublosses = {}
        mse_loss = zero.clone()
        if self.cfg.mse.weight != 0:
            # Mask invalid crops so they don't contribute
            valid_flat = rearrange(valid, "b v -> (b v)")
            diff = (pred_crops - gt_crops) ** 2
            diff = diff * valid_flat.view(-1, 1, 1, 1)
            mse_loss = self.cfg.mse.weight * diff.sum() / (n_valid * pred_crops.shape[1] * (self.face_crop_size ** 2) + 1e-8)
            sublosses["mse"] = mse_loss


        # LPIPS on valid face crops only
        lpips_loss = zero.clone()
        if self.cfg.lpips.weight != 0 and global_step >= self.cfg.lpips.apply_after_step:
            valid_flat = rearrange(valid, "b v -> (b v)")
            lpips_per = self.lpips_net.forward(pred_crops, gt_crops, normalize=True)
            lpips_per = lpips_per.squeeze(-1).squeeze(-1).squeeze(-1)
            lpips_loss = (lpips_per * valid_flat).sum() / (n_valid + 1e-8)
            lpips_loss = self.cfg.lpips.weight * lpips_loss
            sublosses["lpips"] = lpips_loss

        # ArcFace: 1 - cos_sim(pred_emb, gt_emb) when embedder and gt embeddings present
        arcface_loss = zero.clone() # default, this is not implemented
        if self.cfg.arcface.weight != 0 and self.arcface_embedder is not None:
            gt_emb = context_or_target_imgs.get("arcface_mean_embedding")
            if gt_emb is not None and gt_emb.numel() > 0:
                pred_emb = self.arcface_embedder(pred_crops)
                if pred_emb is not None:
                    gt_emb = gt_emb.float()
                    if gt_emb.dim() == 3:
                        gt_emb = rearrange(gt_emb, "b v d -> (b v) d")
                    if pred_emb.shape[0] == gt_emb.shape[0]:
                        pred_emb = F.normalize(pred_emb.float(), p=2, dim=-1)
                        gt_emb = F.normalize(gt_emb, p=2, dim=-1)
                        cos = (pred_emb * gt_emb).sum(dim=-1)
                        valid_flat = rearrange(valid, "b v -> (b v)")
                        arcface_per = (1.0 - cos) * valid_flat
                        arcface_loss = self.cfg.arcface.weight * arcface_per.sum() / (n_valid + 1e-8)
                        sublosses["arcface"] = arcface_loss

        total = mse_loss + lpips_loss + arcface_loss
        return total, sublosses
