import json
from dataclasses import dataclass
from functools import cached_property
from io import BytesIO
from pathlib import Path
from typing import Literal
import numpy as np
import os
from smplx import SMPLX
import cv2

import torch
import torchvision.transforms as tf
from einops import rearrange, repeat
from jaxtyping import Float, UInt8
from PIL import Image
from torch import Tensor
from torch.utils.data import IterableDataset

from ..geometry.projection import get_fov
from .dataset import DatasetCfgCommon
from .shims.augmentation_shim import apply_augmentation_shim
from .shims.crop_shim import apply_crop_shim
from .shims.color_jitter_shim import apply_color_jitter_shim
from .types import Stage
from .view_sampler import ViewSampler
from ..misc.cam_utils import camera_normalization
from ..misc.body_utils import get_canonical_tfms, get_canonical_global_tfms, body_pose_to_body_RTs, get_global_RTs, apply_lbs_to_means


TRAIN_FRAME_ORDERS = []
for i in range(16):
    TRAIN_FRAME_ORDERS.append(i)
    TRAIN_FRAME_ORDERS.append(16 + i * 3)
    TRAIN_FRAME_ORDERS.append(16 + i * 3 + 1)
    TRAIN_FRAME_ORDERS.append(16 + i * 3 + 2)


@dataclass
class DatasetTHumanCfg(DatasetCfgCommon):
    name: str
    roots: list[Path]
    baseline_min: float
    baseline_max: float
    max_fov: float
    make_baseline_1: bool
    augment: bool
    augment_color_jitter: bool
    relative_pose: bool
    skip_bad_shape: bool
    template_image_shape: list[int]
    load_template_uv: bool = False
    load_da_pose: bool = False
    load_supervision: bool = False
    load_lbs_weights: bool = False
    sample_rate: float = 1.0
    noise_scale: float = 0.0
    load_inconsistent_images: bool = False
    vary_poses : bool = False  # if True, uses different body poses for each context view


@dataclass
class DatasetTHumanCfgWrapper:
    thuman: DatasetTHumanCfg


class DatasetTHuman(IterableDataset):
    cfg: DatasetTHumanCfg
    stage: Stage
    view_sampler: ViewSampler

    to_tensor: tf.ToTensor
    chunks: list[Path]
    near: float = 0.1
    far: float = 100.0

    def __init__(
        self,
        cfg: DatasetTHumanCfg,
        stage: Stage,
        view_sampler: ViewSampler,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.stage = stage
        self.view_sampler = view_sampler
        self.to_tensor = tf.ToTensor()

        # Collect chunks.
        self.chunks = []
        for root in cfg.roots:
            root = root / self.data_stage
            root_chunks = sorted(
                [path for path in root.iterdir() if path.suffix == ".torch"]
            )
            self.chunks.extend(root_chunks)
        if self.cfg.overfit_to_scene is not None:
            chunk_path = self.index[self.cfg.overfit_to_scene]
            self.chunks = [chunk_path] * len(self.chunks)

        template_shape = cfg.template_image_shape[0]
        if self.cfg.load_template_uv:
            if self.cfg.load_da_pose:
                suffix = "_da"
            else:
                suffix = ""
            self.template_mask = torch.tensor(np.load(f'assets/templates/mask_res{template_shape}{suffix}.npy')).float()
            self.template_3d = torch.tensor(np.load(f'assets/templates/xyz_res{template_shape}{suffix}.npy')).float()
            self.template_lbs_weights = torch.tensor(
                np.load(f'assets/templates/lbs_weights_res{template_shape}{suffix}.npy')).float()

            self.template_stds = []
            self.template_means = []

            D = self.template_lbs_weights.shape[-1]
            for d in range(D):
                valid = self.template_lbs_weights[..., d].reshape(-1) > 0
                pts = self.template_3d.reshape(-1, 3)[valid]
                weights = self.template_lbs_weights[..., d].reshape(-1)[valid]
                cov = torch.cov(pts.T, aweights=weights)
                L, V = torch.linalg.eig(cov)
                L = torch.sqrt(torch.real(L))
                V = torch.real(V)

                self.template_stds.append(V @ torch.diag(L))
                self.template_means.append(torch.sum(pts * weights[:, None], dim=0) / torch.sum(weights))

            self.template_stds = torch.stack(self.template_stds)
            self.template_means = torch.stack(self.template_means)

            MODEL_DIR = "datasets/smplx/"
            smplx_model = SMPLX(model_path=os.path.join(MODEL_DIR, 'SMPLX_MALE.npz'))
            canonical_poses = smplx_model.body_pose.detach()
            canonical_poses.requires_grad = False
            if self.cfg.load_da_pose:
                canonical_poses[0, 2] = 1.0
                canonical_poses[0, 5] = -1.0
            self.template_tpose_joints = smplx_model(pose=canonical_poses).joints.detach().cpu()[0, :55]
            # UV projection (uv_map / uv_valid) is computed on GPU in the encoder data shim, not in the dataloader.

    def shuffle(self, lst: list) -> list:
        indices = torch.randperm(len(lst))
        return [lst[x] for x in indices]

    def __iter__(self):
        # Chunks must be shuffled here (not inside __init__) for validation to show
        # random chunks.
        if self.stage in ("train", "val"):
            self.chunks = self.shuffle(self.chunks)

        # When testing, the data loaders alternate chunks.
        worker_info = torch.utils.data.get_worker_info()
        if self.stage == "test" and worker_info is not None:
            self.chunks = [
                chunk
                for chunk_index, chunk in enumerate(self.chunks)
                if chunk_index % worker_info.num_workers == worker_info.id
            ]

        for chunk_path in self.chunks:
            # Load the chunk.
            chunk = torch.load(chunk_path, weights_only=False)

            if self.cfg.overfit_to_scene is not None:
                # Support both old format (key) and ID-grouped format (id or scene in keys)
                def matches_overfit(x):
                    if x.get("key") == self.cfg.overfit_to_scene:
                        return True
                    if x.get("id") == self.cfg.overfit_to_scene:
                        return True
                    return self.cfg.overfit_to_scene in x.get("keys", [])
                item = [x for x in chunk if matches_overfit(x)]
                assert len(item) == 1, (
                    f"overfit_to_scene={self.cfg.overfit_to_scene} matched {len(item)} items in chunk. "
                    "With ID-grouped shards, use the identity id (e.g. id_0000) or a constituent scene key."
                )
                chunk = item * len(chunk)

            if self.stage in ("train", "val"):
                chunk = self.shuffle(chunk)

            # check sharded dataset type
            id_grouped_format = 'keys' in chunk[0]
            if not id_grouped_format and self.cfg.vary_poses:
                print("WARNING: vary_poses=True but dataset not in ID-grouped format. Treating as vary_pose=False.")
                id_grouped_format = False

            for example in chunk:
                if id_grouped_format: # new
                    if self.cfg.vary_poses:
                        yield from self.build_batch_id(example)
                    else:
                        # new format without varying pose, just select one scene at random
                        subexamples: dict = example['scenes']
                        all_keys = list(subexamples.keys())
                        chosen = all_keys[torch.randint(0, len(all_keys), []).item()]
                        yield from self.build_batch(subexamples[chosen])
                else: # original format
                    yield from self.build_batch(example)


    def build_batch(self, example: dict) -> dict:
        # implementation before 'vary_pose', no random sampling
        # should work with both old sharded format and new ID-grouped format
        extrinsics, intrinsics = self.convert_poses(example["cameras"])
        Rs, Ts = self.convert_human_poses(example["poses"])
        Rs_tpose, Ts_tpose = self.convert_human_poses(example["poses_tpose"])
        if "poses_angles_all" in example:
            poses_angles = example["poses_angles_all"].detach().numpy()
            noise_pose = np.random.normal(scale=self.cfg.noise_scale, size=poses_angles[0].shape).astype(np.float32)
            poses_angles += noise_pose[None]
            Rs, Ts = [], []
            for pose_angles, tpose_joints in zip(poses_angles, example["tposes_joints"].detach().cpu()):
                cnl_gtfms = get_canonical_global_tfms(tpose_joints, use_smplx=True)
                dst_Rs, dst_Ts = body_pose_to_body_RTs(
                    pose_angles, tpose_joints, use_smplx=True
                )
                global_Rs, global_Ts = get_global_RTs(
                    cnl_gtfms, dst_Rs, dst_Ts,
                    use_smplx=True)
                Rs.append(global_Rs)
                Ts.append(global_Ts)
            Rs = torch.tensor(np.stack(Rs))
            Ts = torch.tensor(np.stack(Ts))

        tpose_joints = example["tposes_joints"].reshape(-1, 55, 3)
        cnl_Rs, cnl_Ts = get_canonical_tfms(self.template_tpose_joints, tpose_joints[0], use_smplx=True)
        cnl_Rs = cnl_Rs[None].repeat(tpose_joints.shape[0], 1, 1, 1)
        cnl_Ts = cnl_Ts[None].repeat(tpose_joints.shape[0], 1, 1)

        scene = example["key"]
        # if self.cfg.load_lbs_weights and self.stage == "train":
        #     lbs_weights = example["lbs_weights"]

        try:
            context_indices, target_indices, overlap = self.view_sampler.sample(
                scene,
                extrinsics,
                intrinsics,
            )
        except ValueError:
            # Skip because the example doesn't have enough frames.
            return None

        # Skip the example if the field of view is too wide.
        if (get_fov(intrinsics).rad2deg() > self.cfg.max_fov).any():
            return None

        # Load the images.
        context_images = [
            example["images"][index.item()] for index in context_indices
        ]
        context_images = self.convert_images(context_images)
        context_images_gt = context_images
        context_masks = [
            example["masks"][index.item()] for index in context_indices
        ]

        context_masks = self.convert_masks(context_masks)
        
        context_face_bboxes = []
        if "face_bbox" in example:
            context_face_bboxes = [example["face_bbox"][index.item()] for index in context_indices]
            context_face_bboxes = torch.stack(context_face_bboxes)

        context_face_confs = []
        if "face_conf" in example:
            context_face_confs = [example["face_conf"][index.item()] for index in context_indices]
            context_face_confs = torch.stack(context_face_confs)

        # NOTE: 'target' images in this case refer to novel views not used as context
        # but still influences the loss
        target_images = [
            example["images"][index.item()] for index in target_indices
        ]  # for iclight, these will be clones of original GT images
        target_images = self.convert_images(target_images)
        target_masks = [
            example["masks"][index.item()] for index in target_indices
        ] 
        target_masks = self.convert_masks(target_masks)

        target_face_bboxes = []
        if "face_bbox" in example:
            target_face_bboxes = [example["face_bbox"][index.item()] for index in target_indices]
            target_face_bboxes = torch.stack(target_face_bboxes)

        target_face_confs = []
        if "face_conf" in example:
            target_face_confs = [example["face_conf"][index.item()] for index in target_indices]
            target_face_confs = torch.stack(target_face_confs)

        # arcface embedding per view
        context_arcface_embeddings = []
        if "arcface_embedding" in example and example["arcface_embedding"] is not None: # need these for context view supervision
            context_arcface_embeddings = [example["arcface_embedding"][index.item()] for index in context_indices]
            context_arcface_embeddings = torch.stack(context_arcface_embeddings)

        target_arcface_embeddings = []
        if "arcface_embedding" in example and example["arcface_embedding"] is not None: # need for target view supervision
            target_arcface_embeddings = [example["arcface_embedding"][index.item()] for index in target_indices]
            target_arcface_embeddings = torch.stack(target_arcface_embeddings)

        ref_mask = torch.zeros(len(context_indices), dtype=torch.bool)
        ref_mask[torch.randint(0, len(context_indices), [1])] = True

        # load inconsistent images for context views for all non-reference views
        if self.cfg.load_inconsistent_images and "ic_images" in example:
            # for now, these share the same masks as context images (possibly change later)
            ic_images = [
                example["ic_images"][index.item()] for index in context_indices
            ]
            ic_images = self.convert_images(ic_images)
            # apply mask and change the context images to inconsistent images
            context_images = torch.stack([context_images[i] if ref_mask[i] else ic_images[i] for i in range(len(context_indices))])
            del ic_images
            # if we ever train on inconsistent POSES, then need a separate 'ic_masks' key for these
        elif self.cfg.load_inconsistent_images:
            print(f"[DEBUG] load_inconsistent_images=True but 'ic_images' not found in example for scene {scene}")

        # Skip the example if the images don't have the right shape.
        context_image_invalid = context_images.shape[1:] != (3, *self.cfg.original_image_shape)
        target_image_invalid = target_images.shape[1:] != (3, *self.cfg.original_image_shape)

        # context_masks = (context_masks * 255.).cpu().numpy().astype(np.uint8)
        # kernel = np.ones((3,3), np.uint8)
        # context_masks = [cv2.erode(context_mask, kernel, iterations=1) for context_mask in context_masks]
        # context_masks = torch.tensor(np.stack(context_masks)) / 255.

        context_images, target_images, bgcolor = self.set_bgcolor(
            context_images, context_masks,
            target_images, target_masks,
            self.cfg.background_color)

        # also set the background color for image_gt
        context_images_gt = context_images_gt * context_masks.unsqueeze(1) + bgcolor[None, :, None, None] * (1 - context_masks.unsqueeze(1))

        if self.cfg.skip_bad_shape and (context_image_invalid or target_image_invalid):
            print(
                f"Skipped bad example {example['key']}. Context shape was "
                f"{context_images.shape} and target shape was "
                f"{target_images.shape}."
            )
            return None

        # Resize the world to make the baseline 1.
        context_extrinsics = extrinsics[context_indices]
        if self.cfg.make_baseline_1:
            a, b = context_extrinsics[0, :3, 3], context_extrinsics[-1, :3, 3]
            scale = (a - b).norm()
            if scale < self.cfg.baseline_min or scale > self.cfg.baseline_max:
                print(
                    f"Skipped {scene} because of baseline out of range: "
                    f"{scale:.6f}"
                )
                return None
            extrinsics[:, :3, 3] /= scale
        else:
            scale = 1

        if self.cfg.relative_pose:
            extrinsics = camera_normalization(extrinsics[context_indices][0:1], extrinsics)

        # build batch

        example = {
            "context": {
                "extrinsics": extrinsics[context_indices],
                "intrinsics": intrinsics[context_indices],
                "Rs": Rs[context_indices],
                "Ts": Ts[context_indices],
                "Rs_tpose": Rs_tpose[context_indices],
                "Ts_tpose": Ts_tpose[context_indices],
                "cnl_Rs": cnl_Rs[context_indices],
                "cnl_Ts": cnl_Ts[context_indices],
                # "lbs_weights": lbs_weights[context_indices],
                "image": context_images,
                "image_gt": context_images_gt, # for context, need these GTs.
                "mask": context_masks,
                "mask_gt": context_masks,
                "near": self.get_bound("near", len(context_indices)) / scale,
                "far": self.get_bound("far", len(context_indices)) / scale,
                "index": context_indices,
                "overlap": overlap,
                "use_smplx": True,
                "ref_mask": ref_mask,
                "face_bbox": context_face_bboxes,
                "face_conf": context_face_confs,
                "arcface_embedding": context_arcface_embeddings, # training only (V,512)
                # Per-subject identity-shaped rest-pose mesh for UV projection.
                # vertex is the fitted canonical mesh for this person (encodes betas implicitly).
                # Only included when present in the chunk (requires regenerated chunks via convert_thuman.py).
                **({
                    "canonical_vertex": torch.tensor(example["vertex"], dtype=torch.float32),
                    "canonical_lbs_weights": torch.tensor(example["lbs_weights"], dtype=torch.float32),
                } if "vertex" in example and "lbs_weights" in example else {}),
            },
            "target": {
                "extrinsics": extrinsics[target_indices],
                "intrinsics": intrinsics[target_indices],
                "Rs": Rs[target_indices],
                "Ts": Ts[target_indices],
                "Rs_tpose": Rs_tpose[target_indices],
                "Ts_tpose": Ts_tpose[target_indices],
                "cnl_Rs": cnl_Rs[target_indices],
                "cnl_Ts": cnl_Ts[target_indices],
                # "lbs_weights": lbs_weights[target_indices],
                "image": target_images, # supervising views
                "mask": target_masks,
                "near": self.get_bound("near", len(target_indices)) / scale,
                "far": self.get_bound("far", len(target_indices)) / scale,
                "index": target_indices,
                "use_smplx": True,
                "ref_mask": ref_mask, # redundant and can remove, but why not
                "face_bbox": target_face_bboxes,
                "face_conf": target_face_confs,
                "arcface_mean_embedding": target_arcface_embeddings, # for training only! (V,512)
            },
            "scene": scene,
            "bgcolor": bgcolor,
            "tpose_joints": tpose_joints[0],
        }

        if self.cfg.load_template_uv:
            example["context"].update({
                "template_mask": self.template_mask,
                "template_3d": self.template_3d,
                "template_lbs_weights": self.template_lbs_weights,
                "template_stds": self.template_stds,
                "template_means": self.template_means,
            })

        # uv_map / uv_valid are computed on GPU in the encoder data shim (see encoder_template_uv_face.get_data_shim).

        if self.cfg.load_supervision:
            # test only; not ready
            supervisions = []
            for idx in context_indices:
                supervision = np.load(f'/home/jw116/codes/generalizable_point-based-human/log/ghg_view3_subdivide_deepf_pointtransformer_iter3_fb_tf_lpips0.5_adam_lr1e-4_5e-5_nolpips_ssim_coeff1.0_lap100.0/supervision/view/scene_{scene}_frame_{idx:06d}.npy')
                supervisions.append(supervision)
            supervisions = np.stack([supervisions[0][0]] + [supervision[1] for supervision in supervisions])
            supervisions = torch.from_numpy(supervisions).float()
            example["context"]["supervisions"] = supervisions

        if self.cfg.load_lbs_weights and self.stage == "train":
            lbs_weights_dir = self.cfg.roots[0] / 'lbs_weights_supervisions' / f"{scene}"

            lbs_weights = []
            for idx in context_indices:
                if os.path.exists(f'{lbs_weights_dir}/frame_{TRAIN_FRAME_ORDERS[idx]:06d}.npy'):
                    lbs_weights_single = np.load(
                        f'{lbs_weights_dir}/frame_{TRAIN_FRAME_ORDERS[idx]:06d}.npy')
                else:
                    lbs_weights_single = np.load(
                        f'{lbs_weights_dir}/frame_{TRAIN_FRAME_ORDERS[idx]:06d}.npz')['lbs_weights']
                lbs_weights.append(lbs_weights_single)
            example["context"].update({
                "lbs_weights": torch.from_numpy(np.stack(lbs_weights)).float()
            })
            # example["context"].update({
            #     "lbs_weights": lbs_weights[context_indices]
            # })
        if self.stage == "train" and self.cfg.augment:
            example = apply_augmentation_shim(example)
            if self.cfg.augment_color_jitter:
                example = apply_color_jitter_shim(example)
        shimmed_data = apply_crop_shim(example, tuple(self.cfg.input_image_shape))
        yield shimmed_data

    def build_batch_id(self, meta_example: dict):
        # Multi-scene sampling: context views can be from any scene; target views are
        # from the same scene as the reference context view only (other views of that key).
        # The view_sampler is not used since there is no single shared camera layout.

        scenes: dict = meta_example["scenes"]       # {scene_key: example_dict, ...}
        scene_keys: list = list(scenes.keys())
        n_scenes = len(scene_keys)

        num_ctx = self.view_sampler.num_context_views
        num_tgt = self.view_sampler.num_target_views

        # extract per-frame data
        def get_frame_data(scene: dict, frame_idx: int) -> dict:
            """Return processed tensors for a single frame of a single scene."""
            cam = scene["cameras"][frame_idx : frame_idx + 1]
            ext, intr = self.convert_poses(cam)   # [1, 4, 4] / [1, 3, 3]

            poses_tpose_raw = scene["poses_tpose"][frame_idx : frame_idx + 1]
            Rs_tpose, Ts_tpose = self.convert_human_poses(poses_tpose_raw)

            tpose_joints_all = scene["tposes_joints"].reshape(-1, 55, 3)
            tpose_joints_frame = tpose_joints_all[frame_idx]   # [55, 3]

            if "poses_angles_all" in scene:
                pose_angles = scene["poses_angles_all"][frame_idx].detach().numpy().copy()
                noise = np.random.normal(
                    scale=self.cfg.noise_scale, size=pose_angles.shape
                ).astype(np.float32)
                pose_angles += noise
                cnl_gtfms = get_canonical_global_tfms(tpose_joints_frame, use_smplx=True)
                dst_Rs, dst_Ts = body_pose_to_body_RTs(
                    pose_angles, tpose_joints_frame, use_smplx=True
                )
                g_Rs, g_Ts = get_global_RTs(cnl_gtfms, dst_Rs, dst_Ts, use_smplx=True)
                Rs = torch.tensor(g_Rs[None])   # [1, joints, 3, 3]
                Ts = torch.tensor(g_Ts[None])   # [1, joints, 3]
            else:
                Rs, Ts = self.convert_human_poses(scene["poses"][frame_idx : frame_idx + 1])

            cnl_Rs_v, cnl_Ts_v = get_canonical_tfms(
                self.template_tpose_joints, tpose_joints_all[0], use_smplx=True
            )
            cnl_Rs_v = cnl_Rs_v[None]   # [1, joints, 3, 3]
            cnl_Ts_v = cnl_Ts_v[None]   # [1, joints, 3]

            face_bbox = (
                scene["face_bbox"][frame_idx]
                if scene.get("face_bbox") is not None else None
            )
            face_conf = (
                scene["face_conf"][frame_idx]
                if scene.get("face_conf") is not None else None
            )
            arcface = (
                scene["arcface_embedding"][frame_idx]
                if scene.get("arcface_embedding") is not None else None
            )

            vertex = scene["vertex"] if scene.get("vertex") is not None else None
            lbs_weights = scene["lbs_weights"] if scene.get("lbs_weights") is not None else None


            element = {
                "extrinsics":  ext,          # [1, 4, 4]
                "intrinsics":  intr,         # [1, 3, 3]
                "Rs":          Rs,           # [1, J, 3, 3]
                "Ts":          Ts,           # [1, J, 3]
                "Rs_tpose":    Rs_tpose,     # [1, J, 3, 3]
                "Ts_tpose":    Ts_tpose,     # [1, J, 3]
                "cnl_Rs":      cnl_Rs_v,     # [1, J, 3, 3]
                "cnl_Ts":      cnl_Ts_v,     # [1, J, 3]
                "image":       scene["images"][frame_idx],
                "image_gt":    scene["images"][frame_idx],  # note that GTs 
                "mask":        scene["masks"][frame_idx],
                "face_bbox":   face_bbox,
                "face_conf":   face_conf,
                "arcface":     arcface,
                "tpose_joints": tpose_joints_all[0],   # [55, 3]  canonical for this scene
                "key":         scene["key"],
                "frame_idx":   frame_idx,
                "vertex":      vertex,
                "lbs_weights": lbs_weights,
            }

            if self.cfg.load_inconsistent_images and scene.get("ic_images") is not None:
                # need to replace with ref later!
                element["image"] = scene["ic_images"][frame_idx]

            return element

        def sample_context_views(n: int) -> list[dict]:
            result = []
            for _ in range(n):
                sk = scene_keys[torch.randint(0, n_scenes, []).item()]  # which scene in shard
                fi = torch.randint(0, len(scenes[sk]["images"]), []).item()  # which frame in scene
                result.append(get_frame_data(scenes[sk], fi))
            return result

        ctx_data = sample_context_views(num_ctx)  # build context views with ic_images and GTs with varying keys
        ref_idx = torch.randint(0, num_ctx, []).item()  # select reference view
        ref_mask = torch.zeros(num_ctx, dtype=torch.bool)
        ref_mask[ref_idx] = True
        ref_tpose_joints = ctx_data[ref_idx]["tpose_joints"]   # [55, 3]
        # Reference scene: targets and (for non-ref context) aligned cameras/GT/masks come from here.
        ref_key = ctx_data[ref_idx]["key"]
        ref_frame_idx = ctx_data[ref_idx]["frame_idx"]
        ref_scene = scenes[ref_key]

        # Input-aligned masks (before SimVS overwrite): encoder / `context["image"]` use this `mask`.
        # After sync, `d["mask"]` is ref-aligned; we keep both via `mask` vs `mask_gt`.
        ctx_masks_input_align = [d["mask"].clone() for d in ctx_data]

        # SimVS-style alignment: non-reference context slots keep their sampled input `image` (e.g. ic/relit from
        # another key) but cameras, poses, masks, GT, etc. come from ref_scene at the same frame index so
        # supervision and geometry match the reference identity.
        n_ref_frames = len(ref_scene["images"])
        if n_ref_frames < 1:  # skip sceens with only a single frame (ref frame)
            return
        for i in range(num_ctx):
            if i == ref_idx:
                continue
            fi = int(ctx_data[i]["frame_idx"])
            fi = min(fi, n_ref_frames - 1)
            orig_image = ctx_data[i]["image"]
            ctx_data[i] = get_frame_data(ref_scene, fi)
            ctx_data[i]["image"] = orig_image

        n_frames = len(ref_scene["images"])
        other_frame_indices = [i for i in range(n_frames) if i != ref_frame_idx] # get all other ref images excluding ref frame
        if len(other_frame_indices) < 1:
            return  # no other views in this scene for target (should not happen for our datasets)
        n_tgt_sample = min(num_tgt, len(other_frame_indices))  # select target views (or all other frames)
        if n_tgt_sample >= num_tgt:  # if we have surplus, sample randomly
            chosen = torch.randperm(len(other_frame_indices))[:num_tgt].tolist()
            tgt_frame_indices = [other_frame_indices[i] for i in chosen]
        else:  # if we have less, randomly sample until we have num_tgt (will be duplicates)
            chosen = torch.randperm(len(other_frame_indices))[:n_tgt_sample].tolist()
            tgt_frame_indices = [other_frame_indices[i] for i in chosen]
            while len(tgt_frame_indices) < num_tgt:
                tgt_frame_indices.append(other_frame_indices[torch.randint(0, len(other_frame_indices), []).item()])
        tgt_data = [get_frame_data(ref_scene, fi) for fi in tgt_frame_indices]

        # stack context
        context_extrinsics  = torch.cat([d["extrinsics"]  for d in ctx_data])
        context_intrinsics  = torch.cat([d["intrinsics"]  for d in ctx_data])
        context_Rs          = torch.cat([d["Rs"]          for d in ctx_data])
        context_Ts          = torch.cat([d["Ts"]          for d in ctx_data])
        context_Rs_tpose    = torch.cat([d["Rs_tpose"]    for d in ctx_data])
        context_Ts_tpose    = torch.cat([d["Ts_tpose"]    for d in ctx_data])
        context_cnl_Rs      = torch.cat([d["cnl_Rs"]      for d in ctx_data])
        context_cnl_Ts      = torch.cat([d["cnl_Ts"]      for d in ctx_data])

        if (get_fov(context_intrinsics).rad2deg() > self.cfg.max_fov).any():
            return

        context_images = self.convert_images([d["image"] for d in ctx_data])
        if self.cfg.load_inconsistent_images:
            context_images_gt = self.convert_images([d["image_gt"] for d in ctx_data])
        else:
            context_images_gt = context_images.clone()

        context_masks = self.convert_masks(ctx_masks_input_align)
        context_masks_gt = self.convert_masks([d["mask"] for d in ctx_data])

        context_face_bboxes = (
            torch.stack([d["face_bbox"] for d in ctx_data])
            if all(d["face_bbox"] is not None for d in ctx_data) else []
        )
        context_face_confs = (
            torch.stack([d["face_conf"] for d in ctx_data])
            if all(d["face_conf"] is not None for d in ctx_data) else []
        )
        context_arcface = (
            torch.stack([d["arcface"] for d in ctx_data])
            if all(d["arcface"] is not None for d in ctx_data) else []
        )

        # Replace non-reference context views with inconsistent (e.g. relit/LBM) images when enabled
        if self.cfg.load_inconsistent_images:
            try:
                context_images = torch.stack([
                    context_images_gt[i] if ref_mask[i] else context_images[i] for i in range(num_ctx)
                ])
            except Exception as e:
                print(f"Error replacing context images with inconsistent images: {e}")
                return

        # stack targets
        target_extrinsics  = torch.cat([d["extrinsics"]  for d in tgt_data])
        target_intrinsics  = torch.cat([d["intrinsics"]  for d in tgt_data])
        target_Rs          = torch.cat([d["Rs"]          for d in tgt_data])
        target_Ts          = torch.cat([d["Ts"]          for d in tgt_data])
        target_Rs_tpose    = torch.cat([d["Rs_tpose"]    for d in tgt_data])
        target_Ts_tpose    = torch.cat([d["Ts_tpose"]    for d in tgt_data])
        target_cnl_Rs      = torch.cat([d["cnl_Rs"]      for d in tgt_data])
        target_cnl_Ts      = torch.cat([d["cnl_Ts"]      for d in tgt_data])

        target_images  = self.convert_images([d["image_gt"] for d in tgt_data])
        target_masks   = self.convert_masks([d["mask"]  for d in tgt_data])

        target_face_bboxes = (
            torch.stack([d["face_bbox"] for d in tgt_data])
            if all(d["face_bbox"] is not None for d in tgt_data) else []
        )
        target_face_confs = (
            torch.stack([d["face_conf"] for d in tgt_data])
            if all(d["face_conf"] is not None for d in tgt_data) else []
        )
        target_arcface = (
            torch.stack([d["arcface"] for d in tgt_data])
            if all(d["arcface"] is not None for d in tgt_data) else []
        )

        # ── background ────────────────────────────────────────────────────────
        context_images, target_images, bgcolor = self.set_bgcolor(
            context_images, context_masks,
            target_images, target_masks,
            self.cfg.background_color,
        )
        context_images_gt = (
            context_images_gt * context_masks_gt.unsqueeze(1)
            + bgcolor[None, :, None, None] * (1 - context_masks_gt.unsqueeze(1))
        )

        # ── shape check ───────────────────────────────────────────────────────
        ctx_invalid = context_images.shape[1:] != (3, *self.cfg.original_image_shape)
        tgt_invalid = target_images.shape[1:]  != (3, *self.cfg.original_image_shape)
        if self.cfg.skip_bad_shape and (ctx_invalid or tgt_invalid):
            return

        # ── baseline scaling (cameras are in canonical body frame; distances are comparable) ──
        if self.cfg.make_baseline_1:
            a, b = context_extrinsics[0, :3, 3], context_extrinsics[-1, :3, 3]
            scale = (a - b).norm()
            if scale < self.cfg.baseline_min or scale > self.cfg.baseline_max:
                return
            context_extrinsics[:, :3, 3] /= scale
            target_extrinsics[:, :3, 3]  /= scale
        else:
            scale = 1

        if self.cfg.relative_pose:
            all_ext = torch.cat([context_extrinsics, target_extrinsics])
            all_ext = camera_normalization(context_extrinsics[0:1], all_ext)
            context_extrinsics = all_ext[:num_ctx]
            target_extrinsics  = all_ext[num_ctx:]

        # ── assemble batch ────────────────────────────────────────────────────
        scene_name = ctx_data[ref_idx]["key"]
        overlap = torch.zeros(num_ctx, dtype=torch.float32)

        # Vertex / lbs_weights per view in same order as images, intrinsics, extrinsics
        context_vertex_lbs = (
            {
                "canonical_vertex": torch.stack([
                    torch.tensor(d["vertex"], dtype=torch.float32) for d in ctx_data
                ]),
                "canonical_lbs_weights": torch.stack([
                    torch.tensor(d["lbs_weights"], dtype=torch.float32) for d in ctx_data
                ]),
            }
            if all(d["vertex"] is not None for d in ctx_data)
            and all(d["lbs_weights"] is not None for d in ctx_data)
            else {}
        )
        target_vertex_lbs = (
            {
                "canonical_vertex": torch.stack([
                    torch.tensor(tgt_data[0]["vertex"], dtype=torch.float32)
                ] * num_tgt),
                "canonical_lbs_weights": torch.stack([
                    torch.tensor(tgt_data[0]["lbs_weights"], dtype=torch.float32)
                ] * num_tgt),
            }
            if tgt_data
            and tgt_data[0].get("vertex") is not None
            and tgt_data[0].get("lbs_weights") is not None
            else {}
        )

        example = {
            "context": {
                "extrinsics":  context_extrinsics,
                "intrinsics":  context_intrinsics,
                "Rs":          context_Rs,
                "Ts":          context_Ts,
                "Rs_tpose":    context_Rs_tpose,
                "Ts_tpose":    context_Ts_tpose,
                "cnl_Rs":      context_cnl_Rs,
                "cnl_Ts":      context_cnl_Ts,
                "image":       context_images,
                "image_gt":    context_images_gt,
                "mask":        context_masks,
                "mask_gt":     context_masks_gt,
                "near":        self.get_bound("near", num_ctx) / scale,
                "far":         self.get_bound("far",  num_ctx) / scale,
                "index":       torch.tensor([d["frame_idx"] for d in ctx_data], dtype=torch.int64),
                "overlap":     overlap,
                "use_smplx":   True,
                "ref_mask":    ref_mask,
                "face_bbox":   context_face_bboxes,
                "face_conf":   context_face_confs,
                "arcface_embedding": context_arcface,
                **context_vertex_lbs,
            },
            "target": {
                "extrinsics":  target_extrinsics,
                "intrinsics":  target_intrinsics,
                "Rs":          target_Rs,
                "Ts":          target_Ts,
                "Rs_tpose":    target_Rs_tpose,
                "Ts_tpose":    target_Ts_tpose,
                "cnl_Rs":      target_cnl_Rs,
                "cnl_Ts":      target_cnl_Ts,
                "image":       target_images,
                "mask":        target_masks,
                "near":        self.get_bound("near", num_tgt) / scale,
                "far":         self.get_bound("far",  num_tgt) / scale,
                "index":       torch.tensor([d["frame_idx"] for d in tgt_data], dtype=torch.int64),
                "use_smplx":   True,
                "ref_mask":    ref_mask,
                "face_bbox":   target_face_bboxes,
                "face_conf":   target_face_confs,
                "arcface_mean_embedding": target_arcface,
                **target_vertex_lbs,
            },
            "scene":       scene_name,
            "bgcolor":     bgcolor,
            "tpose_joints": ref_tpose_joints,
        }

        if self.cfg.load_template_uv:
            example["context"].update({
                "template_mask":        self.template_mask,
                "template_3d":          self.template_3d,
                "template_lbs_weights": self.template_lbs_weights,
                "template_stds":        self.template_stds,
                "template_means":       self.template_means,
            })

        # lbs_weights_supervisions: one dir per scene key (0000, 0063, ...); load per view in same order as images
        if self.cfg.load_lbs_weights and self.stage == "train":
            lbs_ctx = []
            for d in ctx_data:
                scene_key = d["key"]
                frame_idx = d["frame_idx"]
                lbs_dir = self.cfg.roots[0] / "lbs_weights_supervisions" / scene_key
                fname = f"frame_{TRAIN_FRAME_ORDERS[frame_idx]:06d}"
                if (lbs_dir / f"{fname}.npy").exists():
                    w = np.load(lbs_dir / f"{fname}.npy")
                else:
                    w = np.load(lbs_dir / f"{fname}.npz")["lbs_weights"]
                lbs_ctx.append(w)
            example["context"]["lbs_weights"] = torch.from_numpy(np.stack(lbs_ctx)).float()
            lbs_tgt = []
            for d in tgt_data:
                scene_key = d["key"]
                frame_idx = d["frame_idx"]
                lbs_dir = self.cfg.roots[0] / "lbs_weights_supervisions" / scene_key
                fname = f"frame_{TRAIN_FRAME_ORDERS[frame_idx]:06d}"
                if (lbs_dir / f"{fname}.npy").exists():
                    w = np.load(lbs_dir / f"{fname}.npy")
                else:
                    w = np.load(lbs_dir / f"{fname}.npz")["lbs_weights"]
                lbs_tgt.append(w)
            example["target"]["lbs_weights"] = torch.from_numpy(np.stack(lbs_tgt)).float()

        if self.stage == "train" and self.cfg.augment:
            example = apply_augmentation_shim(example)
            if self.cfg.augment_color_jitter:
                example = apply_color_jitter_shim(example)
        shimmed_data = apply_crop_shim(example, tuple(self.cfg.input_image_shape))
        yield shimmed_data

    def convert_poses(
        self,
        poses: Float[Tensor, "batch 18"],
    ) -> tuple[
        Float[Tensor, "batch 4 4"],  # extrinsics
        Float[Tensor, "batch 3 3"],  # intrinsics
    ]:
        b, _ = poses.shape

        # Convert the intrinsics to a 3x3 normalized K matrix.
        intrinsics = torch.eye(3, dtype=torch.float32)
        intrinsics = repeat(intrinsics, "h w -> b h w", b=b).clone()
        fx, fy, cx, cy = poses[:, :4].T
        intrinsics[:, 0, 0] = fx
        intrinsics[:, 1, 1] = fy
        intrinsics[:, 0, 2] = cx
        intrinsics[:, 1, 2] = cy

        # Convert the extrinsics to a 4x4 OpenCV-style W2C matrix.
        w2c = repeat(torch.eye(4, dtype=torch.float32), "h w -> b h w", b=b).clone()
        w2c[:, :3] = rearrange(poses[:, 6:], "b (h w) -> b h w", h=3, w=4)
        return w2c.inverse(), intrinsics

    def convert_human_poses(
        self,
        poses: Float[Tensor, "batch 660"],
    )-> tuple[
        Float[Tensor, "batch joints 3 3"],  # Rs
        Float[Tensor, "batch joints 3"],  # Ts
    ]:
        Rs = poses[:, :55 * 3 * 3]
        Ts = poses[:, 55 * 3 * 3:]
        Rs = rearrange(Rs, "b (j x y) -> b j x y", j=55, x=3, y=3)
        Ts = rearrange(Ts, "b (j x) -> b j x", j=55, x=3)
        return Rs, Ts

    def convert_images(
        self,
        images: list[UInt8[Tensor, "..."]],
    ) -> Float[Tensor, "batch 3 height width"]:
        torch_images = []
        for image in images:
            image = Image.open(BytesIO(image.numpy().tobytes()))
            torch_images.append(self.to_tensor(image))
        return torch.stack(torch_images)

    def convert_masks(
        self,
        images: list[UInt8[Tensor, "..."]],
    ) -> Float[Tensor, "batch height width"]:
        torch_images = []
        for image in images:
            image = Image.open(BytesIO(image.numpy().tobytes()))
            torch_images.append(self.to_tensor(image)[0])
        return torch.stack(torch_images)

    def set_bgcolor(
        self,
        context_images: Float[Tensor, "batch 3 height width"],
        context_masks: Float[Tensor, "batch height width"],
        target_images: Float[Tensor, "batch 3 height width"],
        target_masks: Float[Tensor, "batch height width"],
        bgcolor: list[float],
    ):
        if bgcolor == [-1, -1, -1]:
            bgcolor = torch.randint(0, 255, [3], device=context_images.device) / 255.
            bgcolor = bgcolor.float()
        else:
            bgcolor = torch.tensor(bgcolor).type(context_images.dtype)

        context_images = context_images * context_masks.unsqueeze(1) + bgcolor[None, :, None, None] * (1 - context_masks.unsqueeze(1))
        target_images = target_images * target_masks.unsqueeze(1) + bgcolor[None, :, None, None] * (1 - target_masks.unsqueeze(1))
        return context_images, target_images, bgcolor

    def get_bound(
        self,
        bound: Literal["near", "far"],
        num_views: int,
    ) -> Float[Tensor, " view"]:
        value = torch.tensor(getattr(self, bound), dtype=torch.float32)
        return repeat(value, "-> v", v=num_views)

    @property
    def data_stage(self) -> Stage:
        if self.cfg.overfit_to_scene is not None:
            return "test"
        if self.stage == "val":
            return "test"
        return self.stage

    @cached_property
    def index(self) -> dict[str, Path]:
        merged_index = {}
        data_stages = [self.data_stage]
        if self.cfg.overfit_to_scene is not None:
            data_stages = ("test", "train")
        for data_stage in data_stages:
            for root in self.cfg.roots:
                # Load the root's index.
                with (root / data_stage / "index.json").open("r") as f:
                    index = json.load(f)
                index = {k: Path(root / data_stage / v) for k, v in index.items()}

                # The constituent datasets should have unique keys.
                assert not (set(merged_index.keys()) & set(index.keys()))

                # Merge the root's index into the main index.
                merged_index = {**merged_index, **index}
        return merged_index

    # def __len__(self) -> int:
    #     return len(self.index.keys())