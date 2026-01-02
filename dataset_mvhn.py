"""
Dataset wrapper for MVHumanNet to match THuman dataset format.
Transforms MVHumanNet dataloader output to be compatible with the model.
"""

import torch
import numpy as np
import os
from torch.utils.data import Dataset
from typing import Optional
from dataclasses import dataclass
from pathlib import Path
from smplx import SMPLX

from dataloader import MVHumanNetDataset, custom_collate


@dataclass
class DatasetMVHNCfg:
    """Configuration for MVHumanNet dataset to match THuman interface."""
    name: str = "mvhn"
    root_dir: str = "/path/to/mvhumannet"
    latents_dir: Optional[str] = None
    preload_path: str = None
    num_images: int = 3  # number of context views
    num_target_images: int = 1  # number of target views
    data_limit: Optional[int] = None
    only_include: Optional[list] = None
    exclude: Optional[list] = None
    step_size: int = 60
    random_crop: bool = False
    maximal_crop: bool = False
    white_background: bool = False
    crop_padding: int = 60
    use_inconsistent: bool = True  # Use iclight/infu for non-reference frames
    random_crop_prob: float = 0.3
    ic_sampling_prob: float = 0.7
    iclight_dataset_path: Optional[str] = None
    infu_dataset_path: Optional[str] = None
    face_bbox_dir: Optional[str] = None
    arcface_embeddings_dir: Optional[str] = None
    fixed_sampling_ids: Optional[list] = None
    use_sapiens_conditioning: Optional[list] = None
    sapiens_segmentation_channels_to_use: list = None
    force_face_ref: bool = False
    
    # Camera parameters
    near: float = 0.1
    far: float = 100.0
    
    # Background color: [R, G, B] in [0, 1] or [-1, -1, -1] for random
    background_color: list = None
    
    # Image shape
    original_image_shape: list = None  # [H, W]
    input_image_shape: list = None  # [H, W]
    
    # Template parameters (for canonical space rendering)
    load_template_uv: bool = False
    template_image_shape: list = None  # [H, W] for template resolution
    load_da_pose: bool = False  # Load DA-pose templates
    load_supervision: bool = False
    load_lbs_weights: bool = False


@dataclass
class DatasetMVHNCfgWrapper:
    mvhn: DatasetMVHNCfg


class DatasetMVHN(Dataset):
    """
    MVHumanNet dataset wrapper that outputs data in THuman format.
    
    This dataset:
    - Samples 'n' images (default 3) for context
    - Keeps one as reference (consistent ground truth)
    - Others can be inconsistent data (iclight/infu)
    - Target views are always ground truth
    
    Note: Unlike THuman which uses IterableDataset for chunk-based loading,
    this uses regular Dataset since MVHumanNetDataset is a regular indexed dataset.
    """
    
    def __init__(
        self,
        cfg: DatasetMVHNCfg,
        stage: str,
        view_sampler=None,  # Not used for MVHN, but kept for interface compatibility
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.stage = stage
        self.view_sampler = view_sampler
        
        # Set defaults
        if self.cfg.background_color is None:
            self.cfg.background_color = [0, 0, 0]  # Black background
        if self.cfg.original_image_shape is None:
            self.cfg.original_image_shape = [576, 576]
        if self.cfg.input_image_shape is None:
            self.cfg.input_image_shape = [576, 576]
        if self.cfg.sapiens_segmentation_channels_to_use is None:
            self.cfg.sapiens_segmentation_channels_to_use = []
        if self.cfg.template_image_shape is None:
            self.cfg.template_image_shape = [1024, 1024]  # Default template resolution
        
        # Load static template data (same as THuman dataset)
        if self.cfg.load_template_uv:
            template_shape = self.cfg.template_image_shape[0]
            if self.cfg.load_da_pose:
                suffix = "_da"
            else:
                suffix = ""
            
            # Load template mask, 3D points, and LBS weights
            self.template_mask = torch.tensor(
                np.load(f'assets/templates/mask_res{template_shape}{suffix}.npy')
            ).float()
            self.template_3d = torch.tensor(
                np.load(f'assets/templates/xyz_res{template_shape}{suffix}.npy')
            ).float()
            self.template_lbs_weights = torch.tensor(
                np.load(f'assets/templates/lbs_weights_res{template_shape}{suffix}.npy')
            ).float()
            
            # Compute statistics for each joint
            self.template_stds = []
            self.template_means = []
            
            D = self.template_lbs_weights.shape[-1]  # Number of joints
            for d in range(D):
                valid = self.template_lbs_weights[..., d].reshape(-1) > 0
                pts = self.template_3d.reshape(-1, 3)[valid]
                weights = self.template_lbs_weights[..., d].reshape(-1)[valid]
                cov = torch.cov(pts.T, aweights=weights)
                L, V = torch.linalg.eig(cov)
                L = torch.sqrt(torch.real(L))
                V = torch.real(V)
                
                self.template_stds.append(V @ torch.diag(L))
                self.template_means.append(
                    torch.sum(pts * weights[:, None], dim=0) / torch.sum(weights)
                )
            
            self.template_stds = torch.stack(self.template_stds)
            self.template_means = torch.stack(self.template_means)
            
            # Load SMPLX model for T-pose joints
            MODEL_DIR = "datasets/smplx/"
            smplx_model = SMPLX(model_path=os.path.join(MODEL_DIR, 'SMPLX_MALE.npz'))
            canonical_poses = smplx_model.body_pose.detach()
            canonical_poses.requires_grad = False
            if self.cfg.load_da_pose:
                canonical_poses[0, 2] = 1.0
                canonical_poses[0, 5] = -1.0
            self.template_tpose_joints = smplx_model(pose=canonical_poses).joints.detach().cpu()[0, :55]
        
        # Initialize the MVHumanNet dataset
        self.dataset = MVHumanNetDataset(
            root_dir=self.cfg.root_dir,
            num_images=self.cfg.num_images,
            latents_dir=self.cfg.latents_dir,
            transforms=None,
            pre_scale_intrinsics=0.5,
            data_limit=self.cfg.data_limit,
            only_include=self.cfg.only_include,
            exclude=self.cfg.exclude,
            random_crop=self.cfg.random_crop,
            maximal_crop=self.cfg.maximal_crop,
            white_background=self.cfg.white_background,
            step_size=self.cfg.step_size,
            preload_path=self.cfg.preload_path,
            iclight_dataset_path=self.cfg.iclight_dataset_path,
            infu_dataset_path=self.cfg.infu_dataset_path,
            face_bbox_dir=self.cfg.face_bbox_dir,
            arcface_embeddings_dir=self.cfg.arcface_embeddings_dir,
            crop_padding=self.cfg.crop_padding,
            use_inconsistent=self.cfg.use_inconsistent,
            random_crop_prob=self.cfg.random_crop_prob,
            ic_sampling_prob=self.cfg.ic_sampling_prob,
            fixed_sampling_ids=self.cfg.fixed_sampling_ids,
            use_sapiens_conditioning=self.cfg.use_sapiens_conditioning,
            sapiens_segmentation_channels_to_use=self.cfg.sapiens_segmentation_channels_to_use,
            force_face_ref=self.cfg.force_face_ref,
        )
        
    def __len__(self):
        """Return the length of the dataset."""
        return len(self.dataset)
    
    def __getitem__(self, idx):
        """Get a single example and transform it to THuman format."""
        max_attempts = min(10, len(self))  # Try up to 10 indices or dataset length
        
        for attempt in range(max_attempts):
            try:
                current_idx = (idx + attempt) % len(self)
                example = self.dataset[current_idx]
                
                if example is None:
                    continue
                
                # Transform to THuman format
                transformed = self.transform_to_thuman_format(example)
                
                if transformed is None:
                    continue
                
                return transformed
                
            except Exception as e:
                # Log the error but try next index
                if attempt == 0:  # Only print on first attempt to avoid spam
                    print(f"Warning: Error loading index {current_idx}: {e}. Trying next index...")
                continue
        
        # If we exhausted all attempts, raise an error
        raise RuntimeError(f"Failed to load a valid example after {max_attempts} attempts starting from index {idx}")
    
    def transform_to_thuman_format(self, mvhn_batch: dict) -> dict:
        """
        Transform MVHumanNet batch to THuman format.
        
        MVHumanNet batch contains:
        - frames: [N, 3, H, W] - the images (reference + potentially inconsistent)
        - frames_masks: [N, H, W] - masks
        - c2w: [N, 4, 4] - camera to world matrices
        - K: [N, 3, 3] - intrinsics
        - mask: [N] - bool tensor indicating which frames are inputs
        - ref_mask: [N] - bool tensor indicating which frame is reference
        - ic_rgb: [N, 3, H, W] - inconsistent RGB images
        - subject_id, timestep, etc.
        
        THuman format needs:
        - context: dict with extrinsics, intrinsics, Rs, Ts, etc., images, masks
        - target: similar to context
        - scene: str
        - bgcolor: tensor
        """
        
        try:
            # Validate and convert frames to tensor
            if 'frames' not in mvhn_batch:
                raise ValueError("Missing 'frames' key in batch")
            
            frames = mvhn_batch['frames']
            if not isinstance(frames, torch.Tensor):
                if isinstance(frames, list):
                    # Check if list contains PIL Images or tensors
                    if len(frames) > 0 and hasattr(frames[0], 'size'):  # PIL Image
                        raise ValueError("frames contains PIL Images instead of tensors. "
                                       "This suggests the dataloader didn't apply transforms properly.")
                    frames = torch.stack(frames, dim=0)
                else:
                    raise ValueError(f"frames is not a tensor or list: {type(frames)}")
            
            num_views = frames.shape[0]
            if len(frames.shape) != 4:
                raise ValueError(f"frames should be 4D [N, C, H, W], got shape {frames.shape}")
            _, _, h, w = frames.shape
            
            # Convert c2w to extrinsics (w2c)
            # THuman uses w2c (world to camera), MVHumanNet returns c2w
            extrinsics = torch.linalg.inv(mvhn_batch['c2w'])  # [N, 4, 4]
            intrinsics = mvhn_batch['K']  # [N, 3, 3]
            
            # Get reference mask and input mask
            ref_mask = mvhn_batch['ref_mask']  # [N] bool
            input_mask = mvhn_batch['mask']  # [N] bool
            
            # Context views: all sampled views
            context_indices = torch.arange(num_views)
            
            # For targets, we could sample additional views, but for now
            # we'll use the ground truth of the same views
            # In the future, you might want to sample different views for targets
            target_indices = context_indices.clone()
            
            # Images for context
            # Use inconsistent images (ic_rgb) for non-reference frames if use_inconsistent is True
            if mvhn_batch.get('use_inconsistent', False):
                context_images = mvhn_batch.get('ic_rgb')
                # Ensure ic_rgb is a tensor (it might be a list)
                if isinstance(context_images, list):
                    context_images = torch.stack(context_images, dim=0)
                elif not isinstance(context_images, torch.Tensor):
                    # Fallback to frames if ic_rgb is not available or wrong type
                    context_images = mvhn_batch['frames']
            else:
                context_images = mvhn_batch['frames']  # [N, 3, H, W]
            
            # Ensure context_images is a tensor
            if not isinstance(context_images, torch.Tensor):
                if isinstance(context_images, list):
                    context_images = torch.stack(context_images, dim=0)
                else:
                    raise ValueError(f"context_images is not a tensor or list: {type(context_images)}")
            
            # Images for target - always ground truth
            target_images = frames  # Use the already-converted tensor
            
            # Masks
            context_masks = mvhn_batch['frames_masks']  # [N, H, W]
            target_masks = mvhn_batch['frames_masks']  # [N, H, W]
            
            # Ensure masks are tensors
            if not isinstance(context_masks, torch.Tensor):
                if isinstance(context_masks, list):
                    context_masks = torch.stack(context_masks, dim=0)
                else:
                    raise ValueError(f"context_masks is not a tensor or list: {type(context_masks)}")
            
            if not isinstance(target_masks, torch.Tensor):
                if isinstance(target_masks, list):
                    target_masks = torch.stack(target_masks, dim=0)
                else:
                    raise ValueError(f"target_masks is not a tensor or list: {type(target_masks)}")
            
            # Near and far planes
            near_context = torch.full((num_views,), self.cfg.near, dtype=torch.float32)
            far_context = torch.full((num_views,), self.cfg.far, dtype=torch.float32)
            near_target = torch.full((num_views,), self.cfg.near, dtype=torch.float32)
            far_target = torch.full((num_views,), self.cfg.far, dtype=torch.float32)
            
            # ========================================================================
            # TODO: Add SMPLX parameters here when available
            # These should be loaded from your SMPLX preprocessing/fitting results
            # ========================================================================
            
            # Placeholder Rs and Ts - REPLACE WITH ACTUAL SMPLX PARAMETERS
            # Rs: rotation matrices for each joint [N, num_joints, 3, 3]
            # Ts: translations for each joint [N, num_joints, 3]
            num_joints = 55  # SMPLX has 55 joints (including hand/face joints)
            
            # Initialize with identity rotations and zero translations
            Rs_context = torch.eye(3).unsqueeze(0).unsqueeze(0).repeat(num_views, num_joints, 1, 1)
            Ts_context = torch.zeros(num_views, num_joints, 3)
            Rs_target = torch.eye(3).unsqueeze(0).unsqueeze(0).repeat(num_views, num_joints, 1, 1)
            Ts_target = torch.zeros(num_views, num_joints, 3)
            
            # TODO: Load actual SMPLX pose parameters for the timestep
            # These should come from fitted SMPLX parameters stored for each frame
            # Example loading code (to be implemented):
            # smplx_params = load_smplx_params(mvhn_batch['subject_id'], mvhn_batch['timestep'])
            # Rs_context = smplx_params['global_Rs']  # [N, 55, 3, 3]
            # Ts_context = smplx_params['global_Ts']  # [N, 55, 3]
            
            # Placeholder Rs_tpose and Ts_tpose - T-pose parameters
            Rs_tpose_context = torch.eye(3).unsqueeze(0).unsqueeze(0).repeat(num_views, num_joints, 1, 1)
            Ts_tpose_context = torch.zeros(num_views, num_joints, 3)
            Rs_tpose_target = torch.eye(3).unsqueeze(0).unsqueeze(0).repeat(num_views, num_joints, 1, 1)
            Ts_tpose_target = torch.zeros(num_views, num_joints, 3)
            
            # TODO: Load T-pose parameters
            # These represent the canonical T-pose joint transformations
            # Example:
            # tpose_params = load_tpose_params(mvhn_batch['subject_id'])
            # Rs_tpose_context = tpose_params['Rs']
            # Ts_tpose_context = tpose_params['Ts']
            
            # Placeholder cnl_Rs and cnl_Ts - Canonical transformations
            cnl_Rs_context = torch.eye(3).unsqueeze(0).unsqueeze(0).repeat(num_views, num_joints, 1, 1)
            cnl_Ts_context = torch.zeros(num_views, num_joints, 3)
            cnl_Rs_target = torch.eye(3).unsqueeze(0).unsqueeze(0).repeat(num_views, num_joints, 1, 1)
            cnl_Ts_target = torch.zeros(num_views, num_joints, 3)
            
            # TODO: Compute canonical transformations
            # These transform from canonical space to the current pose
            # They are typically computed from T-pose joints and current joints
            # Example:
            # cnl_Rs, cnl_Ts = get_canonical_tfms(template_tpose_joints, current_tpose_joints)
            
            # ========================================================================
            
            # Background color
            bgcolor = torch.tensor(self.cfg.background_color, dtype=torch.float32)
            if list(bgcolor) == [-1, -1, -1]:
                # Random background
                bgcolor = torch.rand(3)
            
            # Overlap metric (for view pair compatibility) - placeholder
            # This would typically be computed based on view overlap
            overlap = torch.ones(num_views)  # Placeholder
            
            # Scene name
            scene = mvhn_batch['subject_id']
            
            # Construct the output in THuman format
            example = {
                "context": {
                    "extrinsics": extrinsics[context_indices],  # [N, 4, 4]
                    "intrinsics": intrinsics[context_indices],  # [N, 3, 3]
                    "Rs": Rs_context[context_indices],  # [N, 55, 3, 3]
                    "Ts": Ts_context[context_indices],  # [N, 55, 3]
                    "Rs_tpose": Rs_tpose_context[context_indices],  # [N, 55, 3, 3]
                    "Ts_tpose": Ts_tpose_context[context_indices],  # [N, 55, 3]
                    "cnl_Rs": cnl_Rs_context[context_indices],  # [N, 55, 3, 3]
                    "cnl_Ts": cnl_Ts_context[context_indices],  # [N, 55, 3]
                    "image": context_images[context_indices],  # [N, 3, H, W]
                    "mask": context_masks[context_indices],  # [N, H, W]
                    "near": near_context[context_indices],  # [N]
                    "far": far_context[context_indices],  # [N]
                    "index": context_indices,  # [N]
                    "overlap": overlap[context_indices],  # [N]
                    "use_smplx": True,
                },
                "target": {
                    "extrinsics": extrinsics[target_indices],  # [N, 4, 4]
                    "intrinsics": intrinsics[target_indices],  # [N, 3, 3]
                    "Rs": Rs_target[target_indices],  # [N, 55, 3, 3]
                    "Ts": Ts_target[target_indices],  # [N, 55, 3]
                    "Rs_tpose": Rs_tpose_target[target_indices],  # [N, 55, 3, 3]
                    "Ts_tpose": Ts_tpose_target[target_indices],  # [N, 55, 3]
                    "cnl_Rs": cnl_Rs_target[target_indices],  # [N, 55, 3, 3]
                    "cnl_Ts": cnl_Ts_target[target_indices],  # [N, 55, 3]
                    "image": target_images[target_indices],  # [N, 3, H, W]
                    "mask": target_masks[target_indices],  # [N, H, W]
                    "near": near_target[target_indices],  # [N]
                    "far": far_target[target_indices],  # [N]
                    "index": target_indices,  # [N]
                    "use_smplx": True,
                },
                "scene": scene,
                "bgcolor": bgcolor,
            }
            
            # Add static template information if loaded (same as THuman dataset)
            if self.cfg.load_template_uv:
                example["context"].update({
                    "template_mask": self.template_mask,
                    "template_3d": self.template_3d,
                    "template_lbs_weights": self.template_lbs_weights,
                    "template_stds": self.template_stds,
                    "template_means": self.template_means,
                })
            
            return example
            
        except Exception as e:
            print(f"Error transforming batch: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    @property
    def data_stage(self):
        """Return the data stage (train/test/val)."""
        return self.stage


def create_mvhn_dataset(cfg: DatasetMVHNCfg, stage: str):
    """
    Factory function to create MVHumanNet dataset.
    
    Args:
        cfg: Dataset configuration
        stage: 'train', 'val', or 'test'
    
    Returns:
        DatasetMVHN instance
    """
    return DatasetMVHN(cfg, stage)
