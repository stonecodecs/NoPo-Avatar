"""
Wrapper of MVHumanNet dataset to match THuman dataset format for expected model input into NoPo-Avatar.
Main operations borrowed from 'dataloader.py' with MVHumanNetDataset class.
Main use for this file is to use SMPLX parameters for canonical transformations and template data.
"""

import torch
import numpy as np
import os
import cv2
import json
from torch.utils.data import Dataset
from typing import Optional
from dataclasses import dataclass, field
from pathlib import Path
from smplx import SMPLX
from utils.easymocap2smplx import convert_easymocap_to_smplx
from src.misc.body_utils import get_canonical_tfms, get_canonical_global_tfms, body_pose_to_body_RTs, get_global_RTs
from dataloader import MVHumanNetDataset, custom_collate
from .dataset import DatasetCfgCommon
from .view_sampler import ViewSamplerCfg
from src.misc.body_utils import apply_global_tfm_to_camera

def expand_only_include(only_include):
    if isinstance(only_include, str): # in the format ex: "100001-102000,102020-104000"
        only_include = only_include.split(",")
        expanded_includes = []
        for subrange in only_include:
            start, end = [int(num) for num in subrange.split("-")]
            print(start, end)
            expanded_includes.extend([str(i).zfill(6) for i in range(start, end + 1)])
        return expanded_includes
    else:
        return only_include

MISSING_SUBJECTS = "100602-100995"
TRAIN_SUBJECTS = "100001-104500"
VAL_SUBJECTS = "104501-104999" # rest of MVHN pt. 1
# possibly for test, use MVHN pt. 2 (or merge this into train data later)

@dataclass
class DatasetMVHNCfg(DatasetCfgCommon):
    """Configuration for MVHumanNet dataset to match THuman interface."""
    original_image_shape: list[int] = None
    input_image_shape: list[int] = None
    background_color: list[float] = None
    cameras_are_circular: bool = True
    overfit_to_scene: str | None = None
    view_sampler: ViewSamplerCfg = None 
    name: str = "mvhn"
    # optional jsons (lists of subject IDs) to include for the dataset (will overwrite only_include)
    path_to_train_json: Optional[str] = None # optionals 
    path_to_val_json: Optional[str] = None
    path_to_test_json: Optional[str] = None
    root_dir: str = "/workspace/datasetvol/mvhuman_data/mv_captures"
    latents_dir: Optional[str] = None
    preload_path: str = "/workspace/datasetvol/mvhuman_data/preload_paths/shards"
    num_images: int = 3  # number of context views
    data_limit: Optional[int] = None
    only_include: Optional[list] = field(default_factory=lambda: expand_only_include(TRAIN_SUBJECTS))
    exclude: Optional[list] = field(default_factory=lambda: expand_only_include(MISSING_SUBJECTS))
    step_size: int = 60
    random_crop: bool = False # currently, not used, but may be in the future for data augmentation.
    maximal_crop: bool = True # this should be true by default.
    white_background: bool = False
    crop_padding: int = 60 # padding for random crop aligned with IC-light images to mitigate cropping errors in bboxes.
    use_inconsistent: bool = False # Use iclight/infu for non-reference frames. If False, no ic_rgb will be produced (or it will be a copy of 'frames').
    random_crop_prob: float = 0.0
    ic_sampling_prob: float = 1.0 # sample split probability of iclight vs infu
    iclight_dataset_path: Optional[str] = "/workspace/datasetvol/mvhuman_data/relit_images"
    infu_dataset_path: Optional[str] = None # "/workspace/datasetvol/mvhuman_data/inconsistent_images"
    face_bbox_dir: Optional[str] = None # this is not necessary, stay None
    arcface_embeddings_dir: Optional[str] = None # stay None
    fixed_sampling_ids: Optional[list] = None # choose 'N' indices to sample from the original set of 48 views per scene
    use_sapiens_conditioning: Optional[list] = None # stay None
    sapiens_segmentation_channels_to_use: list = None # stay None
    force_face_ref: bool = False # stay None
    target_shape: tuple = (1024,1024)
    # Camera parameters
    near: float = 0.1
    far: float = 100.0
    
    # SMPLX parameters; must run at NoPo-Avatar fork root
    smplx_model_path: str = "datasets/smplx/SMPLX_NEUTRAL.npz"
    
    # Background color: [R, G, B] in [0, 1] or [-1, -1, -1]; this will be black by default
    background_color: list = None
    
    # Image shape
    input_image_shape: list = None  # [H, W]
    
    # Template parameters (for canonical space rendering)
    load_template_uv: bool = False
    template_image_shape: list = None  # [H, W] for template resolution
    load_da_pose: bool = False  # Load DA-pose templates
    load_supervision: bool = False
    load_lbs_weights: bool = False

    # debug
    retry_limit: int = 20


@dataclass
class DatasetMVHNCfgWrapper:
    mvhn: DatasetMVHNCfg


class DatasetMVHN(Dataset):
    """
    MVHumanNet dataset wrapper that outputs data in THuman format.
    
    This dataset:
    - Samples 'n' images (default 3) for context
    - Keeps one as reference (consistent ground truth, reference image)
    - Others can be inconsistent data (iclight/infu)
    - Target views are clean ground truth (no lighting changes)
    
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

        # LET VIEW_SAMPLER OVERWRITE cfg.num_images!
        # would be cleaner to just refine this + dataloader.py but nobody got time for that (yet)
        if self.view_sampler is not None:
            self.cfg.num_images = self.view_sampler.num_context_views + self.view_sampler.num_target_views
        # if no view sampler, keep original num_images
        # and in training loss, we only look at the input views only (no target views)
        # (this is not really desired, so encouraged to use view sampler in the configs)
        
        # Set defaults
        if self.cfg.background_color is None:
            self.cfg.background_color = [0, 0, 0]  # Black background
        if self.cfg.input_image_shape is None:
            self.cfg.input_image_shape = [576, 576]
        if self.cfg.sapiens_segmentation_channels_to_use is None:
            self.cfg.sapiens_segmentation_channels_to_use = []
        if self.cfg.template_image_shape is None:
            self.cfg.template_image_shape = [1024, 1024]  # Default template resolution
        
        # Sync target_shape with input_image_shape if input_image_shape is set
        # This ensures images are resized to the correct resolution
        if self.cfg.input_image_shape is not None:
            # Only override if target_shape is still at default or not explicitly set
            # Convert list to tuple to match target_shape type
            target_shape_tuple = tuple(self.cfg.input_image_shape)
            if self.cfg.target_shape != target_shape_tuple:
                # If target_shape differs from input_image_shape, sync them
                # (User can override by explicitly setting target_shape in config)
                self.cfg.target_shape = target_shape_tuple
        
        # Load SMPLX model for joints computation
        self.smplx_model = SMPLX(
            model_path=self.cfg.smplx_model_path,
            use_pca=False,
            flat_hand_mean=True
        )
        
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
            
        # Get template T-pose joints
        canonical_poses = self.smplx_model.body_pose.detach().clone()
        canonical_poses.requires_grad = False
        if self.cfg.load_da_pose:
            canonical_poses[0, 2] = 1.0
            canonical_poses[0, 5] = -1.0
        self.template_tpose_joints = self.smplx_model(body_pose=canonical_poses).joints.detach().cpu()[0, :55]

        # train/val/test split json lists
        try:
            stage_json_path = None
            if self.cfg.path_to_train_json is not None and self.stage == "train":
                stage_json_path = self.cfg.path_to_train_json
                self.cfg.only_include = json.load(open(self.cfg.path_to_train_json))
            if self.cfg.path_to_val_json is not None and self.stage == "val":
                stage_json_path = self.cfg.path_to_val_json
                self.cfg.only_include = json.load(open(self.cfg.path_to_val_json))
            if self.cfg.path_to_test_json is not None and self.stage == "test":
                stage_json_path = self.cfg.path_to_test_json
                self.cfg.only_include = json.load(open(self.cfg.path_to_test_json))
        except FileNotFoundError as e:
            print(f"For stage {self.stage}, no json file was found. Expected json file at {stage_json_path}. Using defaults. Error: {e}")
            if self.stage == "train":
                self.cfg.only_include = expand_only_include(TRAIN_SUBJECTS)
            elif self.stage == "val" or self.stage == "test":
                self.cfg.only_include = expand_only_include(VAL_SUBJECTS)
            else:
                raise ValueError(f"Invalid stage: {self.stage}. Expected train, val, or test.")
        
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
            target_shape=self.cfg.target_shape,
        )
        
    def __len__(self):
        """Return the length of the dataset."""
        return len(self.dataset)
    
    def __getitem__(self, idx):
        """Get a single example and transform it to THuman format."""
        max_attempts = min(self.cfg.retry_limit, len(self))  # Try up to 10 indices or dataset length
        
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
        - frames_masks: [N, H, W] - binary image masks
        - c2w: [N, 4, 4] - camera to world matrices
        - K: [N, 3, 3] - intrinsics (should be normalized + post-cropping)
        - ref_mask: [N] - bool tensor indicating which frame is reference
        - ic_rgb: [N, 3, H, W] - inconsistent RGB images
        - subject_id: str - subject ID
        - timestep: int - timestep
        
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
            
            frames = mvhn_batch['frames'] # [N,3,H,W]
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
            
            # THuman uses C2W (camera to world) for 'extrinsics'
            # mvhn_batch['c2w'] is already C2W from dataloader
            # these are scaled + centered already by MVHN dataloader (see dataloader.py)
            extrinsics = mvhn_batch['c2w']  # [N, 4, 4]
            intrinsics = mvhn_batch['K']  # [N, 3, 3]
            
            # Get reference mask and input mask
            ref_mask = mvhn_batch['ref_mask']  # [N] bool
            
            # Context views: all sampled views
            if self.view_sampler is not None:
                # all_indices sets the reference view as idx 0
                ref_idx = torch.argmax(ref_mask.int()).item()
                all_indices = torch.roll(torch.arange(num_views), -ref_idx)
                context_indices = all_indices[:self.cfg.view_sampler.num_context_views]
                target_indices = all_indices[self.cfg.view_sampler.num_context_views:]
            else: # use the same views
                context_indices = torch.arange(num_views)
                target_indices = context_indices.clone()
                
            context_images = mvhn_batch['ic_rgb']
            context_images_gt = mvhn_batch['frames']
            target_images = mvhn_batch['frames']
            
            # Masks
            context_masks = mvhn_batch['frames_masks'].squeeze(1)  # [N, H, W]
            target_masks = mvhn_batch['frames_masks'].squeeze(1)  # [N, H, W]
            
            # Near and far planes
            near_context = torch.full((num_views,), self.cfg.near, dtype=torch.float32)
            far_context = torch.full((num_views,), self.cfg.far, dtype=torch.float32)
            near_target = torch.full((num_views,), self.cfg.near, dtype=torch.float32)
            far_target = torch.full((num_views,), self.cfg.far, dtype=torch.float32)
            # cam_scale = mvhn_batch['cam_scale'].numpy()
            # cam_center = mvhn_batch['cam_center'].numpy()
            # cam_scale = mvhn_batch['cam_scale'].numpy()
            # ========================================================================
            # SMPLX parameters
            # ========================================================================
            smplx_params = mvhn_batch['smplx_params']
            
            # Apply global transformation to cameras relative to body pose
            # This adjusts cameras to account for the body's global orientation and translation
            Rh = smplx_params['global_orient']  # Axis-angle rotation [3]
            Th = smplx_params['transl']  # Translation [3]
            
            # Convert c2w to w2c for apply_global_tfm_to_camera (it expects w2c)
            w2cs = torch.linalg.inv(extrinsics)  # [N, 4, 4]
            extrinsics_list = []
            for w2c in w2cs:
                extrinsics_list.append(
                    torch.linalg.inv(
                        torch.from_numpy(apply_global_tfm_to_camera(w2c.numpy(), Rh.numpy(), Th.numpy())).float()
                    )
                )
            extrinsics = torch.stack(extrinsics_list)
            
            # 1. Compute joints for the current pose (RELATIVE to body origin)
            # We set global_orient and transl to zero because the cameras are already body-relative
            with torch.no_grad():
                smplx_output = self.smplx_model(
                    global_orient=torch.zeros((1, 3), dtype=torch.float32),
                    # global_orient=Rh.float().unsqueeze(0),
                    body_pose=torch.from_numpy(smplx_params['body_pose']).float().unsqueeze(0),
                    left_hand_pose=torch.from_numpy(smplx_params['left_hand_pose']).float().unsqueeze(0),
                    right_hand_pose=torch.from_numpy(smplx_params['right_hand_pose']).float().unsqueeze(0),
                    jaw_pose=torch.from_numpy(smplx_params['jaw_pose']).float().unsqueeze(0),
                    leye_pose=torch.from_numpy(smplx_params['left_eye_pose']).float().unsqueeze(0),
                    reye_pose=torch.from_numpy(smplx_params['right_eye_pose']).float().unsqueeze(0),
                    betas=torch.from_numpy(smplx_params['betas']).float().unsqueeze(0),
                    expression=torch.from_numpy(smplx_params['expression']).float().unsqueeze(0),
                    transl=torch.zeros((1, 3), dtype=torch.float32),
                    # transl=Th.float().unsqueeze(0),
                    return_full_pose=True
                )
            
            current_joints = smplx_output.joints.detach().cpu()[0, :55]
            
            # 2. Get T-pose joints for this subject (using current betas)
            with torch.no_grad():
                tpose_output = self.smplx_model(
                    betas=torch.from_numpy(smplx_params['betas']).float().unsqueeze(0),
                    return_full_pose=True
                )
            tpose_joints = tpose_output.joints.detach().cpu()[0, :55]
     
            # 3. Compute Rs and Ts (Global rotation and translation for each joint)
            # We need to reshape the full pose to [55, 3]
            full_pose = smplx_output.full_pose.detach().cpu().numpy().reshape(55, 3)
            # The model expects root rotation to be handled by extrinsics usually, 
            # but here we follow THuman's global RTs computation.
            
            # Compute canonical global transforms for the T-pose joints
            # Note: tpose_joints are now in Template coordinate frame
            cnl_gtfms = get_canonical_global_tfms(tpose_joints.numpy(), use_smplx=True)
            
            # Compute destination Rs and Ts (local)
            dst_Rs, dst_Ts = body_pose_to_body_RTs(full_pose, tpose_joints.numpy(), use_smplx=True)
            
            # Compute global Rs and Ts
            global_Rs, global_Ts = get_global_RTs(cnl_gtfms, dst_Rs, dst_Ts, use_smplx=True)
            
            # 4. Compute T-pose Rs and Ts (Identity rotations, joints as translations)
            # In T-pose, joints are just at their canonical positions
            dst_Rs_tpose, dst_Ts_tpose = body_pose_to_body_RTs(np.zeros((55, 3)), tpose_joints.numpy(), use_smplx=True)
            global_Rs_tpose, global_Ts_tpose = get_global_RTs(cnl_gtfms, dst_Rs_tpose, dst_Ts_tpose, use_smplx=True)
            
            # 5. Compute canonical transformations (cnl_Rs, cnl_Ts)
            # These transform from the template T-pose to the current subject's T-pose
            # * unsqueeze(0) to match THuman format
            cnl_Rs_val, cnl_Ts_val = get_canonical_tfms(self.template_tpose_joints, tpose_joints, use_smplx=True)
            
            # 6. Repeat for all views
            num_joints = 55
            Rs_context = torch.from_numpy(global_Rs).unsqueeze(0).repeat(num_views, 1, 1, 1)
            Ts_context = torch.from_numpy(global_Ts).unsqueeze(0).repeat(num_views, 1, 1)
            Rs_target = Rs_context.clone()
            Ts_target = Ts_context.clone()
            
            Rs_tpose_context = torch.from_numpy(global_Rs_tpose).unsqueeze(0).repeat(num_views, 1, 1, 1)
            Ts_tpose_context = torch.from_numpy(global_Ts_tpose).unsqueeze(0).repeat(num_views, 1, 1)
            Rs_tpose_target = Rs_tpose_context.clone()
            Ts_tpose_target = Ts_tpose_context.clone()
            
            cnl_Rs_context = cnl_Rs_val.unsqueeze(0).repeat(num_views, 1, 1, 1)
            cnl_Ts_context = cnl_Ts_val.unsqueeze(0).repeat(num_views, 1, 1)
            cnl_Rs_target = cnl_Rs_context.clone()
            cnl_Ts_target = cnl_Ts_context.clone()
            
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
            subject_id = mvhn_batch['subject_id']
            timestep = mvhn_batch['timestep']
            ref_mask = mvhn_batch['ref_mask']
            
            # Convert smplx_params to torch tensors for consistency (they're numpy arrays from dataloader)
            # This makes them easier to use in the model and visualization
            smplx_params_torch = {}
            for key, value in smplx_params.items():
                if isinstance(value, np.ndarray):
                    smplx_params_torch[key] = torch.from_numpy(value).float()
                else:
                    smplx_params_torch[key] = torch.tensor(value).float()

            # Construct the output in THuman format; post-dataloader, these will be in the expected shapes [1,...]
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
                    "image_gt": context_images_gt[context_indices], # [N, 3, H, W]
                    "mask": context_masks[context_indices],  # [N, H, W]
                    "near": near_context[context_indices],  # [N]
                    "far": far_context[context_indices],  # [N]
                    "index": context_indices,  # [N]
                    "overlap": overlap[context_indices],  # [N]
                    "use_smplx": True,
                    "ref_mask": ref_mask[context_indices],
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
                "scene": [scene],
                "subject_id": subject_id,
                "timestep": timestep,
                "bgcolor": bgcolor,
                "tpose_joints": tpose_joints,  # Subject-specific T-pose joints
                # ! ==== rest of these are debugging; delete later ====
                "ref_mask": ref_mask,
                "smplx_params": smplx_params_torch,  # Original SMPLX parameters used to generate Rs/Ts
                # "cam_center": cam_center,
                # "cam_scale": cam_scale,
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
