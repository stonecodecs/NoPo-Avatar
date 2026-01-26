"""
Visualize coordinate frames for cameras, origin, and SMPLX joints
for both MVHN and THuman datasets.

This script helps debug coordinate system alignment issues by showing:
- Camera positions and orientations (coordinate frames)
- World origin
- SMPLX joint positions
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import torch
from pathlib import Path
from omegaconf import OmegaConf
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.dataset.dataset_mvhn import DatasetMVHN, DatasetMVHNCfg
from src.dataset.dataset_thuman import DatasetTHuman, DatasetTHumanCfg
from src.dataset.view_sampler import ViewSampler
from smplx import SMPLX


def draw_coordinate_frame(ax, origin, rotation_matrix, scale=0.1, label=""):
    """
    Draw a coordinate frame (X, Y, Z axes) at a given origin with a given rotation.
    
    Args:
        ax: matplotlib 3D axes
        origin: [3] position of the frame
        rotation_matrix: [3, 3] rotation matrix (columns are X, Y, Z axes in world space)
        scale: length of axes
        label: optional label for the frame
    """
    # Axes directions in local frame (X=red, Y=green, Z=blue)
    axes_local = np.array([
        [1, 0, 0],  # X axis (red)
        [0, 1, 0],  # Y axis (green)
        [0, 0, 1],  # Z axis (blue)
    ]) * scale
    
    # Transform to world space
    axes_world = (rotation_matrix @ axes_local.T).T
    
    colors = ['red', 'green', 'blue']
    labels = ['X', 'Y', 'Z']
    
    for i, (axis, color, label_axis) in enumerate(zip(axes_world, colors, labels)):
        end = origin + axis
        ax.plot([origin[0], end[0]], [origin[1], end[1]], [origin[2], end[2]], 
                color=color, linewidth=2, alpha=0.8)
        # Add small label at the end
        ax.text(end[0], end[1], end[2], f'{label_axis}', fontsize=8, color=color)
    
    if label:
        ax.text(origin[0], origin[1], origin[2], f' {label}', fontsize=10)


def extract_camera_info(extrinsics):
    """
    Extract camera position and orientation from C2W matrix.
    
    Args:
        extrinsics: [N, 4, 4] C2W matrices
        
    Returns:
        positions: [N, 3] camera positions
        rotations: [N, 3, 3] camera rotation matrices
    """
    if isinstance(extrinsics, torch.Tensor):
        extrinsics = extrinsics.detach().cpu().numpy()
    
    # Remove batch dimension if present
    if extrinsics.ndim == 4:
        extrinsics = extrinsics[0]  # [N, 4, 4]
    
    # Camera position is the translation part (last column, first 3 rows)
    positions = extrinsics[:, :3, 3]  # [N, 3]
    
    # Camera rotation is the rotation part (first 3x3)
    rotations = extrinsics[:, :3, :3]  # [N, 3, 3]
    
    return positions, rotations


def visualize_dataset_sample(batch, dataset_name, output_path, idx=0):
    """
    Visualize coordinate frames for a dataset sample.
    
    Args:
        batch: dataset batch
        dataset_name: name of dataset ('mvhn' or 'thuman')
        output_path: path to save the plot
        idx: batch index (usually 0 for single sample)
    """
    fig = plt.figure(figsize=(20, 16))
    
    # Create 6 subplots: overview + 3 axis projections (XY, XZ, YZ) + 2 additional views
    ax1 = fig.add_subplot(231, projection='3d')  # Overview
    ax2 = fig.add_subplot(232, projection='3d')  # XY plane (top view)
    ax3 = fig.add_subplot(233, projection='3d')  # XZ plane (front view)
    ax4 = fig.add_subplot(234, projection='3d')  # YZ plane (side view)
    ax5 = fig.add_subplot(235, projection='3d')  # Isometric view 1
    ax6 = fig.add_subplot(236, projection='3d')  # Isometric view 2
    
    axes_list = [ax1, ax2, ax3, ax4, ax5, ax6]
    
    # Extract camera information
    context_extrinsics = batch['context']['extrinsics']  # [B, N, 4, 4] or [N, 4, 4]
    if context_extrinsics.ndim == 4 and context_extrinsics.shape[0] > 1:
        context_extrinsics = context_extrinsics[idx]
    elif context_extrinsics.ndim == 4:
        context_extrinsics = context_extrinsics[0]  # Remove batch dim
    
    cam_positions, cam_rotations = extract_camera_info(context_extrinsics)
    num_cameras = len(cam_positions)
    
    # Extract SMPLX joints if available
    joints = None
    if 'tpose_joints' in batch:
        joints = batch['tpose_joints']
        if isinstance(joints, torch.Tensor):
            joints = joints.detach().cpu().numpy()
        if joints.ndim == 2:
            joints = joints  # [55, 3]
        elif joints.ndim == 3:
            joints = joints[0]  # Remove batch dim
    
    # Also try to get current pose joints from Rs and Ts (preferred - matches what model uses)
    # OR compute from smplx_params if Rs/Ts not available (fallback)
    # CURRENT POSE JOINTS: These are the joints in the ACTUAL POSE from your dataset
    # They should be computed from Rs/Ts matrices (which reflect dataset processing including Rh transformations)
    # OR from raw SMPLX parameters (fallback, but may not reflect dataset transformations)
    # This represents where the joints actually are in the images you captured
    current_joints = None
    
    # Method 1: Compute from Rs and Ts (PREFERRED - matches what the model actually uses)
    # Rs and Ts are global rotation and translation matrices for each joint
    # These reflect any transformations applied in the dataset (e.g., Rh transformations)
    # Apply them to T-pose joints to get current pose joints
    if joints is not None and 'context' in batch:
        context_batch = batch['context']
        if 'Rs' in context_batch and 'Ts' in context_batch:
            try:
                # Get Rs and Ts from first context view (or average if multiple views)
                Rs = context_batch['Rs']  # [N, 55, 3, 3] or [55, 3, 3]
                Ts = context_batch['Ts']  # [N, 55, 3] or [55, 3]
                
                # Handle batch dimension
                if Rs.ndim == 4:
                    Rs = Rs[0]  # Take first view
                if Ts.ndim == 3:
                    Ts = Ts[0]  # Take first view
                
                # Convert to numpy if needed
                if isinstance(Rs, torch.Tensor):
                    Rs = Rs.detach().cpu().numpy()
                if isinstance(Ts, torch.Tensor):
                    Ts = Ts.detach().cpu().numpy()
                
                # Apply global transformations: current_joint = R @ tpose_joint + T
                # Rs and Ts are already in global space (computed via get_global_RTs)
                current_joints = np.zeros_like(joints)
                for i in range(len(joints)):
                    current_joints[i] = Rs[i] @ joints[i] + Ts[i]
                
            except Exception as e:
                print(f"Warning: Could not compute current joints from Rs/Ts: {e}")
    
    # Method 2: Fallback - Compute from smplx_params (if Rs/Ts not available)
    # This is a fallback that doesn't reflect dataset transformations like Rh
    if current_joints is None and 'smplx_params' in batch:
        smplx_params = batch['smplx_params']
        if isinstance(smplx_params, dict):
            # Try to compute joints from SMPLX params
            try:
                from smplx import SMPLX
                smplx_model = SMPLX(
                    model_path="datasets/smplx/SMPLX_NEUTRAL.npz",
                    use_pca=False,
                    flat_hand_mean=True
                )
                
                # Convert to numpy if needed
                def to_numpy(val):
                    if isinstance(val, torch.Tensor):
                        return val.detach().cpu().numpy()
                    return np.array(val)
                
                with torch.no_grad():
                    output = smplx_model(
                        global_orient=torch.from_numpy(to_numpy(smplx_params['global_orient'])).float().unsqueeze(0),
                        body_pose=torch.from_numpy(to_numpy(smplx_params['body_pose'])).float().unsqueeze(0),
                        betas=torch.from_numpy(to_numpy(smplx_params['betas'])).float().unsqueeze(0),
                        transl=torch.from_numpy(to_numpy(smplx_params['transl'])).float().unsqueeze(0),
                        return_full_pose=True
                    )
                    current_joints = output.joints.detach().cpu().numpy()[0, :55]
                    print("Warning: Using smplx_params to compute joints (fallback). "
                          "This may not reflect dataset transformations like Rh. "
                          "Prefer using Rs/Ts from batch if available.")
            except Exception as e:
                print(f"Warning: Could not compute current joints from smplx_params: {e}")
    
    # Plot on all axes
    for ax in axes_list:
        # Draw world origin
        draw_coordinate_frame(ax, np.array([0, 0, 0]), np.eye(3), scale=0.2, label="Origin")
        
        # Draw cameras
        for i, (pos, rot) in enumerate(zip(cam_positions, cam_rotations)):
            # Convert C2W rotation to camera frame rotation
            # C2W rotation matrix columns are camera X, Y, Z axes in world space
            cam_frame_rot = rot  # Already in world space
            draw_coordinate_frame(ax, pos, cam_frame_rot, scale=0.15, label=f"Cam{i}")
            
            # Draw line from origin to camera
            ax.plot([0, pos[0]], [0, pos[1]], [0, pos[2]], 
                   'k--', alpha=0.3, linewidth=1)
        
        # Draw SMPLX joints
        # T-POSE JOINTS: These are the joints in CANONICAL T-POSE (no pose applied)
        # They represent the subject's body shape (betas) in the neutral T-pose
        # This is the canonical/reference pose used for LBS (Linear Blend Skinning)
        if joints is not None:
            ax.scatter(joints[:, 0], joints[:, 1], joints[:, 2], 
                      c='purple', s=20, alpha=0.6, label='T-pose joints (canonical)')
            # Draw pelvis (joint 0) with special marker
            if len(joints) > 0:
                ax.scatter(joints[0:1, 0], joints[0:1, 1], joints[0:1, 2], 
                          c='orange', s=100, marker='*', label='Pelvis (T-pose)', zorder=10)
        
        # CURRENT POSE JOINTS: These are the joints in the ACTUAL POSE from your dataset
        # They show where the joints are positioned in the actual images
        # Computed using: betas + global_orient + body_pose + transl + all pose params
        if current_joints is not None:
            ax.scatter(current_joints[:, 0], current_joints[:, 1], current_joints[:, 2], 
                      c='cyan', s=15, alpha=0.4, marker='^', label='Current pose joints (from dataset)')
            # Draw pelvis in current pose
            if len(current_joints) > 0:
                ax.scatter(current_joints[0:1, 0], current_joints[0:1, 1], current_joints[0:1, 2], 
                          c='yellow', s=100, marker='*', label='Pelvis (current pose)', zorder=10)
            
            # Draw lines connecting corresponding joints to show the transformation
            if joints is not None and len(joints) == len(current_joints):
                # Draw lines for key joints (pelvis, shoulders, hips, head)
                key_joint_indices = [0, 12, 15, 1, 2, 13, 14]  # pelvis, neck, head, hips, shoulders
                for idx in key_joint_indices:
                    if idx < len(joints) and idx < len(current_joints):
                        ax.plot([joints[idx, 0], current_joints[idx, 0]], 
                               [joints[idx, 1], current_joints[idx, 1]], 
                               [joints[idx, 2], current_joints[idx, 2]], 
                               'g--', alpha=0.3, linewidth=1)
        
        # Set labels and title
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.legend(loc='upper left', fontsize=8)
        
        # Set equal aspect ratio
        all_points = []
        if len(cam_positions) > 0:
            all_points.append(cam_positions)
        if joints is not None and len(joints) > 0:
            all_points.append(joints)
        if current_joints is not None and len(current_joints) > 0:
            all_points.append(current_joints)
        
        if all_points:
            all_points = np.concatenate(all_points, axis=0)
            center = all_points.mean(axis=0)
            max_range = np.abs(all_points - center).max()
            if max_range < 1e-6:  # Handle degenerate case
                max_range = 1.0
            max_range *= 1.2
            
            ax.set_xlim(center[0] - max_range, center[0] + max_range)
            ax.set_ylim(center[1] - max_range, center[1] + max_range)
            ax.set_zlim(center[2] - max_range, center[2] + max_range)
        else:
            # Fallback if no points
            ax.set_xlim(-1, 1)
            ax.set_ylim(-1, 1)
            ax.set_zlim(-1, 1)
    
    # Set different viewing angles for each subplot
    ax1.view_init(elev=20, azim=45)  # Isometric overview
    ax1.set_title(f'{dataset_name.upper()} - Overview', fontsize=12, fontweight='bold')
    
    ax2.view_init(elev=90, azim=-90)  # Top view (XY plane, looking down Z)
    ax2.set_title(f'{dataset_name.upper()} - Top View (XY)', fontsize=12, fontweight='bold')
    
    ax3.view_init(elev=0, azim=0)  # Front view (XZ plane, looking along Y)
    ax3.set_title(f'{dataset_name.upper()} - Front View (XZ)', fontsize=12, fontweight='bold')
    
    ax4.view_init(elev=0, azim=90)  # Side view (YZ plane, looking along X)
    ax4.set_title(f'{dataset_name.upper()} - Side View (YZ)', fontsize=12, fontweight='bold')
    
    ax5.view_init(elev=30, azim=135)  # Isometric view 1
    ax5.set_title(f'{dataset_name.upper()} - Isometric 1', fontsize=12, fontweight='bold')
    
    ax6.view_init(elev=30, azim=-45)  # Isometric view 2
    ax6.set_title(f'{dataset_name.upper()} - Isometric 2', fontsize=12, fontweight='bold')
    
    # Add overall title with explanation
    scene_name = batch.get('scene', ['unknown'])[0] if 'scene' in batch else 'unknown'
    title = f'{dataset_name.upper()} Coordinate Frames - Scene: {scene_name}\n'
    title += f'Cameras: {num_cameras}, T-pose joints: {len(joints) if joints is not None else 0}'
    if current_joints is not None:
        title += f', Current pose joints: {len(current_joints)}'
    title += '\nPurple=T-pose (canonical), Cyan=Current pose (from dataset), Lines show transformation'
    fig.suptitle(title, fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved visualization to {output_path}")
    plt.close()


def load_mvhn_sample(cfg_path=None, sample_idx=0):
    """Load a sample from MVHN dataset."""
    if cfg_path is None:
        # Use default config
        cfg = DatasetMVHNCfg(
            root_dir="/workspace/datasetvol/mvhuman_data/mv_captures",
            num_images=3,
            only_include=["100001"],
            exclude=[],
            smplx_model_path="datasets/smplx/SMPLX_NEUTRAL.npz",
            input_image_shape=[576, 576],
            background_color=[0, 0, 0],
        )
        view_sampler = None
    else:
        # Load from config file using OmegaConf with defaults
        config = OmegaConf.load(cfg_path)
        OmegaConf.set_struct(config, False)  # Allow merging
        
        # Resolve defaults
        if 'defaults' in config.dataset.mvhn:
            # Merge with defaults
            base_config = OmegaConf.load('config/dataset/base_dataset.yaml')
            view_sampler_config = OmegaConf.load('config/dataset/view_sampler/uniform.yaml')
            config.dataset.mvhn = OmegaConf.merge(base_config, view_sampler_config, config.dataset.mvhn)
        
        cfg_dict = OmegaConf.to_container(config.dataset.mvhn, resolve=True)
        cfg = OmegaConf.structured(DatasetMVHNCfg(**cfg_dict))
        
        # Create view sampler if specified
        from src.dataset.view_sampler import get_view_sampler
        view_sampler = get_view_sampler(
            cfg.view_sampler,
            'test',
            False,
            cfg.cameras_are_circular,
            None
        )
    
    dataset = DatasetMVHN(cfg, stage='test', view_sampler=view_sampler)
    
    if len(dataset) == 0:
        raise ValueError("MVHN dataset is empty!")
    
    batch = dataset[sample_idx % len(dataset)]
    return batch, 'mvhn'


def load_thuman_sample(cfg_path=None, sample_idx=0):
    """Load a sample from THuman dataset using the same approach as main.py."""
    from src.config import load_typed_config
    from src.dataset import get_dataset
    from src.dataset.dataset_thuman import DatasetTHumanCfgWrapper, DatasetTHumanCfg
    from src.misc.step_tracker import StepTracker
    
    if cfg_path is None:
        # Try to use experimental config first (has all required fields)
        default_paths = [
            'config/experiment/train_thuman2.0_3views_res256.yaml',
            'config/experiment/train_thuman2.0_3views_res512.yaml',
            'config/experiment/train_thuman2.0_3views_res1024.yaml',
        ]
        cfg_path = None
        for path in default_paths:
            if Path(path).exists():
                cfg_path = path
                print(f"  Found config: {cfg_path}")
                break
        
        if cfg_path is None:
            raise ValueError(
                "THuman config not found. Please provide --thuman-config or ensure "
                "one of these exists:\n"
                "  - config/experiment/train_thuman2.0_3views_res256.yaml\n"
                "  - config/experiment/train_thuman2.0_3views_res512.yaml\n"
                "  - config/experiment/train_thuman2.0_3views_res1024.yaml"
            )
    
    # Use Hydra to load config (same as main.py) - this automatically resolves defaults
    cfg_dict = None
    try:
        import hydra
        from hydra import initialize_config_dir, compose
        from hydra.core.global_hydra import GlobalHydra
        
        # Clear any existing Hydra instance
        GlobalHydra.instance().clear()
        
        # Initialize Hydra with the config directory (must be absolute)
        config_dir = Path(cfg_path).parent.parent.resolve()  # Go up to 'config' directory
        with initialize_config_dir(config_dir=str(config_dir), version_base=None):
            # Compose the config using the relative path from config_dir
            config_name = str(Path(cfg_path).relative_to(config_dir).with_suffix(''))
            cfg_dict = compose(config_name=config_name)
    except Exception as e:
        # Fallback: use OmegaConf directly and manually resolve defaults
        print(f"  Warning: Could not use Hydra ({e}), using OmegaConf with manual resolution")
        cfg_dict = OmegaConf.load(cfg_path)
        OmegaConf.set_struct(cfg_dict, False)
        
        # Manually resolve defaults by loading and merging base configs
        if 'defaults' in cfg_dict:
            # This is an experiment config - resolve defaults manually
            config_root = Path(cfg_path).parent.parent
            base_dataset_path = config_root / 'dataset' / 'base_dataset.yaml'
            thuman_default_path = config_root / 'dataset' / 'thuman.yaml'
            
            if base_dataset_path.exists() and thuman_default_path.exists():
                base_config = OmegaConf.load(str(base_dataset_path))
                thuman_default = OmegaConf.load(str(thuman_default_path))
                
                # Resolve thuman_default's defaults too
                if 'defaults' in thuman_default:
                    thuman_default = OmegaConf.to_container(thuman_default, resolve=True)
                    thuman_default = OmegaConf.create(thuman_default)
                
                # Merge: base -> thuman defaults -> experiment overrides
                if 'dataset' in cfg_dict and 'thuman' in cfg_dict.dataset:
                    # Merge the thuman config
                    merged_thuman = OmegaConf.merge(
                        base_config,
                        thuman_default,
                        cfg_dict.dataset.thuman
                    )
                    cfg_dict.dataset.thuman = merged_thuman
                else:
                    # Direct dataset config
                    cfg_dict = OmegaConf.merge(base_config, thuman_default, cfg_dict)
    
    # Extract dataset config - use load_typed_config directly with the specific wrapper type
    # This avoids the dacite Union matching issue
    if 'dataset' in cfg_dict and 'thuman' in cfg_dict.dataset:
        # Get the merged thuman config (already merged with defaults if manual resolution was used)
        thuman_cfg_omega = cfg_dict.dataset.thuman
    else:
        # Direct dataset config
        thuman_cfg_omega = cfg_dict
    
    # Convert to container and filter out OmegaConf-specific keys
    thuman_cfg_dict = OmegaConf.to_container(thuman_cfg_omega, resolve=True)
    # Remove 'defaults' key if present (OmegaConf artifact)
    if 'defaults' in thuman_cfg_dict:
        del thuman_cfg_dict['defaults']
    
    # Manually create view_sampler config to avoid dacite Union matching issues
    # Extract view_sampler dict and create the appropriate config type
    view_sampler_dict = thuman_cfg_dict.get('view_sampler', {})
    if isinstance(view_sampler_dict, dict):
        from dataclasses import fields
        
        view_sampler_name = view_sampler_dict.get('name', 'uniform')
        
        # Import all view sampler configs
        from src.dataset.view_sampler.view_sampler_uniform import ViewSamplerUniformCfg
        from src.dataset.view_sampler.view_sampler_bounded import ViewSamplerBoundedCfg
        from src.dataset.view_sampler.view_sampler_evaluation import ViewSamplerEvaluationCfg
        from src.dataset.view_sampler.view_sampler_aligned import ViewSamplerAlignedCfg
        from src.dataset.view_sampler.view_sampler_arbitrary import ViewSamplerArbitraryCfg
        from src.dataset.view_sampler.view_sampler_all import ViewSamplerAllCfg
        
        # Create the appropriate ViewSamplerCfg based on name
        # Filter dict to only include valid fields for the specific config type
        if view_sampler_name == 'uniform':
            valid_fields = {f.name for f in fields(ViewSamplerUniformCfg)}
            filtered_dict = {k: v for k, v in view_sampler_dict.items() if k in valid_fields}
            filtered_dict['name'] = 'uniform'  # Ensure name is set
            filtered_dict.setdefault('num_context_views', 3)
            filtered_dict.setdefault('num_target_views', 1)
            view_sampler_cfg = ViewSamplerUniformCfg(**filtered_dict)
        elif view_sampler_name == 'bounded':
            valid_fields = {f.name for f in fields(ViewSamplerBoundedCfg)}
            filtered_dict = {k: v for k, v in view_sampler_dict.items() if k in valid_fields}
            filtered_dict['name'] = 'bounded'  # Ensure name is set
            filtered_dict.setdefault('num_context_views', 3)
            filtered_dict.setdefault('num_target_views', 1)
            filtered_dict.setdefault('min_distance_between_context_views', 45)
            filtered_dict.setdefault('max_distance_between_context_views', 90)
            filtered_dict.setdefault('min_distance_to_context_views', 0)
            filtered_dict.setdefault('warm_up_steps', 150000)
            filtered_dict.setdefault('initial_min_distance_between_context_views', 25)
            filtered_dict.setdefault('initial_max_distance_between_context_views', 25)
            view_sampler_cfg = ViewSamplerBoundedCfg(**filtered_dict)
        elif view_sampler_name == 'evaluation':
            valid_fields = {f.name for f in fields(ViewSamplerEvaluationCfg)}
            filtered_dict = {k: v for k, v in view_sampler_dict.items() if k in valid_fields}
            filtered_dict['name'] = 'evaluation'  # Ensure name is set
            filtered_dict.setdefault('num_context_views', 3)
            if 'index_path' not in filtered_dict:
                default_index = Path('assets/evaluation_thuman_view3.json')
                if default_index.exists():
                    filtered_dict['index_path'] = default_index
                else:
                    raise ValueError(f"ViewSamplerEvaluation requires 'index_path' but none provided and default not found")
            view_sampler_cfg = ViewSamplerEvaluationCfg(**filtered_dict)
        elif view_sampler_name == 'aligned':
            valid_fields = {f.name for f in fields(ViewSamplerAlignedCfg)}
            filtered_dict = {k: v for k, v in view_sampler_dict.items() if k in valid_fields}
            filtered_dict['name'] = 'aligned'  # Ensure name is set
            filtered_dict.setdefault('num_context_views', 3)
            filtered_dict.setdefault('num_target_views', 1)
            view_sampler_cfg = ViewSamplerAlignedCfg(**filtered_dict)
        elif view_sampler_name == 'arbitrary':
            valid_fields = {f.name for f in fields(ViewSamplerArbitraryCfg)}
            filtered_dict = {k: v for k, v in view_sampler_dict.items() if k in valid_fields}
            filtered_dict['name'] = 'arbitrary'  # Ensure name is set
            filtered_dict.setdefault('num_context_views', 3)
            filtered_dict.setdefault('num_target_views', 1)
            filtered_dict.setdefault('context_views', None)
            filtered_dict.setdefault('target_views', None)
            view_sampler_cfg = ViewSamplerArbitraryCfg(**filtered_dict)
        elif view_sampler_name == 'all':
            valid_fields = {f.name for f in fields(ViewSamplerAllCfg)}
            filtered_dict = {k: v for k, v in view_sampler_dict.items() if k in valid_fields}
            filtered_dict['name'] = 'all'  # Ensure name is set
            view_sampler_cfg = ViewSamplerAllCfg(**filtered_dict)
        else:
            # Default to uniform
            filtered_dict = {'name': 'uniform', 'num_context_views': 3, 'num_target_views': 1}
            view_sampler_cfg = ViewSamplerUniformCfg(**filtered_dict)
        
        # Replace the dict with the structured config object
        thuman_cfg_dict['view_sampler'] = view_sampler_cfg
    
    # Create DatasetTHumanCfg directly from the dict (avoiding OmegaConf Literal issues)
    # Convert view_sampler back to dict for dacite
    if 'view_sampler' in thuman_cfg_dict and not isinstance(thuman_cfg_dict['view_sampler'], dict):
        # Convert dataclass to dict
        from dataclasses import asdict
        thuman_cfg_dict['view_sampler'] = asdict(thuman_cfg_dict['view_sampler'])
    
    # Use load_typed_config to create DatasetTHumanCfg from the dict
    thuman_cfg_obj = load_typed_config(
        OmegaConf.create(thuman_cfg_dict),
        DatasetTHumanCfg
    )
    
    # Create wrapper manually
    thuman_cfg = DatasetTHumanCfgWrapper(thuman=thuman_cfg_obj)
    dataset_cfg_wrappers = [thuman_cfg]
    
    # Use get_dataset (same as main.py) - this handles everything properly
    step_tracker = None  # Not needed for visualization
    datasets = get_dataset(dataset_cfg_wrappers, 'test', step_tracker)
    
    if not datasets:
        raise ValueError("No THuman datasets were created")
    
    dataset = datasets[0]  # Take first THuman dataset
    
    # THuman is an IterableDataset, so we need to iterate
    print(f"  Iterating through THuman dataset to find sample {sample_idx}...")
    iterator = iter(dataset)
    for i, batch in enumerate(iterator):
        if i == sample_idx:
            return batch, 'thuman'
        if i > sample_idx + 100:  # Safety limit
            break
    
    raise ValueError(f"Could not get sample {sample_idx} from THuman dataset (tried {sample_idx + 1} samples)")


def main():
    parser = argparse.ArgumentParser(
        description='Visualize coordinate frames for MVHN and THuman datasets',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Visualize MVHN with defaults
  python visualize_coordinate_frames.py --mvhn-only
  
  # Visualize both with config files
  python visualize_coordinate_frames.py --mvhn-config config/dataset/mvhn.yaml --thuman-config config/dataset/thuman.yaml
  
  # Visualize specific samples
  python visualize_coordinate_frames.py --mvhn-sample-idx 5 --thuman-sample-idx 10
        """
    )
    parser.add_argument('--mvhn-config', type=str, default=None,
                       help='Path to MVHN config file (optional, uses defaults if not provided)')
    parser.add_argument('--thuman-config', type=str, default=None,
                       help='Path to THuman config file (defaults to config/dataset/thuman.yaml)')
    parser.add_argument('--mvhn-sample-idx', type=int, default=0,
                       help='Sample index for MVHN dataset (default: 0)')
    parser.add_argument('--thuman-sample-idx', type=int, default=0,
                       help='Sample index for THuman dataset (default: 0)')
    parser.add_argument('--output-dir', type=str, default='coordinate_frame_visualizations',
                       help='Output directory for plots (default: coordinate_frame_visualizations)')
    parser.add_argument('--mvhn-only', action='store_true',
                       help='Only visualize MVHN')
    parser.add_argument('--thuman-only', action='store_true',
                       help='Only visualize THuman')
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    success_count = 0
    
    # Visualize MVHN
    if not args.thuman_only:
        try:
            print("=" * 60)
            print("Loading MVHN sample...")
            print("=" * 60)
            mvhn_batch, _ = load_mvhn_sample(args.mvhn_config, args.mvhn_sample_idx)
            print(f"✓ Loaded MVHN sample {args.mvhn_sample_idx}")
            print(f"  Scene: {mvhn_batch.get('scene', ['unknown'])[0]}")
            print(f"  Context views: {mvhn_batch['context']['extrinsics'].shape}")
            
            print("\nVisualizing MVHN...")
            visualize_dataset_sample(
                mvhn_batch, 
                'mvhn', 
                output_dir / f'mvhn_sample_{args.mvhn_sample_idx}.png',
                idx=0
            )
            success_count += 1
        except Exception as e:
            print(f"✗ Error visualizing MVHN: {e}")
            import traceback
            traceback.print_exc()
    
    # Visualize THuman
    if not args.mvhn_only:
        try:
            print("\n" + "=" * 60)
            print("Loading THuman sample...")
            print("=" * 60)
            if args.thuman_config is None:
                print("  Using default config: config/dataset/thuman.yaml")
                print("  (Use --thuman-config to specify a different config)")
            else:
                print(f"  Using config: {args.thuman_config}")
            
            thuman_batch, _ = load_thuman_sample(args.thuman_config, args.thuman_sample_idx)
            print(f"✓ Loaded THuman sample {args.thuman_sample_idx}")
            print(f"  Scene: {thuman_batch.get('scene', 'unknown')}")
            print(f"  Context views: {thuman_batch['context']['extrinsics'].shape}")
            
            print("\nVisualizing THuman...")
            visualize_dataset_sample(
                thuman_batch,
                'thuman',
                output_dir / f'thuman_sample_{args.thuman_sample_idx}.png',
                idx=0
            )
            success_count += 1
        except Exception as e:
            print(f"✗ Error visualizing THuman: {e}")
            print("\nTip: Try providing a config file:")
            print("  --thuman-config config/dataset/thuman.yaml")
            print("  or")
            print("  --thuman-config config/experiment/train_thuman2.0_3views_res1024.yaml")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 60)
    if success_count > 0:
        print(f"✓ Successfully created {success_count} visualization(s)")
        print(f"  Saved to: {output_dir}")
    else:
        print("✗ No visualizations were created. Check errors above.")
    print("=" * 60)


if __name__ == '__main__':
    main()

