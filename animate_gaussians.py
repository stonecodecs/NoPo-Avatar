"""
Standalone script to animate T-pose Gaussians from inference.py output.
Takes tpose_gaussians.npz and a pose sequence directory to create animated videos.

example usage:
python animate_gaussians.py     --gaussians output/tpose_gaussians.npz     --output animation2.mp4 --pose_seq smplx_seq/
"""

import argparse
import os
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn
import moviepy.editor as mpy
from einops import repeat
from PIL import Image
from omegaconf import OmegaConf

# Import necessary modules
from src.model.types import Gaussians
from src.model import decoder as decoder_module
from src.misc.body_utils import (
    get_canonical_global_tfms, 
    body_pose_to_body_RTs, 
    get_global_RTs,
    get_canonical_tfms,
    apply_global_tfm_to_camera,
    _rvec_to_rmtx
)
from src.misc.image_io import save_image
from smplx import SMPLX


def load_tpose_gaussians(npz_path: str, device: torch.device) -> Gaussians:
    """Load T-pose Gaussians from .npz file."""
    print(f"Loading T-pose Gaussians from {npz_path}...")
    data = np.load(npz_path)
    
    # Load data and convert to torch tensors with batch dimension
    means = torch.from_numpy(data['means']).float().to(device).unsqueeze(0)
    covariances = torch.from_numpy(data['covariances']).float().to(device).unsqueeze(0)
    harmonics = torch.from_numpy(data['harmonics']).float().to(device).unsqueeze(0)
    opacities = torch.from_numpy(data['opacities']).float().to(device).unsqueeze(0)
    
    # Load LBS weights if available (required for animation)
    lbs_weights = None
    lbs_weights_bones = None
    idx = None
    if 'lbs_weights' in data:
        lbs_weights = torch.from_numpy(data['lbs_weights']).float().to(device).unsqueeze(0)
        print(f"  ✓ Loaded LBS weights (pose transforms)")
    else:
        print(f"  ⚠ Warning: No LBS weights found - animation may not work correctly")
    
    if 'lbs_weights_bones' in data:
        lbs_weights_bones = torch.from_numpy(data['lbs_weights_bones']).float().to(device).unsqueeze(0)
        print(f"  ✓ Loaded LBS bone weights (identity warping)")
    else:
        print(f"  ⚠ Warning: No LBS bone weights found - identity warping disabled")
    
    if 'idx' in data:
        idx = torch.from_numpy(data['idx']).float().to(device).unsqueeze(0)
    
    gaussians = Gaussians(
        means=means,
        covariances=covariances,
        harmonics=harmonics,
        opacities=opacities,
        lbs_weights=lbs_weights,
        lbs_weights_bones=lbs_weights_bones,
        idx=idx,
    )
    
    print(f"Loaded {means.shape[1]} Gaussians")
    return gaussians


def load_pose_sequence(
    pose_seq_path: str,
    smplx_model: SMPLX,
    template_tpose_joints: torch.Tensor,
    device: torch.device
):
    """
    Load a sequence of poses from JSON files.
    
    Expected JSON format (per frame):
    {
        "betas": [...],
        "trans": [...],
        "root_pose": [...],
        "body_pose": [...],
        "lhand_pose": [...],
        "rhand_pose": [...],
        "jaw_pose": [...],
        "leye_pose": [...],
        "reye_pose": [...],
        "focal": [...],
        "princpt": [...],
        "img_size_wh": [w, h]
    }
    """
    print(f"Loading pose sequence from {pose_seq_path}...")
    
    intrinsics_seq = []
    extrinsics_seq = []
    Rs_seq = []
    Ts_seq = []
    cnl_Rs_seq = []
    cnl_Ts_seq = []
    
    betas = None
    json_files = sorted([f for f in os.listdir(pose_seq_path) if f.endswith('.json')])
    
    if not json_files:
        raise FileNotFoundError(f"No JSON files found in {pose_seq_path}")
    
    for filename in json_files:
        meta = json.load(open(os.path.join(pose_seq_path, filename)))
        if betas is None:
            betas = np.array(meta["betas"])
        
        trans = np.array(meta["trans"])
        global_orient = np.array(meta["root_pose"])
        body_pose = np.array(meta["body_pose"])
        left_hand_pose = np.array(meta["lhand_pose"])
        right_hand_pose = np.array(meta["rhand_pose"])
        jaw_pose = np.array(meta["jaw_pose"])
        leye_pose = np.array(meta["leye_pose"])
        reye_pose = np.array(meta["reye_pose"])
        
        w, h = meta["img_size_wh"]
        intrinsics = np.array([
            [meta["focal"][0], 0, meta["princpt"][0]],
            [0, meta["focal"][1], meta["princpt"][1]],
            [0, 0, 1]
        ], dtype=np.float32)
        # Normalize intrinsics
        intrinsics[0] /= w
        intrinsics[1] /= h
        
        # Get T-pose joints (use template)
        tpose_joints = template_tpose_joints.detach().cpu().numpy()
        
        # Compute camera extrinsics
        Rh = global_orient
        Th = trans
        Th = Th + tpose_joints[0] - _rvec_to_rmtx(Rh) @ tpose_joints[0]
        
        extrinsics = apply_global_tfm_to_camera(
            E=np.eye(4, dtype=np.float32),
            Rh=Rh,
            Th=Th
        )
        
        # Compute body pose transforms
        with torch.no_grad():
            output = smplx_model(
                body_pose=torch.tensor(body_pose)[None].float().to(device),
                betas=torch.tensor(betas)[None].float().to(device),
                left_hand_pose=torch.tensor(left_hand_pose)[None].float().to(device),
                right_hand_pose=torch.tensor(right_hand_pose)[None].float().to(device),
                jaw_pose=torch.tensor(jaw_pose)[None].float().to(device),
                leye_pose=torch.tensor(leye_pose)[None].float().to(device),
                reye_pose=torch.tensor(reye_pose)[None].float().to(device),
                return_full_pose=True,
            )
        full_pose = output.full_pose.detach().cpu().numpy().reshape(55, 3)
        full_pose[0] = 0.  # Zero out root rotation
        
        # Compute Rs and Ts
        cnl_gtfms = get_canonical_global_tfms(tpose_joints, use_smplx=True)
        dst_Rs, dst_Ts = body_pose_to_body_RTs(
            full_pose, tpose_joints, use_smplx=True
        )
        global_Rs, global_Ts = get_global_RTs(
            cnl_gtfms, dst_Rs, dst_Ts, use_smplx=True
        )
        
        # Compute canonical transforms
        cnl_Rs, cnl_Ts = get_canonical_tfms(
            template_tpose_joints, 
            torch.tensor(tpose_joints), 
            use_smplx=True
        )
        
        intrinsics_seq.append(intrinsics)
        extrinsics_seq.append(extrinsics)
        Rs_seq.append(global_Rs)
        Ts_seq.append(global_Ts)
        cnl_Rs_seq.append(cnl_Rs)
        cnl_Ts_seq.append(cnl_Ts)
    
    # Stack and convert to tensors
    extrinsics_seq = torch.tensor(np.stack(extrinsics_seq)).to(device)[None].float().inverse()
    intrinsics_seq = torch.tensor(np.stack(intrinsics_seq)).to(device)[None].float()
    Rs_seq = torch.tensor(np.stack(Rs_seq)).to(device)[None].float()
    Ts_seq = torch.tensor(np.stack(Ts_seq)).to(device)[None].float()
    cnl_Rs_seq = torch.stack(cnl_Rs_seq).to(device)[None].float()
    cnl_Ts_seq = torch.stack(cnl_Ts_seq).to(device)[None].float()
    
    print(f"Loaded {len(json_files)} frames")
    return extrinsics_seq, intrinsics_seq, Rs_seq, Ts_seq, cnl_Rs_seq, cnl_Ts_seq, (h, w)


def render_animation(
    gaussians: Gaussians,
    pose_sequence: tuple,
    decoder,
    output_path: Path,
    fps: int = 20,
    frames_per_batch: int = 20,
    near: float = 0.1,
    far: float = 100.0,
):
    """Render animation using Gaussians and pose sequence."""
    extrinsics, intrinsics, Rs, Ts, cnl_Rs, cnl_Ts, (h, w) = pose_sequence
    num_frames = extrinsics.shape[1]
    
    print(f"Rendering {num_frames} frames...")
    images = []
    
    with torch.no_grad():
        for start_frame in range(0, num_frames, frames_per_batch):
            num_samples = min(frames_per_batch, num_frames - start_frame)
            print(f"  Rendering frames {start_frame} to {start_frame + num_samples - 1}...")
            
            # Prepare camera parameters for this batch
            near_tensor = torch.ones(1, num_samples, device=gaussians.means.device) * near
            far_tensor = torch.ones(1, num_samples, device=gaussians.means.device) * far
            
            # Render
            output, _ = decoder.forward(
                gaussians,
                extrinsics[:, start_frame:start_frame+num_samples],
                intrinsics[:, start_frame:start_frame+num_samples],
                Rs[:, start_frame:start_frame+num_samples],
                Ts[:, start_frame:start_frame+num_samples],
                near_tensor,
                far_tensor,
                (h, w),
                cnl_Rs=cnl_Rs[:, start_frame:start_frame+num_samples],
                cnl_Ts=cnl_Ts[:, start_frame:start_frame+num_samples],
            )
            
            images.extend([
                rgb for rgb in output.color[0].detach().cpu()
            ])
    
    # Save as video
    print(f"Saving video to {output_path}...")
    video = torch.stack(images)
    video = (video.clip(min=0, max=1) * 255).type(torch.uint8).cpu().numpy()
    
    # Convert to list of PIL images for moviepy
    video_frames = [video[i].transpose(1, 2, 0) for i in range(len(video))]
    
    clip = mpy.ImageSequenceClip(video_frames, fps=fps)
    clip.write_videofile(str(output_path), codec='libx264', logger=None)
    
    print(f"✓ Video saved to {output_path}")
    
    # Also save individual frames
    frames_dir = output_path.parent / f"{output_path.stem}_frames"
    frames_dir.mkdir(exist_ok=True)
    for i, img in enumerate(images):
        save_image(img, frames_dir / f"frame_{i:04d}.png")
    print(f"✓ Frames saved to {frames_dir}")


def main():
    parser = argparse.ArgumentParser(description="Animate T-pose Gaussians with pose sequence")
    parser.add_argument(
        '--gaussians',
        type=str,
        required=True,
        help='Path to tpose_gaussians.npz file'
    )
    parser.add_argument(
        '--pose_seq',
        type=str,
        required=True,
        help='Directory containing pose sequence JSON files'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='animation.mp4',
        help='Output video path'
    )
    parser.add_argument(
        '--smplx_model',
        type=str,
        default='datasets/smplx',
        help='Path to SMPLX model directory'
    )
    parser.add_argument(
        '--fps',
        type=int,
        default=20,
        help='Frames per second for output video'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='base_config.yaml',
        help='Path to base config file (for decoder settings)'
    )
    
    args = parser.parse_args()
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load config
    print(f"Loading config from {args.config}...")
    cfg = OmegaConf.load(args.config)
    
    # Initialize SMPLX model
    print(f"Loading SMPLX model from {args.smplx_model}...")
    smplx_model = SMPLX(
        model_path=args.smplx_model,
        gender='neutral',
        use_pca=False,
        flat_hand_mean=True
    ).to(device)
    
    # Get template T-pose joints
    with torch.no_grad():
        template_tpose_joints = smplx_model().joints.detach().cpu()[0, :55]
    
    # Load T-pose Gaussians
    gaussians = load_tpose_gaussians(args.gaussians, device)
    
    # Load pose sequence
    pose_sequence = load_pose_sequence(
        args.pose_seq,
        smplx_model,
        template_tpose_joints,
        device
    )
    
    # Initialize decoder
    print("Initializing decoder...")
    decoder = decoder_module.get_decoder(cfg.model.decoder).to(device)
    decoder.eval()
    
    # Render animation
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    render_animation(
        gaussians,
        pose_sequence,
        decoder,
        output_path,
        fps=args.fps
    )
    
    print("✓ Animation complete!")


if __name__ == '__main__':
    main()
