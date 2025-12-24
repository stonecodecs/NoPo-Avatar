"""
Simple NoPo-Avatar Inference Script
- Takes input directory with images, masks, intrinsics, and extrinsics
- Runs inference and saves results to output directory
- Can render test views if provided
"""

import argparse
import os
import sys
from pathlib import Path
import torch
import numpy as np
from PIL import Image
from typing import List, Optional
from omegaconf import OmegaConf

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

# must patch this BEFORE the encoder imports it
import src.misc.utils

def safe_inverse_normalize(tensor, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)):
    """A shape-aware version of inverse_normalize that handles 5D tensors."""
    device = tensor.device
    dtype = tensor.dtype
    
    # Ensure mean/std are tensors on the correct device
    t_mean = torch.as_tensor(mean, dtype=dtype, device=device)
    t_std = torch.as_tensor(std, dtype=dtype, device=device)
    
    if tensor.ndim == 5: # [B, V, C, H, W]
        t_mean = t_mean.view(1, 1, 3, 1, 1)
        t_std = t_std.view(1, 1, 3, 1, 1)
    else: # [B, C, H, W] or [C, H, W]
        t_mean = t_mean.view(-1, 1, 1)
        t_std = t_std.view(-1, 1, 1)
        
    return tensor.mul(t_std).add(t_mean)

# Apply the patch globally
src.misc.utils.inverse_normalize = safe_inverse_normalize

# Now import the rest of the modules
import src.model.encoder as encoder_module
from src.misc.image_io import save_image
from src.dataset.data_module import get_data_shim

def load_and_preprocess_image(image_path: str, target_size: tuple = (1024, 1024)) -> np.ndarray:
    img = Image.open(image_path).convert('RGB')
    img = img.resize(target_size, Image.LANCZOS)
    return np.array(img).astype(np.float32) / 255.0

def load_and_preprocess_mask(mask_path: str, target_size: tuple = (1024, 1024)) -> np.ndarray:
    mask = Image.open(mask_path)
    if mask.mode != 'L': mask = mask.convert('L')
    mask = mask.resize(target_size, Image.NEAREST)
    return np.array(mask).astype(np.float32) / 255.0

def load_template_data(template_size: int = 1024, device: torch.device = None):
    template_path = Path('assets/templates')
    t_3d = torch.tensor(np.load(template_path / f'xyz_res{template_size}.npy'), dtype=torch.float32)
    t_lbs = torch.tensor(np.load(template_path / f'lbs_weights_res{template_size}.npy'), dtype=torch.float32)
    t_mask = torch.tensor(np.load(template_path / f'mask_res{template_size}.npy'), dtype=torch.float32)
    
    if device:
        t_3d, t_lbs, t_mask = t_3d.to(device), t_lbs.to(device), t_mask.to(device)
    return t_3d[None], t_lbs[None], t_mask[None]

def load_data_dir(input_dir: str, image_size: tuple = (1024, 1024), device: torch.device = 'cpu'):
    """
    Load images, masks, camera parameters (and possible test views) from input directory.
    Expected structure:
    input_dir/
        images/
            img1.jpg
            img2.jpg
            ...
        masks/
            img1.jpg (or .png)
            img2.jpg (or .png)
        (intrinsics.npy)
        (extrinsics.npy)
        (test/)
    """
    img_dir = Path(input_dir) / 'images'
    mask_dir = Path(input_dir) / 'masks'
    img_files = sorted([f for f in img_dir.iterdir() if f.suffix.lower() in ['.jpg', '.jpeg', '.png']])
    masks = sorted([f for f in mask_dir.iterdir() if f.suffix.lower() in ['.jpg', '.jpeg', '.png']])

    intrinsics_path = Path(input_dir) / 'intrinsics.npy'
    extrinsics_path = Path(input_dir) / 'extrinsics.npy'
    test_dir = Path(input_dir) / 'test'

    intrinsics_path = intrinsics_path if intrinsics_path.exists() else None
    extrinsics_path = extrinsics_path if extrinsics_path.exists() else None
    test_dir = test_dir if test_dir.exists() and test_dir.is_dir() else None

    # these are required!
    if not img_files:
        raise FileNotFoundError(f"No images found in {img_dir}")
    if not masks:
        raise FileNotFoundError(f"No masks found in {mask_dir}")

    # actually load images & masks
    images, masks = [], []
    for f in img_files:
        images.append(load_and_preprocess_image(str(f), image_size))
        m_file = mask_dir / f"{f.stem}.png"
        if not m_file.exists(): m_file = mask_dir / f"{f.stem}.jpg"
        masks.append(load_and_preprocess_mask(str(m_file), image_size) if m_file.exists() else np.ones(image_size))

    # load intrinsics if available
    num_v = len(images)
    if intrinsics_path is not None:
        # this will be tensor of shape [num images, 3, 3]
        # these should be normalized
        intrinsics = torch.from_numpy(np.load(intrinsics_path)).float().to(device)
        print(f"Loaded intrinsics from {intrinsics_path}")
    else: # dummy defaults
        print(f"No intrinsics found at {intrinsics_path}. Using dummy intrinsics.")
        intrinsics = torch.eye(3, device=device)
        intrinsics[0, 2] = 0.5
        intrinsics[1, 2] = 0.5
        intrinsics = intrinsics[None, None].repeat(1, num_v, 1, 1)
    
    # load extrinsics if available
    if extrinsics_path is not None:
        # this will be tensor of shape [num images, 4, 4]
        # these should be in OpenCV c2w format
        extrinsics = torch.from_numpy(np.load(extrinsics_path)).float().to(device)
        print(f"Loaded extrinsics from {extrinsics_path}")
    else:
        extrinsics = None

    # can recursively call 'test_dir' to load test views with the same logic above
    return images, masks, intrinsics, extrinsics, test_dir

def run_inference(
    input_dir: str,
    checkpoint_path: str,
    output_dir: str,
    image_size=(1024, 1024),
    render: bool = False,
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # load config
    base_cfg_path = Path(__file__).parent / 'base_config.yaml'
    full_cfg = get_cfg(base_cfg_path)
    encoder_cfg = full_cfg.model.encoder

    # load checkpoint
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    state_dict = ckpt.get('state_dict', ckpt)
    is_template = any('template' in k for k in state_dict.keys())
    
    # override config with expected defaults
    encoder_cfg = update_cfg_with_defaults(encoder_cfg, state_dict, image_size)
    print(f"Detected Encoder: {encoder_cfg.name}")

    # create output dir
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # load image + mask data provided from user (and test/ if provided)
    print(f"Loading data from {input_dir}...")
    images, masks, intrinsics, extrinsics, test_dir = load_data_dir(input_dir, image_size, device)
    num_v = len(images) # number of input views

    # Initialize Model
    encoder, _ = encoder_module.get_encoder(encoder_cfg)
    
    # Extract and load weights
    clean_state = {k.replace('encoder.', ''): v for k, v in state_dict.items() if k.startswith('encoder.')}
    encoder.load_state_dict(clean_state, strict=False)
    encoder.to(device).eval()

    # Prepare Batch
    context = {
        'image': torch.from_numpy(np.stack(images)).unsqueeze(0).to(device).permute(0, 1, 4, 2, 3),
        'mask': torch.from_numpy(np.stack(masks)).unsqueeze(0).to(device),
        'intrinsics': intrinsics,
        'index': torch.arange(num_v, device=device)[None],
        'near': torch.ones(1, num_v, device=device) * 0.1,
        'far': torch.ones(1, num_v, device=device) * 100.0,
        'overlap': torch.ones(1, num_v, num_v, device=device),
        'use_smplx': torch.ones(1, num_v, dtype=torch.bool, device=device)
    }
    
    # Add template data
    if is_template:
        t_3d, t_lbs, t_mask = load_template_data(1024, device)
        context.update({'template_3d': t_3d, 'template_lbs_weights': t_lbs, 'template_mask': t_mask})

    # Apply same preprocessing as main.py
    shim = get_data_shim(encoder)
    batch = shim({'context': context})['context']

    print("Running inference...")
    with torch.no_grad():
        step = ckpt.get('global_step', 100000)
        gaussians, _, _ = encoder(batch, global_step=step, return_complete_gaussians=True)

    # Save results (stored in gaussians.npz)
    means = gaussians.means[0].cpu().numpy()
    valid = np.linalg.norm(means, axis=-1) < 1e7
    np.savez(output_path / 'gaussians.npz', 
             means=means[valid], 
             covariances=gaussians.covariances[0].cpu().numpy()[valid],
             harmonics=gaussians.harmonics[0].cpu().numpy()[valid],
             opacities=gaussians.opacities[0].cpu().numpy()[valid])
    print(f"✓ Success! Output saved to {output_dir}/gaussians.npz")

    # render test views if provided
    if render and test_dir:
        test_images, test_masks, test_intrinsics, test_extrinsics, _ = load_data_dir(test_dir, image_size, device)
        assert test_intrinsics is not None
        assert test_extrinsics is not None

        render_novel_views(
            test_intrinsics,
            test_extrinsics,
            output_dir,
            full_cfg,
            gaussians,
            device
        ) # saves in output_dir with same name


def render_novel_views(test_intrinsics, test_extrinsics, output_dir, full_cfg, gaussians, device):
    """Render novel views using pre-computed Gaussians and provided camera parameters."""
    import src.model.decoder as decoder_module

    print(f"Rendering novel views...")

    # Initialize decoder
    decoder = decoder_module.get_decoder(full_cfg.model.decoder).to(device)

    # Use provided extrinsics (assumed to be in C2W format already)
    # If extrinsics are None, we can't render novel views
    if test_extrinsics is None:
        raise ValueError("Extrinsics are required for novel view rendering")
    
    target_c2w = test_extrinsics
    target_intrinsics = test_intrinsics

    # For arbitrary poses without pose estimation, we use identity transforms
    # This renders the person "as-is" in the novel camera viewpoint
    num_v = target_c2w.shape[0] if target_c2w.ndim > 2 else 1
    if target_c2w.ndim == 2:
        target_c2w = target_c2w[None]
    if target_intrinsics.ndim == 2:
        target_intrinsics = target_intrinsics[None]
    
    # Identity Human Pose (Static - no pose warping) # TODO: use pose estimation
    Rs = torch.eye(3, device=device)[None, None, None].repeat(1, num_v, 55, 1, 1)
    Ts = torch.zeros(1, num_v, 55, 3, device=device)
    cnl_Rs = torch.eye(3, device=device)[None, None, None].repeat(1, num_v, 55, 1, 1)
    cnl_Ts = torch.zeros(1, num_v, 55, 3, device=device)

    # Render
    with torch.no_grad():
        output, _ = decoder.forward(
            gaussians,
            target_c2w[None] if target_c2w.ndim == 3 else target_c2w, # Ensure batch dim
            target_intrinsics[None] if target_intrinsics.ndim == 3 else target_intrinsics,
            Rs, Ts,
            torch.ones(1, num_v, device=device) * 0.1,  # near plane
            torch.ones(1, num_v, device=device) * 100.0,  # far plane
            (1024, 1024),
            cnl_Rs=cnl_Rs,
            cnl_Ts=cnl_Ts,
            context_cnl_Rs=cnl_Rs,
            context_cnl_Ts=cnl_Ts,
        )

    # Save
    render_path = Path(output_dir) / 'renders'
    render_path.mkdir(exist_ok=True)
    for i, img in enumerate(output.color[0]):
        save_image(img, render_path / f"{i:06d}.png")
    print(f"✓ Rendered {len(output.color[0])} views to {render_path}")

def get_cfg(config_path: str):
    # load config
    if not config_path.exists():
        raise FileNotFoundError("base_config.yaml not found in project root!")
    full_cfg = OmegaConf.load(config_path)
    return full_cfg

def update_cfg_with_defaults(encoder_cfg, state_dict: dict, image_size: tuple = (1024, 1024)):
    # update token count depending on config settings
        # original config overrides
    is_template = any('template' in k for k in state_dict.keys())
    encoder_cfg.name = 'template_uv_concat_bone' if is_template else 'noposplat'

    # update token count depending on config settings
    token_count = state_dict.get('encoder.backbone.template_embed', torch.zeros(4096)).shape[0]
    embed_loc = 'encoder' if token_count == 4097 else 'none'
    encoder_cfg.intrinsics_embed_loc = embed_loc
    encoder_cfg.backbone.intrinsics_embed_loc = embed_loc
    encoder_cfg.backbone.intrinsics_embed_type = 'token'
    encoder_cfg.backbone.template_image_size = [image_size[0], image_size[1]]
    encoder_cfg.debug = False
    encoder_cfg.input_mean = [0.5,0.5,0.5]
    encoder_cfg.input_std = [0.5,0.5,0.5]
    encoder_cfg.highres_uv = False
    if not hasattr(encoder_cfg, 'debug'): encoder_cfg.debug = False
    if not hasattr(encoder_cfg, 'separate_xyz_head'): encoder_cfg.separate_xyz_head = False
    if not hasattr(encoder_cfg, 'pretrained_template_reinit'): encoder_cfg.pretrained_template_reinit = False
    encoder_cfg.backbone.template_image_size = [image_size[0], image_size[1]]
    conf_key = 'encoder.downstream_head1_template.dpt.head.4.weight'
    encoder_cfg.has_conf = (state_dict[conf_key].shape[0] == 4) if conf_key in state_dict else False
    return encoder_cfg


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=False,
        default='/workspace/NoPo-Avatar/checkpoint/thuman2.0_inputs3_res1024_iter50000.ckpt')
    parser.add_argument('--output', type=str, default='results')
    parser.add_argument('--render', action='store_true', default=False, help='Render provided test views.')
    args = parser.parse_args()
    run_inference(args.input, args.checkpoint, args.output, render=args.render)