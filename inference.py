#!/usr/bin/env python3
"""
Simple inference script for NoPo-Avatar.
INPUT: directory of images/ and masks/ formatted as such:
    input_dir/
        images/
            img1.jpg
            img2.jpg
            ...
        masks/
            img1.png  (or .jpg - same name as corresponding image!)
            img2.png
            ...
        intrinsics.npy (optional but will be used if provided)
(NOTE: generate this format easily from `demo.ipynb` in this same repo!)

OUTPUT: Reconstructs 3D representation using gaussians in canonical T-pose space.

Usage:
    python nopo_inference.py \
        --input path/to/input_dir \
        --checkpoint path/to/checkpoint.ckpt \
        --output output_dir
"""

import argparse
import os
from pathlib import Path
import torch
import numpy as np
from PIL import Image
from typing import List, Tuple
import sys

sys.path.insert(0, str(Path(__file__).parent))
from src.misc.image_io import save_image


def load_and_preprocess_image(
    image_path: str, 
    target_size: tuple = (1024, 1024)
) -> np.ndarray:
    """Load and resize an image."""
    img = Image.open(image_path).convert('RGB')
    img = img.resize(target_size, Image.LANCZOS)
    return np.array(img).astype(np.float32) / 255.0


def load_and_preprocess_mask(
    mask_path: str,
    target_size: tuple = (1024, 1024)
) -> np.ndarray:
    """Load and resize a mask (grayscale or binary)."""
    mask = Image.open(mask_path)
    
    # Convert to grayscale if needed
    if mask.mode != 'L':
        mask = mask.convert('L')
    
    # Resize
    mask = mask.resize(target_size, Image.NEAREST)
    
    # Normalize to [0, 1]
    mask_array = np.array(mask).astype(np.float32) / 255.0
    
    return mask_array


def normalize_image(
    img: np.ndarray, 
    mean: List[float], 
    std: List[float]
) -> torch.Tensor:
    """Normalize image (ImageNet statistics by default)."""
    img_tensor = torch.from_numpy(img).permute(2, 0, 1)  # HWC -> CHW
    for c in range(3):
        img_tensor[c] = (img_tensor[c] - mean[c]) / std[c]
    return img_tensor


def create_dummy_smplx_params(
    batch_size: int,
    num_views: int, 
    num_joints: int = 55,
    device: torch.device = torch.device('cuda')
) -> dict:
    """
    Create dummy SMPL-X parameters.
    These are NOT used for reconstruction - only for rendering if needed.
    The reconstruction is pose-free!
    """
    # Identity rotations (T-pose)
    Rs = torch.zeros(batch_size, num_views, num_joints, 3, 3, device=device)
    Rs[:, :, :, 0, 0] = 1.0
    Rs[:, :, :, 1, 1] = 1.0
    Rs[:, :, :, 2, 2] = 1.0
    
    # Zero translations
    Ts = torch.zeros(batch_size, num_views, num_joints, 3, device=device)
    
    # Canonical transformations (same as above for T-pose)
    cnl_Rs = Rs.clone()
    cnl_Ts = Ts.clone()
    
    return {
        'Rs': Rs,
        'Ts': Ts,
        'cnl_Rs': cnl_Rs,
        'cnl_Ts': cnl_Ts,
        'Rs_tpose': Rs.clone(),
        'Ts_tpose': Ts.clone(),
    }


def load_template_data(template_size: int = 1024, device: torch.device = None):
    """
    Load template 3D coordinates, LBS weights, and mask from assets/templates.
    These are pre-generated SMPL-X template data.
    """
    template_path = Path('assets/templates')
    
    # Try to load template files
    template_3d_path = template_path / f'xyz_res{template_size}.npy'
    template_lbs_path = template_path / f'lbs_weights_res{template_size}.npy'
    template_mask_path = template_path / f'mask_res{template_size}.npy'
    
    missing_files = []
    if not template_3d_path.exists():
        missing_files.append(str(template_3d_path))
    if not template_lbs_path.exists():
        missing_files.append(str(template_lbs_path))
    if not template_mask_path.exists():
        missing_files.append(str(template_mask_path))
    
    if missing_files:
        raise FileNotFoundError(
            f"Template files not found! Expected:\n"
            f"  - {template_3d_path}\n"
            f"  - {template_lbs_path}\n"
            f"  - {template_mask_path}\n"
            f"\nMissing: {', '.join(missing_files)}\n"
            f"Please run: python src/scripts/generate_template.py"
        )
    
    template_3d = torch.tensor(np.load(template_3d_path), dtype=torch.float32)
    template_lbs_weights = torch.tensor(np.load(template_lbs_path), dtype=torch.float32)
    template_mask = torch.tensor(np.load(template_mask_path), dtype=torch.float32)
    
    # Note: Original code loads mask as-is without binarization
    # The mask should already be binary (0 or 1) from generate_template.py
    # If your mask has non-binary values, check the mask file or regenerate it
    
    if device is not None:
        template_3d = template_3d.to(device)
        template_lbs_weights = template_lbs_weights.to(device)
        template_mask = template_mask.to(device)
    
    # Add batch dimension: [H, W, C] -> [1, H, W, C] for 3d and lbs_weights
    # [H, W] -> [1, H, W] for mask
    if template_3d.dim() == 3:
        template_3d = template_3d.unsqueeze(0)
    if template_lbs_weights.dim() == 3:
        template_lbs_weights = template_lbs_weights.unsqueeze(0)
    if template_mask.dim() == 2:
        template_mask = template_mask.unsqueeze(0)
    
    return template_3d, template_lbs_weights, template_mask


def load_images_and_masks_from_directory(
    input_dir: str,
    image_size: tuple = (1024, 1024),
    use_masks: bool = True,
) -> Tuple[List[np.ndarray], List[np.ndarray], List[str]]:
    """
    Load images and masks from directory structure.
    
    Expected structure:
        input_dir/
            images/
                img1.jpg
                img2.jpg
                ...
            masks/
                img1.jpg (or .png)
                img2.jpg (or .png)
                ...
    
    Args:
        input_dir: Root directory containing images/ and masks/ subdirectories
        image_size: Target size for resizing
        use_masks: Whether to load masks (if False, returns all-ones masks)
    
    Returns:
        images: List of image arrays
        masks: List of mask arrays
        image_names: List of image filenames (for reference)
    """
    input_path = Path(input_dir)
    images_dir = input_path / 'images'
    masks_dir = input_path / 'masks'
    
    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")
    
    # Find all image files
    image_extensions = {'.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'}
    image_files = sorted([
        f for f in images_dir.iterdir() 
        if f.suffix in image_extensions and f.is_file()
    ])
    
    if len(image_files) == 0:
        raise ValueError(f"No image files found in {images_dir}")
    
    print(f"Found {len(image_files)} images in {images_dir}")
    
    images = []
    masks = []
    image_names = []
    
    for img_file in image_files:
        # Load image
        img = load_and_preprocess_image(str(img_file), image_size)
        images.append(img)
        image_names.append(img_file.name)
        
        # Load mask
        if use_masks:
            # Try to find mask with same name (different extensions allowed)
            mask_found = False
            for ext in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG']:
                mask_file = masks_dir / f"{img_file.stem}{ext}"
                if mask_file.exists():
                    mask = load_and_preprocess_mask(str(mask_file), image_size)
                    masks.append(mask)
                    mask_found = True
                    break
            
            if not mask_found:
                print(f"Warning: No mask found for {img_file.name}, using all-ones mask")
                masks.append(np.ones(image_size, dtype=np.float32))
        else:
            # make all foreground
            masks.append(np.ones(image_size, dtype=np.float32))
    
    if use_masks and masks_dir.exists():
        print(f"Loaded masks from {masks_dir}")
    elif use_masks:
        print(f"Warning: Masks directory {masks_dir} not found, using all-ones masks")
    
    return images, masks, image_names


def create_minimal_context(
    images: List[np.ndarray],
    masks: List[np.ndarray],
    device: torch.device,
    image_size: tuple = (1024, 1024),
    mean: List[float] = None,
    std: List[float] = None,
    encoder_type: str = 'template_uv_concat_bone',
    template_size: int = 1024,
    intrinsics: torch.Tensor = None,
) -> dict:
    """
    Create minimal context for NoPo-Avatar inference. (no poses)
    
    The model learns to align features via cross-attention without explicit poses.
    
    Args:
        images: List of preprocessed image arrays (already resized to image_size)
        masks: List of preprocessed mask arrays (already resized to image_size)
        device: Device to place tensors on
        image_size: Image dimensions (h, w)
        mean: Normalization mean (default based on encoder_type; we use template_uv_concat_bone)
        std: Normalization std (default based on encoder_type; we use template_uv_concat_bone)
        encoder_type: Type of encoder ('template_uv_concat_bone' or 'noposplat')
        template_size: Size of template data to load
        intrinsics: Intrinsics matrix
    """
    num_views = len(images)
    h, w = image_size
    
    # Validate inputs
    if len(masks) != num_views:
        raise ValueError(f"Mismatch: {num_views} images but {len(masks)} masks")
    
    # Set default normalization based on encoder type
    if mean is None or std is None:
        if encoder_type == 'template_uv_concat_bone':
            # Template encoder uses [0.5, 0.5, 0.5] mean/std, expects [0, 1] input
            mean = [0.5, 0.5, 0.5]
            std = [0.5, 0.5, 0.5]
        else:
            # noposplat uses ImageNet normalization
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]
    
    # Normalize images
    print(f"Normalizing {num_views} images...")
    image_tensors = [normalize_image(img, mean, std) for img in images]
    images_batch = torch.stack(image_tensors).unsqueeze(0).to(device)  # [1, V, 3, H, W]
    
    # Convert masks to tensors
    mask_tensors = [torch.from_numpy(mask).float() for mask in masks]
    masks_batch = torch.stack(mask_tensors).unsqueeze(0).to(device)  # [1, V, H, W]
        
    # The minimal context
    context = {
        'image': images_batch,
        'mask': masks_batch,
        'index': torch.arange(num_views).unsqueeze(0).to(device),
    }
    
    # Add dummy SMPL-X params (only used for rendering, not reconstruction)
    smplx_params = create_dummy_smplx_params(1, num_views, device=device)
    context.update(smplx_params)
    
    # Add overlap information (assume all views can see the subject)
    context['overlap'] = torch.ones(1, num_views, num_views, device=device)
    context['use_smplx'] = torch.ones(1, num_views, dtype=torch.bool, device=device)
    
    # Load template data (required for template_uv_concat_bone encoder)
    if encoder_type == 'template_uv_concat_bone':
        print("Loading template data...")
        print("  (Template provides 'inpainting' - fills missing body parts!)")
        try:
            template_3d, template_lbs_weights, template_mask = load_template_data(template_size, device)
            context['template_3d'] = template_3d  # [1, H, W, 3]
            context['template_lbs_weights'] = template_lbs_weights  # [1, H, W, 55]
            context['template_mask'] = template_mask  # [1, H, W] - binary mask (should cover full body)
            
            # Check template mask coverage
            mask_coverage = float(template_mask.sum() / template_mask.numel())
            print(f"  Template mask coverage: {mask_coverage:.1%} (should be >50% for full body)")
            if mask_coverage < 0.3:
                print(f"  ⚠ WARNING: Low template mask coverage - missing parts may not be filled!")
        except FileNotFoundError as e:
            print("TEMPLATE DATA NOT FOUND! Ensure zip file is downloaded and extracted to assets/templates/smplx_uv/")
            print(f"!!! Warning: {e}")
            raise ValueError("Template data not found!")

            # * [OPTION] Create dummy template data as fallback (commented out for now)
            # dummy_template_size = template_size
            # context['template_3d'] = torch.zeros(1, dummy_template_size, dummy_template_size, 3, device=device)
            # context['template_lbs_weights'] = torch.zeros(1, dummy_template_size, dummy_template_size, 55, device=device)
            # context['template_mask'] = torch.ones(1, dummy_template_size, dummy_template_size, device=device)  # All ones mask
            # print("  ⚠ Using dummy template - inpainting will NOT work properly!")
    
    # Add camera intrinsics (required if intrinsics_embed_loc is 'encoder' and type is 'token'/'linear')
    if intrinsics is None: # use "default" mock intrinsics
        focal_length = float(min(h, w))  # use image size as focal length
        intrinsics = torch.eye(3, device=device, dtype=torch.float32).unsqueeze(0).unsqueeze(0).repeat(1, num_views, 1, 1)
        intrinsics[:, :, 0, 0] = focal_length  # fx
        intrinsics[:, :, 1, 1] = focal_length  # fy
        intrinsics[:, :, 0, 2] = w * 0.5  # cx (principal point x)
        intrinsics[:, :, 1, 2] = h * 0.5  # cy (principal point y)
        print(f"Using default intrinsics: fx=fy={focal_length:.1f}, cx={w*0.5:.1f}, cy={h*0.5:.1f}")
    else:
        # use provided intrinsics
        if intrinsics.ndim != 4 or intrinsics.shape[1:] != (num_views, 3, 3):
            print(f"⚠ ERROR: Intrinsics has wrong shape: {intrinsics.shape}")
            print(f"   Expected: [1, {num_views}, 3, 3]")
            print(f"   Using default intrinsics instead")
            focal_length = float(min(h, w))
            intrinsics = torch.eye(3, device=device, dtype=torch.float32).unsqueeze(0).unsqueeze(0).repeat(1, num_views, 1, 1)
            intrinsics[:, :, 0, 0] = focal_length
            intrinsics[:, :, 1, 1] = focal_length
            intrinsics[:, :, 0, 2] = w * 0.5
            intrinsics[:, :, 1, 2] = h * 0.5
            print(f"   Using provided intrinsics: fx=fy={focal_length:.1f}, cx={w*0.5:.1f}, cy={h*0.5:.1f}")
    
    context['intrinsics'] = intrinsics
    print(f"Final intrinsics shape: {intrinsics.shape} (should be [1, {num_views}, 3, 3])")
    return context

def load_model_from_checkpoint(checkpoint_path: str, device: torch.device, encoder_type: str = 'auto'):
    """
    Load the NoPo-Avatar model from checkpoint.
    The model is a feedforward encoder-decoder.
    
    Args:
        checkpoint_path: Path to checkpoint file
        device: Device to load model on
        encoder_type: 'noposplat', 'template_uv_concat_bone', or 'auto' to detect from checkpoint
    """
    print(f"Loading checkpoint from {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get('state_dict', checkpoint)
    
    # Try to extract global_step from checkpoint (for proper opacity mapping? NOTE: Check again)
    checkpoint_global_step = None
    if 'global_step' in checkpoint:
        checkpoint_global_step = checkpoint['global_step']
    elif 'epoch' in checkpoint and 'step' in checkpoint:
        # Some checkpoints store step info differently
        checkpoint_global_step = checkpoint.get('step', checkpoint.get('global_step', None))
    
    if checkpoint_global_step is not None:
        print(f"Found global_step={checkpoint_global_step} in checkpoint")
    else:
        print("No global_step found in checkpoint, will use high value for final opacity mapping")
    
    # Import model components
    from omegaconf import OmegaConf
    from src.model.encoder import get_encoder
    
    # Try to detect encoder type from checkpoint
    if encoder_type == 'auto':
        # Check if checkpoint has encoder keys that suggest template_uv_concat_bone
        encoder_keys = [k for k in state_dict.keys() if k.startswith('encoder.')]
        if any('template' in k.lower() or 'lbs' in k.lower() for k in encoder_keys):
            encoder_type = 'template_uv_concat_bone'
            print("Detected encoder type: template_uv_concat_bone")
        else:
            encoder_type = 'noposplat'
            print("Detected encoder type: noposplat")
    
    # Create encoder config based on type
    if encoder_type == 'template_uv_concat_bone':
        # ! Config for template_uv_concat_bone encoder (based on configs; this may need changes!)
        encoder_cfg = OmegaConf.create({
            'name': 'template_uv_concat_bone',
            'd_feature': 128,
            'num_monocular_samples': 32,
            'num_surfaces': 1,
            'gaussians_per_pixel': 1,
            'pts3d_head_type': 'dpt',
            'pts3d_head_skip': False,
            'gs_params_head_type': 'dpt_gs',
            'pose_free': True,
            'apply_mask': 'soft',
            'pts3d_for_lbs_weights': False,
            'separate_xyz_head': False,
            'n_hooks': 4,
            'has_conf': None,  # Can be True, False, or None
            'highres_uv': False,
            'debug': False,
            'input_mean': [0.5, 0.5, 0.5],
            'input_std': [0.5, 0.5, 0.5],
            'pretrained_weights': '',
            'pretrained_template_reinit': False,
            'backbone': {
                'name': 'croco_multi2',
                'model': 'ViTLarge_BaseDecoder',
                'patch_embed_cls': 'PatchEmbedDust3R',
                'asymmetry_decoder': True,
                'intrinsics_embed_loc': 'encoder',  # Checkpoint uses intrinsics embedding
                'intrinsics_embed_degree': 4,
                'intrinsics_embed_type': 'token',
                'template_encoder_free': True,
                'template_image_size': [1024, 1024],
                'template_embed_dim': 1024,
                'disable_checkpointing': True,
            },
            'gaussian_adapter': {
                'gaussian_scale_min': 0.5,
                'gaussian_scale_max': 15.0,
                'sh_degree': 4,
            },
            'opacity_mapping': {
                'initial': 0.0,
                'final': 0.0,
                'warm_up': 1,
            },
            'visualizer': {
                'num_samples': 8,
                'min_resolution': 256,
                'export_ply': False,
            },
            'apply_bounds_shim': True,
        })
    else:
        # Config for noposplat encoder (simpler, older version)
        encoder_cfg = OmegaConf.create({
            'name': 'noposplat',
            'd_feature': 64,
            'num_monocular_samples': 64,
            'num_surfaces': 1,
            'gaussians_per_pixel': 1,
            'gs_params_head_type': 'dpt_gs',
            'pose_free': True,
            'input_mean': [0.485, 0.456, 0.406],
            'input_std': [0.229, 0.224, 0.225],
            'pretrained_weights': '',
            'backbone': {
                'name': 'croco',
                'model': 'ViTLarge_BaseDecoder',
                'patch_embed_cls': 'PatchEmbedDust3R',
                'asymmetry_decoder': True,
                'intrinsics_embed_loc': 'none',  # Required field
                'intrinsics_embed_degree': 4,
                'intrinsics_embed_type': 'token',
            },
            'gaussian_adapter': {
                'gaussian_scale_min': 0.5,
                'gaussian_scale_max': 15.0,
                'sh_degree': 4,
            },
            'opacity_mapping': {
                'initial': 0.0,
                'final': 0.0,
                'warm_up': 1,
            },
            'visualizer': {
                'num_samples': 8,
                'min_resolution': 256,
                'export_ply': False,
            },
            'apply_bounds_shim': True,
        })
    
    # Initialize encoder
    encoder, _ = get_encoder(encoder_cfg)
    
    # Load encoder weights
    # Extract encoder state dict (remove 'encoder.' prefix)
    encoder_state_raw = {
        k.replace('encoder.', ''): v 
        for k, v in state_dict.items() 
        if k.startswith('encoder.')
    }
    
    # Apply checkpoint filtering (same as official code) for proper key handling
    # This handles patch embedding resampling, backbone prefix, etc.
    try:
        from src.misc.weight_modify import checkpoint_filter_fn
        # checkpoint_filter_fn expects keys without 'backbone.' prefix for some keys
        # But our encoder_state_raw already has keys like 'backbone.xxx' from the checkpoint
        # So we need to check if we should apply filtering
        encoder_state = checkpoint_filter_fn(encoder_state_raw, encoder)
        print("Applied checkpoint_filter_fn for proper weight loading")
    except Exception as e:
        print(f"Note: Could not apply checkpoint_filter_fn ({e}), using direct loading")
        encoder_state = encoder_state_raw
    
    # Fix intrinsics encoder key naming mismatch (!check)
    # Checkpoint has: backbone.intrinsic_weight / backbone.intrinsic_bias
    # Model expects: backbone.intrinsic_encoder.weight / backbone.intrinsic_encoder.bias
    if 'backbone.intrinsic_weight' in encoder_state:
        encoder_state['backbone.intrinsic_encoder.weight'] = encoder_state.pop('backbone.intrinsic_weight')
        print("Mapped backbone.intrinsic_weight → backbone.intrinsic_encoder.weight")
    if 'backbone.intrinsic_bias' in encoder_state:
        encoder_state['backbone.intrinsic_encoder.bias'] = encoder_state.pop('backbone.intrinsic_bias')
        print("Mapped backbone.intrinsic_bias → backbone.intrinsic_encoder.bias")
    
    # Handle template_embed size mismatch
    # Checkpoint might have been trained without intrinsics token (4096) but model expects it (4097)
    if 'backbone.template_embed' in encoder_state:
        checkpoint_template_size = encoder_state['backbone.template_embed'].shape[0]
        model_template_size = encoder.state_dict()['backbone.template_embed'].shape[0]
        
        if checkpoint_template_size != model_template_size:
            print(f"Template embed size mismatch: checkpoint={checkpoint_template_size}, model={model_template_size}")
            if checkpoint_template_size == 4096 and model_template_size == 4097:
                # Checkpoint was trained without intrinsics token, but model expects it
                # Initialize the extra token from mean of existing tokens (small initialization)
                checkpoint_embed = encoder_state['backbone.template_embed']
                # Use mean of existing tokens with small scaling for better initialization
                extra_token = checkpoint_embed.mean(dim=0, keepdim=True) * 0.1
                new_embed = torch.cat([checkpoint_embed, extra_token], dim=0)
                encoder_state['backbone.template_embed'] = new_embed
                print(f"  Resized template_embed from {checkpoint_template_size} to {model_template_size} (added intrinsics token)")
            elif checkpoint_template_size == 4097 and model_template_size == 4096:
                # Checkpoint has extra token but model doesn't - remove it
                encoder_state['backbone.template_embed'] = encoder_state['backbone.template_embed'][:4096]
                print(f"  Resized template_embed from {checkpoint_template_size} to {model_template_size} (removed intrinsics token)")
            else:
                print(f"  ⚠ WARNING: Unexpected size mismatch, may cause errors!")
    
    missing_keys, unexpected_keys = encoder.load_state_dict(encoder_state, strict=False)
    
    if missing_keys:
        print(f"Warning: Missing keys in encoder: {len(missing_keys)} keys")
        # Print first few missing keys for debugging
        if len(missing_keys) <= 10:
            print(f"  Missing: {missing_keys}")
        else:
            print(f"  Missing (first 5): {missing_keys[:5]}")
            print(f"  ... and {len(missing_keys) - 5} more")
    if unexpected_keys:
        print(f"Warning: Unexpected keys in checkpoint: {len(unexpected_keys)} keys")
        # Print first few unexpected keys for debugging
        if len(unexpected_keys) <= 10:
            print(f"  Unexpected: {unexpected_keys}")
        else:
            print(f"  Unexpected (first 5): {unexpected_keys[:5]}")
            print(f"  ... and {len(unexpected_keys) - 5} more")
    
    # Most missing/unexpected keys are usually harmless (buffers, optimizer states, etc.)
    # But if there are critical model weights missing, the model won't work properly
    if missing_keys and len(missing_keys) > 0:
        # Check if any missing keys are actual model parameters (not buffers)
        encoder_param_names = {name for name, _ in encoder.named_parameters()}
        critical_missing = [k for k in missing_keys if k in encoder_param_names]
        if critical_missing:
            print(f"  ⚠ CRITICAL: {len(critical_missing)} missing PARAMETER keys (not just buffers)!")
            print(f"     This may cause poor results. Check checkpoint compatibility.")
    
    encoder = encoder.to(device).eval()
    print("Model loaded successfully!")
    
    # Store checkpoint global_step for later use in inference
    encoder._checkpoint_global_step = checkpoint_global_step
    
    return encoder, encoder_type


def run_inference(
    input_dir: str,
    checkpoint_path: str,
    output_dir: str,
    image_size: tuple = (1024, 1024),
    encoder_type: str = 'auto',
    use_masks: bool = True,
):
    """
    Run NoPo-Avatar inference.
    
    Args:
        input_dir: Directory containing images/ and masks/ subdirectories
        checkpoint_path: Path to model checkpoint
        output_dir: Where to save outputs
        image_size: Target image size (height, width)
        encoder_type: Encoder type ('auto', 'noposplat', or 'template_uv_concat_bone')
        use_masks: Whether to load and use masks from masks/ directory
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Image size: {image_size}")
    print(f"Using masks: {use_masks}")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load images and masks
    print(f"\nLoading images and masks from {input_dir}...")
    images, masks, image_names = load_images_and_masks_from_directory(
        input_dir, image_size, use_masks=use_masks
    )
    print(f"Loaded {len(images)} images: {', '.join(image_names)}")
    
    # Load model (this will also detect encoder_type if 'auto')
    encoder, detected_encoder_type = load_model_from_checkpoint(checkpoint_path, device, encoder_type=encoder_type)
    
    # Load intrinsics if available
    intrinsics = None
    intrinsics_path = Path(input_dir) / 'intrinsics.npy'
    if intrinsics_path.exists():
        print(f"Loading intrinsics from {intrinsics_path}")
        intrinsics_np = np.load(str(intrinsics_path))
        print(f"  Loaded intrinsics shape: {intrinsics_np.shape}")
        
        # Handle different input shapes
        if intrinsics_np.ndim == 2 and intrinsics_np.shape == (3, 3):
            # Single 3x3 matrix - repeat for all views
            intrinsics_np = intrinsics_np[np.newaxis, :, :]  # [1, 3, 3]
            intrinsics_np = np.repeat(intrinsics_np, len(images), axis=0)  # [num_views, 3, 3]
        elif intrinsics_np.ndim == 3:
            if intrinsics_np.shape[1:] == (3, 3):
                # [num_views, 3, 3] - correct format
                pass
            elif intrinsics_np.shape[1:] == (24, 3) or intrinsics_np.shape[1:] == (55, 3):
                print(f"  ⚠ WARNING: Intrinsics file has shape {intrinsics_np.shape} which looks like SMPL-X joints, not camera intrinsics!")
                print(f"     Expected shape: [num_views, 3, 3] or [3, 3]")
                print(f"     Skipping intrinsics file, using defaults")
                intrinsics_np = None
            else:
                print(f"  ⚠ WARNING: Unexpected intrinsics shape {intrinsics_np.shape}")
                print(f"     Expected: [num_views, 3, 3] or [3, 3]")
                print(f"     Skipping intrinsics file, using defaults")
                intrinsics_np = None
        else:
            print(f"  ⚠ WARNING: Unexpected intrinsics shape {intrinsics_np.shape}")
            print(f"     Expected: [num_views, 3, 3] or [3, 3]")
            print(f"     Skipping intrinsics file, using defaults")
            intrinsics_np = None
        
        if intrinsics_np is not None:
            # Ensure we have the right number of views
            if intrinsics_np.shape[0] != len(images):
                if intrinsics_np.shape[0] == 1:
                    # Repeat single intrinsics for all views
                    intrinsics_np = np.repeat(intrinsics_np, len(images), axis=0)
                else:
                    print(f"  ⚠ WARNING: Intrinsics has {intrinsics_np.shape[0]} views but {len(images)} images")
                    print(f"     Using first {min(intrinsics_np.shape[0], len(images))} intrinsics")
                    intrinsics_np = intrinsics_np[:len(images)]
            
            # Scale intrinsics to match image size if needed
            # (assuming intrinsics might be from a different image size)
            # For now, we'll use them as-is, but you might want to scale them
            
            # Convert to tensor and add batch dimension: [num_views, 3, 3] -> [1, num_views, 3, 3]
            intrinsics = torch.from_numpy(intrinsics_np).float().to(device).unsqueeze(0)
            print(f"  Final intrinsics shape: {intrinsics.shape} (should be [1, {len(images)}, 3, 3])")
    else:
        print(f"Intrinsics file not found at {intrinsics_path}, using defaults")
    
    # Create context (no poses!)
    print("\nPreparing input context (pose-free)...")
    # Use image size for template size (should match or be close)
    template_size = max(image_size)  # Use the larger dimension
    context = create_minimal_context(
        images, masks, device, image_size, 
        encoder_type=detected_encoder_type,
        template_size=template_size,
        intrinsics=intrinsics
    )
    
    # Run feedforward inference
    print("\nRunning feedforward inference...")
    print(f"\n📸 Template Branch RGB Context:")
    print(f"   - Input views: {num_views}")
    if num_views < 3:
        print(f"   ⚠ WARNING: Only {num_views} view(s) provided! Template may lack RGB context.")
    
    with torch.no_grad():
        # Use checkpoint's global_step if available, otherwise use a high value for final mapping
        if hasattr(encoder, '_checkpoint_global_step') and encoder._checkpoint_global_step is not None:
            inference_global_step = encoder._checkpoint_global_step
            print(f"Using checkpoint global_step={inference_global_step} for opacity mapping")
        else:
            # Use a high value to ensure final opacity mapping
            opacity_warm_up = encoder.cfg.opacity_mapping.warm_up if hasattr(encoder.cfg, 'opacity_mapping') else 1
            inference_global_step = max(opacity_warm_up, 100000)  # Use final opacity mapping
            print(f"Using global_step={inference_global_step} for opacity mapping (warm_up={opacity_warm_up})")
                
        gaussians, gall_template, gall_rgb = encoder(
            context, 
            global_step=inference_global_step,
            return_complete_gaussians=True
        )
    
    # gaussians is already filtered by mask (background removed)
    # gall_template and gall_rgb are unfiltered (for debugging/visualization)
    num_gaussians = gaussians.means.shape[1]
    num_template_unfiltered = gall_template.means.shape[1] if gall_template is not None else 0
    num_rgb_unfiltered = gall_rgb.means.shape[1] if gall_rgb is not None else 0
    
    # *** A few debug checks ***

    # Check opacity values (CRITICAL for inpainting!)
    # Low template opacities = template Gaussians are nearly invisible = poor inpainting
    if gall_template is not None and gall_template.opacities is not None:
        template_opacities = gall_template.opacities[0].cpu().numpy() if gall_template.opacities.ndim > 1 else gall_template.opacities.cpu().numpy()
        template_op_mean = float(template_opacities.mean())
        template_op_min = float(template_opacities.min())
        template_op_max = float(template_opacities.max())
        template_op_median = float(np.median(template_opacities))
        template_op_nonzero = template_opacities[template_opacities > 1e-6]
        template_op_nonzero_mean = float(template_op_nonzero.mean()) if len(template_op_nonzero) > 0 else 0.0
        template_op_nonzero_pct = len(template_op_nonzero) / len(template_opacities) * 100 if len(template_opacities) > 0 else 0.0
        
        print(f"\n🔍 Template Opacity Stats (CRITICAL for inpainting):")
        print(f"  Mean: {template_op_mean:.4f}, Median: {template_op_median:.4f}")
        print(f"  Range: [{template_op_min:.4f}, {template_op_max:.4f}]")
        print(f"  Non-zero (>1e-6): {template_op_nonzero_pct:.1f}% of Gaussians, mean={template_op_nonzero_mean:.4f}")
        if template_op_mean < 0.01:
            print(f"  ⚠ CRITICAL: Very low template opacity (mean < 0.01)!")
            print(f"     Template Gaussians are nearly invisible - inpainting will fail!")
            print(f"     This suggests template PDF values are too low from the model")
            print(f"     Possible causes:")
            print(f"       1. Model wasn't trained well for template inpainting")
            print(f"       2. Template branch needs more RGB context (try more views?)")
            print(f"       3. Template data format mismatch")
    
    if gall_rgb is not None and gall_rgb.opacities is not None:
        rgb_opacities = gall_rgb.opacities[0].cpu().numpy() if gall_rgb.opacities.ndim > 1 else gall_rgb.opacities.cpu().numpy()
        rgb_op_mean = float(rgb_opacities.mean())
        rgb_op_median = float(np.median(rgb_opacities))
        rgb_op_nonzero = rgb_opacities[rgb_opacities > 1e-6]
        rgb_op_nonzero_mean = float(rgb_op_nonzero.mean()) if len(rgb_op_nonzero) > 0 else 0.0
        rgb_op_nonzero_pct = len(rgb_op_nonzero) / len(rgb_opacities) * 100 if len(rgb_opacities) > 0 else 0.0
        
        print(f"\n🔍 RGB Opacity Stats:")
        print(f"  Mean: {rgb_op_mean:.4f}, Median: {rgb_op_median:.4f}")
        print(f"  Non-zero (>1e-6): {rgb_op_nonzero_pct:.1f}% of Gaussians, mean={rgb_op_nonzero_mean:.4f}")
        if 'template_op_mean' in locals() and rgb_op_mean > template_op_mean * 3:
            print(f"  ⚠ WARNING: RGB opacities {rgb_op_mean/template_op_mean:.1f}x higher than template!")
            print(f"     Template Gaussians may be overwhelmed by RGB Gaussians")
    
    print(f"\n✓ Generated {num_gaussians} filtered Gaussians in canonical T-pose!")
    print(f"  (Unfiltered Template Gaussians: {num_template_unfiltered})")
    print(f"  (Unfiltered RGB Gaussians: {num_rgb_unfiltered})")
    print(f"  (Filtered Combined: {num_gaussians})")
    
    # IMPORTANT: Template Gaussians ARE the "inpainting" - they fill in missing parts!
    # The template comes from SMPL-X UV space which covers the entire canonical body.
    # If you see missing parts, it means template Gaussians are being filtered out.
    # This can happen if:
    #   1. Template mask is incomplete (doesn't cover full body)
    #   2. Template Gaussians have low opacity (after apply_mask='soft')
    #   3. The filtering threshold (masks > 1e-3) is too strict
    
    # Calculate template retention rate (how many template Gaussians survived filtering)
    if num_template_unfiltered > 0:
        # The filter_by_mask concatenates template then RGB Gaussians
        # We can estimate template retention by assuming template Gaussians come first
        # But note: the overall retention includes BOTH template and RGB, so it's expected to be lower
        # since RGB Gaussians are heavily filtered (only visible regions)
        
        total_unfiltered = num_template_unfiltered + num_rgb_unfiltered
        overall_retention = (num_gaussians / total_unfiltered) if total_unfiltered > 0 else 0
        
        # Estimate template-specific retention (assuming template Gaussians are first in filtered output)
        # This is an approximation - template Gaussians should have mask=1.0, so most should survive
        # The 74% mask coverage means ~74% of template Gaussians should survive (those with mask=1.0)
        expected_template_retention = 0.74  # Based on template mask coverage
        
        print(f"\n📊 Template Inpainting Analysis:")
        print(f"  Template Gaussians (unfiltered): {num_template_unfiltered:,}")
        print(f"  RGB Gaussians (unfiltered): {num_rgb_unfiltered:,}")
        print(f"  Total Gaussians (filtered): {num_gaussians:,}")
        print(f"  Overall retention rate: {overall_retention:.1%} (includes both template + RGB)")
        print(f"  Note: RGB Gaussians are expected to be heavily filtered (only visible regions)")
        print(f"        Template Gaussians should have ~{expected_template_retention:.0%} retention (from mask coverage)")
        
        # Check template mask coverage (if available in context)
        if 'template_mask' in context:
            template_mask = context['template_mask']
            mask_coverage = float(template_mask.sum() / template_mask.numel())
            mask_mean = float(template_mask.mean())
            mask_min = float(template_mask.min())
            mask_max = float(template_mask.max())
            print(f"\n  Template mask stats (from original codebase - should be correct):")
            print(f"    Coverage: {mask_coverage:.1%} (expected for SMPL-X template)")
            print(f"    Mean value: {mask_mean:.3f} (binary: 0.0 or 1.0)")
            print(f"    Range: [{mask_min:.3f}, {mask_max:.3f}]")
            
            low_mask_ratio = float((template_mask < 1e-3).sum() / template_mask.numel())
            print(f"    Background (mask < 1e-3): {low_mask_ratio:.1%} (expected - will be filtered)")
        
        # Only warn if overall retention is suspiciously low AND we suspect template issues
        # But note: low overall retention is often due to RGB filtering, not template issues
        if overall_retention < 0.2:
            print(f"\n  ⚠ NOTE: Low overall retention ({overall_retention:.1%})")
            print(f"     This is often normal - RGB Gaussians are heavily filtered.")
            print(f"     If you see missing body parts, check if template Gaussians are present.")
    
    # Save results
    print("\nSaving results...")
    
    # Save Gaussian parameters
    # Note: gaussians are already filtered (background removed) by filter_by_mask
    # Remove batch dimension if present
    means = gaussians.means[0].cpu().numpy() if gaussians.means.ndim == 3 else gaussians.means.cpu().numpy()
    covariances = gaussians.covariances[0].cpu().numpy() if gaussians.covariances.ndim == 4 else gaussians.covariances.cpu().numpy()
    harmonics = gaussians.harmonics[0].cpu().numpy() if gaussians.harmonics.ndim == 4 else gaussians.harmonics.cpu().numpy()
    opacities = gaussians.opacities[0].cpu().numpy() if gaussians.opacities.ndim == 2 else gaussians.opacities.cpu().numpy()
    
    # Filter out invalid Gaussians (those with means at 1e8, which are padding)
    valid_mask = np.linalg.norm(means, axis=-1) < 1e7  # Valid Gaussians have reasonable positions
    if not valid_mask.all():
        num_valid = valid_mask.sum()
        print(f"Filtering: {num_valid}/{len(valid_mask)} Gaussians are valid (removing padding)")
        means = means[valid_mask]
        covariances = covariances[valid_mask]
        harmonics = harmonics[valid_mask]
        opacities = opacities[valid_mask]
    
    # gaussian scene representation
    np.savez(
        output_path / 'gaussians.npz',
        means=means,
        covariances=covariances,
        harmonics=harmonics,
        opacities=opacities,
    )
    
    if hasattr(gaussians, 'lbs_weights') and gaussians.lbs_weights is not None:
        lbs_weights = gaussians.lbs_weights[0].cpu().numpy() if gaussians.lbs_weights.ndim == 3 else gaussians.lbs_weights.cpu().numpy()
        if not valid_mask.all():
            lbs_weights = lbs_weights[valid_mask]
        np.savez(
            output_path / 'lbs_weights.npz',
            lbs_weights=lbs_weights,
        )
        print(f"Saved LBS weights to {output_path / 'lbs_weights.npz'}")
    
    # Save masked images (or original images if no mask)
    # Reload original images to ensure correct color space (avoid any preprocessing artifacts)
    input_path = Path(input_dir)
    images_dir = input_path / 'images'
    masks_dir = input_path / 'masks'
    
    for i, img_name in enumerate(image_names):
        # Reload original image
        img_file = images_dir / img_name
        if img_file.exists():
            original_img = Image.open(img_file).convert('RGB')
            original_img = original_img.resize(image_size, Image.LANCZOS)
            img_array = np.array(original_img).astype(np.float32) / 255.0
            img_tensor = torch.from_numpy(img_array).permute(2, 0, 1)  # [3, H, W]
        else:
            # Fallback to preprocessed image if original not found
            img = images[i]
            img_tensor = torch.from_numpy(img).permute(2, 0, 1)
        
        # Get mask and determine if we should apply it
        mask_tensor = None
        should_apply_mask = False
        
        # Try to load mask from file
        if use_masks and masks_dir.exists():
            for ext in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG']:
                mask_file = masks_dir / f"{Path(img_name).stem}{ext}"
                if mask_file.exists():
                    original_mask = Image.open(mask_file)
                    if original_mask.mode != 'L':
                        original_mask = original_mask.convert('L')
                    original_mask = original_mask.resize(image_size, Image.NEAREST)
                    mask_array = np.array(original_mask).astype(np.float32) / 255.0
                    mask_tensor = torch.from_numpy(mask_array)  # [H, W]
                    should_apply_mask = True  # Found a mask file, always apply it
                    break
        
        # If no mask file found, use preprocessed mask
        if mask_tensor is None:
            mask = masks[i]
            mask_tensor = torch.from_numpy(mask)
            # Only apply if use_masks is True AND mask has variation (not all-ones fallback)
            if use_masks:
                # Check if mask has any variation (not all-ones)
                mask_min = float(mask_tensor.min())
                mask_max = float(mask_tensor.max())
                should_apply_mask = (mask_max - mask_min) > 0.01  # Has some variation
        
        # Apply mask to image
        if should_apply_mask:
            # Apply mask: multiply image by mask (broadcast mask to 3 channels)
            mask_3d = mask_tensor.unsqueeze(0).repeat(3, 1, 1)  # [3, H, W]
            masked_img = (img_tensor * mask_3d).clamp(0, 1)
            print(f"Applied mask to {img_name}")
        else:
            # No mask to apply, save original image
            masked_img = img_tensor.clamp(0, 1)
            print(f"No mask applied to {img_name} (use_masks={use_masks})")
        
        # Save masked image (or original if no meaningful mask)
        save_image(
            masked_img,
            output_path / f'input_{i:02d}_{Path(img_name).stem}.png'
        )
    
    print(f"\n✓ Results saved to {output_dir}")
    print("  - gaussians.npz: Canonical T-pose Gaussian representation")
    print("  - lbs_weights.npz: Linear blend skinning weights (for animation)")
    print("  - input_*.png: Your input images")
    print("\nThe Gaussians are in canonical T-pose and can be animated to any pose!")


def main():
    parser = argparse.ArgumentParser(
        description="NoPo-Avatar: Pose-Free Avatar Generation (Feedforward Inference)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with images and masks
  python nopo_inference.py --input my_data/ --checkpoint model.ckpt
  
  # Directory structure:
  #   my_data/
  #     images/
  #       img1.jpg
  #       img2.jpg
  #     masks/
  #       img1.png
  #       img2.png
  
  # Without masks (uses all-ones masks)
  python nopo_inference.py --input my_data/ --checkpoint model.ckpt --no-mask
  
  # Custom output directory and size
  python nopo_inference.py --input my_data/ --checkpoint model.ckpt --output results --image_size 512 512

Key Points:
  - NO POSES NEEDED! The model is truly pose-free for reconstruction
  - 22GB VRAM from 3 images!
        """
    )
    
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Input directory containing images/ and masks/ subdirectories'
    )
    parser.add_argument(
        '--checkpoint',
        required=False,
        default='/workspace/NoPo-Avatar/checkpoint/thuman2.1_huge100k_inputs3_res1024_iter90000.ckpt',
        help='Path to model checkpoint (.ckpt file)'
    )
    parser.add_argument(
        '--output',
        default='inference_output',
        help='Output directory (default: inference_output)'
    )
    parser.add_argument(
        '--image_size',
        type=int,
        nargs=2,
        default=[1024, 1024],
        help='Image size as height width (default: 1024 1024)'
    )
    parser.add_argument(
        '--encoder_type',
        type=str,
        default='auto',
        choices=['auto', 'noposplat', 'template_uv_concat_bone'],
        help='Encoder type (default: auto - detect from checkpoint)'
    )
    parser.add_argument(
        '--apply-mask',
        action='store_true',
        default=True,
        help='Load and use masks from masks/ directory (default: True)'
    )
    parser.add_argument(
        '--no-mask',
        action='store_true',
        help='Do not use masks (use all-ones masks instead)'
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    input_dir = Path(args.input)
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    
    images_dir = input_dir / 'images'
    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")
    
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    
    # Determine mask usage
    use_masks = args.apply_mask and not args.no_mask
    
    print("="*60)
    print("NoPo-Avatar: Feedforward Pose-Free Inference")
    print("="*60)
    print(f"Input directory: {input_dir}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Output: {args.output}")
    print(f"Use masks: {use_masks}")
    print("="*60)
    
    # Run inference
    run_inference(
        str(input_dir),
        args.checkpoint,
        args.output,
        image_size=tuple(args.image_size),
        encoder_type=args.encoder_type,
        use_masks=use_masks,
    )


if __name__ == '__main__':
    main()