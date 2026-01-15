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
import src.model.decoder as decoder_module
from src.misc.body_utils import apply_lbs_to_gaussians
import torch.nn.functional as F

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
import struct

def save_as_ply(means, covariances, harmonics, opacities, output_path):
    """
    Saves Gaussians to a PLY file that is fully compatible with 
    Standard 3DGS viewers (Inria, SIBR, antimatter15, Polycam, etc.)
    """
    from scipy.spatial.transform import Rotation
    import struct

    # 1. Decompose covariances to scales and quaternions
    print("Decomposing covariances...")
    cov_flat = covariances.reshape(-1, 3, 3).cpu().numpy()
    
    quaternions = []
    scales = []
    
    for cov in cov_flat:
        # Eigendecomposition
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        scale = np.sqrt(np.maximum(eigenvalues, 1e-7))
        # Ensure right-handed coordinate system
        if np.linalg.det(eigenvectors) < 0:
            eigenvectors[:, 0] *= -1
        
        rot = Rotation.from_matrix(eigenvectors)
        quat = rot.as_quat()  # (x, y, z, w)
        # Standard GS PLY format expects (w, x, y, z)
        quaternions.append([quat[3], quat[0], quat[1], quat[2]])
        scales.append(scale)
    
    quaternions = np.array(quaternions)
    scales = np.array(scales)
    
    # 2. Prepare Spherical Harmonics
    # Standard format splits DC (first 3) from the rest
    sh_coeffs = harmonics.cpu().numpy() # [N, 3, 16]
    f_dc = sh_coeffs[:, :, 0] # [N, 3]
    f_rest = sh_coeffs[:, :, 1:].transpose(0, 2, 1).reshape(len(sh_coeffs), -1) # [N, 45]
    
    # 3. Prepare Opacity (as logit)
    opacities_np = opacities.cpu().numpy()
    # Standard GS stores the raw logit, which the renderer sigmoids
    opacity_logit = np.log(opacities_np / (1 - opacities_np + 1e-7))
    
    # 4. Write PLY
    N = means.shape[0]
    means_np = means.cpu().numpy()
    num_f_rest = f_rest.shape[1]
    
    print(f"Saving {N} Gaussians to {output_path} (Standard Format)...")
    with open(output_path, 'wb') as f:
        # Header
        f.write(b"ply\n")
        f.write(b"format binary_little_endian 1.0\n")
        f.write(f"element vertex {N}\n".encode())
        f.write(b"property float x\n")
        f.write(b"property float y\n")
        f.write(b"property float z\n")
        f.write(b"property float nx\n") # Normals (unused but standard)
        f.write(b"property float ny\n")
        f.write(b"property float nz\n")
        for i in range(3): f.write(f"property float f_dc_{i}\n".encode())
        for i in range(num_f_rest): f.write(f"property float f_rest_{i}\n".encode())
        f.write(b"property float opacity\n")
        for i in range(3): f.write(f"property float scale_{i}\n".encode())
        for i in range(4): f.write(f"property float rot_{i}\n".encode())
        f.write(b"end_header\n")
        
        # Packing: 3(pos) + 3(norm) + 3(dc) + 45(rest) + 1(opacity) + 3(scale) + 4(rot) = 62 items
        fmt = '<' + 'f' * (3 + 3 + 3 + num_f_rest + 1 + 3 + 4)
        
        for i in range(N):
            data = [
                *means_np[i], 0, 0, 0, # Pos + Dummy Normal
                *f_dc[i],              # SH DC
                *f_rest[i],            # SH Rest
                opacity_logit[i],      # Opacity
                *np.log(scales[i]),    # Scale
                *quaternions[i]        # Rotation
            ]
            f.write(struct.pack(fmt, *data))
            
    print(f"✓ Saved Standard 3DGS PLY to {output_path}")

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
        (smplx_params.json)
        (test/)
    """
    img_dir = Path(input_dir) / 'images'
    mask_dir = Path(input_dir) / 'masks'
    img_files = sorted([f for f in img_dir.iterdir() if f.suffix.lower() in ['.jpg', '.jpeg', '.png']])
    masks = sorted([f for f in mask_dir.iterdir() if f.suffix.lower() in ['.jpg', '.jpeg', '.png']])

    intrinsics_path = Path(input_dir) / 'intrinsics.npy'
    extrinsics_path = Path(input_dir) / 'extrinsics.npy'
    # Support both .json and .npy formats for SMPLX params
    smplx_params_path = None
    for ext in ['.npy', '.json']:
        candidate = Path(input_dir) / f'smplx_params{ext}'
        if candidate.exists():
            smplx_params_path = candidate
            break
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
        if intrinsics.ndim == 3:
            intrinsics = intrinsics[None]
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
        if extrinsics.ndim == 3:
            extrinsics = extrinsics[None]
        print(f"Loaded extrinsics from {extrinsics_path}")
    else:
        extrinsics = None

    # load smplx params if available
    if smplx_params_path is not None:
        smplx_params_path = Path(smplx_params_path)
        if smplx_params_path.suffix == ".npy":
            # Load from .npy file (saved as dictionary with numpy arrays)
            loaded_data = np.load(smplx_params_path, allow_pickle=True)
            if isinstance(loaded_data, np.lib.npyio.NpzFile):
                # .npz file (multiple arrays)
                smplx_params = {k: loaded_data[k] for k in loaded_data.files}
            else:
                # Single .npy file (dictionary saved with allow_pickle=True)
                smplx_params = loaded_data.item() if isinstance(loaded_data, np.ndarray) else loaded_data
            # Convert numpy arrays to torch tensors and ensure proper batch dimensions
            smplx_params_processed = {}
            for key, value in smplx_params.items():
                if isinstance(value, np.ndarray):
                    tensor = torch.from_numpy(value).float().to(device)
                    # Add batch dimension if needed based on expected shapes
                    # Note: batch['context']['Rs'] already has shape [1, num_v, 55, 3, 3]
                    # So we check if batch dimension is missing (ndim == 4 for Rs, ndim == 3 for Ts)
                    if key in ['Rs', 'Rs_tpose', 'cnl_Rs']:
                        # Expected: [1, num_v, 55, 3, 3] or [num_v, 55, 3, 3]
                        if tensor.ndim == 4:
                            # Missing batch dimension: [num_v, 55, 3, 3] -> [1, num_v, 55, 3, 3]
                            tensor = tensor.unsqueeze(0)
                        elif tensor.ndim == 5 and tensor.shape[0] != 1:
                            # Has batch dimension but wrong size, take first or repeat
                            if tensor.shape[0] > 1:
                                tensor = tensor[0:1]  # Take first batch
                    elif key in ['Ts', 'Ts_tpose', 'cnl_Ts']:
                        # Expected: [1, num_v, 55, 3] or [num_v, 55, 3]
                        if tensor.ndim == 3:
                            # Missing batch dimension: [num_v, 55, 3] -> [1, num_v, 55, 3]
                            tensor = tensor.unsqueeze(0)
                        elif tensor.ndim == 4 and tensor.shape[0] != 1:
                            # Has batch dimension but wrong size, take first or repeat
                            if tensor.shape[0] > 1:
                                tensor = tensor[0:1]  # Take first batch
                    smplx_params_processed[key] = tensor
                else:
                    smplx_params_processed[key] = value
            smplx_params = smplx_params_processed
            print(f"Loaded smplx params from {smplx_params_path} (.npy format)")
        elif smplx_params_path.suffix == ".json":
            import json
            smplx_params = json.load(open(smplx_params_path))
            # Convert JSON lists to torch tensors if needed
            smplx_params_processed = {}
            for key, value in smplx_params.items():
                if isinstance(value, list):
                    tensor = torch.tensor(value).float().to(device)
                    # Add batch dimension if needed (same logic as .npy loading)
                    if key in ['Rs', 'Rs_tpose', 'cnl_Rs']:
                        if tensor.ndim == 4:
                            tensor = tensor.unsqueeze(0)
                        elif tensor.ndim == 5 and tensor.shape[0] != 1:
                            if tensor.shape[0] > 1:
                                tensor = tensor[0:1]
                    elif key in ['Ts', 'Ts_tpose', 'cnl_Ts']:
                        if tensor.ndim == 3:
                            tensor = tensor.unsqueeze(0)
                        elif tensor.ndim == 4 and tensor.shape[0] != 1:
                            if tensor.shape[0] > 1:
                                tensor = tensor[0:1]
                    smplx_params_processed[key] = tensor
                else:
                    smplx_params_processed[key] = value
            smplx_params = smplx_params_processed
            print(f"Loaded smplx params from {smplx_params_path} (.json format)")
        else:
            raise ValueError(f"Unsupported SMPLX params file format: {smplx_params_path.suffix}")
    else:
        smplx_params = None

    # can recursively call 'test_dir' to load test views with the same logic above
    return images, masks, intrinsics, extrinsics, smplx_params, test_dir

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
    images, masks, intrinsics, extrinsics, smplx_params, test_dir = load_data_dir(input_dir, image_size, device)
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
        'extrinsics': extrinsics,
        'index': torch.arange(num_v, device=device)[None],
        'near': torch.ones(1, num_v, device=device) * 0.1,
        'far': torch.ones(1, num_v, device=device) * 100.0,
        'overlap': torch.ones(1, num_v, num_v, device=device),
        'use_smplx': torch.ones(1, num_v, dtype=torch.bool, device=device),
    }

    # if SMPLX parameters are provided, update context:
    # NOTE: we assume that these smplx_parameters are the same ones to be used for target rendering
    if smplx_params is not None:
        context.update(smplx_params)
    
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
        gaussians = encoder(batch, global_step=step)

    # Save canonical results (stored in gaussians.npz)
    # ! this saves gaussians in canonical average body T-pose
    means = gaussians.means[0].cpu().numpy()
    valid = np.linalg.norm(means, axis=-1) < 1e7
    np.savez(output_path / 'tpose_gaussians.npz', 
             means=means[valid], 
             covariances=gaussians.covariances[0].cpu().numpy()[valid],
             harmonics=gaussians.harmonics[0].cpu().numpy()[valid],
             opacities=gaussians.opacities[0].cpu().numpy()[valid])
             
    print(f"✓ Success! Output saved to {output_dir}/tpose_gaussians.npz")

    # * PLY export that uses smplx 'betas' to warp the template body to subject identity
    # curr_cnl_Rs = context['cnl_Rs'][:, 0].clone().detach().to(device)
    # curr_cnl_Ts = context['cnl_Ts'][:, 0].clone().detach().to(device)
    shaped_means, shaped_covs = apply_lbs_to_gaussians(
        gaussians.means,
        gaussians.covariances,
        context['cnl_Rs'][:,0].clone().detach().to(device),
        context['cnl_Ts'][:,0].clone().detach().to(device),
        F.softmax(gaussians.lbs_weights_bones, dim=-1) # Shape-specific weights
    )

    save_as_ply(
        shaped_means[0][valid], 
        shaped_covs[0][valid], 
        gaussians.harmonics[0][valid], 
        gaussians.opacities[0][valid], 
        output_path / 'gaussians.ply'
    )

    print(f"✓ Success! PLY saved to {output_dir}/gaussians.ply")

    # render test views if provided
    if render and test_dir:
        # this overwrites smplx params with the test views (but these are essentially the same.)
        test_images, test_masks, test_intrinsics, test_extrinsics, smplx_params, _ = load_data_dir(test_dir, image_size, device)
        assert test_intrinsics is not None
        assert test_extrinsics is not None

        target = {
            'image': torch.from_numpy(np.stack(test_images)).unsqueeze(0).to(device).permute(0, 1, 4, 2, 3),
            'mask': torch.from_numpy(np.stack(test_masks)).unsqueeze(0).to(device),
            'intrinsics': test_intrinsics,
            'extrinsics': test_extrinsics,
            'index': torch.arange(num_v, device=device)[None],
            'near': torch.ones(1, num_v, device=device) * 0.1,
            'far': torch.ones(1, num_v, device=device) * 100.0,
            'overlap': torch.ones(1, num_v, num_v, device=device),
            'use_smplx': torch.ones(1, num_v, dtype=torch.bool, device=device),
        }
        if smplx_params is not None:
            target.update(smplx_params)

        render_novel_views(
            target,
            smplx_params,
            output_dir,
            full_cfg,
            gaussians,
            device
        ) # saves in output_dir with same name


def render_novel_views(target_context, smplx_params, output_dir, full_cfg, gaussians, device):
    """Render novel views using pre-computed Gaussians and provided camera parameters."""
    import src.model.decoder as decoder_module

    print(f"Rendering novel views...")

    # Initialize decoder
    decoder = decoder_module.get_decoder(full_cfg.model.decoder).to(device)

    # Use provided extrinsics (assumed to be in C2W format already)
    # If extrinsics are None, we can't render novel views
    if target_context.get('extrinsics', None) is None:
        raise ValueError("Extrinsics are required for novel view rendering")
    if smplx_params is None:
        raise ValueError("SMPLX parameters are required for novel view rendering")

    target_c2w = target_context['extrinsics'].to(device)
    target_intrinsics = target_context['intrinsics'].to(device)

    # For arbitrary poses without pose estimation, we use identity transforms
    # This renders the person "as-is" in the novel camera viewpoint
    num_v = target_c2w.shape[0] if target_c2w.ndim > 2 else 1
    if target_c2w.ndim == 2:
        target_c2w = target_c2w[None]
    if target_intrinsics.ndim == 2:
        target_intrinsics = target_intrinsics[None]
    
    # Identity Human Pose (Static - no pose warping) # TODO: use pose estimation
    # Rs = torch.eye(3, device=device)[None, None, None].repeat(1, num_v, 55, 1, 1)
    # Ts = torch.zeros(1, num_v, 55, 3, device=device)
    # cnl_Rs = torch.eye(3, device=device)[None, None, None].repeat(1, num_v, 69, 1, 1)
    # cnl_Ts = torch.zeros(1, num_v, 69, 3, device=device)

    Rs = smplx_params['Rs'].to(device)
    Ts = smplx_params['Ts'].to(device)
    cnl_Rs = smplx_params['cnl_Rs'].to(device) # (V,69,3,3)
    cnl_Ts = smplx_params['cnl_Ts'].to(device) # (V,69,3)

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
    embed_loc = 'none'
    encoder_cfg.intrinsics_embed_loc = embed_loc
    encoder_cfg.backbone.intrinsics_embed_loc = embed_loc
    encoder_cfg.backbone.intrinsics_embed_type = 'none'
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