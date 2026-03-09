import os
from pathlib import Path

import numpy as np
import torch
from pytorch3d.structures import Meshes
from pytorch3d.renderer import OrthographicCameras, RasterizationSettings, MeshRasterizer
from pytorch3d.ops import interpolate_face_attributes


def load_smplx_uv_mesh_constants(obj_path: Path | str, smplx_npz_path: str):
    """
    Load constant SMPL-X mesh for UV projection: vertices, faces, verts_uv, and LBS weights.
    Returns (vertices, faces, verts_uv, lbs_weights) as numpy arrays, or None if loading fails.
    Used when computing uv_map/uv_valid outside the dataloader (e.g. on GPU in a data shim).
    """
    obj_path = Path(obj_path)
    if not obj_path.exists():
        return None
    vertices = []
    uvs = []
    face_verts = []
    face_uvs = []
    with open(obj_path, "r") as f:
        for line in f:
            tokens = line.split()
            if not tokens:
                continue
            if tokens[0] == "v":
                vertices.append([float(tokens[1]), float(tokens[2]), float(tokens[3])])
            elif tokens[0] == "vt":
                uvs.append([float(tokens[1]), float(tokens[2])])
            elif tokens[0] == "f":
                for t in tokens[1:]:
                    parts = t.split("/")
                    vi = int(parts[0]) - 1
                    vti = int(parts[1]) - 1 if len(parts) > 1 and parts[1] else -1
                    face_verts.append(vi)
                    face_uvs.append(uvs[vti] if vti >= 0 else [0.0, 0.0])
    vertices = np.array(vertices, dtype=np.float32)
    uvs = np.array(uvs, dtype=np.float32)
    face_verts = np.array(face_verts, dtype=np.int64)
    face_uvs = np.array(face_uvs, dtype=np.float32)
    F = len(face_verts) // 3
    faces = face_verts.reshape(F, 3)
    n_verts = vertices.shape[0]
    verts_uv = np.zeros((n_verts, 2), dtype=np.float32)
    for i in range(0, len(face_verts), 3):
        for k in range(3):
            vi = face_verts[i + k]
            verts_uv[vi] = face_uvs[i + k]
    try:
        from smplx import SMPLX
        smplx_model = SMPLX(model_path=smplx_npz_path)
        lbs_weights = smplx_model.lbs_weights.detach().cpu().numpy()
    except Exception:
        return None
    if lbs_weights.shape[0] != n_verts:
        return None
    return vertices, faces, verts_uv, lbs_weights

def create_static_uv_lookup(uv_coords, uv_faces, uv_resolution=1024, device="cuda"):
    """
    Run this ONCE. 
    Returns a (H_uv, W_uv) tensor where each pixel contains the integer Face ID it belongs to.
    """
    # Convert UV [0, 1] to NDC [-1, 1] and pad Z
    uv_verts_ndc = (uv_coords * 2.0) - 1.0 
    flat_uv_verts = torch.cat([uv_verts_ndc, torch.zeros_like(uv_verts_ndc[..., :1])], dim=-1)
    
    uv_mesh = Meshes(verts=flat_uv_verts, faces=uv_faces).to(device)
    # UV layout lies in XY plane at z=0. OrthographicCameras looks along +Z, so world z=0 must map to z_cam > 0.
    # T = (0,0,1) gives world origin -> (0,0,1) in camera space so the UV mesh is in front.
    batch_size = flat_uv_verts.shape[0]
    R_uv = torch.eye(3, device=device).unsqueeze(0).expand(batch_size, -1, -1)
    T_uv = torch.zeros(batch_size, 3, device=device)
    T_uv[:, 2] = 1.0  # z_cam = 1 for world z=0
    uv_cameras = OrthographicCameras(device=device, R=R_uv, T=T_uv)
    raster_settings_uv = RasterizationSettings(
        image_size=uv_resolution, 
        faces_per_pixel=1,
        cull_backfaces=False 
    )
    
    uv_rasterizer = MeshRasterizer(cameras=uv_cameras, raster_settings=raster_settings_uv)
    
    # Render the static layout
    uv_fragments = uv_rasterizer(uv_mesh)
    static_uv_pix_to_face = uv_fragments.pix_to_face[0, ..., 0] # Shape: (1024, 1024)
    
    return static_uv_pix_to_face

def get_batched_multiview_uv_masks(
    posed_vertices,          # Shape: (B, V_views, Num_Verts, 3)
    faces,                   # Shape: (Num_Faces, 3) - Standard SMPL-X faces
    cameras,                 # PyTorch3D cameras, batched to size (B * V_views)
    static_uv_pix_to_face,   # Shape: (H_uv, W_uv) - From the setup function
    image_size=(512, 512), 
    num_total_faces=20908,   # Total faces in SMPL-X
    device="cuda"
):
    B, V_views, V, _ = posed_vertices.shape
    B_flat = B * V_views
    
    # 1. Flatten the batch and view dimensions for PyTorch3D
    flat_vertices = posed_vertices.view(B_flat, V, 3)
    
    # Expand the single faces tensor to match the batch size
    flat_faces = faces.unsqueeze(0).expand(B_flat, -1, -1)
    
    # 2. Camera-Space Rasterization (The Z-Buffer)
    body_meshes = Meshes(verts=flat_vertices, faces=flat_faces).to(device)
    
    raster_settings_img = RasterizationSettings(
        image_size=image_size, faces_per_pixel=1
    )
    img_rasterizer = MeshRasterizer(cameras=cameras, raster_settings=raster_settings_img)
    
    # fragments.pix_to_face is Shape: (B_flat, H, W, 1)
    fragments = img_rasterizer(body_meshes)
    cam_pix_to_face = fragments.pix_to_face[..., 0] # Shape: (B_flat, H, W)
    
    # 3. Fast Batched Visibility Masking (Avoiding Ragged Tensors)
    # We create a boolean matrix tracking which faces are visible per flattened batch item
    visible_faces = torch.zeros((B_flat, num_total_faces), dtype=torch.bool, device=device)
    
    # Ignore the background (-1)
    valid_mask = cam_pix_to_face > -1 
    
    # Get the batch indices and face IDs for every valid hit
    b_indices = torch.arange(B_flat, device=device).view(-1, 1, 1).expand(-1, image_size[0], image_size[1])[valid_mask]
    f_indices = cam_pix_to_face[valid_mask]
    
    # Scatter True into the specific faces that were hit by the camera rays
    visible_faces[b_indices, f_indices] = True
    
    # 4. Paint the UV Masks Instantly via Broadcasting
    H_uv, W_uv = static_uv_pix_to_face.shape
    uv_valid = static_uv_pix_to_face > -1
    
    uv_masks_flat = torch.zeros((B_flat, H_uv, W_uv), dtype=torch.float32, device=device)
    
    # For every valid UV pixel, look up its Face ID, and check if that Face ID is visible in `visible_faces`
    uv_masks_flat[:, uv_valid] = visible_faces[:, static_uv_pix_to_face[uv_valid]].float()
    
    # 5. Unflatten back to the original Multi-View shape
    uv_masks = uv_masks_flat.view(B, V_views, H_uv, W_uv)

    return uv_masks


def build_pytorch3d_cameras_from_w2c_k(
    w2c: torch.Tensor,
    K: torch.Tensor,
    image_size: tuple[int, int],
    device: torch.device,
):
    """
    Build PyTorch3D cameras for mesh rasterization from world-to-camera 4x4 and 3x3 K.

    w2c: (B, 4, 4) or (B, 3, 4) world-to-camera.
    K: (B, 3, 3) intrinsics (fx, fy, cx, cy).
    image_size: (H, W) in pixels.
    Returns a PyTorch3D PerspectiveCameras-compatible object for the rasterizer.
    """
    from pytorch3d.utils.camera_conversions import cameras_from_opencv_projection

    if w2c.dim() == 3 and w2c.shape[1] == 4:
        R = w2c[:, :3, :3].to(device)
        tvec = w2c[:, :3, 3].to(device)
    else:
        R = w2c[:, :3, :3].to(device)
        tvec = w2c[:, :3, 3].to(device)

    image_sizes = torch.tensor(
        [image_size] * R.shape[0], dtype=torch.float32, device=device
    )  # (B, 2) H, W
    cameras = cameras_from_opencv_projection(
        R=R, tvec=tvec, camera_matrix=K.to(device), image_size=image_sizes
    )
    return cameras


def get_batched_image_space_uv(
    posed_vertices: torch.Tensor,
    faces: torch.Tensor,
    verts_uv: torch.Tensor,
    cameras,
    image_size: tuple[int, int],
    device: torch.device = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Rasterize the posed mesh per view and compute per-image-pixel (u, v) and validity.

    Compatible with the backbone: uv_map (B, V, H, W, 2), uv_valid (B, V, H, W).
    Image size (H, W) must match the input image resolution used in the model.

    Args:
        posed_vertices: (B, V_views, N_verts, 3) in world coordinates.
        faces: (F, 3) vertex indices, same for all batches.
        verts_uv: (N_verts, 2) UV coordinates in [0, 1] per vertex.
        cameras: PyTorch3D cameras, batch size B * V_views (same order as posed_vertices flattened).
        image_size: (H, W) target image size.
    Returns:
        uv_map: (B, V_views, H, W, 2) float, (u, v) at each pixel; invalid pixels are 0.
        uv_valid: (B, V_views, H, W) bool, True where the ray hit the mesh.
    """
    if device is None:
        device = posed_vertices.device
    B, V_views, N_verts, _ = posed_vertices.shape
    B_flat = B * V_views

    flat_vertices = posed_vertices.reshape(B_flat, N_verts, 3).to(device)
    flat_faces = faces.unsqueeze(0).expand(B_flat, -1, -1).to(device)

    meshes = Meshes(verts=flat_vertices, faces=flat_faces)

    raster_settings = RasterizationSettings(
        image_size=image_size,
        faces_per_pixel=1,
    )
    rasterizer = MeshRasterizer(cameras=cameras, raster_settings=raster_settings)
    fragments = rasterizer(meshes)

    pix_to_face = fragments.pix_to_face  # (B_flat, H, W, 1)
    bary_coords = fragments.bary_coords   # (B_flat, H, W, 1, 3)

    # Packed face attributes: (B_flat * F, 3, 2) for UV per face vertex
    F = faces.shape[0]
    face_uvs = verts_uv[faces].to(device)  # (F, 3, 2)
    face_uvs_packed = face_uvs.unsqueeze(0).expand(B_flat, -1, -1, -1).reshape(
        B_flat * F, 3, 2
    ).contiguous()

    # Interpolate (u, v) per pixel: (B_flat, H, W, 1, 2)
    uv_at_pixels = interpolate_face_attributes(
        pix_to_face, bary_coords, face_uvs_packed
    )

    uv_map_flat = uv_at_pixels.squeeze(3)  # (B_flat, H, W, 2)
    valid_flat = (pix_to_face[..., 0] >= 0)  # (B_flat, H, W)

    uv_map_flat = uv_map_flat.clamp(0.0, 1.0)
    uv_map = uv_map_flat.view(B, V_views, image_size[0], image_size[1], 2)
    uv_valid = valid_flat.view(B, V_views, image_size[0], image_size[1])

    return uv_map, uv_valid


import torch
import matplotlib.pyplot as plt
import numpy as np

def visualize_visibility_outputs(
    rgb_crop_tensor,        # (3, H, W) - Your original input image crop
    cam_pix_to_face,        # (H, W)    - From the PyTorch3D rasterizer output
    uv_mask_tensor,         # (H_uv, W_uv) - The final output of our batched function
    uv_texture_image=None   # (Optional) A reference SMPL-X texture map to overlay on
):
    """
    Pulls tensors off the GPU and plots the pipeline step-by-step.
    """
    # 1. Move everything to CPU and convert to NumPy for Matplotlib
    # Assuming rgb_crop is normalized [0, 1]
    rgb_img = rgb_crop_tensor.detach().cpu().permute(1, 2, 0).numpy() 
    
    # Create the silhouette: 1 where the mesh exists, 0 for background
    silhouette = (cam_pix_to_face.detach().cpu() > -1).float().numpy()
    
    uv_mask = uv_mask_tensor.detach().cpu().numpy()

    # 2. Setup the Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # --- Panel 1: Original Cropped Image ---
    axes[0].imshow(rgb_img)
    axes[0].set_title("1. Target Camera View (Crop Bounds)")
    axes[0].axis('off')

    # --- Panel 2: PyTorch3D Rendered Silhouette ---
    # Overlay the silhouette on the RGB image to verify camera alignment
    axes[1].imshow(rgb_img)
    axes[1].imshow(silhouette, cmap='jet', alpha=0.5) # Alpha blend
    axes[1].set_title("2. Rendered Silhouette Overlay\n(If this doesn't match, intrinsics are wrong)")
    axes[1].axis('off')

    # --- Panel 3: The UV Visibility Mask ---
    if uv_texture_image is not None:
        # If you have a base texture, show the mask painted over it
        axes[2].imshow(uv_texture_image)
        axes[2].imshow(uv_mask, cmap='magma', alpha=0.7)
    else:
        # Otherwise, just show the raw binary mask
        axes[2].imshow(uv_mask, cmap='magma')
    
    axes[2].set_title("3. Unrolled UV Visibility Mask")
    axes[2].axis('off')

    plt.tight_layout()
    plt.show()