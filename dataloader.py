"""
Standalone dataloader for MVHumanNet dataset.
Adapted from seva/data modules to be self-contained.
"""

import os
import json
import pickle
import glob
from collections import defaultdict
from einops import repeat 
from tqdm import tqdm
from typing import Tuple, Optional, Dict, Union, Callable, List
from math import isclose

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler
from PIL import Image
import torchvision.transforms.v2 as T
import torch.nn.functional as F

# Optional dependencies
try:
    import pytorch_lightning as pl
    HAS_PL = True
except ImportError:
    HAS_PL = False
    pl = None

try:
    import h5py
    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False

try:
    from datasets import load_from_disk
    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False
    load_from_disk = None

# ============================================================================
# Preprocessing Functions (from seva/data/preprocessing.py)
# ============================================================================

def load_pickle(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)

def load_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

def save_json(data, file_path):
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)

def normalize_camera_poses(camera_poses, scale_factor=1.0):
    """
    Normalize camera poses to a smaller scale while preserving their relative positions.
    """
    positions = np.array([pose[:3, 3] for pose in camera_poses])
    centroid = np.mean(positions, axis=0)
    max_distance = np.max(np.linalg.norm(positions - centroid, axis=1))
    
    normalized_poses = []
    for pose in camera_poses:
        new_pose = pose.copy()
        new_pose[:3, 3] = new_pose[:3, 3] - centroid
        new_pose[:3, 3] = new_pose[:3, 3] / max_distance * scale_factor
        normalized_poses.append(new_pose)
    
    return normalized_poses

def normalize_intrinsics(intrinsics, H, W):
    """
    Normalize intrinsics to a smaller scale while preserving their relative positions.
    """
    new_intrinsics = intrinsics.copy() if isinstance(intrinsics, np.ndarray) else intrinsics.clone().numpy()
    if len(new_intrinsics.shape) == 2:
        new_intrinsics = new_intrinsics[None, ...]
    new_intrinsics[:, 0, 0] = new_intrinsics[:, 0, 0] / W
    new_intrinsics[:, 1, 1] = new_intrinsics[:, 1, 1] / H
    new_intrinsics[:, 0, 2] = new_intrinsics[:, 0, 2] / W
    new_intrinsics[:, 1, 2] = new_intrinsics[:, 1, 2] / H
    if len(intrinsics.shape) == 2:
        new_intrinsics = new_intrinsics[0]
    return new_intrinsics if isinstance(intrinsics, np.ndarray) else new_intrinsics

def create_transform_matrix(R, t, homogeneous=True):
    """Create a 3x4 (4x4 if homogeneous) transform matrix from translation and rotation."""
    transform = np.eye(4) if homogeneous else np.zeros((3, 4))
    transform[:3, :3] = R
    transform[:3, 3] = t
    return transform

def get_bbox_center_and_size(bbox):
    """
    Get center point and size of bbox.
    Returns (center_x, center_y), (width, height)
    """
    if isinstance(bbox, torch.Tensor):
        x1, y1, x2, y2 = bbox.T  # (4, B)
    else:
        x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
        if isinstance(bbox, np.ndarray) and bbox.ndim > 1:
            x1, y1, x2, y2 = bbox.T
    
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    width = x2 - x1
    height = y2 - y1
    return (center_x, center_y), (width, height)

def update_intrinsics(K, crop_x=0, crop_y=0, scale=1.0, crop_first=True, padding_mode=False):
    """
    Update intrinsic matrix for the crop and resizes.
    """
    K_new = K.copy() if type(K) == np.ndarray else K.clone()
    K_new = torch.as_tensor(K_new)
    if crop_first:
        K_new = update_intrinsics_crop(K_new, crop_x, crop_y)
        if not isclose(scale, 1):
            K_new = update_intrinsics_resize(K_new, scale)
    else:
        if not isclose(scale, 1):
            K_new = update_intrinsics_resize(K_new, scale)
        K_new = update_intrinsics_crop(K_new, crop_x, crop_y)
    return K_new

def update_intrinsics_crop(K, crop_x, crop_y):
    """
    Update intrinsic matrix for the crop.
    This can also be used for padding if crop_x and crop_y are negative.
    """
    is_numpy = isinstance(K, np.ndarray)
    crop_x = np.array(crop_x) if isinstance(crop_x, torch.Tensor) else crop_x
    crop_y = np.array(crop_y) if isinstance(crop_y, torch.Tensor) else crop_y
    K_new = K.copy() if is_numpy else K.clone()
    K_new = K_new if is_numpy else K_new.numpy()
    if len(K.shape) == 2:
        K_new = K_new[None, ...]
    K_new[:, 0, 2] = K_new[:, 0, 2] - crop_x
    K_new[:, 1, 2] = K_new[:, 1, 2] - crop_y
    if len(K.shape) == 2:
        K_new = K_new[0, ...]
    return K_new if is_numpy else torch.from_numpy(K_new)

def update_intrinsics_resize(K, scale):
    """
    Update intrinsic matrix for the resize.
    scale is a float or tensor of shape (B,).
    """
    is_numpy = isinstance(K, np.ndarray)
    scale = np.array(scale) if isinstance(scale, torch.Tensor) else scale
    K_new = K.copy() if is_numpy else K.clone()
    K_new = K_new if is_numpy else K_new.numpy()
    if len(K.shape) == 2:
        K_new = K_new[None, ...]
    scale_tensor = np.array(scale).reshape(-1, 1, 1)
    K_new = K_new * scale_tensor
    K_new[:, -1, :] = np.array([0, 0, 1])
    if len(K.shape) == 2:
        K_new = K_new[0, ...]
    return K_new if is_numpy else torch.from_numpy(K_new)

def get_mvhumannet_extrinsics(extrinsics_dict, scale):
    """Get extrinsics for all cameras in a subject directory."""
    extrinsics = create_transform_matrix(
        extrinsics_dict['rotation'], extrinsics_dict['translation'],
        homogeneous=False
    )
    extrinsics[:3, 3] = extrinsics[:3, 3] * scale
    return extrinsics

# ============================================================================
# Geometry Functions (from seva/geometry.py - simplified)
# ============================================================================

DEFAULT_FOV_RAD = 0.9424777960769379  # 54 degrees

def to_hom(X):
    """Get homogeneous coordinates of the input."""
    X_hom = torch.cat([X, torch.ones_like(X[..., :1])], dim=-1)
    return X_hom

def to_hom_pose(pose):
    """Get homogeneous coordinates of the input pose."""
    if pose.shape[-2:] == (3, 4):
        pose_hom = torch.eye(4, device=pose.device)[None].repeat(pose.shape[0], 1, 1)
        pose_hom[:, :3, :] = pose
        return pose_hom
    return pose

def get_default_intrinsics(fov_rad=DEFAULT_FOV_RAD, aspect_ratio=1.0):
    """Get default intrinsics matrix."""
    if not isinstance(fov_rad, torch.Tensor):
        fov_rad = torch.tensor([fov_rad] if isinstance(fov_rad, (int, float)) else fov_rad)
    if aspect_ratio >= 1.0:
        focal_x = 0.5 / torch.tan(0.5 * fov_rad)
        focal_y = focal_x * aspect_ratio
    else:
        focal_y = 0.5 / torch.tan(0.5 * fov_rad)
        focal_x = focal_y / aspect_ratio
    intrinsics = focal_x.new_zeros((focal_x.shape[0], 3, 3))
    intrinsics[:, torch.eye(3, device=focal_x.device, dtype=bool)] = torch.stack(
        [focal_x, focal_y, torch.ones_like(focal_x)], dim=-1
    )
    intrinsics[:, :, -1] = torch.tensor([0.5, 0.5, 1.0], device=focal_x.device, dtype=focal_x.dtype)
    return intrinsics

def get_image_grid(img_h, img_w):
    """Get image grid coordinates."""
    y_range = torch.arange(img_h, dtype=torch.float32).add_(0.5)
    x_range = torch.arange(img_w, dtype=torch.float32).add_(0.5)
    Y, X = torch.meshgrid(y_range, x_range, indexing="ij")
    xy_grid = torch.stack([X, Y], dim=-1).view(-1, 2)
    return to_hom(xy_grid)

def img2cam(X, cam_intr):
    """Convert image coordinates to camera coordinates."""
    return X @ cam_intr.inverse().transpose(-1, -2)

def cam2world(X, pose):
    """Convert camera coordinates to world coordinates."""
    X_hom = to_hom(X)
    pose_inv = torch.linalg.inv(to_hom_pose(pose))[..., :3, :4]
    return X_hom @ pose_inv.transpose(-1, -2)

def get_center_and_ray(img_h, img_w, pose, intr):
    """Given intrinsic/extrinsic matrices, get camera center and ray directions."""
    grid_img = get_image_grid(img_h, img_w)
    grid_3D_cam = img2cam(grid_img.to(intr.device), intr.float())
    center_3D_cam = torch.zeros_like(grid_3D_cam)
    grid_3D = cam2world(grid_3D_cam, pose)
    center_3D = cam2world(center_3D_cam, pose)
    ray = grid_3D - center_3D
    return center_3D, ray, grid_3D_cam

def get_plucker_coordinates(
    extrinsics_src,
    extrinsics,
    intrinsics=None,
    fov_rad=DEFAULT_FOV_RAD,
    target_size=[72, 72],
):
    """
    Compute Plucker coordinates for camera rays.
    
    Args:
        extrinsics_src: Source camera extrinsics (w2c) [4, 4]
        extrinsics: Target camera extrinsics (w2c) [N, 4, 4]
        intrinsics: Camera intrinsics [N, 3, 3] (optional)
        fov_rad: Field of view in radians
        target_size: Target image size [H, W]
    
    Returns:
        plucker: Plucker coordinates [N, 6, H, W]
    """
    if intrinsics is None:
        intrinsics = get_default_intrinsics(fov_rad).to(extrinsics.device)
    
    c2w_src = torch.linalg.inv(extrinsics_src)
    extrinsics_rel = torch.einsum(
        "vnm,vmp->vnp", extrinsics, c2w_src[None].repeat(extrinsics.shape[0], 1, 1)
    )
    
    intrinsics[:, :2] *= extrinsics.new_tensor([target_size[1], target_size[0]]).view(1, -1, 1)
    centers, rays, grid_cam = get_center_and_ray(
        img_h=target_size[0],
        img_w=target_size[1],
        pose=extrinsics_rel[:, :3, :],
        intr=intrinsics,
    )
    
    rays = torch.nn.functional.normalize(rays, dim=-1)
    plucker = torch.cat((rays, torch.cross(centers, rays, dim=-1)), dim=-1)
    plucker = plucker.permute(0, 2, 1).reshape(plucker.shape[0], -1, *target_size)
    return plucker

# ============================================================================
# Cropper Class (from seva/data/cropper.py)
# ============================================================================

def percent_to_absolute(arr, abs_arr):
    """Convert percentage values to absolute pixel values."""
    _arr = torch.as_tensor(arr)
    orig_shape = _arr.shape
    _arr = _arr.reshape(-1)
    decimal_mask = (torch.where((_arr <= 1) & (_arr >= 0))[0]).to(torch.int32)
    if len(decimal_mask) == 0:
        return _arr.reshape(orig_shape).to(torch.float32)
    _arr[decimal_mask] = _arr[decimal_mask] * abs_arr
    return _arr.reshape(orig_shape).to(torch.float32)

class RandomBBoxCropper(object):
    """
    Random (Gaussian) crop transform centered around a 2D bounding box.
    NOTE: images are NOT resized to (576, 576) here!
    - padding: [left, top, right, bottom] (in pixels) only for deterministic crop!
    """
    def __init__(self, random_crop=True, random_crop_prob=1.0, crop_size_bounds=None, padding=[0,0,0,0]):
        self.crop_size_bounds = crop_size_bounds
        self.random_crop = random_crop
        self.random_crop_prob = random_crop_prob
        if not self.random_crop:
            self.random_crop_prob = 0.0

        if isinstance(padding, int) or isinstance(padding, float):
            self.padding = [padding, padding, padding, padding]
        elif isinstance(padding, list):
            self.padding = padding
        else:
            raise ValueError(f"Invalid padding type: {type(padding)}")

    def _get_crop_params(self, bbox: torch.Tensor, K: torch.Tensor, options: dict):
        """Calculate crop parameters based on bbox and intrinsics."""
        W = options["W"]
        H = options["H"]
        B = bbox.shape[0]

        center, size = get_bbox_center_and_size(bbox)
        centers = torch.stack(center, dim=1)
        sizes = torch.stack(size, dim=1)
        center_x, center_y = centers.T
        bbox_W, bbox_H = sizes.T

        bbox_max_dim = torch.maximum(bbox_W, bbox_H)
        
        x1 = torch.floor(center_x - (bbox_max_dim // 2) - self.padding[0]).int()
        y1 = torch.ceil(center_y - (bbox_max_dim // 2) - self.padding[1]).int()
        
        total_width = bbox_max_dim + self.padding[0] + self.padding[2]
        total_height = bbox_max_dim + self.padding[1] + self.padding[3]
        
        x2 = x1 + total_width
        y2 = y1 + total_height
        
        rel_bbox = torch.zeros(B, 4)

        if self.random_crop and options.get("to_crop", False):
            center_mean = options.get("center_mean", centers)
            center_std = options.get("center_std", torch.stack([(W - bbox_W) / 6, (H - bbox_H) / 6], dim=1))
            crop_size_mean = options.get("crop_size_mean", (bbox_W + bbox_H) * 3 / 4)
            crop_size_std = options.get("crop_size_std", (bbox_W + bbox_H) / 2)
            min_crop_size = options.get("min_crop_size", (bbox_max_dim * 3) // 4)

            center_mean = percent_to_absolute(center_mean, torch.tensor([H, W]))
            center_std = torch.as_tensor(center_std)
            crop_size_mean = percent_to_absolute(crop_size_mean, torch.tensor([min(H, W)]))
            crop_size_std = torch.as_tensor(crop_size_std)

            size_sample = torch.clamp(torch.randn(B) * crop_size_std + crop_size_mean, min=min_crop_size, max=bbox_max_dim)

            if self.crop_size_bounds is not None:
                size_sample = torch.clamp(
                    size_sample,
                    min=percent_to_absolute(self.crop_size_bounds[0], torch.tensor([min(H, W)])),
                    max=percent_to_absolute(self.crop_size_bounds[1], torch.tensor([min(H, W)]))
                )

            size_sample_int = size_sample.int()

            x_offset = torch.clamp(
                torch.randn(B,1) * center_std[:,0].view(-1,1) + center_mean[:,0].view(-1,1),
                min=(x1 + size_sample_int // 2).view(-1, 1),
                max=(x2 - size_sample_int // 2).view(-1, 1)
            )
            y_offset = torch.clamp(
                torch.randn(B,1) * center_std[:,1].view(-1,1) + center_mean[:,1].view(-1,1) - 200,
                min=(y1 + size_sample_int // 2).view(-1, 1),
                max=(y2 - size_sample_int // 2).view(-1, 1)
            )

            x1_new = torch.floor(x_offset - (size_sample_int // 2).view(-1, 1)).int().view(-1)
            y1_new = torch.floor(y_offset - (size_sample_int // 2).view(-1, 1)).int().view(-1)
            
            x2_new = x1_new + size_sample_int.view(-1)
            y2_new = y1_new + size_sample_int.view(-1)

            rel_bbox[:, 0] = x1_new - x1
            rel_bbox[:, 1] = y1_new - y1
            rel_bbox[:, 2] = x2_new - x2
            rel_bbox[:, 3] = y2_new - y2
            x1, y1, x2, y2 = x1_new, y1_new, x2_new, y2_new

        if len(K.shape) == 2:
            K_ = repeat(K, 'd1 d2 -> n d1 d2', n=B).detach().clone()
        else:
            K_ = K

        K_new = update_intrinsics(
            torch.as_tensor(K_), 
            crop_x=x1,
            crop_y=y1,
            scale=1,
            crop_first=False,
            padding_mode=True
        )

        scale = 576.0 / (bbox_max_dim + self.padding[0] + self.padding[2])
        rel_bbox = (rel_bbox * scale.view(-1, 1)).int()

        return {
            "bbox": torch.stack([x1, y1, x2, y2], dim=1),
            "K": K_new,
            "relative_bbox": rel_bbox
        }

    def _possibly_pad_img(self, images, x1, y1, x2, y2):
        """Pad the image if the crop parameters extend beyond the image."""
        H, W = images.shape[-2:]
        pad_left = torch.maximum(torch.zeros_like(x1), -x1)
        pad_top = torch.maximum(torch.zeros_like(y1), -y1)
        pad_right = torch.maximum(torch.zeros_like(x2), x2 - W)
        pad_bottom = torch.maximum(torch.zeros_like(y2), y2 - H)
        
        if torch.any(pad_left > 0) or torch.any(pad_top > 0) or torch.any(pad_right > 0) or torch.any(pad_bottom > 0):
            image_list = []
            padding = torch.stack([pad_left.int(), pad_right.int(), pad_top.int(), pad_bottom.int()], dim=1)

            for i, image in enumerate(images):
                image = torch.nn.functional.pad(image, padding[i].tolist(), mode="constant", value=0)
                image_list.append(image)

            new_bbox = torch.stack([x1 + pad_left, y1 + pad_top, x2 + pad_left, y2 + pad_top], dim=1)
            return image_list, new_bbox
        else:
            return images, torch.stack([x1, y1, x2, y2], dim=1)

    def crop_images(self, images, x1, y1, x2, y2):
        """Crop images based on bounding box."""
        cropped_images = []
        for i in range(len(images)):
            if len(images[i].shape) == 2:
                cropped_img = images[i][int(y1[i]):int(y2[i]), int(x1[i]):int(x2[i])]
            else:
                cropped_img = images[i][:, int(y1[i]):int(y2[i]), int(x1[i]):int(x2[i])]
            cropped_images.append(cropped_img)
        return cropped_images

    def __call__(self, images: torch.Tensor, bbox: torch.Tensor, K: torch.Tensor, 
                 face_bboxes: torch.Tensor = None, **kwargs):
        """
        Args:
            images: Tensor of shape (B, C, H, W)
            bbox: Tensor of shape (B, 4) with [x1, y1, x2, y2]
            K: Intrinsics matrix of shape (B, 3, 3) or (3, 3)
            face_bboxes: Tensor of shape (B, 4) with [x1, y1, x2, y2]
        Returns:
            Cropped images, updated intrinsics, relative bbox, face_bboxes, new_bbox, prev_pad_bbox
        """
        options = {
            "H": images.shape[-2],
            "W": images.shape[-1],
            "to_crop": True if torch.rand(1) < self.random_crop_prob else False
        }
        options.update(kwargs)

        crop_params = self._get_crop_params(bbox, K, options)
        bbox = crop_params["bbox"]
        K_new = crop_params["K"]
        rel_bbox = crop_params["relative_bbox"]

        x1, y1, x2, y2 = bbox.T
        prev_pad_bbox = bbox.clone()
        images, bbox = self._possibly_pad_img(images, x1, y1, x2, y2)
        x1, y1, x2, y2 = bbox.T

        if face_bboxes is not None:
            face_bboxes_new = face_bboxes.to(torch.float32)
            no_face_mask = face_bboxes[:,0] != -1
            face_bboxes_new[no_face_mask, 0] = face_bboxes_new[no_face_mask, 0] - x1[no_face_mask].to(torch.float32)
            face_bboxes_new[no_face_mask, 1] = face_bboxes_new[no_face_mask, 1] - y1[no_face_mask].to(torch.float32)
            face_bboxes_new[no_face_mask, 2] = face_bboxes_new[no_face_mask, 2] - x1[no_face_mask].to(torch.float32)
            face_bboxes_new[no_face_mask, 3] = face_bboxes_new[no_face_mask, 3] - y1[no_face_mask].to(torch.float32)
            face_bboxes_new[no_face_mask] = face_bboxes_new[no_face_mask] * (576.0 / torch.maximum((x2 - x1)[no_face_mask].to(torch.float32), (y2 - y1)[no_face_mask].to(torch.float32)).unsqueeze(-1))
            face_bboxes_new[~no_face_mask] = -1
            oob_mask = (face_bboxes_new < 0).any(dim=-1) | (face_bboxes_new > 576.0).any(dim=1)
            face_bboxes_new[oob_mask] = -1
            face_bboxes_new = face_bboxes_new.to(torch.int32)
        else:
            face_bboxes_new = None

        cropped_images = self.crop_images(images, x1, y1, x2, y2)
        return cropped_images, K_new, rel_bbox, face_bboxes_new, bbox.int(), prev_pad_bbox.int()

# ============================================================================
# Camera Constants
# ============================================================================

TOP_RUNG = [
    'CC32871A043', 'CC32871A018', 'CC32871A012', 'CC32871A021',
    'CC32871A060', 'CC32871A006', 'CC32871A042', 'CC32871A041', 
    'CC32871A049', 'CC32871A036', 'CC32871A047', 'CC32871A019',
    'CC32871A020', 'CC32871A056', 'CC32871A009', 'CC32871A014'
]

MIDDLE_RUNG = [
    'CC32871A005', 'CC32871A033', 'CC32871A050', 'CC32871A059',
    'CC32871A017', 'CC32871A034', 'CC32871A032', 'CC32871A052',
    'CC32871A039', 'CC32871A058', 'CC32871A013', 'CC32871A004',
    'CC32871A044', 'CC32871A031', 'CC32871A055', 'CC32871A029'
]

BOTTOM_RUNG = [
    'CC32871A035', 'CC32871A016', 'CC32871A030', 'CC32871A038',
    'CC32871A023', 'CC32871A027', 'CC32871A051', 'CC32871A015',
    'CC32871A022', 'CC32871A057', 'CC32871A048', 'CC32871A008',
    'CC32871A046', 'CC32871A010', 'CC32871A040', 'CC32871A037'
]

CAMERA_RUNGS = [TOP_RUNG, MIDDLE_RUNG, BOTTOM_RUNG]
ALL_CAMERAS = sorted([cam for rung in CAMERA_RUNGS for cam in rung])
CAMERA_TO_INDEX = {cam: idx for idx, cam in enumerate(ALL_CAMERAS)}

# ============================================================================
# Utility Functions
# ============================================================================

def center_cameras(all_c2ws, c2ws):
    """Finds mean position of all_c2ws, then centers cameras by subtracting the mean."""
    ref_c2ws = all_c2ws
    camera_dist_2med = torch.norm(
        ref_c2ws[:, :3, 3] - ref_c2ws[:, :3, 3].median(0, keepdim=True).values,
        dim=-1,
    )
    valid_mask = camera_dist_2med <= torch.clamp(
        torch.quantile(camera_dist_2med, 0.97) * 10,
        max=1e6,
    )
    c2ws[:, :3, 3] -= ref_c2ws[valid_mask, :3, 3].mean(0, keepdim=True)

def scale_cameras(c2ws, camera_scale=2.0):
    """Scale camera positions."""
    camera_dists = c2ws[:, :3, 3].clone()
    translation_scaling_factor = (
        camera_scale
        if torch.isclose(
            torch.norm(camera_dists[0]),
            torch.zeros(1),
            atol=1e-5,
        ).any()
        else (camera_scale / torch.norm(camera_dists[0]))
    )
    c2ws[:, :3, 3] *= translation_scaling_factor

def read_from_hdf5(hdf5_file, *args):
    """Read from HDF5 file (nested keys)."""
    if not HAS_H5PY:
        raise ImportError("h5py is required for HDF5 support. Install with: pip install h5py")
    try:
        with h5py.File(hdf5_file, 'r') as f:
            current = f
            for arg in args:
                current = current[arg]
            
            if isinstance(current, h5py.Dataset):
                return np.array(current)
            elif isinstance(current, h5py.Group):
                return {key: current[key] for key in current.keys()}
            else:
                return current
    except KeyError:
        return None
    except Exception as e:
        print(f"Error reading HDF5 file: {e}")
        return None

def one_hot_encode_segmentation(seg_map: torch.Tensor, num_classes: int, classes_to_use: list = []) -> torch.Tensor:
    """Converts a segmentation label map to a one-hot encoded tensor."""
    if len(classes_to_use) > 0:
        seg_map_ = seg_map[classes_to_use]
    else:
        seg_map_ = seg_map
    if seg_map_.dim() == 3 and seg_map_.shape[0] == 1:
        seg_map_ = seg_map_.squeeze(0)
    
    seg_map_long_ = seg_map_.long()
    one_hot = F.one_hot(seg_map_long_, num_classes=num_classes)
    one_hot_ = one_hot.permute(2, 0, 1)
    return one_hot_.float()

# ============================================================================
# Dataset Class (from seva/data/mvh_dataloader.py - adapted)
# ============================================================================

class MVHumanNetDataset(Dataset):
    def __init__(
        self,
        root_dir,
        num_images,
        latents_dir=None,
        transforms=None,
        pre_scale_intrinsics=0.5,
        data_limit=None,
        only_include=None,
        exclude=None,
        random_crop=False,
        maximal_crop=False,
        white_background=False,
        step_size=60,
        preload_path=None,
        iclight_dataset_path=None,
        infu_dataset_path=None,
        face_bbox_dir=None,
        arcface_embeddings_dir=None,
        crop_padding=60,
        use_inconsistent=False,
        random_crop_prob=0.3,
        ic_sampling_prob=0.7,
        fixed_sampling_ids=None,
        use_sapiens_conditioning=None,
        sapiens_segmentation_channels_to_use=[],
        force_face_ref=False
    ):
        self.root_dir = root_dir
        self.latents_dir = latents_dir
        self.num_images = num_images
        self.transforms = transforms
        self.pre_scale_intrinsics = pre_scale_intrinsics
        self.only_include = set(only_include) if only_include is not None else None
        self.exclude = set(exclude) if exclude is not None else None
        self.data_limit = data_limit
        self.step_size = step_size
        self.random_crop = random_crop
        self.random_crop_prob = random_crop_prob
        self.maximal_crop = maximal_crop
        self.use_inconsistent = use_inconsistent
        self.ic_sampling_prob = ic_sampling_prob
        self.use_sapiens_conditioning = use_sapiens_conditioning
        assert self.use_sapiens_conditioning is None or all(cond in ["depth", "seg_masks", "latents"] for cond in self.use_sapiens_conditioning), "Invalid sapiens conditioning!"
        self.sapiens_segmentation_channels_to_use = sapiens_segmentation_channels_to_use
        self.fixed_sampling_ids = fixed_sampling_ids
        self.adjacent_frame_sampling_prob = 0.2
        self.all_inputs_prob = 0.85
        self.white_background = white_background
        self.preload_path = preload_path
        self.iclight_dataset_path = iclight_dataset_path
        self.infu_dataset_path = infu_dataset_path
        self.face_bbox_dir = face_bbox_dir
        self.arcface_embeddings_dir = arcface_embeddings_dir
        self.force_face_ref = force_face_ref
        
        self.infu_num_images = {}
        if self.infu_dataset_path is not None:
            subjects_to_parse = os.listdir(self.infu_dataset_path)
            for subject_id in subjects_to_parse:
                if self.exclude is not None and subject_id in self.exclude:
                    continue
                if self.only_include is not None and subject_id not in self.only_include:
                    continue
                self.infu_num_images[subject_id] = len(os.listdir(os.path.join(self.infu_dataset_path, subject_id))) - 2

        if self.num_images > 16:
            self.adjacent_frame_sampling_prob = 0.0
        self.crop_padding = crop_padding
        
        self.cam_params = {}
        self.face_bboxes = self._load_face_bboxes() if face_bbox_dir is not None else None
        
        self.is_arrow = False
        self.dataset = None
        
        # Determine if preload_path is an Arrow dataset or JSON file
        if self.preload_path:
            if os.path.isdir(self.preload_path):
                # Check for Arrow dataset indicators
                has_arrow_files = len(glob.glob(os.path.join(self.preload_path, '*.arrow'))) > 0
                has_dataset_info = os.path.exists(os.path.join(self.preload_path, 'dataset_info.json'))
                has_state = os.path.exists(os.path.join(self.preload_path, 'state.json'))
                has_parquet = len(glob.glob(os.path.join(self.preload_path, '*.parquet'))) > 0
                
                if has_arrow_files or has_dataset_info or has_state or has_parquet:
                    if not HAS_DATASETS:
                        raise ImportError(
                            f"Arrow dataset detected at {self.preload_path}, but 'datasets' library is not installed. "
                            "Install with: pip install datasets"
                        )
                    self.is_arrow = True
                    print(f"Detected Arrow dataset at {self.preload_path}")
                elif os.path.exists(os.path.join(self.preload_path, 'metadata.json')):
                    # Might be a JSON file directory structure, check if it's actually a JSON file
                    if os.path.isfile(self.preload_path):
                        # It's actually a file, not a directory
                        if self.preload_path.endswith('.json'):
                            self.is_arrow = False
                        else:
                            raise ValueError(f"Unknown preload_path format: {self.preload_path}")
                    else:
                        # It's a directory but not an Arrow dataset - check if it contains a JSON file
                        json_files = glob.glob(os.path.join(self.preload_path, '*.json'))
                        if json_files:
                            # Use the first JSON file found
                            self.preload_path = json_files[0]
                            self.is_arrow = False
                        else:
                            raise ValueError(
                                f"Directory {self.preload_path} does not appear to be an Arrow dataset "
                                "(no .arrow files, dataset_info.json, state.json, or .parquet files) "
                                "and does not contain JSON files."
                            )
            elif self.preload_path.endswith('.arrow'):
                if not HAS_DATASETS:
                    raise ImportError(
                        f"Arrow dataset file detected: {self.preload_path}, but 'datasets' library is not installed. "
                        "Install with: pip install datasets"
                    )
                self.is_arrow = True
                print(f"Detected Arrow dataset file: {self.preload_path}")
            elif self.preload_path.endswith('.json'):
                self.is_arrow = False
            else:
                # Try to determine by attempting to load
                if os.path.isfile(self.preload_path):
                    # Assume JSON if it's a file
                    self.is_arrow = False
                else:
                    raise ValueError(f"Unknown preload_path format: {self.preload_path}")
        
        if self.is_arrow:
            self.scenes = self._load_scenes_arrow()
        else:
            if self.preload_path and os.path.isdir(self.preload_path):
                raise ValueError(
                    f"preload_path is a directory ({self.preload_path}) but was not detected as an Arrow dataset. "
                    "Please ensure it contains Arrow dataset files (.arrow, dataset_info.json, state.json, or .parquet) "
                    "or provide a path to a JSON file instead."
                )
            self.scenes = self._load_preloaded_filepaths()
            
        self.image_shape = (1500, 2048)

        self.downsample_factor = 8
        self.scale_factor = 0.18215
        self.target_shape = (576, 576)
        self.latent_shape = (self.num_images, 4, self.target_shape[0] // self.downsample_factor, 
                          self.target_shape[1] // self.downsample_factor)

        if self.transforms is None:
            self.transform = T.Compose([
                T.CenterCrop(self.image_shape[0]),
                T.Resize(self.target_shape),
                T.Compose([T.ToImage(), T.ToDtype(torch.float32, scale=True)]),
                T.Normalize([0.5], [0.5])
            ])
            self.mask_transform = T.Compose([
                T.CenterCrop(self.image_shape[0]),
                T.Resize(self.target_shape),
                T.Compose([T.ToImage(), T.ToDtype(torch.float32, scale=True)]),
            ])

        if self.random_crop or self.maximal_crop:
            self.cropper = RandomBBoxCropper(
                random_crop=self.random_crop,
                random_crop_prob=self.random_crop_prob,
                padding=self.crop_padding
            )
            self.transform = T.Compose([
                T.Resize(self.target_shape),
                T.Compose([T.ToImage(), T.ToDtype(torch.float32, scale=True)]),
                T.Normalize([0.5], [0.5])
            ])
            self.mask_transform = T.Compose([
                T.Resize(self.target_shape),
                T.Compose([T.ToImage(), T.ToDtype(torch.float32, scale=True)]),
            ])

        if self.pre_scale_intrinsics != 0.5:
            print("WARNING: pre_scale_intrinsics is not 0.5, which is expected for MVHumanNet!")
        print("MVHN::init done!")

    def _clean_camera_keys(self, data):
        """Clean camera keys from extrinsics dict."""
        cleaned_data = {}
        for key, value in data.items():
            camera_id = key[2:-4]
            cleaned_data[camera_id] = value
        return cleaned_data

    def _load_face_bboxes(self):
        """Load face bboxes from JSON files."""
        all_face_info = {}
        for face_json in os.listdir(self.face_bbox_dir):
            all_face_info.update(load_json(os.path.join(self.face_bbox_dir, face_json)))
        return all_face_info

    def _read_arcface_embeddings(self, *args):
        """Reads from chosen HDF5 file."""
        return read_from_hdf5(os.path.join(self.arcface_embeddings_dir, "arcface_embeddings_merged.hdf5"), *args)

    def _load_preloaded_filepaths(self):
        """Load preloaded filepaths from JSON."""
        assert self.preload_path is not None, "Preload path must be provided!"
        print("Loading preloaded filepaths...")
        preload_path = self.preload_path
        subjects = load_json(preload_path)
        scenes = []

        subjects_with_latents = None
        if self.latents_dir is not None:
            subjects_with_latents = set([subject for subject in os.listdir(self.latents_dir) if os.path.exists(os.path.join(self.latents_dir, subject, f"{subject}.npz"))])
            print(f"Found {len(subjects_with_latents)} subjects with latents")

        for i, subject in tqdm(enumerate(subjects), total=len(subjects), desc="Loading scenes"):
            if subject == "metadata":
                continue
            subject_path = os.path.join(self.root_dir, subject)
            if self.only_include is not None and subject not in self.only_include:
                continue
            if self.exclude is not None and subject in self.exclude:
                continue
            if self.data_limit is not None and i >= self.data_limit:
                break
            if len(subjects[subject]['cameras']) != 48:
                print(f"Skipping subject {subject} because it does not have all 48 cameras!")
                continue
            if (subjects_with_latents is not None and subject not in subjects_with_latents):
                print(f"Skipping subject {subject} because it does not have latents precomputed!")
                continue

            extrinsics = subjects[subject]['extrinsics']
            intrinsics = subjects[subject]['intrinsics']
            camera_scale = subjects[subject]['camera_scale']
            annots = subjects[subject]['annots']

            self.cam_params[subject] = {
                'extrinsics': extrinsics,
                'intrinsics': intrinsics,
                'camera_scale': camera_scale
            }

            num_timesteps = subjects[subject]['timesteps']
            cameras = [cam for cam in subjects[subject]['cameras'] if cam in annots['bbox']]
            step_size = subjects["metadata"]["step_size"]
            subject_map = defaultdict(dict)

            iterator = range(1, num_timesteps, self.step_size) if isinstance(num_timesteps, int) else num_timesteps
            for timestep in iterator:
                try:
                    for camera in cameras:
                        if isinstance(timestep, str) and timestep.endswith("_img.jpg"):
                            timestep = int(timestep.split("_")[0])
                        time_id = f"{timestep * 5:04d}"
                        image_path = os.path.join(subject_path, "images_lr", camera, f"{time_id}_img.jpg")
                        mask_path = os.path.join(subject_path, "fmask_lr", camera, f"{time_id}_img_fmask.png")
                        bbox = annots['bbox'][camera][time_id]
                        face_bbox = [-1, -1, -1, -1]
                        arcface_embedding = None
                        
                        if self.face_bboxes is not None:
                            try:
                                face_bbox_dict = self.face_bboxes[subject][camera][f"{time_id}_img.jpg"]
                                if face_bbox_dict == {}:
                                    face_bbox = [-1, -1, -1, -1]
                                else:
                                    face_bbox = [face_bbox_dict['x1'], face_bbox_dict['y1'], face_bbox_dict['x2'], face_bbox_dict['y2']]
                            except KeyError:
                                face_bbox = [-1, -1, -1, -1]

                            if face_bbox != [-1, -1, -1, -1] and ((bbox[2] - bbox[0]) == 0 or (bbox[3] - bbox[1]) == 0):
                                print(f"Skipping subject {subject} camera {camera} timestep {timestep} because bbox is invalid")
                                continue

                        subject_map[time_id][camera] = {
                                    'image_path': image_path,
                                    'mask_path': mask_path,
                                    'annots': {
                                        'bbox': bbox,
                                        'bbox_face': face_bbox if self.face_bboxes is not None else None,
                                    }
                                }
                except Exception as e:
                    print(f"Error loading subject {subject} camera {camera} timestep {timestep}: {e}")
                    subject_map.pop(time_id, None)
                    break 

            sorted_timesteps = sorted(subject_map.keys())
            for i in range(0, len(sorted_timesteps)):
                timestep = sorted_timesteps[i]
                frames_info = subject_map[timestep]

                if len(frames_info.keys()) < self.num_images:
                    continue

                scenes.append({
                    'subject_id': subject,
                    'frames_info': frames_info,
                    'timestep': timestep
                })
        print("Loading preloaded filepaths completed!")
        return scenes

    def _load_scenes_arrow(self):
        """Load scenes from Arrow dataset."""
        if not HAS_DATASETS:
            raise ImportError("datasets library is required for Arrow dataset support. Install with: pip install datasets")
        print("Loading Arrow dataset...")
        
        # Handle directory with shards - load_from_disk should handle this automatically
        if os.path.isdir(self.preload_path):
            # Check if it's a directory with shard files
            if any(f.endswith('.arrow') for f in os.listdir(self.preload_path)):
                self.dataset = load_from_disk(self.preload_path)
            else:
                # Try to find the first shard or use the directory directly
                self.dataset = load_from_disk(self.preload_path)
        else:
            self.dataset = load_from_disk(self.preload_path)
        
        print(f"Loaded Arrow dataset with {len(self.dataset)} subjects")
        
        # Pre-load camera parameters for all subjects to avoid repeated parsing
        print("Pre-loading camera parameters from Arrow dataset...")
        for i in tqdm(range(len(self.dataset)), desc="Loading camera params"):
            row = self.dataset[i]
            subject_id = row['subject_id']
            
            # Parse and cache camera parameters
            if isinstance(row['extrinsics'], str):
                extrinsics = json.loads(row['extrinsics'])
            else:
                extrinsics = row['extrinsics']
            
            if isinstance(row['intrinsics'], str):
                intrinsics = json.loads(row['intrinsics'])
            else:
                intrinsics = row['intrinsics']
            
            if isinstance(row.get('camera_scale'), str):
                camera_scale = float(row['camera_scale'])
            else:
                camera_scale = float(row['camera_scale']) if 'camera_scale' in row else 1.0
            
            # Clean extrinsics keys if needed
            if extrinsics and isinstance(extrinsics, dict):
                # Check if keys need cleaning (format: "1_XXXX.png")
                if any(key.startswith('1_') and key.endswith('.png') for key in extrinsics.keys()):
                    extrinsics = self._clean_camera_keys(extrinsics)
            
            # Store in cam_params for later use
            self.cam_params[subject_id] = {
                'extrinsics': extrinsics,
                'intrinsics': intrinsics if isinstance(intrinsics, dict) else intrinsics,
                'camera_scale': camera_scale
            }
        
        scenes = []
        
        subjects_with_latents = None
        if self.latents_dir is not None:
            subjects_with_latents = set([subject for subject in os.listdir(self.latents_dir) if os.path.exists(os.path.join(self.latents_dir, subject, f"{subject}.npz"))])
            print(f"Found {len(subjects_with_latents)} subjects with latents")
            
        for i in tqdm(range(len(self.dataset)), desc="Indexing Arrow dataset"):
            row = self.dataset[i]
            subject_id = row['subject_id']
            
            if self.only_include is not None and subject_id not in self.only_include:
                continue
            if self.exclude is not None and subject_id in self.exclude:
                continue
            if self.data_limit is not None and len(scenes) >= self.data_limit * 100:
                pass 
                 
            if (subjects_with_latents is not None and subject_id not in subjects_with_latents):
                continue

            # Get timesteps - handle both list and other formats
            timesteps = row['timesteps']
            if isinstance(timesteps, str):
                # If it's a JSON string, parse it
                try:
                    timesteps = json.loads(timesteps)
                except:
                    timesteps = [timesteps]
            
            for timestep in timesteps:
                if isinstance(timestep, str) and timestep.endswith("_img.jpg"):
                    timestep = int(timestep.split("_")[0])
                
                time_id = f"{timestep * 5:04d}"                
                if isinstance(timestep, str):
                    # e.g. "0005_img.jpg" -> "0005"
                    if "_" in timestep:
                        time_id = timestep.split("_")[0]
                    else:
                        time_id = timestep
                else:
                    # Integer timestep
                    time_id = f"{timestep:04d}"

                scenes.append({
                    'subject_id': subject_id,
                    'timestep': time_id,
                    'arrow_idx': i,
                    'is_arrow': True
                })

        print(f"Loaded {len(scenes)} scenes from Arrow dataset")
        return scenes

    def _get_infu_path(self, subject_id, timestep):
        if self.infu_dataset_path is None:
            return None
        return self.infu_dataset_path + f"/{subject_id}/{timestep}_{subject_id}_img.png"

    def _get_iclight_path(self, subject_id, timestep, camera):
        if self.iclight_dataset_path is None:
            return None
        return self.iclight_dataset_path + f"/{subject_id}/images_lr/{camera}/{timestep}_img.png"

    def _sapiens_get(self, cond, subject_id, camera, timestep, dataset_type="mvhn"):
        try:
            if dataset_type == "mvhn":
                npz_path =  os.path.join(self.root_dir, subject_id, f"{subject_id}_{cond}.npz")
            elif dataset_type == "infu":
                npz_path = os.path.join(self.infu_dataset_path, subject_id, f"{subject_id}_{cond}.npz")
            elif dataset_type == "iclight":
                npz_path = os.path.join(self.iclight_dataset_path, subject_id, f"{subject_id}_{cond}.npz")
            else:
                raise ValueError(f"Invalid dataset type: {dataset_type}")
            data = np.load(npz_path)
            query = timestep if dataset_type == "infu" else f"{camera}_{timestep}"
            return torch.tensor(data[query])
        except Exception as e:
            print(f"Error loading sapiens conditioning: {e}")
            return None

    def _load_raw_frames(self, image_paths, mask_paths):
        """Load multi-view frames from paths."""
        frames = torch.zeros((self.num_images, 3, self.image_shape[0],  self.image_shape[1]))
        img_masks = torch.zeros((self.num_images, self.image_shape[0], self.image_shape[1]))
        for i, (img_path, mask_path) in enumerate(zip(image_paths, mask_paths)):
            image = Image.open(img_path).convert("RGB")
            img_mask = Image.open(mask_path)

            if img_mask.size != image.size:
                image = image.resize(img_mask.size, Image.BILINEAR)

            background = Image.new(
                'RGB', image.size, (255, 255, 255) if self.white_background else (0, 0, 0)
            )

            masked_image = Image.composite(image, background, img_mask)
            frames[i] = T.Compose([T.ToImage(), T.ToDtype(torch.float32, scale=True)])(masked_image)
            img_masks[i] = T.Compose([T.ToImage()])(img_mask)
            del image, img_mask, masked_image
        return frames, img_masks

    def _sample_multiview_image_paths(self, frames_info: dict, use_iclight: bool = False, use_infu: bool = False):
        """Sample multi-view frame paths from frames_info dictionary."""
        camera_order = [cam for cam in list(frames_info.keys())]
        sampled_image_paths = [frames_info[cam]['image_path'] for cam in camera_order]
        sampled_image_mask_paths = [frames_info[cam]['mask_path'] for cam in camera_order]

        if np.random.rand() <= self.adjacent_frame_sampling_prob:
            which_rung = np.random.randint(0, len(CAMERA_RUNGS))
            rung_of_cameras = CAMERA_RUNGS[which_rung]
            start_idx = np.random.randint(0, len(rung_of_cameras))
            images_permutation = np.roll(np.arange(len(rung_of_cameras)), -start_idx)[:self.num_images]
            images_permutation = [CAMERA_TO_INDEX[rung_of_cameras[i]] for i in images_permutation]
        else:
            images_permutation = np.random.choice(len(sampled_image_paths), self.num_images, replace=False)

        if self.fixed_sampling_ids is not None:
            images_permutation = self.fixed_sampling_ids

        camera_order = [camera_order[i] for i in images_permutation]
        sampled_image_paths = [sampled_image_paths[i] for i in images_permutation]
        sampled_image_mask_paths = [sampled_image_mask_paths[i] for i in images_permutation]

        return sampled_image_paths, sampled_image_mask_paths, camera_order, images_permutation

    def _sample_all_masks(self, use_iclight: bool = False, use_infu: bool = False):
        """Sample input/target frame split."""
        if not self.use_inconsistent:
            num_input_frames = np.random.randint(1, self.num_images)
        else:
            if np.random.rand() <= self.all_inputs_prob:
                num_input_frames = self.num_images
            else:
                num_input_frames = np.random.randint(1, self.num_images)

        input_frames_indices = np.random.choice(self.num_images, num_input_frames, replace=False)
        input_target_mask = torch.zeros(self.num_images, dtype=torch.bool)
        input_target_mask[input_frames_indices] = True

        if not self.use_inconsistent:
            ref_mask = input_target_mask.clone()
        else:
            ref_mask = torch.zeros(self.num_images, dtype=torch.bool)
            fix_frame_idx = input_frames_indices[np.random.choice(len(input_frames_indices), 1).item()]
            ref_mask[fix_frame_idx] = True

        ic_masks = {}
        if self.use_inconsistent:
            if use_iclight and use_infu:
                ic_masks['iclight'] = torch.rand(self.num_images) < self.ic_sampling_prob
                ic_masks['infu'] = ~ic_masks['iclight']
                ic_masks['iclight'] = ic_masks['iclight'] * input_target_mask
                ic_masks['infu'] = ic_masks['infu'] * input_target_mask
                ic_masks['iclight'][ref_mask] = False
                ic_masks['infu'][ref_mask] = False
            elif use_iclight and not use_infu:
                ic_masks['iclight'] = ~ref_mask * input_target_mask
                ic_masks['infu'] = torch.zeros(self.num_images, dtype=torch.bool)
            elif not use_iclight and use_infu:
                ic_masks['iclight'] = torch.zeros(self.num_images, dtype=torch.bool)
                ic_masks['infu'] = ~ref_mask * input_target_mask
            else:
                raise ValueError(f"use_inconsistent is True, but neither iclight or infu paths are provided!")
        else:
            ic_masks['iclight'] = torch.zeros(self.num_images, dtype=torch.bool)
            ic_masks['infu'] = torch.zeros(self.num_images, dtype=torch.bool)
        return input_target_mask, ref_mask, ic_masks

    def _load_inconsistent_frames(self, img_paths, img_mask_paths, cam_order, subject_id, timestep, ref_mask, ic_masks):
        """Replace sampled MVHN image paths with inconsistent paths."""
        if not self.use_inconsistent:
            return torch.zeros((self.num_images, 3, self.target_shape[0], self.target_shape[1]), dtype=torch.float32)

        ic_paths = []
        num_infu_frames = ic_masks['infu'].sum().item() if isinstance(ic_masks['infu'], torch.Tensor) else ic_masks['infu'].sum()
        if ic_masks['infu'].sum() > 0 and num_infu_frames > 0:
            infu_num_images_in_directory = self.infu_num_images[subject_id]
            num_samples = min(num_infu_frames, infu_num_images_in_directory)
            infu_indices = list(np.random.choice(infu_num_images_in_directory, num_samples, replace=False) + 1)
        else:
            infu_indices = []
        
        for path, is_iclight, is_infu, is_ref, camera in zip(img_paths, ic_masks['iclight'], ic_masks['infu'], ref_mask, cam_order):
            if is_ref:
                ic_path = path
            elif is_iclight:
                ic_path = self._get_iclight_path(subject_id, timestep, camera)
            elif is_infu:
                infu_random_index = infu_indices.pop(0)
                ic_path = self._get_infu_path(subject_id, f"{infu_random_index:06d}")
            else:
                ic_path = None
            ic_paths.append(ic_path)

        ic_rgb = []
        tensorize = T.Compose([T.ToImage(), T.ToDtype(torch.float32, scale=True)])
        for ic_path in ic_paths:
            if ic_path is not None:
                ic_image = Image.open(ic_path).convert("RGB")
                ic_rgb.append(tensorize(ic_image))
            else:
                ic_rgb.append(torch.zeros((3, self.target_shape[0], self.target_shape[1]), dtype=torch.float32))
        return ic_rgb, ic_paths

    def _crop_and_transform_frames_and_intrinsics(
        self, frames_info, frames, image_masks, ic_rgb, subject_id, cam_order, intrinsics, ref_mask, ic_masks, sapiens_conditionings):
        """Crop and transform frames while updating intrinsics."""
        if self.random_crop or self.maximal_crop:
            annots_jsons = [frames_info[cam]["annots"] for cam in cam_order]
            crop_params = []
            face_bboxes_adjusted = []
            for annots_json in annots_jsons:
                bbox = annots_json['bbox'][:4]
                face_bbox = annots_json['bbox_face'][:4] if self.face_bboxes is not None else [-1, -1, -1, -1]
                crop_params.append(bbox)
                face_bboxes_adjusted.append(face_bbox)
            bbox_annot_scale = 0.5 if int(subject_id) < 103000 else 1.0
            bbox_params = torch.stack([torch.tensor(bbox) * bbox_annot_scale for bbox in crop_params])

            face_params = torch.stack([torch.tensor(face_bbox) * bbox_annot_scale for face_bbox in face_bboxes_adjusted])
            frames, Ks, rel_bbox, face_bboxes_result, new_bbox, bbox_before_pad = self.cropper(frames, bbox_params, torch.from_numpy(intrinsics).float(), face_bboxes=face_params)
            image_masks, _ = self.cropper._possibly_pad_img(image_masks.unsqueeze(1), bbox_before_pad[:,0], bbox_before_pad[:,1], bbox_before_pad[:,2], bbox_before_pad[:,3])
            image_masks = self.cropper.crop_images(image_masks, new_bbox[:,0], new_bbox[:,1], new_bbox[:,2], new_bbox[:,3])
            if face_bboxes_result is not None:
                face_bboxes_adjusted = face_bboxes_result

            scale = np.array([self.target_shape[0] / cropped_img.shape[-2] for cropped_img in frames])
            Ks = update_intrinsics_resize(Ks, scale)
            Ks = normalize_intrinsics(Ks, self.target_shape[0], self.target_shape[1])
            if len(Ks.shape) == 2:
                Ks = repeat(Ks, 'd1 d2 -> n d1 d2', n=self.num_images)
            Ks = torch.from_numpy(Ks).float()
        else:
            min_dim = min(*self.image_shape)
            max_dim = max(*self.image_shape)
            crop_amount  = (max_dim - min_dim) // 2
            scale_amount = (self.target_shape[0] / min_dim)
            Ks = update_intrinsics(np.array(intrinsics), crop_x=crop_amount, crop_y=0, scale=scale_amount)
            Ks = normalize_intrinsics(Ks, self.target_shape[0], self.target_shape[1])
            Ks = repeat(Ks, 'd1 d2 -> n d1 d2', n=self.num_images)
            Ks = torch.from_numpy(Ks).float()

            annots_jsons = [frames_info[cam]["annots"] for cam in cam_order]
            face_bboxes_adjusted = []
            for annots_json in annots_jsons:
                face_bbox = annots_json['bbox_face'][:4]
                face_bboxes_adjusted.append(face_bbox)
                if face_bbox != [-1, -1, -1, -1]:
                    face_bbox[0] = face_bbox[0] - crop_amount
                    face_bbox[2] = face_bbox[2] - crop_amount
            face_params = torch.stack([torch.tensor(face_bbox) for face_bbox in face_bboxes_adjusted])
            face_bboxes_adjusted = face_params * scale_amount

        if self.random_crop or self.maximal_crop:
            # Ensure all frames are tensors before transforming
            transformed_frames = []
            for frame in frames:
                # If frame is a PIL Image, convert it first
                if hasattr(frame, 'size') and not isinstance(frame, torch.Tensor):
                    # It's a PIL Image, convert to tensor first
                    frame = T.Compose([T.ToImage(), T.ToDtype(torch.float32, scale=True)])(frame)
                transformed_frame = self.transform(frame)
                # Ensure the result is a tensor
                if not isinstance(transformed_frame, torch.Tensor):
                    transformed_frame = T.Compose([T.ToImage(), T.ToDtype(torch.float32, scale=True)])(transformed_frame)
                transformed_frames.append(transformed_frame)
            frames = torch.stack(transformed_frames, dim=0)
            
            # Ensure all masks are tensors before transforming
            transformed_masks = []
            for img_mask in image_masks:
                # If mask is a PIL Image, convert it first
                if hasattr(img_mask, 'size') and not isinstance(img_mask, torch.Tensor):
                    # It's a PIL Image, convert to tensor first
                    img_mask = T.Compose([T.ToImage()])(img_mask)
                transformed_mask = self.mask_transform(img_mask)
                # Ensure the result is a tensor
                if not isinstance(transformed_mask, torch.Tensor):
                    transformed_mask = T.Compose([T.ToImage()])(transformed_mask)
                transformed_masks.append(transformed_mask)
            image_masks = torch.stack(transformed_masks, dim=0)
            
            ic_rgb_tensor = torch.zeros((self.num_images, 3, self.target_shape[0], self.target_shape[1]), dtype=torch.float32)
            for i, (ic_image, bbox, is_ref, is_iclight, is_infu) in enumerate(zip(ic_rgb, rel_bbox, ref_mask, ic_masks['iclight'], ic_masks['infu'])):
                ic_image = torch.nn.functional.interpolate(ic_image.unsqueeze(0), size=(self.target_shape[0], self.target_shape[1]), mode='bilinear', align_corners=False).squeeze(0)
                dx1, dy1, dx2, dy2 = bbox.int()
                ic_image_ = ic_image[:,0+dy1:self.target_shape[0]+dy2, 0+dx1:self.target_shape[1]+dx2]
                ic_image_ = self.transform(ic_image_)
                ic_rgb_tensor[i] = ic_image_

                if self.use_sapiens_conditioning is not None:
                    ref_idx = torch.where(ref_mask == True)[0][0].item()
                    for cond in self.use_sapiens_conditioning:
                        cond_tensor = sapiens_conditionings[cond][i]
                        if ref_idx == i:
                            padded_img, refbbox = self.cropper._possibly_pad_img(
                                cond_tensor.unsqueeze(0), 
                                new_bbox[ref_idx][0].unsqueeze(0), 
                                new_bbox[ref_idx][1].unsqueeze(0), 
                                new_bbox[ref_idx][2].unsqueeze(0), 
                                new_bbox[ref_idx][3].unsqueeze(0)
                            )
                            if isinstance(padded_img, list):
                                padded_img = padded_img[0]
                            else:
                                padded_img = padded_img.squeeze(0)
                            refbbox = refbbox.squeeze(0).int()
                            cropped = padded_img[:, refbbox[1]:refbbox[3], refbbox[0]:refbbox[2]]
                            sapiens_conditionings[cond][ref_idx] = T.Resize((self.target_shape[0], self.target_shape[1]))(cropped)
                        else:
                            scale = cond_tensor.shape[-2] / self.target_shape[0]
                            cropped_cond_tensor = cond_tensor[:, 0+int(dy1*scale):int((self.target_shape[0]+dy2)*scale), 0+int(dx1*scale):int((self.target_shape[1]+dx2)*scale)]
                            sapiens_conditionings[cond][i] = T.Resize((self.target_shape[0], self.target_shape[1]))(cropped_cond_tensor)
                        if cond == "seg_masks":
                            sapiens_conditionings[cond][i] = one_hot_encode_segmentation(sapiens_conditionings[cond][i], 28)

            sapiens_conditionings = {cond: torch.stack(cond_tensor, dim=0) for cond, cond_tensor in sapiens_conditionings.items()}
            ic_rgb = ic_rgb_tensor
        else:
            frames = self.transform(frames)
            # Ensure all ic_rgb images are tensors before transforming and stacking
            transformed_ic_rgb = []
            for ic_image in ic_rgb:
                # If ic_image is a PIL Image, convert it first
                if hasattr(ic_image, 'size') and not isinstance(ic_image, torch.Tensor):
                    # It's a PIL Image, convert to tensor first
                    ic_image = T.Compose([T.ToImage(), T.ToDtype(torch.float32, scale=True)])(ic_image)
                transformed_ic = self.transform(ic_image)
                # Ensure the result is a tensor
                if not isinstance(transformed_ic, torch.Tensor):
                    transformed_ic = T.Compose([T.ToImage(), T.ToDtype(torch.float32, scale=True)])(transformed_ic)
                transformed_ic_rgb.append(transformed_ic)
            ic_rgb = torch.stack(transformed_ic_rgb, dim=0)
            image_masks = self.mask_transform(image_masks)
            # image_masks should already be a tensor from mask_transform, but ensure it is
            if not isinstance(image_masks, torch.Tensor):
                # If it's a list, transform each and stack
                if isinstance(image_masks, list):
                    transformed_masks = []
                    for mask in image_masks:
                        if hasattr(mask, 'size') and not isinstance(mask, torch.Tensor):
                            mask = T.Compose([T.ToImage()])(mask)
                        transformed_mask = self.mask_transform(mask) if not isinstance(mask, torch.Tensor) else mask
                        if not isinstance(transformed_mask, torch.Tensor):
                            transformed_mask = T.Compose([T.ToImage()])(transformed_mask)
                        transformed_masks.append(transformed_mask)
                    image_masks = torch.stack(transformed_masks, dim=0)

            if self.use_sapiens_conditioning is not None:
                for cond in self.use_sapiens_conditioning:
                    for is_ref, cond_tensor in zip(ref_mask, sapiens_conditionings[cond]):
                        cond_tensor = T.Resize((self.target_shape[0], self.target_shape[1]))(cond_tensor)
                        if cond == "seg_masks":
                            cond_tensor = one_hot_encode_segmentation(cond_tensor, 28)
            sapiens_conditionings = {cond: torch.stack(cond_tensor, dim=0) for cond, cond_tensor in sapiens_conditionings.items()}

        return frames, image_masks, ic_rgb, Ks, sapiens_conditionings, face_bboxes_adjusted

    def _get_arcface_embeddings(self, subject_id, timestep, cam_order, input_target_mask, ref_mask, ic_masks):
        """Get ArcFace embeddings."""
        arcface_embeddings = []
        if self.arcface_embeddings_dir is not None:
            for is_ref, is_iclight, is_infu, cam in zip(ref_mask, ic_masks['iclight'], ic_masks['infu'], cam_order):   
                try:
                    if is_ref:
                        arcface_embedding = self._read_arcface_embeddings("mvhn", subject_id, cam, f"{timestep}_img.jpg")
                    elif is_infu:
                        arcface_embedding = self._read_arcface_embeddings("infu", subject_id, cam, f"{timestep}_img.png")
                    elif is_iclight:
                        arcface_embedding = self._read_arcface_embeddings("iclight", subject_id, cam, f"{timestep}_img.png")
                    else:
                        arcface_embedding = None
                except KeyError:
                    arcface_embedding = None
                except Exception as e:
                    print(f"Error reading arcface embedding: {e}")
                    arcface_embedding = None
                arcface_embeddings.append(arcface_embedding)
            
            arcface_embeddings = [
                torch.tensor(emb) if emb is not None else torch.zeros(512, dtype=torch.float32)
                for emb in arcface_embeddings
            ]
            arcface_embeddings = torch.stack(arcface_embeddings)
            arcface_embeddings[~input_target_mask] *= 0
        else:
            arcface_embeddings = torch.zeros((self.num_images, 512), dtype=torch.float32)
        
        return arcface_embeddings

    def _get_sapiens_conditionings(self, subject_id, timestep, cam_order, ic_paths, input_target_mask, ref_mask, ic_masks):
        """Get sapiens conditionings."""
        sapiens_conditionings = {}
        if self.use_sapiens_conditioning is not None:
            for cond in self.use_sapiens_conditioning:
                sapiens_conditionings[cond] = []
                for is_ref, is_iclight, is_infu, camera, ic_path in zip(ref_mask, ic_masks['iclight'], ic_masks['infu'], cam_order, ic_paths):
                    try: 
                        cond_tensor = None
                        if is_ref:
                            cond_tensor = self._sapiens_get(cond, subject_id, camera, timestep, dataset_type="mvhn")
                        elif is_iclight: 
                            cond_tensor = self._sapiens_get(cond, subject_id, camera, timestep, dataset_type="iclight")
                        else:
                            cond_tensor = self._sapiens_get(cond, subject_id, camera, f"{os.path.basename(ic_path).split('_')[0]}", dataset_type="infu")

                        cond_tensor = torch.nan_to_num(cond_tensor, nan=0)
                    except Exception as e:
                        pass
                    if cond_tensor is None:
                        if cond == "depth":
                            cond_tensor = torch.zeros((1, self.target_shape[0], self.target_shape[1]), dtype=torch.float32)
                        elif cond == "seg_masks":
                            cond_tensor = torch.zeros((1, self.target_shape[0], self.target_shape[1]), dtype=torch.float32)
                        elif cond == "latents":
                            cond_tensor = torch.zeros((4, self.target_shape[0], self.target_shape[1]), dtype=torch.float32)
                    else:
                        cond_tensor = cond_tensor.unsqueeze(0)
                    sapiens_conditionings[cond].append(cond_tensor)
        return sapiens_conditionings

    def __len__(self):
        return len(self.scenes)

    def __getitem__(self, idx):
        try:
            return self.create_batch(idx)
        except Exception as e:
            print(f"Error creating batch at index {idx}: {e}. Skipping this scene.")
            return None
    
    def create_batch(self, idx):
        """Collect multi-views + conditioning data for a scene at a fixed timestep."""
        scene = self.scenes[idx]
        subject_id = scene['subject_id']
        timestep = scene['timestep']
        
        if self.is_arrow:
            row = self.dataset[scene['arrow_idx']]
            
            # Parse JSON strings if needed, otherwise use directly
            if isinstance(row['extrinsics'], str):
                extrinsics = json.loads(row['extrinsics'])
            else:
                extrinsics = row['extrinsics']
            
            if isinstance(row['intrinsics'], str):
                intrinsics = json.loads(row['intrinsics'])
            else:
                intrinsics = row['intrinsics']
            
            if isinstance(row.get('camera_scale'), str):
                camera_scale = float(row['camera_scale'])
            else:
                camera_scale = float(row.get('camera_scale', 1.0))
            
            # Parse annotation bboxes
            if isinstance(row.get('annots_bbox'), str):
                annots_bbox = json.loads(row['annots_bbox'])
            else:
                annots_bbox = row.get('annots_bbox', {})
            
            if isinstance(row.get('annots_bbox_face2d'), str):
                annots_bbox_face = json.loads(row['annots_bbox_face2d'])
            else:
                annots_bbox_face = row.get('annots_bbox_face2d', {})
            
            # Get cameras list
            cameras = row.get('cameras', [])
            if isinstance(cameras, str):
                try:
                    cameras = json.loads(cameras)
                except:
                    cameras = [cameras]
            
            # Use cached camera params if available, otherwise use parsed ones
            if subject_id in self.cam_params:
                extrinsics = self.cam_params[subject_id]['extrinsics']
                intrinsics = self.cam_params[subject_id]['intrinsics']
                camera_scale = self.cam_params[subject_id]['camera_scale']
            
            frames_info = {}
            subject_path = os.path.join(self.root_dir, subject_id)
            
            for camera in cameras:
                time_id = timestep 
                
                if camera in annots_bbox and time_id in annots_bbox[camera]:
                    bbox = annots_bbox[camera][time_id]
                    bbox_face = [-1,-1,-1,-1]
                    if camera in annots_bbox_face and time_id in annots_bbox_face[camera]:
                        bbox_face = annots_bbox_face[camera][time_id]
                        
                    if (bbox[2] - bbox[0]) == 0 or (bbox[3] - bbox[1]) == 0:
                        continue

                    image_filename = f"{time_id}_img.jpg"
                    mask_filename = f"{time_id}_img_fmask.png"
                    
                    frames_info[camera] = {
                        'image_path': os.path.join(subject_path, "images_lr", camera, image_filename),
                        'mask_path': os.path.join(subject_path, "fmask_lr", camera, mask_filename),
                        'annots': {
                            'bbox': bbox,
                            'bbox_face': bbox_face
                        }
                    }
            
            # Convert intrinsics to numpy array
            if isinstance(intrinsics, dict):
                intrinsics = np.array(intrinsics.get('intrinsics', intrinsics))
            else:
                intrinsics = np.array(intrinsics)
            
        else:
            frames_info = dict(sorted(scene['frames_info'].items()))
            subject_path = os.path.join(self.root_dir, subject_id)
            extrinsics = self.cam_params[subject_id]['extrinsics']
            intrinsics = np.array(self.cam_params[subject_id]['intrinsics'])
            camera_scale = self.cam_params[subject_id]['camera_scale'] 

        if self.pre_scale_intrinsics != 1:
            intrinsics = update_intrinsics_resize(intrinsics, scale=self.pre_scale_intrinsics)
            
        if isinstance(intrinsics, list):
             intrinsics = np.array(intrinsics)

        img_paths, img_mask_paths, cam_order, sample_permutation = self._sample_multiview_image_paths(frames_info)
        
        input_target_mask, ref_mask, ic_masks = self._sample_all_masks(
            use_iclight=self.iclight_dataset_path is not None,
            use_infu=self.infu_dataset_path is not None,
        )
        iclight_mask = ic_masks["iclight"]
        infu_mask = ic_masks["infu"]

        frames, image_masks = self._load_raw_frames(img_paths, img_mask_paths)

        ic_rgb, ic_paths = self._load_inconsistent_frames(
            img_paths, img_mask_paths, cam_order,
            subject_id, timestep, ref_mask, ic_masks
        )

        arcface_embeddings = self._get_arcface_embeddings(
            subject_id, timestep, cam_order, input_target_mask, ref_mask, ic_masks
        )

        sapiens_conditionings = self._get_sapiens_conditionings(
            subject_id, timestep, cam_order, ic_paths, input_target_mask, ref_mask, ic_masks
        )

        frames, img_masks, ic_rgb, Ks, sapiens_conditionings, face_bboxes_adjusted = self._crop_and_transform_frames_and_intrinsics(
            frames_info, frames, image_masks, ic_rgb,
            subject_id, cam_order, intrinsics,
            ref_mask, ic_masks, sapiens_conditionings
        )
        img_masks = img_masks / 255.0

        camera_mask = torch.ones(self.num_images, dtype=torch.bool)

        def get_c2w(cam):
            tf_matrix = create_transform_matrix(
                np.array(extrinsics[cam]['rotation']),
                np.array(extrinsics[cam]['translation']) * camera_scale,
                homogeneous=True
            )
            return np.linalg.inv(tf_matrix)

        all_c2ws = np.array([
            get_c2w(cam) for cam in frames_info.keys()
        ])

        all_c2ws = torch.from_numpy(all_c2ws).float()
        c2ws = all_c2ws[sample_permutation]
        center_cameras(all_c2ws, c2ws)
        scale_cameras(c2ws)

        w2cs = torch.linalg.inv(c2ws)
        src_camera_idx = input_target_mask.to(torch.int).argmax().item()
        pluckers = get_plucker_coordinates(
            extrinsics_src=w2cs[src_camera_idx],
            extrinsics=w2cs,
            intrinsics=Ks.clone(),
            target_size=(self.target_shape[0] // self.downsample_factor, 
                         self.target_shape[1] // self.downsample_factor),
        )

        if self.latents_dir is not None and os.path.exists(os.path.join(self.latents_dir, subject_id, f"{subject_id}.npz")) and not self.maximal_crop:
            npz_file = os.path.join(self.latents_dir, subject_id, f"{subject_id}.npz")
            with np.load(npz_file) as npz_data:
                latent_tensors = [npz_data[f"{sample_cam}.{timestep}"] for sample_cam in cam_order]
                clean_latents = torch.stack([torch.from_numpy(latent_tensor) for latent_tensor in latent_tensors])
        else:
            clean_latents = 0

        concat = torch.cat([
            repeat(input_target_mask, "n -> n 1 h w", h=pluckers.shape[2], w=pluckers.shape[3]),
            pluckers,
            repeat(ref_mask, "n -> n 1 h w", h=pluckers.shape[2], w=pluckers.shape[3]),
        ], dim=1)

        if type(clean_latents) == int and clean_latents == 0:
            replace = 0
        else:
            replace = torch.cat([
                clean_latents * self.scale_factor,
                repeat(ref_mask, "n -> n 1 h w", h=pluckers.shape[2], w=pluckers.shape[3]),
            ], dim=1)

        try:
            output_dict = {
                "clean_latent": clean_latents,
                "mask": input_target_mask,
                "ref_mask": ref_mask,
                "ic_rgb": ic_rgb,
                "plucker": pluckers,
                "camera_mask": camera_mask,
                "concat": concat,
                "frames": frames,
                "frames_masks": img_masks,
                "replace": replace,
                "c2w": c2ws,
                "K": Ks,
                "use_inconsistent": self.use_inconsistent,
                "face_bbox": face_bboxes_adjusted,
                "subject_id": subject_id,
                "timestep": timestep,
            }

            if self.arcface_embeddings_dir is not None:
                output_dict["arcface_embedding"] = arcface_embeddings

            if self.use_sapiens_conditioning is not None:
                output_dict["sapiens_conditioning"] = sapiens_conditionings
        except Exception as e:
            print(f"Error creating output_dict: {e}")
            raise

        return output_dict

def custom_collate(batch):
    """Custom collate function that filters None values."""
    batch = list(filter(lambda x: x is not None, batch))
    if not batch:
        return None
    return torch.utils.data.default_collate(batch)

def expand_only_include(only_include):
    """Expand range strings like '100001-102000' to list of IDs."""
    if isinstance(only_include, str):
        only_include = only_include.split(",")
        expanded_includes = []
        for subrange in only_include:
            start, end = [int(num) for num in subrange.split("-")]
            expanded_includes.extend([str(i).zfill(6) for i in range(start, end + 1)])
        return expanded_includes
    else:
        return only_include

# ============================================================================
# DataModule Class (PyTorch Lightning - optional)
# ============================================================================

if HAS_PL:
    class MVHumanNetLoader(pl.LightningDataModule):
        """PyTorch Lightning DataModule wrapper (optional)."""
        def __init__(
            self,
            root_dir: str,
            num_images: int,
            batch_size: int,
            latents_dir: str = None,
            num_workers: int = 0,
            shuffle: bool = True,
            image_size: int = 576,
            data_limit: int = None,
            only_include: list = None,
            exclude: list = None,
            step_size: int = 150,
            preload_path: str = None,
            iclight_dataset_path: str = None,
            infu_dataset_path: str = None,
            face_bbox_dir: str = None,
            arcface_embeddings_dir: str = None,
            random_crop: bool = False,
            maximal_crop: bool = True,
            val_include: list = None,
            use_inconsistent: bool = False,
            random_crop_prob: float = 0.3,
            ic_sampling_prob: float = 0.7,
            fixed_sampling_ids: list = None,
            use_sapiens_conditioning: list = None,
            sapiens_segmentation_channels_to_use: list = None,
        ):
            super().__init__()
            print("init of DATALOADER")
            self.root_dir = root_dir
            self.latents_dir = latents_dir
            self.num_images = num_images
            self.batch_size = batch_size
            self.num_workers = num_workers
            self.shuffle = shuffle
            self.data_limit = data_limit
            self.only_include = only_include
            self.exclude = exclude
            self.step_size = step_size
            self.preload_path = preload_path
            self.iclight_dataset_path = iclight_dataset_path
            self.infu_dataset_path = infu_dataset_path
            self.face_bbox_dir = face_bbox_dir
            self.arcface_embeddings_dir = arcface_embeddings_dir
            self.random_crop = random_crop
            self.maximal_crop = maximal_crop
            self.val_include = val_include
            self.use_inconsistent = use_inconsistent
            self.random_crop_prob = random_crop_prob
            self.ic_sampling_prob = ic_sampling_prob
            self.fixed_sampling_ids = fixed_sampling_ids
            self.use_sapiens_conditioning = use_sapiens_conditioning
            self.sapiens_segmentation_channels_to_use = sapiens_segmentation_channels_to_use
            self.transform = None
            
            if isinstance(self.only_include, str):
                if os.path.exists(self.only_include):
                    with open(self.only_include, 'r') as f:
                        self.only_include = [line.strip() for line in f]
                else:
                    self.only_include = expand_only_include(self.only_include)
            if isinstance(self.exclude, str):
                if os.path.exists(self.exclude):
                    with open(self.exclude, 'r') as f:
                        self.exclude = [line.strip() for line in f]
                else:
                    self.exclude = expand_only_include(self.exclude)
            if isinstance(self.val_include, str):
                if os.path.exists(self.val_include):
                    with open(self.val_include, 'r') as f:
                        self.val_include = [line.strip() for line in f]
                else:
                    self.val_include = expand_only_include(self.val_include)

        def setup(self, stage: Optional[str] = None):
            print("setup of DATALOADER")
            print("stage: ", stage)
            if stage == "fit" or stage is None:
                print("train is reached")
                self.train_dataset = MVHumanNetDataset(
                    root_dir=os.path.join(self.root_dir),
                    latents_dir=self.latents_dir,
                    num_images=self.num_images,
                    transforms=self.transform,
                    data_limit=self.data_limit,
                    only_include=self.only_include,
                    exclude=self.exclude,
                    step_size=self.step_size,
                    preload_path=self.preload_path,
                    iclight_dataset_path=self.iclight_dataset_path,
                    infu_dataset_path=self.infu_dataset_path,
                    face_bbox_dir=self.face_bbox_dir,
                    arcface_embeddings_dir=self.arcface_embeddings_dir,
                    random_crop=self.random_crop,
                    maximal_crop=self.maximal_crop,
                    use_inconsistent=self.use_inconsistent,
                    random_crop_prob=self.random_crop_prob,
                    ic_sampling_prob=self.ic_sampling_prob,
                    fixed_sampling_ids=self.fixed_sampling_ids,
                    use_sapiens_conditioning=self.use_sapiens_conditioning,
                    sapiens_segmentation_channels_to_use=self.sapiens_segmentation_channels_to_use,
                )

            if stage == "validate" or stage is None:
                print("val_dataset reached")
                self.val_dataset = MVHumanNetDataset(
                    root_dir=os.path.join(self.root_dir),
                    latents_dir=self.latents_dir,
                    num_images=self.num_images,
                    transforms=self.transform,
                    data_limit=self.data_limit,
                    only_include=self.val_include,
                    exclude=self.exclude,
                    step_size=self.step_size,
                    preload_path=self.preload_path,
                    iclight_dataset_path=self.iclight_dataset_path,
                    infu_dataset_path=self.infu_dataset_path,
                    face_bbox_dir=self.face_bbox_dir,
                    arcface_embeddings_dir=self.arcface_embeddings_dir,
                    random_crop=self.random_crop,
                    maximal_crop=self.maximal_crop,
                    use_inconsistent=self.use_inconsistent,
                    ic_sampling_prob=self.ic_sampling_prob,
                    random_crop_prob=self.random_crop_prob,
                    fixed_sampling_ids=self.fixed_sampling_ids,
                    use_sapiens_conditioning=self.use_sapiens_conditioning,
                    sapiens_segmentation_channels_to_use=self.sapiens_segmentation_channels_to_use,
                )
            if stage == "test" or stage is None:
                self.test_dataset = MVHumanNetDataset(
                    root_dir=os.path.join(self.root_dir, "test"),
                    latents_dir=self.latents_dir,
                    num_images=self.num_images,
                    transforms=self.transform,
                    data_limit=self.data_limit,
                    only_include=self.only_include,
                    exclude=self.exclude,
                    step_size=self.step_size,
                    preload_path=self.preload_path,
                    iclight_dataset_path=self.iclight_dataset_path,
                    infu_dataset_path=self.infu_dataset_path,
                    face_bbox_dir=self.face_bbox_dir,
                    arcface_embeddings_dir=self.arcface_embeddings_dir,
                    random_crop=self.random_crop,
                    maximal_crop=self.maximal_crop,
                    use_inconsistent=self.use_inconsistent,
                    ic_sampling_prob=self.ic_sampling_prob,
                    random_crop_prob=self.random_crop_prob,
                    fixed_sampling_ids=self.fixed_sampling_ids,
                    use_sapiens_conditioning=self.use_sapiens_conditioning,
                    sapiens_segmentation_channels_to_use=self.sapiens_segmentation_channels_to_use,
                )
                
        def prepare_data(self):
            pass

        def train_dataloader(self) -> DataLoader:
            print("dataloader train_dataloader")
            return DataLoader(
                self.train_dataset,
                batch_size=self.batch_size,
                shuffle=self.shuffle,
                num_workers=self.num_workers,
                drop_last=True,
                pin_memory=True,
                persistent_workers=True if self.num_workers > 0 else False,
                prefetch_factor=2 if self.num_workers > 0 else None,
                collate_fn=custom_collate,
            )

        def val_dataloader(self) -> DataLoader:
            if not hasattr(self, 'val_dataset'):
                self.setup("validate")
            k = 1
            sampler = RandomSampler(self.val_dataset, num_samples=self.batch_size * k, replacement=True)
            return DataLoader(
                self.val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                sampler=sampler,
                num_workers=self.num_workers,
                drop_last=True,
                pin_memory=True,
                persistent_workers=True if self.num_workers > 0 else False,
                prefetch_factor=2 if self.num_workers > 0 else None,
                collate_fn=custom_collate,
            )

        def test_dataloader(self) -> DataLoader:
            return DataLoader(
                self.test_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                drop_last=True,
                pin_memory=True,
                persistent_workers=True if self.num_workers > 0 else False,
                prefetch_factor=2 if self.num_workers > 0 else None,
                collate_fn=custom_collate,
            )
else:
    # Dummy class if PyTorch Lightning is not available
    class MVHumanNetLoader:
        """Dummy class when PyTorch Lightning is not available."""
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch Lightning is required for MVHumanNetLoader. Install with: pip install pytorch-lightning")

