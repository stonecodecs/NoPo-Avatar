from pathlib import Path
from typing import Optional
import torch
from jaxtyping import Float, Int64
from torch import Tensor
import torchvision.transforms as tf
from PIL import Image

from ..types import AnyExample, AnyViews
from ...misc.image_io import load_image


def load_inconsistent_image(
    inconsistent_base_path: Path,
    scene: str,
    view_index: Int64[Tensor, ""] | int,
    stage: str = "train",
    frame_name_format: str = "frame_{:06d}",
    image_extensions: tuple[str, ...] = (".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG"),
) -> Optional[Float[Tensor, "3 height width"]]:
    """
    Load an inconsistent image from disk.
    
    Expected folder structure:
        inconsistent_base_path/stage/scene/images/frame_XXXXXX.png
    
    Args:
        inconsistent_base_path: Base path to inconsistent images folder
        scene: Scene/object name (e.g., "subject_001")
        view_index: View/frame index (integer or tensor)
        stage: Dataset stage ("train", "val", "test")
        frame_name_format: Format string for frame names (default: "frame_{:06d}")
        image_extensions: Allowed image file extensions
        
    Returns:
        Loaded inconsistent image tensor [3, H, W] in range [0, 1], or None if not found
    """
    if isinstance(view_index, Tensor):
        view_index = view_index.item()
    
    # Construct the path: base_folder/stage/scene/images/frame_XXXXXX.png
    images_dir = inconsistent_base_path / stage / scene / "images"
    
    # Try different frame name formats and extensions
    frame_name = frame_name_format.format(view_index)
    
    for ext in image_extensions:
        image_path = images_dir / f"{frame_name}{ext}"
        if image_path.exists():
            try:
                return load_image(image_path)
            except Exception as e:
                print(f"Warning: Failed to load inconsistent image from {image_path}: {e}")
                continue
    
    # If not found with standard format, try alternative formats
    # Try with TRAIN_FRAME_ORDERS mapping if needed
    alt_formats = [
        f"{view_index:06d}",
        f"frame_{view_index:06d}",
        f"{view_index:04d}",
        f"frame_{view_index:04d}",
    ]
    
    for alt_name in alt_formats:
        for ext in image_extensions:
            image_path = images_dir / f"{alt_name}{ext}"
            if image_path.exists():
                try:
                    return load_image(image_path)
                except Exception as e:
                    print(f"Warning: Failed to load inconsistent image from {image_path}: {e}")
                    continue
    
    return None


def load_inconsistent_views(
    views: AnyViews,
    inconsistent_base_path: Path,
    scene: str,
    view_indices: Int64[Tensor, "view"],
    stage: str = "train",
    frame_name_format: str = "frame_{:06d}",
) -> AnyViews:
    """
    Load inconsistent versions of images for all views from disk.
    
    Args:
        views: Dictionary containing view data with "image" key
        inconsistent_base_path: Base path to inconsistent images folder
        scene: Scene/object name
        view_indices: Tensor of view indices [V]
        stage: Dataset stage ("train", "val", "test")
        frame_name_format: Format string for frame names
        
    Returns:
        Updated views dictionary with "image_original" and "image_inconsistent" added
    """
    images = views["image"]  # [V, 3, H, W] or [3, H, W]
    
    # Handle both batched and unbatched cases
    if images.ndim == 3:  # Unbatched: [3, H, W]
        images = images.unsqueeze(0)  # [1, 3, H, W]
        view_indices = view_indices.unsqueeze(0) if isinstance(view_indices, Tensor) else [view_indices]
        squeeze_output = True
    else:
        squeeze_output = False
    
    num_views = images.shape[0]
    inconsistent_images = []
    
    # Load inconsistent version for each view
    for v in range(num_views):
        view_idx = view_indices[v] if isinstance(view_indices, Tensor) else view_indices[v]
        inconsistent_img = load_inconsistent_image(
            inconsistent_base_path,
            scene,
            view_idx,
            stage=stage,
            frame_name_format=frame_name_format,
        )
        
        if inconsistent_img is not None:
            inconsistent_images.append(inconsistent_img)
        else:
            # If inconsistent image not found, use original as fallback
            # This allows training to continue even if some inconsistent images are missing
            inconsistent_images.append(images[v].clone())
    
    inconsistent_images = torch.stack(inconsistent_images, dim=0)  # [V, 3, H, W]
    
    if squeeze_output:
        inconsistent_images = inconsistent_images.squeeze(0)  # [3, H, W]
        images = images.squeeze(0)  # [3, H, W]
    
    return {
        **views,
        "image_original": images,           # Store original images
        "image_inconsistent": inconsistent_images,  # Store inconsistent versions
    }


def apply_inconsistent_image_shim(
    example: AnyExample,
    inconsistent_base_path: Path | str,
    stage: str = "train",
    frame_name_format: str = "frame_{:06d}",
) -> AnyExample:
    """
    Apply inconsistent image loading to an example.
    
    This shim loads pre-generated inconsistent images from disk to simulate real-world
    variations. The original images are preserved as "image_original" and inconsistent
    versions are stored as "image_inconsistent".
    
    Expected folder structure:
        inconsistent_base_path/stage/scene/images/frame_XXXXXX.png
    
    Args:
        example: Data example with "context" and "target" views, and "scene" key
        inconsistent_base_path: Base path to inconsistent images folder
        stage: Dataset stage ("train", "val", "test")
        frame_name_format: Format string for frame names (default: "frame_{:06d}")
        
    Returns:
        Modified example with "image_original" and "image_inconsistent" added to views
    """
    if isinstance(inconsistent_base_path, str):
        inconsistent_base_path = Path(inconsistent_base_path)
    
    # Get scene name from example
    if isinstance(example["scene"], list):
        scene = example["scene"][0]  # For batched examples, take first scene
    else:
        scene = example["scene"]  # For unbatched examples
    
    # Load inconsistent versions for context views
    context_views = load_inconsistent_views(
        example["context"],
        inconsistent_base_path,
        scene,
        example["context"]["index"],
        stage=stage,
        frame_name_format=frame_name_format,
    )
    
    # Load inconsistent versions for target views
    target_views = load_inconsistent_views(
        example["target"],
        inconsistent_base_path,
        scene,
        example["target"]["index"],
        stage=stage,
        frame_name_format=frame_name_format,
    )
    
    return {
        **example,
        "context": context_views,
        "target": target_views,
    }
