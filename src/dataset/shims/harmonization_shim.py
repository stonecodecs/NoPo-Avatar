from typing import Optional
import torch
from jaxtyping import Float
from torch import Tensor

from ..types import AnyExample, AnyViews
from ...misc.reference_mask_utils import create_reference_mask


def apply_harmonization_to_views(
    views: AnyViews,
    reference_mask: Optional[Float[Tensor, "..."]] = None,
    default_reference_indices: Optional[list[int]] = None,
) -> AnyViews:
    """
    Apply harmonization to views: mix original and inconsistent images based on reference mask.
    
    This function modifies the "image" field to contain:
    - Reference views: original images (from image_original)
    - Non-reference views: inconsistent images (from image_inconsistent)
    
    Args:
        views: Dictionary containing view data with "image", "image_original", and "image_inconsistent"
        reference_mask: Boolean tensor [V] or [B, V] indicating which views are reference (True = reference)
        default_reference_indices: If reference_mask not provided, use these indices as reference (default: [0])
        
    Returns:
        Updated views dictionary with "image" modified to mix original and inconsistent
    """
    # Check if we have the required fields
    if "image_original" not in views or "image_inconsistent" not in views:
        # If inconsistent images not loaded, return views as-is
        return views
    
    images_original = views["image_original"]  # [V, 3, H, W] or [B, V, 3, H, W] or [3, H, W]
    images_inconsistent = views["image_inconsistent"]  # [V, 3, H, W] or [B, V, 3, H, W] or [3, H, W]
    
    # Determine if batched (5D: [B, V, 3, H, W]) or unbatched (4D: [V, 3, H, W] or 3D: [3, H, W])
    if images_original.ndim == 5:  # Batched: [B, V, 3, H, W]
        is_batched = True
        batch_size, num_views = images_original.shape[:2]
        squeeze_batch = False
    elif images_original.ndim == 4:  # Unbatched multi-view: [V, 3, H, W]
        is_batched = False
        num_views = images_original.shape[0]
        batch_size = 1
        # Add batch dimension for consistent processing
        images_original = images_original.unsqueeze(0)  # [1, V, 3, H, W]
        images_inconsistent = images_inconsistent.unsqueeze(0)  # [1, V, 3, H, W]
        squeeze_batch = True
    else:  # Unbatched single view: [3, H, W]
        is_batched = False
        num_views = 1
        batch_size = 1
        # Add batch and view dimensions
        images_original = images_original.unsqueeze(0).unsqueeze(0)  # [1, 1, 3, H, W]
        images_inconsistent = images_inconsistent.unsqueeze(0).unsqueeze(0)  # [1, 1, 3, H, W]
        squeeze_batch = True
    
    # Create or use reference mask
    if reference_mask is None:
        if default_reference_indices is None:
            default_reference_indices = [0]  # Default: first view is reference
        device = images_original.device
        reference_mask = create_reference_mask(
            batch_size=batch_size,
            num_views=num_views,
            reference_view_indices=default_reference_indices,
            device=device,
        )  # [B, V] or [1, V]
    else:
        # Ensure reference_mask is 2D [B, V]
        if reference_mask.ndim == 1:
            # [V] -> [1, V]
            reference_mask = reference_mask.unsqueeze(0)
        elif reference_mask.ndim > 2:
            reference_mask = reference_mask.squeeze()
            if reference_mask.ndim == 1:
                reference_mask = reference_mask.unsqueeze(0)
        
        # Ensure batch dimension matches
        if reference_mask.shape[0] != batch_size:
            if reference_mask.shape[0] == 1 and batch_size > 1:
                # Broadcast single batch mask to all batches
                reference_mask = reference_mask.expand(batch_size, -1)
            elif reference_mask.shape[0] > batch_size:
                reference_mask = reference_mask[:batch_size]
    
    # Ensure reference_mask matches number of views
    if reference_mask.shape[1] != num_views:
        raise ValueError(
            f"Reference mask views dimension {reference_mask.shape[1]} doesn't match number of views {num_views}"
        )
    
    # Ensure reference_mask is boolean
    reference_mask = reference_mask.to(torch.bool)  # [B, V]
    
    # Expand reference_mask for broadcasting: [B, V] -> [B, V, 1, 1, 1]
    ref_mask_expanded = reference_mask.view(batch_size, num_views, 1, 1, 1)  # [B, V, 1, 1, 1]
    
    # Mix: where reference_mask is True, use original; where False, use inconsistent
    harmonized_images = torch.where(
        ref_mask_expanded,
        images_original,
        images_inconsistent,
    )  # [B, V, 3, H, W] or [1, V, 3, H, W] or [1, 1, 3, H, W]
    
    # Remove added batch dimension if needed
    if squeeze_batch:
        if harmonized_images.shape[0] == 1:
            harmonized_images = harmonized_images.squeeze(0)  # [V, 3, H, W] or [3, H, W]
            if harmonized_images.ndim == 4 and harmonized_images.shape[0] == 1:
                harmonized_images = harmonized_images.squeeze(0)  # [3, H, W]
    
    return {
        **views,
        "image": harmonized_images,  # Modified: mixed original and inconsistent
        # Keep image_original and image_inconsistent for reference
    }


def apply_harmonization_shim(
    example: AnyExample,
    default_reference_indices: Optional[list[int]] = None,
) -> AnyExample:
    """
    Apply harmonization to an example for SimVS-style training.
    
    This shim implements the SimVS harmonization strategy:
    - Input (context): Reference images (original) + Non-reference images (inconsistent)
    - Output (target): All original images
    
    The model learns to harmonize inconsistent observations into a coherent 3D scene,
    with reference images serving as the "ground truth" that should be reconstructed.
    
    Args:
        example: Data example with "context" and "target" views
        default_reference_indices: If reference_mask not provided, use these indices as reference
        
    Returns:
        Modified example with:
        - context["image"]: Mixed (reference=original, non-reference=inconsistent)
        - target["image"]: All original images
    """
    # Get reference mask from context if available
    context_reference_mask = example["context"].get("reference_mask", None)
    target_reference_mask = example["target"].get("reference_mask", None)
    
    # Apply harmonization to context views (input)
    context_views = apply_harmonization_to_views(
        example["context"],
        reference_mask=context_reference_mask,
        default_reference_indices=default_reference_indices,
    )
    
    # For target views, always use original images (they're the ground truth)
    # If image_original exists, use it; otherwise keep current image
    target_views = example["target"].copy()
    if "image_original" in target_views:
        # Use all original images for targets
        target_views["image"] = target_views["image_original"]
    # If image_original not available, target["image"] stays as-is (should already be original)
    
    return {
        **example,
        "context": context_views,
        "target": target_views,
    }
