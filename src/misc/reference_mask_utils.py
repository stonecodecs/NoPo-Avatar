"""
Utility functions for creating and handling reference masks in NoPo-Avatar.

Reference masks are used to identify reference images in inconsistent datasets 
where images have different lighting and poses. The mask acts as an indicator 
to the model about which image should be treated as the reference for loss computation.

The mask is a simple view-level boolean indicator with shape (batch, view).
"""

import torch
from torch import Tensor


def create_reference_mask(
    batch_size: int,
    num_views: int,
    reference_view_indices: list[int],
    device: torch.device = torch.device("cpu"),
) -> Tensor:
    """
    Create a reference mask tensor for the given batch and view configuration.
    
    Args:
        batch_size: Number of samples in the batch
        num_views: Total number of views per sample
        reference_view_indices: List of view indices (0-indexed) that are reference images
        device: Device to create the tensor on
        
    Returns:
        Reference mask tensor of shape (batch_size, num_views) as boolean/float tensor
        where reference views are marked with 1.0 and non-reference views with 0.0
        
    Example:
        >>> # Mark the first view as reference in a batch of 2 samples with 3 views each
        >>> mask = create_reference_mask(
        ...     batch_size=2, 
        ...     num_views=3, 
        ...     reference_view_indices=[0],
        ... )
        >>> mask.shape
        torch.Size([2, 3])
        >>> mask[0, 0]  # First view is reference
        tensor(1.)
        >>> mask[0, 1]  # Second view is not reference
        tensor(0.)
    """
    # Initialize mask with zeros (batch, view)
    mask = torch.zeros(batch_size, num_views, device=device)
    
    # Set reference views to 1
    for ref_idx in reference_view_indices:
        if 0 <= ref_idx < num_views:
            mask[:, ref_idx] = 1.0
        else:
            raise ValueError(
                f"Reference index {ref_idx} is out of range for {num_views} views"
            )
    
    return mask


def add_reference_mask_to_batch(
    batch: dict,
    reference_view_indices: list[int],
    view_key: str = "context",
) -> dict:
    """
    Add reference mask to an existing batch dictionary.
    
    Args:
        batch: Batch dictionary containing image data
        reference_view_indices: List of view indices that are reference images
        view_key: Key in batch dict to add mask to ("context" or "target")
        
    Returns:
        Updated batch dictionary with reference_mask added
        
    Example:
        >>> batch = {
        ...     "context": {
        ...         "image": torch.randn(2, 3, 3, 256, 256),
        ...         "extrinsics": torch.eye(4).unsqueeze(0).unsqueeze(0).expand(2, 3, -1, -1),
        ...         "intrinsics": torch.eye(3).unsqueeze(0).unsqueeze(0).expand(2, 3, -1, -1),
        ...     }
        ... }
        >>> batch = add_reference_mask_to_batch(batch, reference_view_indices=[0])
        >>> "reference_mask" in batch["context"]
        True
        >>> batch["context"]["reference_mask"].shape
        torch.Size([2, 3])
    """
    if view_key not in batch:
        raise ValueError(f"Key '{view_key}' not found in batch")
    
    if "image" not in batch[view_key]:
        raise ValueError(f"No 'image' key found in batch['{view_key}']")
    
    # Get dimensions from image tensor
    b, v, _, _, _ = batch[view_key]["image"].shape
    device = batch[view_key]["image"].device
    
    # Create mask
    mask = create_reference_mask(b, v, reference_view_indices, device)
    
    # Add to batch
    batch[view_key]["reference_mask"] = mask
    
    return batch


def get_reference_view_indices_from_mask(
    reference_mask: Tensor,
) -> list[list[int]]:
    """
    Extract reference view indices from a reference mask tensor.
    
    Args:
        reference_mask: Reference mask tensor of shape (batch, view)
        
    Returns:
        List of lists, where each inner list contains the reference view indices 
        for the corresponding batch element
        
    Example:
        >>> mask = create_reference_mask(2, 3, [0, 2])
        >>> indices = get_reference_view_indices_from_mask(mask)
        >>> indices
        [[0, 2], [0, 2]]
    """
    # reference_mask shape: (batch, view)
    result = []
    for batch_idx in range(reference_mask.shape[0]):
        ref_indices = (reference_mask[batch_idx] > 0).nonzero(as_tuple=False).squeeze(-1).tolist()
        if isinstance(ref_indices, int):
            ref_indices = [ref_indices]
        result.append(ref_indices)
    
    return result


# Example usage in a dataset or dataloader
if __name__ == "__main__":
    print("Example: Creating a reference mask")
    print("=" * 60)
    
    # Create a reference mask for a batch
    batch_size = 2
    num_views = 4
    reference_indices = [0]  # First view is the reference
    
    mask = create_reference_mask(
        batch_size=batch_size,
        num_views=num_views,
        reference_view_indices=reference_indices,
    )
    
    print(f"Created mask shape: {mask.shape}")
    print(f"Reference view indices: {reference_indices}")
    print(f"Mask values for first sample:")
    for view_idx in range(num_views):
        print(f"  View {view_idx}: {mask[0, view_idx].item()}")
    
    print("\n" + "=" * 60)
    print("Example: Adding reference mask to batch")
    print("=" * 60)
    
    # Create a dummy batch
    batch = {
        "context": {
            "image": torch.randn(2, 4, 3, 256, 256),
            "extrinsics": torch.eye(4).unsqueeze(0).unsqueeze(0).expand(2, 4, -1, -1),
            "intrinsics": torch.eye(3).unsqueeze(0).unsqueeze(0).expand(2, 4, -1, -1),
        },
        "target": {
            "image": torch.randn(2, 2, 3, 256, 256),
        }
    }
    
    # Add reference mask to context views
    batch = add_reference_mask_to_batch(batch, reference_view_indices=[0, 1])
    
    print(f"Reference mask added to batch['context']")
    print(f"Mask shape: {batch['context']['reference_mask'].shape}")
    
    # Extract reference indices back from mask
    ref_indices = get_reference_view_indices_from_mask(batch['context']['reference_mask'])
    print(f"Extracted reference indices: {ref_indices}")
