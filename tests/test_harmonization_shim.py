"""
Tests for harmonization_shim.py
"""
import torch
from jaxtyping import Float
from torch import Tensor

from src.dataset.shims.harmonization_shim import (
    apply_harmonization_to_views,
    apply_harmonization_shim,
)
from src.dataset.types import AnyExample, AnyViews
from src.misc.reference_mask_utils import create_reference_mask


def test_apply_harmonization_to_views_with_reference_mask():
    """Test harmonization with explicit reference mask."""
    # Create test data
    num_views = 4
    original_images = torch.rand(num_views, 3, 256, 256)  # [V, C, H, W]
    inconsistent_images = torch.rand(num_views, 3, 256, 256)  # Different from original
    
    views: AnyViews = {
        "image": original_images.clone(),
        "image_original": original_images.clone(),
        "image_inconsistent": inconsistent_images.clone(),
    }
    
    # Create reference mask: views 0 and 2 are reference
    reference_mask = torch.tensor([True, False, True, False], dtype=torch.bool)
    
    # Apply harmonization
    result = apply_harmonization_to_views(views, reference_mask=reference_mask)
    
    assert "image" in result, "Should have image field"
    assert result["image"].shape == (num_views, 3, 256, 256), "Shape should match"
    
    # Check that reference views use original images
    assert torch.allclose(result["image"][0], original_images[0]), "Reference view 0 should use original"
    assert torch.allclose(result["image"][2], original_images[2]), "Reference view 2 should use original"
    
    # Check that non-reference views use inconsistent images
    assert torch.allclose(result["image"][1], inconsistent_images[1]), "Non-reference view 1 should use inconsistent"
    assert torch.allclose(result["image"][3], inconsistent_images[3]), "Non-reference view 3 should use inconsistent"


def test_apply_harmonization_to_views_default_reference():
    """Test harmonization with default reference indices (first view)."""
    num_views = 3
    original_images = torch.rand(num_views, 3, 256, 256)
    inconsistent_images = torch.rand(num_views, 3, 256, 256)
    
    views: AnyViews = {
        "image": original_images.clone(),
        "image_original": original_images.clone(),
        "image_inconsistent": inconsistent_images.clone(),
    }
    
    # Apply without reference_mask (should default to first view)
    result = apply_harmonization_to_views(views, default_reference_indices=[0])
    
    # First view should use original, others should use inconsistent
    assert torch.allclose(result["image"][0], original_images[0]), "First view should be reference (original)"
    assert torch.allclose(result["image"][1], inconsistent_images[1]), "Second view should use inconsistent"
    assert torch.allclose(result["image"][2], inconsistent_images[2]), "Third view should use inconsistent"


def test_apply_harmonization_to_views_no_inconsistent_images():
    """Test harmonization when inconsistent images are not loaded (should return as-is)."""
    num_views = 2
    original_images = torch.rand(num_views, 3, 256, 256)
    
    views: AnyViews = {
        "image": original_images.clone(),
        # No image_original or image_inconsistent
    }
    
    # Apply harmonization
    result = apply_harmonization_to_views(views)
    
    # Should return views unchanged
    assert torch.allclose(result["image"], original_images), "Should return unchanged when no inconsistent images"


def test_apply_harmonization_shim():
    """Test the full harmonization shim on an example."""
    num_context_views = 3
    num_target_views = 1
    
    context_original = torch.rand(1, num_context_views, 3, 256, 256)  # [B, V, C, H, W]
    context_inconsistent = torch.rand(1, num_context_views, 3, 256, 256)
    target_original = torch.rand(1, num_target_views, 3, 256, 256)
    
    # Create reference mask for context
    reference_mask = torch.tensor([[True, False, False]], dtype=torch.bool)  # [B, V]
    
    example: AnyExample = {
        "context": {
            "image": context_original.clone(),
            "image_original": context_original.clone(),
            "image_inconsistent": context_inconsistent.clone(),
            "reference_mask": reference_mask,
        },
        "target": {
            "image": target_original.clone(),
            "image_original": target_original.clone(),
        },
        "scene": ["test_scene"],
    }
    
    # Apply harmonization
    result = apply_harmonization_shim(example)
    
    # Context should have mixed images
    assert "image" in result["context"], "Context should have image"
    # View 0 (reference) should use original
    assert torch.allclose(result["context"]["image"][0, 0], context_original[0, 0]), \
        "Reference view should use original"
    # View 1 (non-reference) should use inconsistent
    assert torch.allclose(result["context"]["image"][0, 1], context_inconsistent[0, 1]), \
        "Non-reference view should use inconsistent"
    
    # Target should use all original images
    assert torch.allclose(result["target"]["image"], target_original), \
        "Target should use all original images"


def test_apply_harmonization_shim_batched():
    """Test harmonization with batched examples."""
    batch_size = 2
    num_views = 2
    
    context_original = torch.rand(batch_size, num_views, 3, 256, 256)
    context_inconsistent = torch.rand(batch_size, num_views, 3, 256, 256)
    
    # First batch: view 0 is reference
    # Second batch: view 1 is reference
    reference_mask = torch.tensor([
        [True, False],  # Batch 0
        [False, True],  # Batch 1
    ], dtype=torch.bool)
    
    example: AnyExample = {
        "context": {
            "image": context_original.clone(),
            "image_original": context_original.clone(),
            "image_inconsistent": context_inconsistent.clone(),
            "reference_mask": reference_mask,
        },
        "target": {
            "image": torch.rand(batch_size, 1, 3, 256, 256),
            "image_original": torch.rand(batch_size, 1, 3, 256, 256),
        },
        "scene": ["scene1", "scene2"],
    }
    
    result = apply_harmonization_shim(example)
    
    # Batch 0: view 0 should be original, view 1 should be inconsistent
    assert torch.allclose(result["context"]["image"][0, 0], context_original[0, 0])
    assert torch.allclose(result["context"]["image"][0, 1], context_inconsistent[0, 1])
    
    # Batch 1: view 0 should be inconsistent, view 1 should be original
    assert torch.allclose(result["context"]["image"][1, 0], context_inconsistent[1, 0])
    assert torch.allclose(result["context"]["image"][1, 1], context_original[1, 1])


if __name__ == "__main__":
    test_apply_harmonization_to_views_with_reference_mask()
    print("✓ test_apply_harmonization_to_views_with_reference_mask passed")
    
    test_apply_harmonization_to_views_default_reference()
    print("✓ test_apply_harmonization_to_views_default_reference passed")
    
    test_apply_harmonization_to_views_no_inconsistent_images()
    print("✓ test_apply_harmonization_to_views_no_inconsistent_images passed")
    
    test_apply_harmonization_shim()
    print("✓ test_apply_harmonization_shim passed")
    
    test_apply_harmonization_shim_batched()
    print("✓ test_apply_harmonization_shim_batched passed")
    
    print("\nAll tests passed! ✓")
