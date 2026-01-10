"""
Tests for inconsistent_image_shim.py
"""
import tempfile
import shutil
from pathlib import Path
import torch
from PIL import Image
import numpy as np

from src.dataset.shims.inconsistent_image_shim import (
    load_inconsistent_image,
    load_inconsistent_views,
    apply_inconsistent_image_shim,
)
from src.dataset.types import AnyExample, AnyViews


def create_test_image(path: Path, color: tuple = (255, 0, 0)):
    """Create a test image file."""
    img = Image.new("RGB", (256, 256), color)
    img.save(path)
    return path


def test_load_inconsistent_image():
    """Test loading a single inconsistent image."""
    with tempfile.TemporaryDirectory() as tmpdir:
        base_path = Path(tmpdir)
        stage = "train"
        scene = "subject_001"
        view_index = 5
        
        # Create directory structure
        images_dir = base_path / stage / scene / "images"
        images_dir.mkdir(parents=True, exist_ok=True)
        
        # Create test image
        image_path = images_dir / "frame_000005.png"
        create_test_image(image_path, color=(128, 128, 128))
        
        # Test loading
        result = load_inconsistent_image(
            inconsistent_base_path=base_path,
            scene=scene,
            view_index=view_index,
            stage=stage,
        )
        
        assert result is not None, "Should load image successfully"
        assert result.shape == (3, 256, 256), f"Expected shape (3, 256, 256), got {result.shape}"
        assert result.dtype == torch.float32, "Should be float32"
        assert (result >= 0).all() and (result <= 1).all(), "Values should be in [0, 1]"


def test_load_inconsistent_image_not_found():
    """Test loading when image doesn't exist."""
    with tempfile.TemporaryDirectory() as tmpdir:
        base_path = Path(tmpdir)
        result = load_inconsistent_image(
            inconsistent_base_path=base_path,
            scene="nonexistent",
            view_index=0,
            stage="train",
        )
        assert result is None, "Should return None when image not found"


def test_load_inconsistent_views():
    """Test loading inconsistent images for multiple views."""
    with tempfile.TemporaryDirectory() as tmpdir:
        base_path = Path(tmpdir)
        stage = "train"
        scene = "subject_001"
        
        # Create directory structure
        images_dir = base_path / stage / scene / "images"
        images_dir.mkdir(parents=True, exist_ok=True)
        
        # Create test images for views 0, 1, 2
        for i in range(3):
            image_path = images_dir / f"frame_{i:06d}.png"
            create_test_image(image_path, color=(i * 50, i * 50, i * 50))
        
        # Create views dictionary
        original_images = torch.rand(3, 3, 256, 256)  # [V, C, H, W]
        views: AnyViews = {
            "image": original_images,
            "index": torch.tensor([0, 1, 2]),
        }
        
        # Test loading
        result = load_inconsistent_views(
            views=views,
            inconsistent_base_path=base_path,
            scene=scene,
            view_indices=torch.tensor([0, 1, 2]),
            stage=stage,
        )
        
        assert "image_original" in result, "Should have image_original"
        assert "image_inconsistent" in result, "Should have image_inconsistent"
        assert result["image_original"].shape == (3, 3, 256, 256), "Original images shape should match"
        assert result["image_inconsistent"].shape == (3, 3, 256, 256), "Inconsistent images shape should match"
        assert torch.allclose(result["image_original"], original_images), "Original images should be preserved"


def test_apply_inconsistent_image_shim():
    """Test the full shim function."""
    with tempfile.TemporaryDirectory() as tmpdir:
        base_path = Path(tmpdir)
        stage = "train"
        scene = "subject_001"
        
        # Create directory structure
        images_dir = base_path / stage / scene / "images"
        images_dir.mkdir(parents=True, exist_ok=True)
        
        # Create test images
        for i in range(2):
            image_path = images_dir / f"frame_{i:06d}.png"
            create_test_image(image_path, color=(100 + i * 50, 100 + i * 50, 100 + i * 50))
        
        # Create example
        example: AnyExample = {
            "context": {
                "image": torch.rand(1, 2, 3, 256, 256),  # [B, V, C, H, W]
                "index": torch.tensor([[0, 1]]),  # [B, V]
            },
            "target": {
                "image": torch.rand(1, 1, 3, 256, 256),
                "index": torch.tensor([[0]]),
            },
            "scene": [scene],
        }
        
        # Apply shim
        result = apply_inconsistent_image_shim(
            example,
            inconsistent_base_path=base_path,
            stage=stage,
        )
        
        assert "image_original" in result["context"], "Context should have image_original"
        assert "image_inconsistent" in result["context"], "Context should have image_inconsistent"
        assert "image_original" in result["target"], "Target should have image_original"
        assert "image_inconsistent" in result["target"], "Target should have image_inconsistent"


def test_apply_inconsistent_image_shim_missing_images():
    """Test shim when some inconsistent images are missing (should use original as fallback)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        base_path = Path(tmpdir)
        stage = "train"
        scene = "subject_001"
        
        # Create directory structure
        images_dir = base_path / stage / scene / "images"
        images_dir.mkdir(parents=True, exist_ok=True)
        
        # Create only one test image (view 0 exists, view 1 doesn't)
        image_path = images_dir / "frame_000000.png"
        create_test_image(image_path, color=(200, 200, 200))
        
        # Create example
        original_context_images = torch.rand(1, 2, 3, 256, 256)
        example: AnyExample = {
            "context": {
                "image": original_context_images,
                "index": torch.tensor([[0, 1]]),
            },
            "target": {
                "image": torch.rand(1, 1, 3, 256, 256),
                "index": torch.tensor([[0]]),
            },
            "scene": [scene],
        }
        
        # Apply shim
        result = apply_inconsistent_image_shim(
            example,
            inconsistent_base_path=base_path,
            stage=stage,
        )
        
        # Should still work, with missing images using original as fallback
        assert "image_inconsistent" in result["context"], "Should have image_inconsistent even with missing images"
        # View 0 should have loaded inconsistent image, view 1 should use original
        assert result["image_inconsistent"].shape == (1, 2, 3, 256, 256), "Should have correct shape"


if __name__ == "__main__":
    test_load_inconsistent_image()
    print("✓ test_load_inconsistent_image passed")
    
    test_load_inconsistent_image_not_found()
    print("✓ test_load_inconsistent_image_not_found passed")
    
    test_load_inconsistent_views()
    print("✓ test_load_inconsistent_views passed")
    
    test_apply_inconsistent_image_shim()
    print("✓ test_apply_inconsistent_image_shim passed")
    
    test_apply_inconsistent_image_shim_missing_images()
    print("✓ test_apply_inconsistent_image_shim_missing_images passed")
    
    print("\nAll tests passed! ✓")
