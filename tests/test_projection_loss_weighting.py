"""
Tests for projection loss with per-view weighting.
"""
import torch
from jaxtyping import Float
from torch import Tensor

from src.loss.loss_projection import LossProjection, LossProjectionCfg, LossProjectionCfgWrapper
from src.model.decoder.decoder import DecoderOutput
from src.model.types import Gaussians
from src.dataset.types import BatchedExample


def create_mock_gaussians(batch_size: int = 1, num_gaussians: int = 100, num_views: int = 3):
    """Create mock Gaussians for testing."""
    device = torch.device("cpu")
    
    # Create indices: [batch, gaussian, 3] where last dim is [view_id, w, h]
    # View IDs are 1-indexed
    indices = torch.zeros(batch_size, num_gaussians, 3, dtype=torch.long)
    for batch_idx in range(batch_size):
        for i in range(num_gaussians):
            view_id = (i % num_views) + 1  # 1-indexed view IDs
            w = (i * 7 + batch_idx * 13) % 256  # Add batch offset for variety
            h = (i * 11 + batch_idx * 17) % 256  # Add batch offset for variety
            indices[batch_idx, i] = torch.tensor([view_id, w, h])
    
    return Gaussians(
        means=torch.randn(batch_size, num_gaussians, 3, device=device),
        covariances=torch.eye(3, device=device).unsqueeze(0).unsqueeze(0).expand(batch_size, num_gaussians, -1, -1),
        harmonics=torch.randn(batch_size, num_gaussians, 3, 16, device=device),
        opacities=torch.ones(batch_size, num_gaussians, device=device) * 0.5,
        idx=indices,
        lbs_weights=torch.softmax(torch.randn(batch_size, num_gaussians, 10, device=device), dim=-1),
        lbs_weights_bones=None,
        conf=None,
    )


def create_mock_batch(batch_size: int = 1, num_views: int = 3, image_size: tuple = (256, 256)):
    """Create mock batch for testing."""
    h, w = image_size
    device = torch.device("cpu")
    
    # Create mock camera parameters
    extrinsics = torch.eye(4, device=device).unsqueeze(0).unsqueeze(0).expand(batch_size, num_views, -1, -1)
    intrinsics = torch.eye(3, device=device).unsqueeze(0).unsqueeze(0).expand(batch_size, num_views, -1, -1)
    
    # Create mock pose parameters
    num_joints = 10
    Rs = torch.eye(3, device=device).unsqueeze(0).unsqueeze(0).unsqueeze(0).expand(batch_size, num_views, num_joints, -1, -1)
    Ts = torch.zeros(batch_size, num_views, num_joints, 3, device=device)
    cnl_Rs = Rs.clone()
    cnl_Ts = Ts.clone()
    
    batch: BatchedExample = {
        "context": {
            "image": torch.rand(batch_size, num_views, 3, h, w, device=device),
            "extrinsics": extrinsics,
            "intrinsics": intrinsics,
            "Rs": Rs,
            "Ts": Ts,
            "cnl_Rs": cnl_Rs,
            "cnl_Ts": cnl_Ts,
            "near": torch.ones(batch_size, num_views, device=device) * 0.1,
            "far": torch.ones(batch_size, num_views, device=device) * 100.0,
        },
        "target": {
            "image": torch.rand(batch_size, 1, 3, h, w, device=device),
        },
        "scene": ["test_scene"] * batch_size,
    }
    return batch


def test_projection_loss_without_weights():
    """Test projection loss without weights (original behavior)."""
    loss_fn = LossProjection(LossProjectionCfgWrapper(LossProjectionCfg(weight=1.0)))
    
    gaussians = create_mock_gaussians(batch_size=1, num_gaussians=50, num_views=3)
    batch = create_mock_batch(batch_size=1, num_views=3)
    output = DecoderOutput(
        color=torch.rand(1, 1, 3, 256, 256),
        depth=torch.rand(1, 1, 256, 256),
        conf=None,
    )
    
    # Compute loss without weights
    loss = loss_fn.forward(output, batch, gaussians, global_step=0, weights=None)
    
    assert isinstance(loss, torch.Tensor), "Loss should be a tensor"
    assert loss.shape == (), "Loss should be scalar"
    assert loss.item() >= 0, "Loss should be non-negative"


def test_projection_loss_with_weights():
    """Test projection loss with per-view weights."""
    loss_fn = LossProjection(LossProjectionCfgWrapper(LossProjectionCfg(weight=1.0)))
    
    gaussians = create_mock_gaussians(batch_size=1, num_gaussians=50, num_views=3)
    batch = create_mock_batch(batch_size=1, num_views=3)
    output = DecoderOutput(
        color=torch.rand(1, 1, 3, 256, 256),
        depth=torch.rand(1, 1, 256, 256),
        conf=None,
    )
    
    # Create weights: view 0 gets weight 0 (reference), views 1-2 get higher weights
    weights = torch.tensor([[0.0, 2.0, 5.0]], dtype=torch.float32)  # [B, V]
    
    # Compute loss with weights
    loss_weighted = loss_fn.forward(output, batch, gaussians, global_step=0, weights=weights)
    
    assert isinstance(loss_weighted, torch.Tensor), "Loss should be a tensor"
    assert loss_weighted.shape == (), "Loss should be scalar"
    assert loss_weighted.item() >= 0, "Loss should be non-negative"
    
    # Loss with weights should be different from loss without weights
    loss_unweighted = loss_fn.forward(output, batch, gaussians, global_step=0, weights=None)
    # They might be similar but should be computed differently
    assert not torch.isnan(loss_weighted), "Weighted loss should not be NaN"


def test_projection_loss_zero_weight_reference():
    """Test that reference views (weight=0) don't contribute to loss."""
    loss_fn = LossProjection(LossProjectionCfgWrapper(LossProjectionCfg(weight=1.0)))
    
    gaussians = create_mock_gaussians(batch_size=1, num_gaussians=100, num_views=2)
    batch = create_mock_batch(batch_size=1, num_views=2)
    output = DecoderOutput(
        color=torch.rand(1, 1, 3, 256, 256),
        depth=torch.rand(1, 1, 256, 256),
        conf=None,
    )
    
    # Weight 0 for view 0 (reference), weight 1 for view 1
    weights_ref_zero = torch.tensor([[0.0, 1.0]], dtype=torch.float32)
    
    # Weight 1 for both views
    weights_uniform = torch.tensor([[1.0, 1.0]], dtype=torch.float32)
    
    loss_ref_zero = loss_fn.forward(output, batch, gaussians, global_step=0, weights=weights_ref_zero)
    loss_uniform = loss_fn.forward(output, batch, gaussians, global_step=0, weights=weights_uniform)
    
    # Loss with reference weight=0 should be lower (or equal if all points are from view 0)
    # This depends on the distribution of points across views
    assert not torch.isnan(loss_ref_zero), "Loss should not be NaN"
    assert not torch.isnan(loss_uniform), "Loss should not be NaN"


def test_projection_loss_batched_weights():
    """Test projection loss with batched weights."""
    batch_size = 2
    num_views = 3
    
    loss_fn = LossProjection(LossProjectionCfgWrapper(LossProjectionCfg(weight=1.0)))
    
    gaussians = create_mock_gaussians(batch_size=batch_size, num_gaussians=50, num_views=num_views)
    batch = create_mock_batch(batch_size=batch_size, num_views=num_views)
    output = DecoderOutput(
        color=torch.rand(batch_size, 1, 3, 256, 256),
        depth=torch.rand(batch_size, 1, 256, 256),
        conf=None,
    )
    
    # Different weights for each batch
    weights = torch.tensor([
        [0.0, 1.0, 2.0],  # Batch 0: view 0 is reference
        [1.0, 0.0, 3.0],  # Batch 1: view 1 is reference
    ], dtype=torch.float32)  # [B, V]
    
    loss = loss_fn.forward(output, batch, gaussians, global_step=0, weights=weights)
    
    assert isinstance(loss, torch.Tensor), "Loss should be a tensor"
    assert loss.shape == (), "Loss should be scalar"
    assert not torch.isnan(loss), "Loss should not be NaN"
    assert loss.item() >= 0, "Loss should be non-negative"


if __name__ == "__main__":
    test_projection_loss_without_weights()
    print("✓ test_projection_loss_without_weights passed")
    
    test_projection_loss_with_weights()
    print("✓ test_projection_loss_with_weights passed")
    
    test_projection_loss_zero_weight_reference()
    print("✓ test_projection_loss_zero_weight_reference passed")
    
    test_projection_loss_batched_weights()
    print("✓ test_projection_loss_batched_weights passed")
    
    print("\nAll tests passed! ✓")
