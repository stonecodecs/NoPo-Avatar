"""
Loss weighting functions for multi-view training with reference frames.

Based on stable-virtual-camera implementation:
https://github.com/stonecodecs/stable-virtual-camera
"""
from abc import ABC, abstractmethod
import torch
from jaxtyping import Float
from torch import Tensor


class LossWeighting(ABC):
    """Abstract base class for loss weighting strategies."""
    
    @abstractmethod
    def __call__(
        self, 
        ref_mask: Float[Tensor, "batch view"] | None = None,
        **kwargs
    ) -> Float[Tensor, "batch view"]:
        """
        Compute per-view loss weights.
        
        Args:
            ref_mask: Boolean tensor indicating reference frames (True = reference)
            **kwargs: Additional arguments for specific weighting strategies
            
        Returns:
            Weight tensor of shape (batch, view)
        """
        pass


class UniformWeighting(LossWeighting):
    """Uniform weighting - all frames get equal weight."""
    
    def __call__(
        self, 
        ref_mask: Float[Tensor, "batch view"] | None = None,
        **kwargs
    ) -> Float[Tensor, "batch view"]:
        """Return uniform weights of 1.0 for all frames."""
        if ref_mask is None:
            raise ValueError("ref_mask is required for UniformWeighting")
        return torch.ones_like(ref_mask, dtype=torch.float)


class SimVSWeighting(LossWeighting):
    """
    SimVS (Simulated Virtual Scene) weighting strategy.
    Combines distance-based weighting with reference frame exclusion.
    
    - Reference frames get weight 0 (they're clean/correct)
    - Other frames weighted by distance from input frames
    - Farther from inputs = higher weight (learn novel view synthesis)
    """
    
    def __init__(self, max_weight: float = 5.0):
        """
        Args:
            max_weight: Maximum weight for frames farthest from inputs
        """
        self.max_weight = max_weight
    
    def __call__(
        self, 
        ref_mask: Float[Tensor, "batch view"] | None = None,
        input_mask: Float[Tensor, "batch view"] | None = None,
        **kwargs
    ) -> Float[Tensor, "batch view"]:
        """
        Compute SimVS weights: distance-based + reference exclusion.
        
        Args:
            ref_mask: Boolean tensor (batch, view) indicating reference frames
            input_mask: Boolean tensor (batch, view) indicating input frames
            
        Returns:
            Weight tensor: 0 for reference frames, distance-based for others
            
        Notes:
            - Reference frames (ref_mask=True) always get weight 0
            - Input frames get weight 0 (they're the conditioning)
            - Target frames weighted by distance from nearest input
            - All weights clamped to [0, max_weight]
        """
        if ref_mask is None:
            raise ValueError("ref_mask is required for SimVSWeighting")
        if input_mask is None:
            raise ValueError("input_mask is required for SimVSWeighting")
        
        bools = input_mask.to(torch.bool)
        ref_bools = ref_mask.to(torch.bool)
        batch_size, N = bools.shape
        device = bools.device
        
        indices = torch.arange(N, device=device).unsqueeze(0).expand(batch_size, N)
        weights = torch.full(
            (batch_size, N), self.max_weight, dtype=torch.float, device=device
        )
        
        # Compute distance-based weights for non-reference frames
        for b in range(batch_size):
            # Get indices of input frames
            true_idx = indices[b][bools[b]]
            
            if len(true_idx) > 0:
                # Compute distance to nearest input frame
                dists = torch.stack([
                    torch.abs(indices[b] - t) for t in true_idx
                ]).min(dim=0).values
                
                # Input frames get 0 distance
                dists[bools[b]] = 0
                
                # Normalize and scale
                max_dist = dists.max()
                if max_dist > 0:
                    weights[b] = dists / max_dist * self.max_weight
                else:
                    # All frames are inputs
                    weights[b] = torch.zeros_like(weights[b])
                
                # Handle NaN values (shouldn't happen but just in case)
                if torch.any(weights[b].isnan()):
                    weights[b] = torch.zeros_like(weights[b])
            else:
                # No input frames, use max weight for all
                weights[b] = self.max_weight
        
        # Create reference exclusion weights: 0 for ref frames, 1 for others
        ref_weights = 1.0 - ref_bools.float()
        
        # Combine: multiply distance weights by ref_weights
        # This ensures reference frames always get 0 weight
        weights = weights * ref_weights
        
        # Clamp to valid range
        weights = torch.clamp(weights, min=0.0, max=self.max_weight)
        
        return weights


def get_loss_weighting(
    strategy: str = "uniform",
    max_weight: float = 5.0
) -> LossWeighting:
    """
    Factory function to get loss weighting strategy.
    
    Args:
        strategy: One of ["uniform", "simvs"]
        max_weight: Maximum weight for SimVS strategy
        
    Returns:
        LossWeighting instance
    """
    strategies = {
        "uniform": UniformWeighting,
        "simvs": lambda: SimVSWeighting(max_weight=max_weight),
    }
    
    if strategy not in strategies:
        raise ValueError(
            f"Unknown weighting strategy: {strategy}. "
            f"Choose from {list(strategies.keys())}"
        )
    
    weighting_cls = strategies[strategy]
    return weighting_cls() if callable(weighting_cls) else weighting_cls
