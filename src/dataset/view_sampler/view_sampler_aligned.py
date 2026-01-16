from dataclasses import dataclass
from typing import Literal

import torch
from jaxtyping import Float, Int64
from torch import Tensor

from .view_sampler import ViewSampler

"""
For SimVS behavior, we want to ensure that the corresponding context view GTs are used as target views.
This ensures that the indices are aligned.
"""

@dataclass
class ViewSamplerAlignedCfg:
    name: Literal["aligned"]
    num_context_views: int
    num_target_views: int


class ViewSamplerAligned(ViewSampler[ViewSamplerAlignedCfg]):
    def schedule(self, initial: int, final: int) -> int:
        fraction = self.global_step / self.cfg.warm_up_steps
        return min(initial + int((final - initial) * fraction), final)

    def sample(
        self,
        scene: str,
        extrinsics: Float[Tensor, "view 4 4"],
        intrinsics: Float[Tensor, "view 3 3"],
        device: torch.device = torch.device("cpu"),
    ) -> tuple[
        Int64[Tensor, " context_view"],  # indices for context views
        Int64[Tensor, " target_view"],  # indices for target views
        Float[Tensor, " overlap"],  # overlap
    ]:
        num_views, _, _ = extrinsics.shape
    
        # randomly sample without replacement (assuming unique images)
        index_context = torch.randperm(num_views, device=device)[:self.cfg.num_context_views]
        index_target = index_context[:self.cfg.num_target_views] # same as context indices
        
        overlap = torch.tensor([0.5], dtype=torch.float32, device=device)  # dummy
        
        return index_context, index_target, overlap

    @property
    def num_context_views(self) -> int:
        return self.cfg.num_context_views

    @property
    def num_target_views(self) -> int:
        return self.cfg.num_target_views