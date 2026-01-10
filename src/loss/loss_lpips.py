from dataclasses import dataclass

import torch
from einops import rearrange
from jaxtyping import Float
from lpips import LPIPS
from torch import Tensor

from ..dataset.types import BatchedExample
from ..misc.nn_module_tools import convert_to_buffer
from ..model.decoder.decoder import DecoderOutput
from ..model.types import Gaussians
from .loss import Loss


@dataclass
class LossLpipsCfg:
    weight_template: float
    weight_rgb: float
    weight: float
    apply_after_step: int


@dataclass
class LossLpipsCfgWrapper:
    lpips: LossLpipsCfg


class LossLpips(Loss[LossLpipsCfg, LossLpipsCfgWrapper]):
    lpips: LPIPS

    def __init__(self, cfg: LossLpipsCfgWrapper) -> None:
        super().__init__(cfg)

        self.lpips = LPIPS(net="vgg")
        convert_to_buffer(self.lpips, persistent=False)

    def forward(
        self,
        prediction: DecoderOutput,
        batch: BatchedExample,
        gaussians: Gaussians,
        global_step: int,
        compare_target: bool = True,
        weight: str = "",
        reference_weights: Float[Tensor, "batch view"] | None = None,
    ) -> Float[Tensor, ""] | float:

        weight_suffix = "" if weight == "" else f"_{weight}"
        if getattr(self.cfg, "weight" + weight_suffix) == 0:
            return 0.0

        if compare_target:
            image = batch["target"]["image"]
        else:
            # Use original images for context comparison (not harmonized)
            if "image_original" in batch["context"]:
                image = batch["context"]["image_original"]
            else:
                image = batch["context"]["image"]  # Fallback for backward compatibility

        # Before the specified step, don't apply the loss.
        if global_step < self.cfg.apply_after_step:
            return torch.tensor(0, dtype=torch.float32, device=image.device)

        loss = self.lpips.forward(
            rearrange(prediction.color, "b v c h w -> (b v) c h w"),
            rearrange(image, "b v c h w -> (b v) c h w"),
            normalize=True,
        )

        # Apply reference weights if provided
        if reference_weights is not None:
            # loss: [B*V], reference_weights: [B, V]
            # Reshape loss to [B, V], multiply by weights, keep as [B, V] for mean() at return
            B, V = reference_weights.shape
            loss = loss.reshape(B, V) * reference_weights

        return getattr(self.cfg, "weight" + weight_suffix) * loss.mean()
