from dataclasses import dataclass

from jaxtyping import Float
from torch import Tensor

from ..dataset.types import BatchedExample
from ..model.decoder.decoder import DecoderOutput
from ..model.types import Gaussians
from .loss import Loss


@dataclass
class LossMseCfg:
    weight: float
    use_conf: bool
    alpha: float


@dataclass
class LossMseCfgWrapper:
    mse: LossMseCfg


class LossMse(Loss[LossMseCfg, LossMseCfgWrapper]):
    def forward(
        self,
        prediction: DecoderOutput,
        batch: BatchedExample,
        gaussians: Gaussians,
        global_step: int,
        compare_target: bool = True,
        weight: str = "",
        reference_weights: Float[Tensor, "batch view"] | None = None,
    ) -> Float[Tensor, ""]:
        if compare_target:
            image = batch["target"]["image"]
        else:
            # Use original images for context comparison (not harmonized)
            if "image_original" in batch["context"]:
                image = batch["context"]["image_original"]
            else:
                image = batch["context"]["image"]  # Fallback for backward compatibility
        delta = prediction.color - image
        dist = delta ** 2

        if self.cfg.use_conf:
            conf = prediction.conf.clamp(min=1.)
            dist = conf * dist - self.cfg.alpha * conf.log()

        # Apply reference weights if provided
        if reference_weights is not None:
            # dist: [B, V, 3, H, W], reference_weights: [B, V]
            # Need to broadcast: [B, V, 1, 1, 1]
            B, V = reference_weights.shape
            dist = dist * reference_weights.view(B, V, 1, 1, 1)

        return self.cfg.weight * dist.mean()
