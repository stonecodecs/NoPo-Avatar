import torch
from jaxtyping import Float
from torch import Tensor
import torchvision.transforms.functional as TF

from ..types import AnyExample, AnyViews


def jitter_view(
    views: AnyViews,
    brightness_factor: Float,
    contrast_factor: Float,
    saturation_factor: Float,
    hue_factor: Float,
    bgcolor: Tensor | None = None,
    ref_mask: Tensor | None = None,
) -> AnyViews:
    """Apply color jitter."""
    mask = views["mask"].unsqueeze(1)

    def jitter_and_composite(img: Tensor) -> Tensor:
        out = TF.adjust_brightness(img, brightness_factor)
        out = TF.adjust_contrast(out, contrast_factor)
        out = TF.adjust_saturation(out, saturation_factor)
        out = TF.adjust_hue(out, hue_factor)
        if bgcolor is not None and bgcolor.numel() >= 3:
            bg = bgcolor.to(device=img.device, dtype=img.dtype).view(-1, 3, 1, 1)
            if bg.shape[0] == 1:
                bg = bg.expand_as(img)
            out = out * mask + bg * (1.0 - mask)
        else:
            out = out * mask + img * (1.0 - mask)
        return out

    img = views["image"]
    if ref_mask is not None and ref_mask.numel() > 0:
        # Keep reference view unchanged; jitter only non-reference views
        ref = ref_mask.to(device=img.device, dtype=img.dtype)
        while ref.dim() < img.dim():
            ref = ref.unsqueeze(-1)
        jittered = jitter_and_composite(img)
        image_out = img * ref + jittered * (1.0 - ref)
    else:
        image_out = jitter_and_composite(img)

    result = {**views, "image": image_out}
    # image_gt stays the same
    # if "image_gt" in views and views["image_gt"] is not None:
    #     result["image_gt"] = jitter_and_composite(views["image_gt"])
    return result


def apply_color_jitter_shim(
    example: AnyExample,
    generator: torch.Generator | None = None,
) -> AnyExample:
    """Randomly augment the training images. Uses example bgcolor for mask=0 so image and image_gt match.
    For context, the reference view (ref_mask) is left unchanged; non-reference views are jittered. Target: all jittered."""
    brightness_factor = 1 + (torch.rand(tuple(), generator=generator) * 0.8 - 0.4)
    contrast_factor = 1 + (torch.rand(tuple(), generator=generator) * 0.8 - 0.4)
    saturation_factor = 1 + (torch.rand(tuple(), generator=generator) * 0.8 - 0.4)
    hue_factor = torch.rand(tuple(), generator=generator) * 0.8 - 0.4

    bgcolor = example.get("bgcolor")
    context_ref_mask = example.get("context", {}).get("ref_mask")

    return {
        **example,
        "context": jitter_view(
            example["context"],
            brightness_factor,
            contrast_factor,
            saturation_factor,
            hue_factor,
            bgcolor=bgcolor,
            ref_mask=context_ref_mask,
        ),
        "target": jitter_view(
            example["target"],
            brightness_factor,
            contrast_factor,
            saturation_factor,
            hue_factor,
            bgcolor=bgcolor,
        ),
    }