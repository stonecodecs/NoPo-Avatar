import numpy as np
import torch
from einops import rearrange, repeat
from jaxtyping import Float
from PIL import Image
import cv2
from torch import Tensor
import torch.nn.functional as F
import torchvision.transforms.functional as tvf
from torchvision.transforms import InterpolationMode
import random

from ..types import AnyExample, AnyViews, Callable


def rescale(
    image: Float[Tensor, "3 h_in w_in"],
    mask: Float[Tensor, "h_in w_in"],
    lbs_weight: Float[Tensor, "h_in w_in d"] | None,
    shape: tuple[int, int],
) -> Float[Tensor, "3 h_out w_out"]:
    h, w = shape
    image_new = (image * 255).clip(min=0, max=255).type(torch.uint8)
    image_new = rearrange(image_new, "c h w -> h w c").detach().cpu().numpy()
    image_new = Image.fromarray(image_new)
    image_new = image_new.resize((w, h), Image.LANCZOS)
    # image_new = cv2.resize(image_new, (w, h), interpolation=cv2.INTER_LANCZOS4)
    image_new = np.array(image_new) / 255
    image_new = torch.tensor(image_new, dtype=image.dtype, device=image.device)
    image_new = rearrange(image_new, "h w c -> c h w")

    mask_new = (mask * 255).clip(min=0, max=255).type(torch.uint8)
    mask_new = mask_new.detach().cpu().numpy()
    mask_new = Image.fromarray(mask_new)
    mask_new = mask_new.resize((w, h), Image.NEAREST)
    # mask_new = cv2.resize(mask_new, (w, h), interpolation=cv2.INTER_LINEAR)
    mask_new = np.array(mask_new) / 255
    mask_new = torch.tensor(mask_new, dtype=mask.dtype, device=mask.device)

    if lbs_weight is not None:
        if lbs_weight.shape[:2] == shape:
            lbs_weight_new = lbs_weight
        else:
            lbs_weight_new = F.interpolate(lbs_weight[None].permute(0, 3, 1, 2), (h, w), mode='bilinear', antialias=True)
            lbs_weight_new = lbs_weight_new.permute(0, 2, 3, 1)[0]
    else:
        lbs_weight_new = None
    return image_new, mask_new, lbs_weight_new


def center_pad(
    images: Float[Tensor, "*#batch c h w"],
    masks: Float[Tensor, "*#batch h w"],
    lbs_weights: Float[Tensor, "*#batch h w d"] | None,
    intrinsics: Float[Tensor, "*#batch 3 3"],
    shape: tuple[int, int],
    bgcolor: Float[Tensor, "*#batch 3"] | None = None,
) -> tuple[
    Float[Tensor, "*#batch c h_out w_out"],  # updated images
    Float[Tensor, "*#batch h_out w_out"],  # updated masks
    Float[Tensor, "*#batch h_out w_out d"] | None,  # updated lbs_weights
    Float[Tensor, "*#batch 3 3"],  # updated intrinsics
]:
    *_, h_in, w_in = images.shape
    h_out, w_out = shape

    # Note that odd input dimensions induce half-pixel misalignments.
    row = (h_out - h_in) // 2
    col = (w_out - w_in) // 2

    # Center-crop the image.
    if bgcolor is not None:
        images_pad = repeat(bgcolor, "... -> n ... h_out w_out", n=images.shape[0], h_out=h_out, w_out=w_out).clone()
        images_pad[..., row:row + h_in, col:col + w_in] = images
        images = images_pad
    else:
        images = F.pad(images, (col, w_out - w_in - col, row, h_out - h_in - row), "replicate")
    masks = F.pad(masks, (col, w_out - w_in - col, row, h_out - h_in - row))
    if lbs_weights is not None:
        if lbs_weights.shape[:2] != shape:
            lbs_weights = F.pad(lbs_weights, (0, 0, col, w_out - w_in - col, row, h_out - h_in - row))

    # Adjust the intrinsics to account for the cropping.
    intrinsics = intrinsics.clone()
    intrinsics[..., 0, 0] *= w_in / w_out  # fx
    intrinsics[..., 1, 1] *= h_in / h_out  # fy
    intrinsics[..., 0, 2] = intrinsics[..., 0, 2] * w_in / w_out + col / w_out
    intrinsics[..., 1, 2] = intrinsics[..., 1, 2] * h_in / h_out + row / h_out

    # return images, masks, lbs_weights, intrinsics
    return images, masks, lbs_weights, intrinsics


def center_crop(
    images: Float[Tensor, "*#batch c h w"],
    masks: Float[Tensor, "*#batch h w"],
    lbs_weights: Float[Tensor, "*#batch h w d"] | None,
    intrinsics: Float[Tensor, "*#batch 3 3"],
    shape: tuple[int, int],
) -> tuple[
    Float[Tensor, "*#batch c h_out w_out"],  # updated images
    Float[Tensor, "*#batch h_out w_out"],  # updated masks
    Float[Tensor, "*#batch h_out w_out d"] | None,  # updated lbs_weights
    Float[Tensor, "*#batch 3 3"],  # updated intrinsics
]:
    *_, h_in, w_in = images.shape
    h_out, w_out = shape

    # Note that odd input dimensions induce half-pixel misalignments.
    row = (h_in - h_out) // 2
    col = (w_in - w_out) // 2

    # Center-crop the image.
    images = images[..., :, row : row + h_out, col : col + w_out]
    masks = masks[..., :, row : row + h_out, col : col + w_out]
    if lbs_weights is not None:
        if lbs_weights.shape[:2] != shape:
            lbs_weights = lbs_weights[..., row : row + h_out, col : col + w_out, :]

    # Adjust the intrinsics to account for the cropping.
    intrinsics = intrinsics.clone()
    intrinsics[..., 0, 0] *= w_in / w_out  # fx
    intrinsics[..., 1, 1] *= h_in / h_out  # fy
    intrinsics[..., 0, 2] = intrinsics[..., 0, 2] * w_in / w_out - col / w_out
    intrinsics[..., 1, 2] = intrinsics[..., 1, 2] * h_in / h_out - row / h_out

    # return images, masks, lbs_weights, intrinsics
    return images, masks, lbs_weights, intrinsics


def get_rescale_and_crop_transform(
    h_in: int, w_in: int, h_out: int, w_out: int, pad: bool = False
) -> tuple[float, float, float, float]:
    """
    Return (scale_x, scale_y, offset_x, offset_y) to transform pixel coords from
    input image (h_in, w_in) to output image (h_out, w_out), matching rescale_and_crop.
    Transform: x_out = x_in * scale_x + offset_x, y_out = y_in * scale_y + offset_y.
    """
    if h_out <= h_in and w_out <= w_in:
        if pad:
            scale_factor = min(h_out / h_in, w_out / w_in)
        else:
            scale_factor = max(h_out / h_in, w_out / w_in)
        h_scaled = round(h_in * scale_factor)
        w_scaled = round(w_in * scale_factor)
        scale_x = w_scaled / w_in
        scale_y = h_scaled / h_in
        if pad:
            col = (w_out - w_scaled) // 2
            row = (h_out - h_scaled) // 2
            return scale_x, scale_y, float(col), float(row)
        else:
            col = (w_scaled - w_out) // 2
            row = (h_scaled - h_out) // 2
            return scale_x, scale_y, float(-col), float(-row)
    else:
        # No scaling; only pad (or crop if output smaller in one dim – rare)
        scale_x, scale_y = 1.0, 1.0
        col = (w_out - w_in) // 2
        row = (h_out - h_in) // 2
        return scale_x, scale_y, float(col), float(row)


def rescale_and_crop(
    images: Float[Tensor, "*#batch c h w"],
    masks: Float[Tensor, "*#batch h w"],
    lbs_weights: Float[Tensor, "*#batch h w d"] | None,
    intrinsics: Float[Tensor, "*#batch 3 3"],
    shape: tuple[int, int],
    pad: bool = False,
    bgcolor: Float[Tensor, "*#batch 3"] | None = None,
) -> tuple[
    Float[Tensor, "*#batch c h_out w_out"],  # updated images
    Float[Tensor, "*#batch h_out w_out"],  # updated masks
    Float[Tensor, "*#batch h w d"] | None, # updated lbs_weights
    Float[Tensor, "*#batch 3 3"],  # updated intrinsics
]:
    *_, h_in, w_in = images.shape
    h_out, w_out = shape
    if h_out <= h_in and w_out <= w_in:
        assert h_out <= h_in and w_out <= w_in

        if pad:
            scale_factor = min(h_out / h_in, w_out / w_in)
        else:
            scale_factor = max(h_out / h_in, w_out / w_in)
        h_scaled = round(h_in * scale_factor)
        w_scaled = round(w_in * scale_factor)
        assert h_scaled == h_out or w_scaled == w_out

        # Reshape the images to the correct size. Assume we don't have to worry about
        # changing the intrinsics based on how the images are rounded.
        *batch, c, h, w = images.shape
        images = images.reshape(-1, c, h, w)
        masks = masks.reshape(-1, h, w)
        if lbs_weights is not None:
            d = lbs_weights.shape[-1]
            lbs_weights = lbs_weights.reshape(-1, h, w, d)
        else:
            lbs_weights = [None] * images.shape[0]
        images_new, masks_new, lbs_weights_new = [], [], []
        for (image, mask, lbs_weight) in zip(images, masks, lbs_weights):
            image_new, mask_new, lbs_weight_new = rescale(image, mask, lbs_weight, (h_scaled, w_scaled))
            images_new.append(image_new)
            masks_new.append(mask_new)
            lbs_weights_new.append(lbs_weight_new)
        images = torch.stack(images_new)
        images = images.reshape(*batch, c, h_scaled, w_scaled)
        masks = torch.stack(masks_new)
        masks = masks.reshape(*batch, h_scaled, w_scaled)
        if lbs_weights_new[0] is not None:
            lbs_weights = torch.stack(lbs_weights_new)
            lbs_weights = lbs_weights.reshape(*batch, h_scaled, w_scaled, d)
        else:
            lbs_weights = None

    if pad:
        return center_pad(images, masks, lbs_weights, intrinsics, shape, bgcolor)
    else:
        return center_crop(images, masks, lbs_weights, intrinsics, shape)

def apply_bbox_crop_to_views(
    views: AnyViews,
    crop_params_list: list[tuple[int, int, int, int]],
    target_shape: tuple[int, int],
) -> AnyViews:
    """
    Crop each view using a pre-specified (top, left, crop_h, crop_w) bbox and resize to
    target_shape.  All spatial tensors (image, mask, image_gt, lbs_weights, uv_map,
    uv_valid) are cropped in the same way.  Intrinsics and face_bbox are updated
    analytically.

    Args:
        views: dict of view tensors, images expected as (V, C, H, W).
        crop_params_list: list of (top, left, ch, cw) per view, len == V.
        target_shape: (h_out, w_out) to resize each crop to.
    """
    h_out, w_out = target_shape
    images = views["image"]        # (V, C, H, W)
    masks = views["mask"]          # (V, H, W)
    V, C, H, W = images.shape

    assert len(crop_params_list) == V, (
        f"crop_params_list has {len(crop_params_list)} entries but there are {V} views"
    )

    def _crop_resize_chw(t_chw: torch.Tensor, top: int, left: int, ch: int, cw: int) -> torch.Tensor:
        """Crop a (C, H, W) tensor and resize to (h_out, w_out)."""
        t = t_chw[:, top:top + ch, left:left + cw].unsqueeze(0).float()  # (1, C, ch, cw)
        return F.interpolate(t, (h_out, w_out), mode="bilinear", align_corners=False).squeeze(0)

    def _crop_resize_hw(t_hw: torch.Tensor, top: int, left: int, ch: int, cw: int) -> torch.Tensor:
        """Crop a (H, W) tensor and resize to (h_out, w_out) with nearest-neighbour."""
        t = t_hw[top:top + ch, left:left + cw].unsqueeze(0).unsqueeze(0).float()
        return F.interpolate(t, (h_out, w_out), mode="nearest").squeeze(0).squeeze(0)

    imgs_out, masks_out = [], []
    for i, (top, left, ch, cw) in enumerate(crop_params_list):
        imgs_out.append(_crop_resize_chw(images[i], top, left, ch, cw))
        masks_out.append(_crop_resize_hw(masks[i], top, left, ch, cw))

    new_views: AnyViews = {
        **views,
        "image": torch.stack(imgs_out),
        "mask": torch.stack(masks_out),
    }

    # image_gt (same crop as image)
    if "image_gt" in views:
        imgs_gt_out = []
        for i, (top, left, ch, cw) in enumerate(crop_params_list):
            imgs_gt_out.append(_crop_resize_chw(views["image_gt"][i], top, left, ch, cw))
        new_views["image_gt"] = torch.stack(imgs_gt_out)

    # lbs_weights: (V, H, W, D)
    if "lbs_weights" in views and views["lbs_weights"] is not None:
        lbs = views["lbs_weights"]  # (V, H, W, D)
        D = lbs.shape[-1]
        lbs_out = []
        for i, (top, left, ch, cw) in enumerate(crop_params_list):
            t = lbs[i, top:top + ch, left:left + cw, :].permute(2, 0, 1).unsqueeze(0).float()
            t = F.interpolate(t, (h_out, w_out), mode="bilinear", align_corners=False)
            lbs_out.append(t.squeeze(0).permute(1, 2, 0))  # (h_out, w_out, D)
        new_views["lbs_weights"] = torch.stack(lbs_out)

    # uv_map: (V, H, W, 2) / uv_valid: (V, H, W)
    if "uv_map" in views and "uv_valid" in views:
        uv_map_out, uv_valid_out = [], []
        uv_map = views["uv_map"]
        uv_valid = views["uv_valid"]
        for i, (top, left, ch, cw) in enumerate(crop_params_list):
            um = uv_map[i, top:top + ch, left:left + cw, :].permute(2, 0, 1).unsqueeze(0).float()
            um = F.interpolate(um, (h_out, w_out), mode="bilinear", align_corners=False)
            uv_map_out.append(um.squeeze(0).permute(1, 2, 0))

            uv_v = uv_valid[i, top:top + ch, left:left + cw].unsqueeze(0).unsqueeze(0).float()
            uv_v = F.interpolate(uv_v, (h_out, w_out), mode="nearest").squeeze(0).squeeze(0).bool()
            uv_valid_out.append(uv_v)
        new_views["uv_map"] = torch.stack(uv_map_out)
        new_views["uv_valid"] = torch.stack(uv_valid_out)

    # Intrinsics (normalized): fx_new = fx * W / cw, cx_new = (cx * W - left) / cw
    intrinsics = views["intrinsics"].clone()  # (V, 3, 3)
    orig_K = views["intrinsics"]
    for i, (top, left, ch, cw) in enumerate(crop_params_list):
        # Safety floor for ch/cw to avoid zero division.
        cw = max(cw, 1)
        ch = max(ch, 1)
        intrinsics[i, 0, 0] = orig_K[i, 0, 0] * W / cw
        intrinsics[i, 1, 1] = orig_K[i, 1, 1] * H / ch
        intrinsics[i, 0, 2] = (orig_K[i, 0, 2] * W - left) / cw
        intrinsics[i, 1, 2] = (orig_K[i, 1, 2] * H - top) / ch
    new_views["intrinsics"] = intrinsics

    # face_bbox: (V, 4) x1,y1,x2,y2 in original image pixel coords → output pixel coords
    if "face_bbox" in views and views["face_bbox"] is not None and len(views["face_bbox"]) > 0:
        bboxes = views["face_bbox"].float()  # (V, 4)
        new_bboxes = []
        for i, (top, left, ch, cw) in enumerate(crop_params_list):
            cw = max(cw, 1)
            ch = max(ch, 1)
            sx = w_out / cw
            sy = h_out / ch
            b = bboxes[i]
            new_bboxes.append(torch.stack([
                (b[0] - left) * sx,
                (b[1] - top) * sy,
                (b[2] - left) * sx,
                (b[3] - top) * sy,
            ]))
        new_views["face_bbox"] = torch.stack(new_bboxes)

    return new_views


def apply_crop_shim_to_views(views: AnyViews, shape: tuple[int, int], pad: bool = False, bgcolor: torch.Tensor | None = None) -> AnyViews:
    if "lbs_weights" in views and views["lbs_weights"].shape[-3:-1] != shape:
        lbs_weights = views["lbs_weights"]
    else:
        lbs_weights = None

    images, masks, lbs_weights, intrinsics = rescale_and_crop(
            views["image"], views["mask"], lbs_weights, views["intrinsics"], shape, pad, bgcolor)
        
    new_views = {
        **views,
        "image": images,
        "mask": masks,
        "intrinsics": intrinsics,
    }
    if lbs_weights is not None:
        new_views["lbs_weights"] = lbs_weights
    
    # Also resize image_gt if it exists (used for context loss)
    if "image_gt" in views:
        # Supervision-aligned mask (mask_gt) when present; else same as input mask
        gt_mask = views["mask_gt"] if "mask_gt" in views else views["mask"]
        image_gt, mask_gt_out, _, _ = rescale_and_crop(
            views["image_gt"], gt_mask, None, views["intrinsics"], shape, pad, bgcolor)
        new_views["image_gt"] = image_gt
        if "mask_gt" in views:
            new_views["mask_gt"] = mask_gt_out

    # Transform face_bbox from input image coords to resized/cropped image coords
    if "face_bbox" in views and views["face_bbox"] is not None and len(views["face_bbox"]) > 0:
        *_, h_in, w_in = views["image"].shape
        h_out, w_out = shape
        scale_x, scale_y, offset_x, offset_y = get_rescale_and_crop_transform(
            int(h_in), int(w_in), int(h_out), int(w_out), pad
        )
        bbox = views["face_bbox"].float()  # (..., 4) x1, y1, x2, y2
        device = bbox.device
        scale_x = torch.tensor(scale_x, device=device, dtype=bbox.dtype)
        scale_y = torch.tensor(scale_y, device=device, dtype=bbox.dtype)
        offset_x = torch.tensor(offset_x, device=device, dtype=bbox.dtype)
        offset_y = torch.tensor(offset_y, device=device, dtype=bbox.dtype)
        # x1,y1,x2,y2 -> scale and shift
        new_views["face_bbox"] = torch.stack([
            bbox[..., 0] * scale_x + offset_x,
            bbox[..., 1] * scale_y + offset_y,
            bbox[..., 2] * scale_x + offset_x,
            bbox[..., 3] * scale_y + offset_y,
        ], dim=-1)

    # Resize/crop uv_map and uv_valid to match image shape (same transform as image).
    if "uv_map" in views and "uv_valid" in views:
        uv_map = views["uv_map"]
        uv_valid = views["uv_valid"]
        *_, h_in, w_in = views["image"].shape
        h_out, w_out = shape
        if h_out <= h_in and w_out <= w_in:
            scale_factor = min(h_out / h_in, w_out / w_in) if pad else max(h_out / h_in, w_out / w_in)
            h_scaled = round(h_in * scale_factor)
            w_scaled = round(w_in * scale_factor)
            # uv_map (V, H, W, 2), uv_valid (V, H, W)
            uv_map = F.interpolate(
                uv_map.permute(0, 3, 1, 2), (h_scaled, w_scaled), mode="bilinear", align_corners=False
            ).permute(0, 2, 3, 1)
            uv_valid = F.interpolate(
                uv_valid.unsqueeze(1).float(), (h_scaled, w_scaled), mode="nearest"
            ).squeeze(1).bool()
            row = (h_scaled - h_out) // 2
            col = (w_scaled - w_out) // 2
            uv_map = uv_map[:, row : row + h_out, col : col + w_out, :]
            uv_valid = uv_valid[:, row : row + h_out, col : col + w_out]
        else:
            row = (h_out - h_in) // 2
            col = (w_out - w_in) // 2
            uv_map = F.pad(
                uv_map, (0, 0, col, w_out - w_in - col, row, h_out - h_in - row), mode="constant", value=0.0
            )
            uv_valid = F.pad(
                uv_valid.float(), (col, w_out - w_in - col, row, h_out - h_in - row), mode="constant", value=0.0
            ).bool()
        new_views["uv_map"] = uv_map
        new_views["uv_valid"] = uv_valid

    return new_views


def apply_crop_shim(
    example: AnyExample,
    shape: tuple[int, int],
    pad: bool = False,
    context_bbox_crops: list[tuple[int, int, int, int]] | None = None,
) -> AnyExample:
    """Crop images in the example.

    When context_bbox_crops is provided (a list of (top, left, ch, cw) per context
    view), context views are cropped via apply_bbox_crop_to_views and then the result
    is returned at 'shape' resolution.  Target views always receive the standard
    center-rescale crop so their camera geometry is undisturbed.
    """
    if context_bbox_crops is not None:
        context_views = apply_bbox_crop_to_views(example["context"], context_bbox_crops, shape)
    else:
        context_views = apply_crop_shim_to_views(example["context"], shape, pad, example["bgcolor"])
    return {
        **example,
        "context": context_views,
        "target": apply_crop_shim_to_views(example["target"], shape, pad, example["bgcolor"]),
    }


def random_scale_and_crop(image: torch.Tensor, mask, lbs_weights, intrinsics, bgcolor, scale_range=(0.8, 1.2)) -> torch.Tensor:
    """
    Randomly scale the input image and crop/pad to maintain original size.

    Args:
        image: Input image tensor of shape [H, W, 3]
        scale_range: Range for scaling factor, default (0.8, 1.2)

    Returns:
        Scaled and cropped/padded image tensor of shape [H, W, 3]
    """
    is_numpy = False
    if not torch.is_tensor(image):
        image = torch.from_numpy(image)
        is_numpy = True
    # 获取图像的高度和宽度
    h, w = image.shape[1:]

    # 生成随机缩放因子
    scale_factor = random.uniform(*scale_range)

    # 计算新的高度和宽度
    new_h = int(h * scale_factor)
    new_w = int(w * scale_factor)

    # 使用 torchvision.transforms.functional.resize 进行缩放
    scaled_image = tvf.resize(image, [new_h, new_w])
    scaled_mask = tvf.resize(mask, [new_h, new_w], interpolation=InterpolationMode.NEAREST)
    scaled_lbs_weights = tvf.resize(lbs_weights, [new_h, new_w])
    # intrinsics[..., 0, 0] *= new_w / w
    # intrinsics[..., 1, 1] *= new_h / h

    # 如果缩放后的图像比原图大，进行居中裁剪
    if new_h > h or new_w > w:
        top = (new_h - h) // 2
        left = (new_w - w) // 2
        top = random.randint(0, new_h - h)
        left = random.randint(0, new_w - w)
        scaled_image = scaled_image[:, top:top + h, left:left + w]
        scaled_mask = scaled_mask[:, top:top + h, left:left + w]
        scaled_lbs_weights = scaled_lbs_weights[:, top:top + h, left:left + w]
        intrinsics[..., 0, 0] *= new_w / w
        intrinsics[..., 1, 1] *= new_h / h
        intrinsics[..., 0, 2] = (intrinsics[..., 0, 2] * new_w - left) / w
        intrinsics[..., 1, 2] = (intrinsics[..., 1, 2] * new_h - top) / h
    else:
        # 如果缩放后的图像比原图小，进行居中填充
        # padded_image = torch.ones((3, h, w), dtype=image.dtype)
        padded_image = bgcolor.unsqueeze(-1).unsqueeze(-1).repeat(1, h, w)
        padded_mask = torch.zeros(1, h, w, dtype=mask.dtype)
        padded_lbs_weights = torch.zeros(55, h, w, dtype=scaled_lbs_weights.dtype)
        # print(padded_image.shape, scaled_image.shape)
        top = h-new_h #(h - new_h) // 2 # H不应该居中
        left = (w - new_w) // 2
        top = random.randint(0, h - new_h)
        left = random.randint(0, w - new_w)
        padded_image[:, top:top + new_h, left:left + new_w] = scaled_image
        scaled_image = padded_image
        padded_mask[:, top:top + new_h, left:left + new_w] = scaled_mask
        scaled_mask = padded_mask
        padded_lbs_weights[:, top:top + new_h, left:left + new_w] = scaled_lbs_weights
        scaled_lbs_weights = padded_lbs_weights
        intrinsics[..., 0, 0] *= new_w / w
        intrinsics[..., 1, 1] *= new_h / h
        intrinsics[..., 0, 2] = (intrinsics[..., 0, 2] * new_w + left) / w
        intrinsics[..., 1, 2] = (intrinsics[..., 1, 2] * new_h + top) / h
    if is_numpy:
        scaled_image = scaled_image.numpy()
    return scaled_image, scaled_mask, scaled_lbs_weights, intrinsics


def apply_crop_shim2(example: AnyExample, shape: tuple[int, int], pad: bool = False) -> AnyExample:
    """This augmentation borrowed from IDOL"""
    cond_imgs = example["context"]["image"]
    cond_masks = example["context"]["mask"][None]
    cond_lbs_weights = example["context"]["lbs_weights"].permute(0, 3, 1, 2)
    cond_imgs[0], cond_masks[0], cond_lbs_weights[0], example["context"]["intrinsics"] = random_scale_and_crop(
        cond_imgs[0], cond_masks[0], cond_lbs_weights[0], example["context"]["intrinsics"], example["bgcolor"],
        (0.7, 1.1))

    example["context"]["image"] = cond_imgs
    example["context"]["mask"] = cond_masks.squeeze(1)
    example["context"]["lbs_weights"] = cond_lbs_weights.permute(0, 2, 3, 1)

    return apply_crop_shim(example, shape, pad)
