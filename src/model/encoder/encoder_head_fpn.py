"""
Head FPN encoder: DINOv2 feature pyramid on a face crop -> token sequence.

- Input: head crop (B, 3, H, W) RGB in [0, 1]. Non-square is fine: we pad to square then resize.
- Optional ESRGAN: set use_esrgan=True and add esrgan_utils.py to this package (copy from LHM's
  ESRGANer_utils.py + RealESRGANer / RRDBNet and basicsr deps). Provides RealESRGAN 4x face upscaling.
"""
import os
import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Literal, Optional
import kornia
import torch.nn.functional as F
import numpy as np

@dataclass
class EncoderHeadDINOv2FPNCfg:
    name: Literal["dino_v2_fpn"]
    dino_model_name: Literal["dinov2_vitl14_reg"]
    out_dim: int
    resolution: int  # must be divisible by DINOv2 patch_size (14)
    freeze: bool = True
    # Optional face super-resolution (RealESRGAN). Requires esrgan_utils in this package if True.
    use_esrgan: bool = False
    esrgan_model_path: Optional[str] = None  # e.g. "pretrained_models/RealESRGAN_x4plus.pth"


class DPTHead(nn.Module):
    def __init__(
        self,
        in_channels,
        inner_channels,
        use_clstoken=False,
        out_channel=1024,
    ):
        super(DPTHead, self).__init__()

        self.use_clstoken = use_clstoken
        self.projects = nn.ModuleList(
            [
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channel,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )
                for out_channel in inner_channels
            ]
        )

        if use_clstoken:
            self.readout_projects = nn.ModuleList()
            for _ in range(len(self.projects)):
                self.readout_projects.append(
                    nn.Sequential(nn.Linear(2 * in_channels, in_channels), nn.GELU())
                )

        self.output_conv = nn.Conv2d(
            sum(inner_channels), out_channel, kernel_size=1, stride=1, padding=0
        )

    def forward(self, out_features, patch_h, patch_w):
        out = []
        for i, x in enumerate(out_features):
            if self.use_clstoken:
                x, cls_token = x[0], x[1]
                readout = cls_token.unsqueeze(1).expand_as(x)
                x = self.readout_projects[i](torch.cat((x, readout), -1))
            else:
                x = x[0]

            x = x.permute(0, 2, 1).reshape((x.shape[0], x.shape[-1], patch_h, patch_w))
            x = self.projects[i](x)
            out.append(x)

        fusion_feats = torch.cat(out, dim=1) # these are each (B,1024,16,16)
        fusion_feats = self.output_conv(fusion_feats)

        return fusion_feats

# references: LHM/models/encoders/dinov2_fusion_wrapper.py
class EncoderHeadDINOv2FPN(nn.Module):
    def __init__(self, cfg: EncoderHeadDINOv2FPNCfg):
        super().__init__()
        self.cfg = cfg
        self.dino_model = torch.hub.load("facebookresearch/dinov2", cfg.dino_model_name, pretrained=True)
        if cfg.freeze:
            self._freeze()

        self.intermediate_layer_idx_info = {
            "dinov2_vits14_reg": [2, 5, 8, 11],
            "dinov2_vitb14_reg": [2, 5, 8, 11],
            "dinov2_vitl14_reg": [4, 11, 17, 23],
            "dinov2_vitg14_reg": [9, 19, 29, 39],
        }
        self.intermediate_layer_idx = self.intermediate_layer_idx_info[cfg.dino_model_name]

        embed_dim = self.dino_model.embed_dim
        self.fusion_head = DPTHead(
            in_channels=embed_dim,
            inner_channels=[embed_dim] * 4,
            out_channel=cfg.out_dim,
        )

        # Optional face super-resolution (RealESRGAN). Uses local .esrgan_utils if use_esrgan.
        self._face_sr = None
        if getattr(cfg, "use_esrgan", False):
            self._face_sr = self._make_face_sr(getattr(cfg, "esrgan_model_path", None))
            if self._face_sr is None:
                print(
                    "EncoderHeadDINOv2FPN: use_esrgan=True but ESRGAN not available. "
                    "Add esrgan_utils.py to this package (copy from LHM/models/ESRGANer_utils.py and install basicsr). Disabling."
                )

    def _freeze(self):
        print("======== Freezing EncoderHeadDINOv2FPN ========")
        self.dino_model.eval()
        for param in self.dino_model.parameters():
            param.requires_grad = False

    def _make_face_sr(self, model_path: Optional[str]):
        """Build face SR model from local esrgan_utils (works when run as package or as script)."""
        try:
            try:
                from .esrgan_utils import ESRGANEasyModel
            except ImportError:
                # Run as script (e.g. python encoder_head_fpn.py): no parent package
                import importlib.util
                _path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "esrgan_utils.py")
                spec = importlib.util.spec_from_file_location("esrgan_utils", _path)
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                ESRGANEasyModel = mod.ESRGANEasyModel
            return ESRGANEasyModel(
                model_path=model_path or "pretrained_models/RealESRGAN_x4plus.pth",
                face_enhance=False,
            )
        except Exception as e:
            print(f"EncoderHeadDINOv2FPN: could not load ESRGAN: {e}")
            return None

    def _maybe_face_sr(self, x: torch.Tensor) -> torch.Tensor:
        """Optional face super-resolution. x: (B,3,H,W) RGB [0,1]. Returns tensor (e.g. 4x upscaled) same format."""
        if self._face_sr is None:
            return x
        device = x.device
        dtype = x.dtype
        with torch.no_grad():
            x_np = x.permute(0, 2, 3, 1).detach().cpu().numpy()
            x_np = (x_np * 255).astype(np.uint8)[..., ::-1]  # RGB -> BGR
            out_list = []
            for i in range(x_np.shape[0]):
                out_list.append(self._face_sr(x_np[i]))
            out_np = np.stack(out_list, axis=0).astype(np.float32) / 255.0
        out_np = out_np[..., ::-1]  # BGR -> RGB
        return torch.from_numpy(out_np).permute(0, 3, 1, 2).to(device=device, dtype=dtype)

    def _preprocess_image(
        self, image: torch.Tensor, antialias: bool = True
    ) -> torch.Tensor:
        # Non-square crops are supported: we pad to square then resize (no aspect-ratio distortion of content).
        resolution = self.cfg.resolution  # must be divisible by patch_size (14)
        _, __, H, W = image.shape
        max_size = max(H, W)
        H_pad = max_size - H
        W_pad = max_size - W
        pad_size = (
            W_pad // 2,
            max_size - (W + W_pad // 2),
            H_pad // 2,
            max_size - (H + H_pad // 2),
            0,
            0,
            0,
            0,
        )

        image = F.pad(image, pad_size, value=1)

        image = kornia.geometry.resize(
            image,
            (resolution, resolution),
            interpolation="bicubic",
            align_corners=True,
            antialias=antialias,
        )

        return image

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input: head crop (B,3,H,W) RGB [0,1]. Non-square is ok (padded to square then resized).
        # Output: (B, N_tokens, out_dim) where N_tokens = (resolution // patch_size)^2
        #   e.g. resolution=224, patch_size=14 -> 16*16 = 256 tokens; 256 is the spatial (patch) count.
        device = x.device
        if next(self.dino_model.parameters()).device != device:
            self.to(device)
        x = self._maybe_face_sr(x)
        image = self._preprocess_image(x)
        patch_size = getattr(self.dino_model, "patch_size", 14)
        patch_h = image.shape[-2] // patch_size
        patch_w = image.shape[-1] // patch_size
        features = self.dino_model.get_intermediate_layers(
            image, self.intermediate_layer_idx, return_class_token=True
        ) # returns tuple of 
        out_local = self.fusion_head(features, patch_h, patch_w)
        out_global = None
        if out_global is not None:
            ret = torch.cat(
                [out_local.permute(0, 2, 3, 1).flatten(1, 2), out_global.unsqueeze(1)],
                dim=1,
            )
        else:
            ret = out_local.permute(0, 2, 3, 1).flatten(1, 2)
        return ret


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = EncoderHeadDINOv2FPNCfg(
        name="dino_v2_fpn",
        dino_model_name="dinov2_vitl14_reg",
        out_dim=1024,
        resolution=448,
        freeze=True,
        use_esrgan=False,  # ! doesn't work yet; debug later if we decide to SR
    )
    model = EncoderHeadDINOv2FPN(cfg).to(device)
    x = torch.randn(1, 3, 512, 512, device=device) # output: (1, 256, 384)
    print(model(x).shape)