# Copied and adapted from LHM/models/ESRGANer_utils.py for use in NoPo-Avatar (no LHM dependency).
# Requires: pip install basicsr (and optionally gfpgan for face_enhance=True).
# Place RealESRGAN weights at repo_root/pretrained_models/RealESRGAN_x4plus.pth or pass model_path.

import math
import os

import cv2
import numpy as np
import torch
from torch.nn import functional as F

# NoPo-Avatar repo root: encoder -> model -> src -> repo root
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PRETRAINED_DIR = os.path.normpath(os.path.join(_THIS_DIR, "..", "..", "..", "pretrained_models"))

try:
    from basicsr.utils.download_util import load_file_from_url
    from basicsr.archs.rrdbnet_arch import RRDBNet
except ImportError as e:
    raise ImportError(
        "esrgan_utils requires basicsr. Install with: pip install basicsr"
    ) from e


def _device():
    if torch.cuda.is_available():
        return torch.device(f"cuda:{torch.cuda.current_device()}")
    return torch.device("cpu")


class RealESRGANer:
    """Upsampling with RealESRGAN. Expects model_path as local path or URL."""

    def __init__(
        self,
        scale,
        model_path,
        dni_weight=None,
        model=None,
        tile=0,
        tile_pad=10,
        pre_pad=10,
        half=False,
        device=None,
        gpu_id=None,
    ):
        self.scale = scale
        self.tile_size = tile
        self.tile_pad = tile_pad
        self.pre_pad = pre_pad
        self.mod_scale = None
        self.half = half

        if gpu_id is not None:
            self.device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu") if device is None else device
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device is None else device

        if isinstance(model_path, list):
            assert dni_weight is not None and len(model_path) == len(dni_weight)
            loadnet = self._dni(model_path[0], model_path[1], dni_weight)
        else:
            if model_path.startswith("https://"):
                model_path = load_file_from_url(
                    url=model_path,
                    model_dir=PRETRAINED_DIR,
                    progress=True,
                    file_name=None,
                )
            loadnet = torch.load(model_path, map_location=torch.device("cpu"))

        keyname = "params_ema" if "params_ema" in loadnet else "params"
        model.load_state_dict(loadnet[keyname], strict=True)
        model.eval()
        self.model = model.to(self.device)
        if self.half:
            self.model = self.model.half()

    def _dni(self, net_a, net_b, dni_weight, key="params", loc="cpu"):
        net_a = torch.load(net_a, map_location=torch.device(loc))
        net_b = torch.load(net_b, map_location=torch.device(loc))
        for k, v_a in net_a[key].items():
            net_a[key][k] = dni_weight[0] * v_a + dni_weight[1] * net_b[key][k]
        return net_a

    def pre_process(self, img):
        img = torch.from_numpy(np.transpose(img, (2, 0, 1))).float()
        self.img = img.unsqueeze(0).to(self.device)
        if self.half:
            self.img = self.img.half()
        if self.pre_pad != 0:
            self.img = F.pad(self.img, (0, self.pre_pad, 0, self.pre_pad), "reflect")
        if self.scale == 2:
            self.mod_scale = 2
        elif self.scale == 1:
            self.mod_scale = 4
        if self.mod_scale is not None:
            self.mod_pad_h, self.mod_pad_w = 0, 0
            _, _, h, w = self.img.size()
            if h % self.mod_scale != 0:
                self.mod_pad_h = self.mod_scale - h % self.mod_scale
            if w % self.mod_scale != 0:
                self.mod_pad_w = self.mod_scale - w % self.mod_scale
            self.img = F.pad(self.img, (0, self.mod_pad_w, 0, self.mod_pad_h), "reflect")

    def process(self):
        self.output = self.model(self.img)

    def tile_process(self):
        batch, channel, height, width = self.img.shape
        output_height = height * self.scale
        output_width = width * self.scale
        self.output = self.img.new_zeros((batch, channel, output_height, output_width))
        tiles_x = math.ceil(width / self.tile_size)
        tiles_y = math.ceil(height / self.tile_size)
        for y in range(tiles_y):
            for x in range(tiles_x):
                ofs_x, ofs_y = x * self.tile_size, y * self.tile_size
                input_end_x = min(ofs_x + self.tile_size, width)
                input_end_y = min(ofs_y + self.tile_size, height)
                input_start_x_pad = max(ofs_x - self.tile_pad, 0)
                input_end_x_pad = min(input_end_x + self.tile_pad, width)
                input_start_y_pad = max(ofs_y - self.tile_pad, 0)
                input_end_y_pad = min(input_end_y + self.tile_pad, height)
                input_tile_width = input_end_x - ofs_x
                input_tile_height = input_end_y - ofs_y
                input_tile = self.img[:, :, input_start_y_pad:input_end_y_pad, input_start_x_pad:input_end_x_pad]
                with torch.no_grad():
                    output_tile = self.model(input_tile)
                output_start_x = ofs_x * self.scale
                output_end_x = input_end_x * self.scale
                output_start_y = ofs_y * self.scale
                output_end_y = input_end_y * self.scale
                output_start_x_tile = (ofs_x - input_start_x_pad) * self.scale
                output_end_x_tile = output_start_x_tile + input_tile_width * self.scale
                output_start_y_tile = (ofs_y - input_start_y_pad) * self.scale
                output_end_y_tile = output_start_y_tile + input_tile_height * self.scale
                self.output[:, :, output_start_y:output_end_y, output_start_x:output_end_x] = output_tile[
                    :, :, output_start_y_tile:output_end_y_tile, output_start_x_tile:output_end_x_tile
                ]

    def post_process(self):
        if self.mod_scale is not None:
            _, _, h, w = self.output.size()
            self.output = self.output[:, :, 0 : h - self.mod_pad_h * self.scale, 0 : w - self.mod_pad_w * self.scale]
        if self.pre_pad != 0:
            _, _, h, w = self.output.size()
            self.output = self.output[:, :, 0 : h - self.pre_pad * self.scale, 0 : w - self.pre_pad * self.scale]
        return self.output

    @torch.no_grad()
    def enhance(self, img, outscale=None, alpha_upsampler="realesrgan"):
        h_input, w_input = img.shape[0:2]
        img = img.astype(np.float32)
        max_range = 65535 if np.max(img) > 256 else 255
        img = img / max_range
        if len(img.shape) == 2:
            img_mode = "L"
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.shape[2] == 4:
            img_mode = "RGBA"
            alpha = img[:, :, 3]
            img = img[:, :, 0:3]
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if alpha_upsampler == "realesrgan":
                alpha = cv2.cvtColor(alpha, cv2.COLOR_GRAY2RGB)
        else:
            img_mode = "RGB"
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        self.pre_process(img)
        if self.tile_size > 0:
            self.tile_process()
        else:
            self.process()
        output_img = self.post_process()
        output_img = output_img.data.squeeze().float().cpu().clamp_(0, 1).numpy()
        output_img = np.transpose(output_img[[2, 1, 0], :, :], (1, 2, 0))
        if img_mode == "L":
            output_img = cv2.cvtColor(output_img, cv2.COLOR_BGR2GRAY)

        if img_mode == "RGBA":
            if alpha_upsampler == "realesrgan":
                self.pre_process(alpha)
                if self.tile_size > 0:
                    self.tile_process()
                else:
                    self.process()
                output_alpha = self.post_process()
                output_alpha = output_alpha.data.squeeze().float().cpu().clamp_(0, 1).numpy()
                output_alpha = np.transpose(output_alpha[[2, 1, 0], :, :], (1, 2, 0))
                output_alpha = cv2.cvtColor(output_alpha, cv2.COLOR_BGR2GRAY)
            else:
                h, w = alpha.shape[0:2]
                output_alpha = cv2.resize(alpha, (w * self.scale, h * self.scale), interpolation=cv2.INTER_LINEAR)
            output_img = cv2.cvtColor(output_img, cv2.COLOR_BGR2BGRA)
            output_img[:, :, 3] = output_alpha

        output = (output_img * 65535.0).round().astype(np.uint16) if max_range == 65535 else (output_img * 255.0).round().astype(np.uint8)
        if outscale is not None and outscale != float(self.scale):
            output = cv2.resize(output, (int(w_input * outscale), int(h_input * outscale)), interpolation=cv2.INTER_LANCZOS4)
        return output, img_mode


# Default RealESRGAN x4 URL (used if model_path missing or file not found)
_REALESRGAN_X4_URL = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"


class ESRGANEasyModel:
    """Face/image super-resolution with RealESRGAN (and optional GFPGAN). Call with BGR numpy image (H, W, 3)."""

    def __init__(
        self,
        model_path: str = "pretrained_models/RealESRGAN_x4plus.pth",
        face_enhance: bool = True,
    ):
        model = RRDBNet(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=64,
            num_block=23,
            num_grow_ch=32,
            scale=4,
        )
        self.net_scale = 4
        self.face_enhance = face_enhance

        # Resolve path: if not absolute and not found, try repo pretrained_models, then download
        if model_path and not os.path.isabs(model_path) and not os.path.isfile(model_path):
            candidate = os.path.join(PRETRAINED_DIR, os.path.basename(model_path))
            if os.path.isfile(candidate):
                model_path = candidate
            else:
                model_path = load_file_from_url(
                    url=_REALESRGAN_X4_URL,
                    model_dir=PRETRAINED_DIR,
                    progress=True,
                    file_name=os.path.basename(model_path) or "RealESRGAN_x4plus.pth",
                )
        elif not model_path or not os.path.isfile(model_path):
            model_path = load_file_from_url(
                url=_REALESRGAN_X4_URL,
                model_dir=PRETRAINED_DIR,
                progress=True,
                file_name="RealESRGAN_x4plus.pth",
            )

        self.upsampler = RealESRGANer(
            scale=self.net_scale,
            model_path=model_path,
            dni_weight=None,
            model=model,
            tile=0,
            tile_pad=10,
            pre_pad=0,
            half=False,
        )
        self.upsampler.model.to(_device())

        if face_enhance:
            try:
                from gfpgan import GFPGANer
                self.face_enhancer = GFPGANer(
                    model_path="https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth",
                    upscale=4,
                    arch="clean",
                    channel_multiplier=2,
                    bg_upsampler=self.upsampler,
                )
            except Exception:
                self.face_enhancer = None
        else:
            self.face_enhancer = None

    @torch.no_grad()
    def __call__(self, img):
        """img: BGR numpy (H, W, 3). Returns BGR numpy upscaled (e.g. 4x)."""
        if self.face_enhancer is not None:
            _, _, output = self.face_enhancer.enhance(
                img, has_aligned=False, only_center_face=False, paste_back=True
            )
        else:
            output, _ = self.upsampler.enhance(img, outscale=4)
        return output

    def __repr__(self):
        return f"ESRGANEasyModel(face_enhance={self.face_enhance})\n{self.upsampler}"
