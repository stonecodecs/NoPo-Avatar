# Copyright (C) 2022-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).


# --------------------------------------------------------
# Position embedding utils
# --------------------------------------------------------



import numpy as np

import torch

# --------------------------------------------------------
# 2D sine-cosine position embedding
# References:
# MAE: https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py
# Transformer: https://github.com/tensorflow/models/blob/master/official/nlp/transformer/model_utils.py
# MoCo v3: https://github.com/facebookresearch/moco-v3
# --------------------------------------------------------
def get_2d_sincos_pos_embed(embed_dim, grid_size, n_cls_token=0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [n_cls_token+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if n_cls_token>0:
        pos_embed = np.concatenate([np.zeros([n_cls_token, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=float)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


# --------------------------------------------------------
# Interpolate position embeddings for high-resolution
# References:
# MAE: https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------
def interpolate_pos_embed(model, checkpoint_model):
    if 'pos_embed' in checkpoint_model:
        pos_embed_checkpoint = checkpoint_model['pos_embed']
        embedding_size = pos_embed_checkpoint.shape[-1]
        num_patches = model.patch_embed.num_patches
        num_extra_tokens = model.pos_embed.shape[-2] - num_patches
        # height (== width) for the checkpoint position embedding
        orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
        # height (== width) for the new position embedding
        new_size = int(num_patches ** 0.5)
        # class_token and dist_token are kept unchanged
        if orig_size != new_size:
            print("Position interpolate from %dx%d to %dx%d" % (orig_size, orig_size, new_size, new_size))
            extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
            # only the position tokens are interpolated
            pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
            pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
            pos_tokens = torch.nn.functional.interpolate(
                pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
            pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
            new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
            checkpoint_model['pos_embed'] = new_pos_embed


# -----------------------------------------------------------------------
# UV-space Integrated Positional Encoding (Mean-of-Embeddings / Mip-NeRF)
# -----------------------------------------------------------------------

def uv_sincos_embed(uv: torch.Tensor, embed_dim: int) -> torch.Tensor:
    """
    2D sin/cos positional embedding for continuous UV coordinates in [0, 1].

    embed_dim must be divisible by 4: embed_dim/4 frequencies per UV axis,
    each represented by both sin and cos (4 * D = embed_dim total).

    Designed for Mean-of-Embeddings (IPE) use: average pixel embeddings per
    patch *before* projecting so that high-frequency components attenuate
    naturally over spatially broad or seam-spanning patches.

    Args:
        uv:        (..., 2) float tensor, UV coordinates in [0, 1].
        embed_dim: int, output dimension; must be divisible by 4.
    Returns:
        (..., embed_dim) float tensor.
    """
    assert embed_dim % 4 == 0, f"uv_sincos_embed: embed_dim must be % 4, got {embed_dim}"
    D = embed_dim // 4
    # Same frequency schedule as the standard 1-D sincos PE (base = 10000).
    omega = 1.0 / (10000.0 ** (torch.arange(D, device=uv.device, dtype=uv.dtype) / D))  # (D,)
    u_ang = uv[..., 0:1] * omega  # (..., D)
    v_ang = uv[..., 1:2] * omega  # (..., D)
    return torch.cat([u_ang.sin(), u_ang.cos(), v_ang.sin(), v_ang.cos()], dim=-1)


class UVPatchPositionalEncoder(torch.nn.Module):
    """
    Per-patch UV positional embeddings via Mean-of-Embeddings (Integrated PE).

    For each image patch:
    * Every pixel whose SMPL-X UV is visible is embedded with sin/cos Fourier
      features via uv_sincos_embed (same dimension as tokens).
    * Those per-pixel embeddings are mean-pooled.  High-frequency components
      naturally attenuate over large or seam-spanning patches.
    * Fully-background patches (no visible surface) fall back to a learned
      null embedding that is distinct from any real UV point.

    Usage:
        enc = UVPatchPositionalEncoder(enc_embed_dim=1024)
        uv_pe = enc(uv_coords, uv_valid)   # (B, N_patches, 1024)
        x = x + uv_pe

        uv_pe_tmpl = enc.from_uv_grid(n_h, n_w, device)  # (1, N_patches, 1024)
        template = template + uv_pe_tmpl
    """

    def __init__(self, enc_embed_dim: int, patch_size: int = 16):
        super().__init__()
        assert enc_embed_dim % 4 == 0, f"enc_embed_dim must be divisible by 4, got {enc_embed_dim}"
        self.enc_embed_dim = enc_embed_dim
        self.patch_size = patch_size

        # Learned null embedding for fully-background patches.
        self.bg_embed = torch.nn.Parameter(torch.zeros(1, 1, enc_embed_dim))

    def forward(self, uv_coords: torch.Tensor, uv_valid: torch.Tensor) -> torch.Tensor:
        """
        Args:
            uv_coords: (B, H, W, 2)  float, UV coordinates in [0, 1].
            uv_valid:  (B, H, W)     bool or float, True/1 where pixel has visible surface.
        Returns:
            (B, N_patches, enc_embed_dim) positional embeddings to add to patch tokens.
        """
        B, H, W, _ = uv_coords.shape
        P = self.patch_size
        n_h, n_w = H // P, W // P
        N = n_h * n_w
        D = self.enc_embed_dim

        with torch.no_grad():
            # Patchify UV coords: (B, n_h, P, n_w, P, 2) → (B, N, P*P, 2)
            uv_p = uv_coords.reshape(B, n_h, P, n_w, P, 2)
            uv_p = uv_p.permute(0, 1, 3, 2, 4, 5).reshape(B, N, P * P, 2)

            # Patchify validity: (B, n_h, P, n_w, P) → (B, N, P*P)
            valid = uv_valid.float().reshape(B, n_h, P, n_w, P)
            valid = valid.permute(0, 1, 3, 2, 4).reshape(B, N, P * P)

            # Embed every pixel: (B, N, P*P, D). Computing inside no_grad so
            # this large intermediate is not stored for the backward pass.
            pixel_emb = uv_sincos_embed(uv_p, D)  # (B, N, P*P, D)

            # Weighted mean pool (IPE / Mip-NeRF style): (B, N, D)
            valid_count = valid.sum(dim=2, keepdim=True).clamp(min=1e-8)
            patch_emb = (pixel_emb * valid.unsqueeze(-1)).sum(dim=2) / valid_count # (BV, N, D)

            has_any_valid = valid.sum(dim=2) > 0  # (B, N) bool

        # bg_embed is a learned parameter; must stay in the autograd graph.
        bg = self.bg_embed.expand(B, N, D)
        patch_emb = torch.where(has_any_valid.unsqueeze(-1), patch_emb, bg)

        return patch_emb  # (B, N, enc_embed_dim)

    def from_uv_grid(self, n_h: int, n_w: int, device: torch.device, template_mask: torch.Tensor=None) -> torch.Tensor:
        """
        UV PE for a template that is already defined in UV space.
        
        To maintain mathematical parity with the image branch (IPE), we
        integrate (mean-pool) the embeddings over the area of each patch
        instead of sampling just the center point.
        """
        P = self.patch_size
        H, W = n_h * P, n_w * P
        
        with torch.no_grad():
            # 1. Create a full-resolution UV grid for the template (0 to 1)
            # Each pixel (i, j) represents the center of that pixel area.
            u_lin = (torch.arange(H, device=device, dtype=torch.float32) + 0.5) / H
            v_lin = (torch.arange(W, device=device, dtype=torch.float32) + 0.5) / W
            u, v = torch.meshgrid(u_lin, v_lin, indexing='ij')
            uv_grid = torch.stack([u, v], dim=-1).unsqueeze(0)  # (1, H, W, 2)
            
            # 2. Use the same logic as forward() but with all pixels valid
            if template_mask is not None: # use UV template mask (0=empty space, 1=body part regions)
                uv_valid = template_mask.float().reshape(-1, H, W)[0]  # template_masks should be all the same throughout batches
            else: # default all 1s
                uv_valid = torch.ones((1, H, W), device=device, dtype=torch.float32)
            
            # We call the internal logic of forward
            # (Alternatively, we can just call self.forward(uv_grid, uv_valid))
            emb = self.forward(uv_grid, uv_valid)
            
        return emb.detach()  # (1, N_patches, enc_embed_dim)


# -----------------------------------------------------------------------
# RoPE2D: RoPE implementation in 2D
# -----------------------------------------------------------------------

try:
    from .curope import cuRoPE2D
    RoPE2D = cuRoPE2D
except ImportError:
    print('Warning, cannot find cuda-compiled version of RoPE2D, using a slow pytorch version instead')

    class RoPE2D(torch.nn.Module):
        
        def __init__(self, freq=100.0, F0=1.0):
            super().__init__()
            self.base = freq 
            self.F0 = F0
            self.cache = {}

        def get_cos_sin(self, D, seq_len, device, dtype):
            if (D,seq_len,device,dtype) not in self.cache:
                inv_freq = 1.0 / (self.base ** (torch.arange(0, D, 2).float().to(device) / D))
                t = torch.arange(seq_len, device=device, dtype=inv_freq.dtype)
                freqs = torch.einsum("i,j->ij", t, inv_freq).to(dtype)
                freqs = torch.cat((freqs, freqs), dim=-1)
                cos = freqs.cos() # (Seq, Dim)
                sin = freqs.sin()
                self.cache[D,seq_len,device,dtype] = (cos,sin)
            return self.cache[D,seq_len,device,dtype]
            
        @staticmethod
        def rotate_half(x):
            x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
            return torch.cat((-x2, x1), dim=-1)
            
        def apply_rope1d(self, tokens, pos1d, cos, sin):
            assert pos1d.ndim==2
            cos = torch.nn.functional.embedding(pos1d, cos)[:, None, :, :]
            sin = torch.nn.functional.embedding(pos1d, sin)[:, None, :, :]
            return (tokens * cos) + (self.rotate_half(tokens) * sin)
            
        def forward(self, tokens, positions):
            """
            input:
                * tokens: batch_size x nheads x ntokens x dim
                * positions: batch_size x ntokens x 2 (y and x position of each token)
            output:
                * tokens after appplying RoPE2D (batch_size x nheads x ntokens x dim)
            """
            assert tokens.size(3)%2==0, "number of dimensions should be a multiple of two"
            D = tokens.size(3) // 2
            assert positions.ndim==3 and positions.shape[-1] == 2 # Batch, Seq, 2
            cos, sin = self.get_cos_sin(D, int(positions.max())+1, tokens.device, tokens.dtype)
            # split features into two along the feature dimension, and apply rope1d on each half
            y, x = tokens.chunk(2, dim=-1)
            y = self.apply_rope1d(y, positions[:,:,0], cos, sin)
            x = self.apply_rope1d(x, positions[:,:,1], cos, sin)
            tokens = torch.cat((y, x), dim=-1)
            return tokens