# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# PatchEmbed implementation for DUST3R,
# in particular ManyAR_PatchEmbed that Handle images with non-square aspect ratio
# --------------------------------------------------------
import torch

from .blocks import PatchEmbed


def get_patch_embed(patch_embed_cls, img_size, patch_size, enc_embed_dim, in_chans=3):
    assert patch_embed_cls in ['PatchEmbedDust3R', 'ManyAR_PatchEmbed', 'PrunedPatchEmbed']
    patch_embed = eval(patch_embed_cls)(img_size, patch_size, in_chans, enc_embed_dim)
    return patch_embed

def pad_tokens_to_max_sequence(foreground_tokens, max_tok):
    """
    Pad foreground tokens to the max token sequence length of the batch.

    Args:
        foreground_tokens: list[B,(fg_toks_b, C)]
        max_tok: int -> max token sequence length across batch

    Returns:
        new_tokens: (B, max_tok, C) padded tokens (padding filled with zeros)
        attn_mask: (B, max_tok, 1) bool mask — True for real tokens, False for padding
    """
    B, embed_dim = len(foreground_tokens), foreground_tokens[0].shape[-1]
    device = foreground_tokens[0].device
    dtype = foreground_tokens[0].dtype
    new_tokens = torch.zeros((B, max_tok, embed_dim), dtype=dtype, device=device)
    attn_mask = torch.zeros((B, max_tok, 1), dtype=torch.bool, device=device)
    for b in range(B):
        n = foreground_tokens[b].shape[0]
        new_tokens[b, :n] = foreground_tokens[b]
        attn_mask[b, :n] = True
    return new_tokens, attn_mask


def pad_positions_to_max_sequence(foreground_positions, max_tok):
    """
    Pad foreground positions to the max token sequence length of the batch.
    Padding slots are filled with zeros (they are ignored via the attention mask).

    Args:
        foreground_positions: list of length B, each element (fg_toks_b, 2)
        max_tok: int -> max token sequence length across batch

    Returns:
        new_positions: (B, max_tok, 2) padded positions
    """
    B = len(foreground_positions)
    device = foreground_positions[0].device
    dtype = foreground_positions[0].dtype
    new_positions = torch.zeros((B, max_tok, 2), dtype=dtype, device=device)
    for b in range(B):
        n = foreground_positions[b].shape[0]
        new_positions[b, :n] = foreground_positions[b]
    return new_positions

def prune_background_tokens(x, pos, foreground_mask, patch_size):
    """
    Given a binary, spatial foreground_mask, prune all background tokens.
    Only keep tokens that spatially correspond to foreground pixels.

    Args:
        x: (B, N, C) — all patch tokens
        pos: (B, N, 2) — patch positions (row, col) for downstream PE
        foreground_mask: (B, 1, H, W) — binary foreground mask at image resolution
        patch_size: int or (int, int) — patch size used to downsample the mask

    Returns:
        new_tokens:    (B, max_tok, C)   — foreground tokens, zero-padded
        new_positions: (B, max_tok, 2)   — corresponding positions, zero-padded
        attn_mask:     (B, max_tok, 1)   — True for real tokens, False for padding
    """
    B = foreground_mask.shape[0]
    # Downsample mask to patch grid: (B, 1, H, W) -> (B, N=HW/patch_size**2)
    patch_mask = torch.nn.functional.max_pool2d(
        foreground_mask.float(), patch_size, stride=patch_size
    ).bool().squeeze(1).flatten(1)  # (B, N)

    all_foreground_tokens = []
    all_foreground_positions = []
    max_tok = 0
    for b in range(B):
        # nonzero returns (fg_toks,) 1D index tensor of foreground patch positions
        foreground_idx = patch_mask[b].nonzero(as_tuple=True)[0]  # (fg_toks,)
        max_tok = max(max_tok, foreground_idx.shape[0])
        all_foreground_tokens.append(x[b][foreground_idx])        # (fg_toks, C)
        all_foreground_positions.append(pos[b][foreground_idx])   # (fg_toks, 2)

    new_tokens, attn_mask = pad_tokens_to_max_sequence(all_foreground_tokens, max_tok)
    new_positions = pad_positions_to_max_sequence(all_foreground_positions, max_tok)
    return new_tokens, new_positions, attn_mask  # (B, max_tok, C), (B, max_tok, 2), (B, max_tok, 1)


class PrunedPatchEmbed(PatchEmbed):
    """
    Patch embedder that optionally prunes background tokens using a foreground mask.

    When 'foreground_mask' (B, 1, H, W) is passed as a keyword argument, tokens
    corresponding to background patches are dropped. The remaining tokens are
    zero-padded to the maximum foreground token count in the batch, and an
    attention mask is returned so callers can ignore padding slots.

    Always returns (x, pos, attn_mask):
        x:         (B, N', C)    — pruned+padded tokens  (N' == N when no mask)
        pos:       (B, N', 2)    — matching pruned+padded positions for RoPE
        attn_mask: (B, N', 1)    — True for real tokens; None when no mask given
    """
    def forward(self, x, **kw):
        B, C, H, W = x.shape
        assert H % self.patch_size[0] == 0, f"Input image height ({H}) is not a multiple of patch size ({self.patch_size[0]})."
        assert W % self.patch_size[1] == 0, f"Input image width ({W}) is not a multiple of patch size ({self.patch_size[1]})."
        x = self.proj(x)  # (B, embed_dim, h_tok, w_tok)
        pos = self.position_getter(B, x.size(2), x.size(3), x.device)  # (B, N, 2)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # (B, N, embed_dim)
        attn_mask = None
        if 'foreground_mask' in kw: # prune background tokens (if no mask, default PatchEmbedDust3R behavior)
            foreground_mask = kw['foreground_mask']  # (B, 1, H, W)
            x, pos, attn_mask = prune_background_tokens(x, pos, foreground_mask, self.patch_size)
        x = self.norm(x)
        return x, pos, attn_mask


class PatchEmbedDust3R(PatchEmbed):
    def forward(self, x, **kw):
        B, C, H, W = x.shape
        assert H % self.patch_size[0] == 0, f"Input image height ({H}) is not a multiple of patch size ({self.patch_size[0]})."
        assert W % self.patch_size[1] == 0, f"Input image width ({W}) is not a multiple of patch size ({self.patch_size[1]})."            
        x = self.proj(x) # patched tokens (B,C,H,W)
        pos = self.position_getter(B, x.size(2), x.size(3), x.device)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x, pos


class ManyAR_PatchEmbed (PatchEmbed):
    """ Handle images with non-square aspect ratio.
        All images in the same batch have the same aspect ratio.
        true_shape = [(height, width) ...] indicates the actual shape of each image.
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, norm_layer=None, flatten=True):
        self.embed_dim = embed_dim
        super().__init__(img_size, patch_size, in_chans, embed_dim, norm_layer, flatten)

    def forward(self, img, true_shape):
        B, C, H, W = img.shape
        assert W >= H, f'img should be in landscape mode, but got {W=} {H=}'
        assert H % self.patch_size[0] == 0, f"Input image height ({H}) is not a multiple of patch size ({self.patch_size[0]})."
        assert W % self.patch_size[1] == 0, f"Input image width ({W}) is not a multiple of patch size ({self.patch_size[1]})."
        assert true_shape.shape == (B, 2), f"true_shape has the wrong shape={true_shape.shape}"

        # size expressed in tokens
        W //= self.patch_size[0]
        H //= self.patch_size[1]
        n_tokens = H * W

        height, width = true_shape.T
        is_landscape = (width >= height)
        is_portrait = ~is_landscape

        # allocate result
        x = img.new_zeros((B, n_tokens, self.embed_dim))
        pos = img.new_zeros((B, n_tokens, 2), dtype=torch.int64)

        # linear projection, transposed if necessary
        x[is_landscape] = self.proj(img[is_landscape]).permute(0, 2, 3, 1).flatten(1, 2).float()
        x[is_portrait] = self.proj(img[is_portrait].swapaxes(-1, -2)).permute(0, 2, 3, 1).flatten(1, 2).float()

        pos[is_landscape] = self.position_getter(1, H, W, pos.device)
        pos[is_portrait] = self.position_getter(1, W, H, pos.device)

        x = self.norm(x)
        return x, pos
