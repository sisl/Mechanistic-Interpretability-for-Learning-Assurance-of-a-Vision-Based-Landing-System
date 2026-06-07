"""Runway L/R/C cue painter (main.typ @sec:cue).

Paints a ~quarter-patch colored block in the top-left of each image encoding
the pilot-intent side: red=L, blue=R, green=C, untouched for runways with no
L/R/C suffix. Applied in-place to ImageNet-normalized tensors; the cue color
is first mapped through the same normalization so the painted pixels stay
consistent with the rest of the input.
"""

from __future__ import annotations

import torch

from .constants import IMAGENET_MEAN, IMAGENET_STD


_CUE_COLORS_RAW = {
    1: torch.tensor([255.0,   0.0,   0.0]),  # L = red
    2: torch.tensor([  0.0,   0.0, 255.0]),  # R = blue
    3: torch.tensor([  0.0, 255.0,   0.0]),  # C = green
}


def render_runway_cue(images: torch.Tensor, runway_side: torch.Tensor,
                      patch_size: int = 14) -> torch.Tensor:
    """Paint the L/R/C cue in-place.

    Args:
        images: [B, 3, H, W] ImageNet-normalized tensor.
        runway_side: [B] long tensor with entries in {0, 1, 2, 3}; 0 means no
            cue (runway has no L/R/C suffix) so the image is left untouched.
        patch_size: ViT patch edge length in pixels; the cue fills the
            top-left ``patch_size // 2`` square so it occupies the first
            patch token and nothing more.
    """
    device = images.device
    mean = IMAGENET_MEAN.to(device)
    std  = IMAGENET_STD.to(device)
    cue_size = patch_size // 2
    for side_idx, raw_rgb in _CUE_COLORS_RAW.items():
        mask = (runway_side == side_idx)
        if not mask.any():
            continue
        norm_color = (raw_rgb.to(device) / 255.0 - mean) / std
        images[mask, :, :cue_size, :cue_size] = norm_color[:, None, None]
    return images
