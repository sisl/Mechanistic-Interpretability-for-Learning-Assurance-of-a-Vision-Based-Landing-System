"""Masked Huber loss for keypoint regression (main.typ @sec:reg-setup).

Breakpoint delta is in normalized [0, 1] coordinates; with images at 224x224
and delta = 8/224 the loss transitions from quadratic to linear at 8 pixels,
which we justify in the paper as the threshold above which we consider a
residual an "outlier" driven by occlusion rather than a recoverable error.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def masked_huber(pred: torch.Tensor, target: torch.Tensor,
                 visibility: torch.Tensor, beta: float = 8.0 / 224.0) -> torch.Tensor:
    """Smooth L1 loss restricted to visible keypoints.

    Args:
        pred: [B, K, 2] predictions in [0, 1].
        target: [B, K, 2] targets in [0, 1].
        visibility: [B, K] with entries > 0 for visible keypoints.
        beta: Huber breakpoint in [0, 1] coordinate units.
    """
    mask = (visibility > 0).unsqueeze(-1).expand_as(pred).float()
    n = mask.sum()
    if n == 0:
        return pred.sum() * 0.0
    loss = F.smooth_l1_loss(pred, target, reduction="none", beta=beta)
    return (loss * mask).sum() / n
