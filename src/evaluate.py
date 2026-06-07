"""Paper-local evaluator: mean absolute pixel error and median pixel error.

The paper does not report PCK. The evaluator returns only the two pixel-error
metrics reported in tab:training. Called on both train and test loaders by
scripts/eval.py.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import torch

from .constants import IMG_SIZE


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    dataloader,
    device,
    preprocess_fn: Callable | None = None,
) -> dict[str, float]:
    """Return {"mae_px": ..., "median_px": ...} on ``dataloader``.

    ``preprocess_fn(images, batch)`` is applied in-place before the forward
    pass and is used for rendering the L/R/C cue; if None the images are
    passed through unchanged.
    """
    model.eval()
    preds, targets, vis = [], [], []
    for batch in dataloader:
        images = batch["image"].to(device)
        if preprocess_fn is not None:
            images = preprocess_fn(images, batch)
        pred = model(images)
        preds.append(pred.cpu().numpy())
        targets.append(batch["keypoints"].numpy())
        vis.append(batch["visibility"].numpy())

    pred = np.concatenate(preds)
    target = np.concatenate(targets)
    visibility = np.concatenate(vis)

    # per-sample L2 error in pixels, averaged over visible keypoints
    dist = np.linalg.norm(pred - target, axis=-1) * IMG_SIZE           # [B, K]
    mask = visibility > 0
    dist_masked = np.where(mask, dist, 0.0)
    n_vis = mask.sum(axis=-1).clip(min=1)
    per_sample = dist_masked.sum(axis=-1) / n_vis                      # [B]
    per_sample[~mask.any(axis=-1)] = np.nan

    mae_px = float(dist[mask].mean()) if mask.any() else 0.0
    median_px = float(np.nanmedian(per_sample))
    return {"mae_px": mae_px, "median_px": median_px}
