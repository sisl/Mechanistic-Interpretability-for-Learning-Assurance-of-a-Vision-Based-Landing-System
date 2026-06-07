"""Paper-local source tree for the DASC 2026 mechinterp submission.

Structure mirrors the paper's methods: `data` (LARDv2 + BOGO + cue), `model`
(ViT backbone + soft-argmax head), `losses` (masked Huber), `evaluate`
(MAE + median only). Downstream analyses (content/style, effective score,
atom viz, IMS sweep) live in scripts/ as typer CLIs that compose these.
"""

from .constants import IMG_SIZE, NUM_KEYPOINTS, IMAGENET_MEAN, IMAGENET_STD

__all__ = ["IMG_SIZE", "NUM_KEYPOINTS", "IMAGENET_MEAN", "IMAGENET_STD"]
