"""ViT + soft-argmax keypoint regressor (main.typ @eq:softargmax).

Predicts keypoint coordinates as a differentiable weighted sum over a fixed
linspace grid of anchor positions::

    coords_k = sum_i softmax(head(z_i))_{i,k} * grid_i

where z_i are patch token embeddings and grid_i spans [0, 1]^2. No
uncertainty head: the paper does not consume per-sample variance.
"""

from __future__ import annotations

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F


class ViTKeypointRegressor(nn.Module):
    def __init__(
        self,
        num_keypoints: int,
        backbone: str = "vit_small_patch14_dinov2.lvd142m",
        pretrained: bool = True,
        image_size: int = 224,
    ):
        super().__init__()
        self.num_keypoints = num_keypoints

        self.backbone = timm.create_model(
            backbone, pretrained=pretrained, img_size=image_size,
            num_classes=0, global_pool="",
        )

        pe = getattr(self.backbone, "patch_embed", None)
        if pe is not None and hasattr(pe, "patch_size"):
            ps = pe.patch_size[0] if isinstance(pe.patch_size, tuple) else pe.patch_size
        else:
            ps = 16
        self.patch_size = ps
        grid_h, grid_w = image_size // ps, image_size // ps
        self.num_patches = grid_h * grid_w

        D = self.backbone.num_features
        # Per-patch head that yields one attention logit per keypoint.
        self.global_head = nn.Linear(D, num_keypoints)

        # Anchor grid: linspace [0, 1] so the convex combination can reach edges.
        gy, gx = torch.meshgrid(
            torch.linspace(0, 1, grid_h),
            torch.linspace(0, 1, grid_w),
            indexing="ij",
        )
        self.register_buffer(
            "grid", torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=-1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, 3, H, W] ImageNet-normalized images (cue already painted).
        Returns:
            coords [B, K, 2] in [0, 1] as (x, y).
        """
        tokens = self.backbone(x)                                     # [B, 1+N, D]
        n_prefix = getattr(self.backbone, "num_prefix_tokens", 1)
        patch_tokens = tokens[:, n_prefix:]                           # [B, N, D]
        alpha = F.softmax(self.global_head(patch_tokens), dim=1)      # [B, N, K]
        coords = torch.einsum("bnk,nd->bkd", alpha, self.grid)        # [B, K, 2]
        return coords
