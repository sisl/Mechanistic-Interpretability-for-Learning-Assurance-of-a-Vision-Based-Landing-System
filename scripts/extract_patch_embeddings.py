"""Extract patch token embeddings and the importance-sampled K-SVD training pool.

Produces three files (main.typ @sec:ksvd-setup):

- ``{out}_all_patches.h5``: one row per (image, patch) for every test sample,
  with the top-left cue patch excluded so no atom can encode the cue. Used by
  ``scripts/julia/sparse_code_patches.jl`` to matching-pursuit encode every patch.
- ``{out}_ksvd_pool.h5``: the importance-sampled training pool for K-SVD,
  consisting of one CLS token plus the top-K patch tokens (by sum of head
  attention magnitudes across keypoints) per image. Default K=4 → ~5 vectors
  per test image, ~240k vectors total.
- ``{out}_head.npz``: the regression head's weight W and bias b, saved so
  downstream analyses can compute head-weight statistics without loading
  the full model.

The cue patch is always painted on the input before the forward pass so the
embeddings match the training distribution.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Annotated

import h5py
import numpy as np
import torch
import torch.nn.functional as F
import typer
from torch.utils.data import DataLoader

from src.constants import IMG_SIZE, NUM_KEYPOINTS
from src.cue import render_runway_cue
from src.data import LARDv2Dataset, LARDTrainDataset
from src.model import ViTKeypointRegressor


app = typer.Typer(add_completion=False)

SKIPPED_PATCH_INDICES = (0,)  # top-left patch carries the cue


@app.command()
def main(
    checkpoint: Annotated[Path, typer.Option()],
    out_prefix: Annotated[Path, typer.Option(help="Output prefix; writes {out}_all_patches.h5, {out}_ksvd_pool.h5, {out}_head.npz")],
    split: Annotated[str, typer.Option()] = "test",
    k_sampled: Annotated[int, typer.Option(help="Top-K attention-ranked patches to add to the K-SVD pool per image")] = 4,
    batch_size: Annotated[int, typer.Option()] = 32,
    num_workers: Annotated[int, typer.Option()] = 4,
    device: Annotated[str, typer.Option()] = "cuda:0",
):
    dry = bool(os.environ.get("DRY"))
    max_samples = 50 if dry else None
    if dry:
        num_workers = 0

    dev = torch.device(device)
    ckpt = torch.load(checkpoint, map_location=dev, weights_only=False)
    model = ViTKeypointRegressor(
        num_keypoints=NUM_KEYPOINTS, backbone=ckpt["config"]["backbone"],
        pretrained=False, image_size=IMG_SIZE,
    ).to(dev)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    ps = model.patch_size
    W_head = model.global_head.weight.detach().cpu().numpy()
    b_head = model.global_head.bias.detach().cpu().numpy()
    grid = model.grid.detach().cpu().numpy()
    print(f"[EXTRACT] backbone dim={model.backbone.num_features}, patches={model.num_patches}, K={NUM_KEYPOINTS}", flush=True)

    ds = LARDv2Dataset(subset="all", split=split, max_samples=max_samples)
    loader = DataLoader(LARDTrainDataset(ds, image_size=IMG_SIZE),
                        batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=True)

    D = model.backbone.num_features
    N_per_image = model.num_patches - len(SKIPPED_PATCH_INDICES)
    N_total = len(ds) * N_per_image
    all_patches = np.empty((N_total, D), dtype=np.float32)
    ksvd_pool: list[np.ndarray] = []

    row = 0
    keep_mask = np.ones(model.num_patches, dtype=bool)
    for idx in SKIPPED_PATCH_INDICES:
        keep_mask[idx] = False
    with torch.no_grad():
        for bi, batch in enumerate(loader):
            images = batch["image"].to(dev)
            render_runway_cue(images, batch["runway_side"].to(dev), patch_size=ps)
            tokens = model.backbone(images)                                    # [B, 1+N, D]
            n_prefix = getattr(model.backbone, "num_prefix_tokens", 1)
            cls = tokens[:, 0]                                                 # [B, D]
            patches = tokens[:, n_prefix:]                                     # [B, N, D]
            alpha_logits = model.global_head(patches)                          # [B, N, K]
            alpha = F.softmax(alpha_logits, dim=1)

            # importance-sampled pool: CLS + top-k patches per image
            # rank patches by the sum of per-keypoint attention weights
            attn_rank = alpha.sum(dim=-1)                                      # [B, N]
            attn_rank[:, ~torch.from_numpy(keep_mask).to(attn_rank.device)] = -1.0
            topk_idx = attn_rank.topk(k_sampled, dim=1).indices                # [B, k]
            for i in range(patches.shape[0]):
                ksvd_pool.append(cls[i:i+1].cpu().numpy())
                ksvd_pool.append(patches[i, topk_idx[i]].cpu().numpy())

            # full patch set (minus cue) for matching-pursuit coding downstream
            filtered = patches[:, keep_mask, :].cpu().numpy()                  # [B, N', D]
            end = row + filtered.shape[0] * filtered.shape[1]
            all_patches[row:end] = filtered.reshape(-1, D)
            row = end

            if (bi + 1) % 50 == 0 or (bi + 1) == len(loader):
                print(f"[EXTRACT] {(bi+1) * batch_size}/{len(ds)}", flush=True)

    all_patches = all_patches[:row]
    print(f"[EXTRACT] all_patches: {all_patches.shape}", flush=True)
    pool = np.concatenate(ksvd_pool, axis=0)
    print(f"[EXTRACT] ksvd_pool:    {pool.shape}", flush=True)

    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    # all_patches.h5: [N_total, D] so Julia H5 -> [D, N_total] after transpose
    ap_path = out_prefix.with_name(out_prefix.name + "_all_patches.h5")
    with h5py.File(ap_path, "w") as f:
        f.create_dataset("embeddings", data=all_patches, dtype=np.float32)
        f.attrs["skipped_patches"] = np.array(SKIPPED_PATCH_INDICES, dtype=np.int64)
        f.attrs["patches_per_image"] = N_per_image
        f.attrs["num_images"] = len(ds)
        f.attrs["subset_names"] = np.array(ds.subset_names, dtype="S")
        f.attrs["subset_sizes"] = np.array(ds.subset_sizes, dtype=np.int64)
    print(f"[EXTRACT] wrote {ap_path}")

    pool_path = out_prefix.with_name(out_prefix.name + "_ksvd_pool.h5")
    with h5py.File(pool_path, "w") as f:
        f.create_dataset("embeddings", data=pool, dtype=np.float32)
        f.attrs["k_sampled"] = k_sampled
    print(f"[EXTRACT] wrote {pool_path}")

    head_path = out_prefix.with_name(out_prefix.name + "_head.npz")
    np.savez(head_path, W=W_head, b=b_head, patch_centers=grid)
    print(f"[EXTRACT] wrote {head_path}")


if __name__ == "__main__":
    app()
