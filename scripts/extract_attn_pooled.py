"""Attention-pooled image codes for the IMS classifier (main.typ @sec:ims-lr).

For each image the paper's IMS classifier consumes the sparse-coded
attention-pooled summary ``u = (1/K) sum_k sum_i alpha_{ik} z_i``. This
script computes ``u`` for every BOGO test sample and, in a second pass,
for every inverse-BOGO sample, then matching-pursuit encodes each against
the K-SVD dictionary at the same sparsity k=8 used in the rest of the paper.

Output: a single .npz with ``bogo_codes`` and ``inverse_bogo_codes`` of shape
``[N, M]`` each, ready for ``scripts/ims_sweep.py``.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Annotated

import numpy as np
import torch
import torch.nn.functional as F
import typer
from sklearn.decomposition import sparse_encode
from torch.utils.data import DataLoader

from src.constants import IMG_SIZE, NUM_KEYPOINTS
from src.cue import render_runway_cue
from src.data import LARDv2Dataset, LARDTrainDataset
from src.model import ViTKeypointRegressor


app = typer.Typer(add_completion=False)

SKIPPED_PATCH_INDICES = (0,)  # top-left patch carries the cue


def _pooled_summary(model, loader, device) -> np.ndarray:
    """Mean over K of attention-pooled non-cue patch tokens, per main.typ @sec:ims-lr."""
    ps = model.patch_size
    pooled = []
    n_prefix = getattr(model.backbone, "num_prefix_tokens", 1)
    keep_mask = np.ones(model.num_patches, dtype=bool)
    for idx in SKIPPED_PATCH_INDICES:
        keep_mask[idx] = False
    keep_mask_t = torch.from_numpy(keep_mask).to(device)
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            render_runway_cue(images, batch["runway_side"].to(device), patch_size=ps)
            tokens = model.backbone(images)
            patches = tokens[:, n_prefix:]                                    # [B, N, D]
            patches = patches[:, keep_mask_t, :]                              # [B, N-1, D]
            alpha = F.softmax(model.global_head(patches), dim=1)              # [B, N-1, K]
            uk = torch.einsum("bnk,bnd->bkd", alpha, patches)                 # [B, K, D]
            pooled.append(uk.mean(dim=1).cpu().numpy())                       # [B, D]
    return np.concatenate(pooled, axis=0)


@app.command()
def main(
    checkpoint: Annotated[Path, typer.Option()],
    dictionary: Annotated[Path, typer.Option(help="KSVD dict .npy (D x M)")],
    out: Annotated[Path, typer.Option(help="Output .npz with bogo_codes / inverse_bogo_codes")],
    split: Annotated[str, typer.Option()] = "test",
    nnz: Annotated[int, typer.Option()] = 8,
    batch_size: Annotated[int, typer.Option()] = 64,
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

    D_matrix = np.load(dictionary)
    # Normalize to [D, M]; accept [D, M] or [M, D].
    if D_matrix.shape[0] != model.backbone.num_features:
        D_matrix = D_matrix.T
    assert D_matrix.shape[0] == model.backbone.num_features, \
        f"dictionary {D_matrix.shape} incompatible with backbone dim {model.backbone.num_features}"
    print(f"[ATTN] dictionary {D_matrix.shape}", flush=True)

    def _codes_for(ood: bool) -> np.ndarray:
        ds = LARDv2Dataset(subset="all", split=split, max_samples=max_samples, ood=ood)
        loader = DataLoader(LARDTrainDataset(ds, image_size=IMG_SIZE),
                            batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
        pooled = _pooled_summary(model, loader, dev)
        print(f"[ATTN] {'ibogo' if ood else 'bogo'} pooled {pooled.shape}", flush=True)
        codes = sparse_encode(pooled, D_matrix.T.astype(np.float64),
                              n_nonzero_coefs=nnz, algorithm="omp").astype(np.float32)
        print(f"[ATTN] codes {codes.shape}", flush=True)
        return codes

    bogo = _codes_for(ood=False)
    ibogo = _codes_for(ood=True)

    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, bogo_codes=bogo, inverse_bogo_codes=ibogo)
    print(f"[ATTN] wrote {out}")


if __name__ == "__main__":
    app()
