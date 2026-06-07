"""Atom-level top-activating-patch grids (main.typ @sec:visualization, @fig:atoms).

For each atom we pick the patches with the largest absolute sparse
coefficient and render them at their native resolution with a small
three-patch context window, target patch outlined. Output: one PNG per atom
in ``out_dir/atom_{j:04d}.png``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Annotated

import h5py
import numpy as np
import torch
import typer
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader

from src.constants import IMG_SIZE
from src.data import LARDv2Dataset


app = typer.Typer(add_completion=False)


def _atom_grid(sample_tiles: list[np.ndarray], target_boxes: list[tuple[int, int, int, int]],
               cols: int = 3, cell: int = 96) -> Image.Image:
    rows = (len(sample_tiles) + cols - 1) // cols
    grid = Image.new("RGB", (cols * cell, rows * cell), (255, 255, 255))
    draw = ImageDraw.Draw(grid)
    for i, (tile, box) in enumerate(zip(sample_tiles, target_boxes)):
        r, c = divmod(i, cols)
        im = Image.fromarray(tile).resize((cell, cell))
        grid.paste(im, (c * cell, r * cell))
        x0, y0, x1, y1 = box
        scale_x = cell / tile.shape[1]; scale_y = cell / tile.shape[0]
        draw.rectangle([c * cell + int(x0 * scale_x), r * cell + int(y0 * scale_y),
                         c * cell + int(x1 * scale_x), r * cell + int(y1 * scale_y)],
                        outline=(220, 30, 30), width=2)
    return grid


@app.command()
def main(
    sparse_codes: Annotated[Path, typer.Option()],
    atoms: Annotated[Path, typer.Option(help="atoms.json from scripts/atoms.py")],
    out_dir: Annotated[Path, typer.Option()],
    split: Annotated[str, typer.Option()] = "test",
    top_k: Annotated[int, typer.Option(help="Top-K activating patches per atom")] = 9,
    atom_ids: Annotated[str, typer.Option(help="Comma-separated atom ids to render; default: top 32 effective")] = "",
):
    dry = bool(os.environ.get("DRY"))
    max_samples = 50 if dry else None

    with h5py.File(sparse_codes, "r") as f:
        indices = f["indices"][:]                                        # [k, n_cols]
        values  = f["values"][:]
        num_patches_per_image = int(f.attrs.get("patches_per_image", 255))
        skipped = list(f.attrs.get("skipped_patches", [0]))
    nnz_k, n_cols = indices.shape

    meta = json.loads(atoms.read_text())
    if atom_ids:
        ids = [int(x) for x in atom_ids.split(",")]
    else:
        ids = meta["effective_order"][:32]

    ds = LARDv2Dataset(subset="all", split=split, max_samples=max_samples)
    # Map column -> (image_idx, patch_idx in the full grid before cue removal).
    keep_mask = np.ones(num_patches_per_image + len(skipped), dtype=bool)
    for s in skipped:
        keep_mask[s] = False
    patch_to_full = np.flatnonzero(keep_mask)                              # [N']

    # For each requested atom, find top-K absolute sparse codes.
    out_dir.mkdir(parents=True, exist_ok=True)
    # flatten indices/values to a sparse lookup for just the atoms we need.
    for j in ids:
        # scan for columns where atom j appears in the nnz list.
        hits_col = []
        hits_val = []
        for k_row in range(nnz_k):
            col_mask = indices[k_row] == j
            cols = np.flatnonzero(col_mask)
            hits_col.append(cols)
            hits_val.append(values[k_row, cols])
        cols = np.concatenate(hits_col) if hits_col else np.array([], dtype=np.int64)
        vals = np.concatenate(hits_val) if hits_val else np.array([], dtype=np.float32)
        if len(cols) == 0:
            print(f"[VIZ] atom {j}: never fires, skipping")
            continue
        order = np.argsort(-np.abs(vals))[:top_k]
        top_cols = cols[order]

        tiles: list[np.ndarray] = []
        boxes: list[tuple[int, int, int, int]] = []
        grid_h = int(np.sqrt(num_patches_per_image + len(skipped)))
        patch_size = IMG_SIZE // grid_h
        for col in top_cols:
            img_idx = int(col // num_patches_per_image)
            patch_local = int(col % num_patches_per_image)
            patch_full = int(patch_to_full[patch_local])
            if img_idx >= len(ds):
                continue
            image = ds[img_idx].image                                        # [H, W, 3] uint8
            row, c = divmod(patch_full, grid_h)
            # three-patch context window
            y0 = max(0, (row - 1) * patch_size); y1 = min(image.shape[0], (row + 2) * patch_size)
            x0 = max(0, (c - 1) * patch_size);   x1 = min(image.shape[1], (c + 2) * patch_size)
            tile = image[y0:y1, x0:x1]
            tiles.append(tile)
            # target patch outline within the tile
            bx0 = (c * patch_size) - x0
            by0 = (row * patch_size) - y0
            boxes.append((bx0, by0, bx0 + patch_size, by0 + patch_size))
        grid = _atom_grid(tiles, boxes)
        grid.save(out_dir / f"atom_{j:04d}.png")
    print(f"[VIZ] wrote {len(ids)} grids to {out_dir}")


if __name__ == "__main__":
    app()
