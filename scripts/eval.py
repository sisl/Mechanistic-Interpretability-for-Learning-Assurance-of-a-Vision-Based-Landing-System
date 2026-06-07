"""Post-hoc evaluation of a finished checkpoint on train and test splits.

Writes {results_dir}/{run_name}_eval.json with:
    {
        "train": {mae_px, median_px},
        "test":  {mae_px, median_px},
    }

The paper reports both columns in tab:training. Kept separate from
scripts/train.py so train/test metrics can be regenerated from the saved
final checkpoint without rerunning training.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Annotated

import numpy as np
import torch
import typer
from torch.utils.data import DataLoader

from src.constants import IMG_SIZE, NUM_KEYPOINTS
from src.cue import render_runway_cue
from src.data import LARDv2Dataset, LARDTrainDataset
from src.evaluate import evaluate
from src.model import ViTKeypointRegressor


app = typer.Typer(add_completion=False)


def _make_loader(split: str, batch_size: int, num_workers: int, max_samples: int | None):
    ds = LARDv2Dataset(subset="all", split=split, max_samples=max_samples)
    return DataLoader(LARDTrainDataset(ds, image_size=IMG_SIZE),
                      batch_size=batch_size, shuffle=False,
                      num_workers=num_workers, pin_memory=True)


@app.command()
def main(
    checkpoint: Annotated[Path, typer.Option()],
    run_name: Annotated[str, typer.Option(help="Output basename; default derived from checkpoint")] = "",
    results_dir: Annotated[Path, typer.Option()] = Path("results"),
    device: Annotated[str, typer.Option()] = "cuda:0",
    batch_size: Annotated[int, typer.Option()] = 128,
    num_workers: Annotated[int, typer.Option()] = 8,
):
    import os
    dry = bool(os.environ.get("DRY"))
    max_samples = 50 if dry else None
    if dry:
        num_workers = 0

    dev = torch.device(device)
    ckpt = torch.load(checkpoint, map_location=dev, weights_only=False)
    seed = int(ckpt.get("config", {}).get("seed", 0))
    torch.manual_seed(seed); random.seed(seed); np.random.seed(seed)
    backbone = ckpt["config"]["backbone"]
    model = ViTKeypointRegressor(
        num_keypoints=NUM_KEYPOINTS, backbone=backbone,
        pretrained=False, image_size=IMG_SIZE,
    ).to(dev)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    run_name = run_name or Path(checkpoint).stem.replace("_best", "")
    print(f"[EVAL] {run_name}", flush=True)

    def preprocess(imgs, b):
        render_runway_cue(imgs, b["runway_side"].to(imgs.device), patch_size=model.patch_size)
        return imgs

    out = {"run_name": run_name}
    for split in ("train", "test"):
        loader = _make_loader(split, batch_size, num_workers, max_samples)
        m = evaluate(model, loader, dev, preprocess_fn=preprocess)
        out[split] = {"mae_px": m["mae_px"], "median_px": m["median_px"]}
        print(f"[EVAL] {split}: median={m['median_px']:.2f}px  MAE={m['mae_px']:.2f}px", flush=True)

    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / f"{run_name}_eval.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"[EVAL] wrote {out_path}")


if __name__ == "__main__":
    app()
