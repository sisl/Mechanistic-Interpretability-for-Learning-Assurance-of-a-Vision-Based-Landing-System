"""Training CLI for the DASC 2026 keypoint regressor (main.typ @sec:reg-setup).

The minimal dual of the paper's training setup: DINOv2-S backbone,
soft-argmax head, masked Huber on [0,1] keypoints, AdamW with linear warmup
then cosine decay, runway cue painted in-place before the forward pass.
No uncertainty head. Training always saves the final-epoch checkpoint.

Usage:
    uv run python -u scripts/train.py --variant pretrained --seed 0 --device cuda:0
    DRY=1 uv run python -u scripts/train.py --variant pretrained --seed 0 --device cpu
"""

from __future__ import annotations

import json
import math
import os
import random
import time
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
from src.losses import masked_huber
from src.model import ViTKeypointRegressor


VARIANT_DEFAULTS = {
    "pretrained": dict(epochs=30, batch_size=128, lr=5e-4,
                       backbone_lr_scale=0.1, warmup_epochs=5, pretrained=True),
    "scratch":    dict(epochs=60, batch_size=128, lr=5e-4,
                       backbone_lr_scale=1.0, warmup_epochs=5, pretrained=False),
}


app = typer.Typer(add_completion=False)


def _cue_preprocess(images, batch, patch_size):
    render_runway_cue(images, batch["runway_side"].to(images.device), patch_size=patch_size)
    return images


def _make_loader(dataset_name: str, split: str, batch_size: int,
                 num_workers: int, max_samples: int | None):
    """Single-split loader; retained as a thin helper for ad-hoc callers."""
    subset = "all" if dataset_name == "lard" else dataset_name.split("_", 1)[1]
    underlying = "train" if split in ("train", "train_eval") else split
    ds = LARDv2Dataset(subset=subset, split=underlying, max_samples=max_samples)
    wrapped = LARDTrainDataset(ds, image_size=IMG_SIZE)
    shuffle = (split == "train")
    drop_last = (split == "train")
    return DataLoader(wrapped, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, pin_memory=True,
                      drop_last=drop_last)


def _torch_loaders(dataset_name: str, batch_size: int, num_workers: int,
                   max_samples: int | None):
    """Return (train, train_eval, test) DataLoaders.

    Materializes each split once. The train split is wrapped in two
    DataLoaders: SGD-style (shuffle + drop_last) for training and
    deterministic (shuffle=False, drop_last=False) for the per-epoch
    train-side metric pass.
    """
    subset = "all" if dataset_name == "lard" else dataset_name.split("_", 1)[1]
    train_ds = LARDv2Dataset(subset=subset, split="train", max_samples=max_samples)
    test_ds  = LARDv2Dataset(subset=subset, split="test",  max_samples=max_samples)
    train_wrapped = LARDTrainDataset(train_ds, image_size=IMG_SIZE)
    test_wrapped  = LARDTrainDataset(test_ds,  image_size=IMG_SIZE)
    common = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=True)
    train_loader      = DataLoader(train_wrapped, shuffle=True,  drop_last=True,  **common)
    train_eval_loader = DataLoader(train_wrapped, shuffle=False, drop_last=False, **common)
    test_loader       = DataLoader(test_wrapped,  shuffle=False, drop_last=False, **common)
    return train_loader, train_eval_loader, test_loader


def _build_optimizer(model, lr: float, backbone_lr_scale: float, weight_decay: float):
    if backbone_lr_scale != 1.0:
        bb_ids = {id(p) for p in model.backbone.parameters()}
        return torch.optim.AdamW([
            {"params": [p for p in model.parameters() if id(p) not in bb_ids], "lr": lr},
            {"params": list(model.backbone.parameters()), "lr": lr * backbone_lr_scale},
        ], weight_decay=weight_decay)
    return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)


def _schedule_lr(optimizer, base_lrs, step, total_steps, warmup_steps):
    if step < warmup_steps and warmup_steps > 0:
        mult = step / max(1, warmup_steps)
    else:
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        mult = 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
    for pg, base in zip(optimizer.param_groups, base_lrs):
        pg["lr"] = base * mult


@app.command()
def main(
    variant: Annotated[str, typer.Option(help="pretrained or scratch")] = "pretrained",
    seed: Annotated[int, typer.Option()] = 0,
    run_name: Annotated[str, typer.Option(help="Checkpoint basename; default lard_<variant>_seed<seed>")] = "",
    dataset: Annotated[str, typer.Option(help="lard or lard_<subset>")] = "lard",
    backbone: Annotated[str, typer.Option()] = "vit_small_patch14_dinov2.lvd142m",
    device: Annotated[str, typer.Option()] = "cuda:0",
    num_workers: Annotated[int, typer.Option()] = 8,
    weight_decay: Annotated[float, typer.Option()] = 1e-4,
    grad_clip: Annotated[float, typer.Option()] = 1.0,
    huber_delta_px: Annotated[float, typer.Option(help="Huber breakpoint in pixels; converted to [0,1] via /IMG_SIZE")] = 8.0,
    output_dir: Annotated[Path, typer.Option()] = Path("models"),
    results_dir: Annotated[Path, typer.Option()] = Path("results"),
    log_every: Annotated[int, typer.Option()] = 50,
    epochs: Annotated[int, typer.Option(help="Override variant default")] = -1,
    batch_size: Annotated[int, typer.Option(help="Override variant default")] = -1,
    lr: Annotated[float, typer.Option(help="Override variant default")] = -1.0,
    backbone_lr_scale: Annotated[float, typer.Option(help="Override variant default")] = -1.0,
    warmup_epochs: Annotated[int, typer.Option(help="Override variant default")] = -1,
):
    """Train the paper-local keypoint regressor. `variant` picks the default schedule."""
    if variant not in VARIANT_DEFAULTS:
        raise typer.BadParameter(f"variant must be one of {list(VARIANT_DEFAULTS)}")
    d = dict(VARIANT_DEFAULTS[variant])
    if epochs > 0:            d["epochs"] = epochs
    if batch_size > 0:        d["batch_size"] = batch_size
    if lr > 0:                d["lr"] = lr
    if backbone_lr_scale > 0: d["backbone_lr_scale"] = backbone_lr_scale
    if warmup_epochs >= 0:    d["warmup_epochs"] = warmup_epochs

    dry = bool(os.environ.get("DRY"))
    max_samples = 50 if dry else None
    if dry:
        d["epochs"] = 1
        d["batch_size"] = min(d["batch_size"], 8)
        num_workers = 0

    run_name = run_name or f"lard_{variant}_seed{seed}"
    output_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = output_dir / f"{run_name}_best.pt"

    torch.manual_seed(seed); random.seed(seed); np.random.seed(seed)
    dev = torch.device(device)

    print(f"[TRAIN] variant={variant} run={run_name} device={dev} dry={bool(dry)}", flush=True)
    print(f"[TRAIN] {d}", flush=True)

    # Loading LARD is the expensive part; share the train-split materialization
    # between the SGD loader and the deterministic train-eval loader.
    train_loader, train_eval_loader, val_loader = _torch_loaders(
        dataset, d["batch_size"], num_workers=num_workers, max_samples=max_samples,
    )

    model = ViTKeypointRegressor(
        num_keypoints=NUM_KEYPOINTS, backbone=backbone,
        pretrained=d["pretrained"], image_size=IMG_SIZE,
    ).to(dev)
    print(f"[TRAIN] {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M params", flush=True)

    optimizer = _build_optimizer(model, d["lr"], d["backbone_lr_scale"], weight_decay)
    base_lrs = [pg["lr"] for pg in optimizer.param_groups]

    steps_per_epoch = max(1, len(train_loader))
    total_steps = steps_per_epoch * d["epochs"]
    warmup_steps = steps_per_epoch * d["warmup_epochs"]
    delta = huber_delta_px / IMG_SIZE

    step = 0; t_start = time.time()
    trace: list[dict] = []  # per-epoch {epoch, train, test} metrics

    preprocess = lambda imgs, b: _cue_preprocess(imgs, b, patch_size=model.patch_size)

    for epoch in range(1, d["epochs"] + 1):
        model.train()
        for batch in train_loader:
            images = batch["image"].to(dev)
            targets = batch["keypoints"].to(dev)
            vis = batch["visibility"].to(dev)
            render_runway_cue(images, batch["runway_side"].to(dev), patch_size=model.patch_size)
            pred = model(images)
            loss = masked_huber(pred, targets, vis, beta=delta)

            optimizer.zero_grad()
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            _schedule_lr(optimizer, base_lrs, step, total_steps, warmup_steps)
            optimizer.step()

            if step % log_every == 0:
                lr_now = optimizer.param_groups[0]["lr"]
                print(f"[TRAIN] step={step} ep={epoch}/{d['epochs']} loss={loss.item():.4f} "
                      f"lr={lr_now:.2e} [{int(time.time() - t_start)}s]", flush=True)
            step += 1

        train_m = evaluate(model, train_eval_loader, dev, preprocess_fn=preprocess)
        test_m  = evaluate(model, val_loader,        dev, preprocess_fn=preprocess)
        trace.append({"epoch": epoch, "train": train_m, "test": test_m})
        print(f"[TRAIN] ep={epoch}  "
              f"train_median={train_m['median_px']:.2f}px train_mae={train_m['mae_px']:.2f}px  "
              f"test_median={test_m['median_px']:.2f}px test_mae={test_m['mae_px']:.2f}px",
              flush=True)

    # Final-epoch checkpoint; no test-set best selection.
    final = trace[-1] if trace else {"train": {}, "test": {}}
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": {"backbone": backbone, "variant": variant,
                   "huber_delta_px": huber_delta_px, "seed": seed},
        "epoch": d["epochs"],
        "metrics": final,
    }, ckpt_path)
    print(f"[TRAIN] saved final {ckpt_path} "
          f"(test_median={final['test'].get('median_px', float('nan')):.2f}px)", flush=True)

    out_json = results_dir / f"{run_name}_train.json"
    out_json.write_text(json.dumps({
        "run_name": run_name, "variant": variant, "seed": seed,
        "epochs": d["epochs"],
        "final": {"train": final["train"], "test": final["test"]},
        "trace": trace,
        "total_seconds": int(time.time() - t_start),
    }, indent=2))
    print(f"[TRAIN] done. {out_json}", flush=True)


if __name__ == "__main__":
    app()
