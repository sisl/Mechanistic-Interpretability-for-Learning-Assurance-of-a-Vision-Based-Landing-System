"""LARDv2 loader with BOGO crop (main.typ @sec:bogo).

Scope: only what the paper consumes.

- ``LARDv2Dataset(subset, split, ood=False)`` materializes BOGO-cropped
  samples. Train uses a random valid crop (data augmentation); val and test
  use the deterministic center of the valid range so metrics are reproducible.
- ``LARDv2Dataset(..., ood=True)`` produces inverse-BOGO crops where at
  least one runway corner falls outside the crop window: these are the
  negatives for the IMS classifier of main.typ @sec:ims-lr.
- ``LARDTrainDataset`` wraps a LARDv2Dataset into a torch Dataset with
  resize + ImageNet normalization + batchable runway_side index.

The loader is intentionally eager: we materialize all samples up front so
downstream analyses can walk the test set in a deterministic order. The
class supports ``max_samples`` for dry-run debugging.
"""

from __future__ import annotations

import math
import os
import random
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset

from .constants import IMAGENET_MEAN, IMAGENET_STD, IMG_SIZE


LARD_KEYPOINT_NAMES = ["top_left", "top_right", "bottom_left", "bottom_right"]
LARD_SKELETON = [(0, 1), (1, 3), (3, 2), (2, 0)]  # TL-TR-BR-BL quadrilateral
LARDV2_SUBSETS = ["xplane", "ges", "arcgis", "bingmaps"]

_HF_DATASET = "DEEL-AI/LARD_V2"


@dataclass
class KeypointSample:
    """BOGO-cropped image with 4 normalized runway corners."""
    image: np.ndarray       # [H, W, 3] uint8
    keypoints: np.ndarray   # [4, 2] float32 in [0, 1]
    visibility: np.ndarray  # [4] uint8 (0 = absent, 2 = visible)
    image_id: Any
    metadata: dict = field(default_factory=dict)


# ----------------------------------------------------------------------
# Crop helpers
# ----------------------------------------------------------------------

def _parse_runway_side(runway: str | None) -> str:
    """Extract L/R/C suffix from a runway designator (e.g. '08R' -> 'R')."""
    if not runway:
        return ""
    runway = runway.strip().upper()
    if runway and runway[-1] in ("L", "R", "C"):
        return runway[-1]
    return ""


def _fixup_corners(xs: np.ndarray, ys: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Ensure TL.x <= TR.x and BL.x <= BR.x so the corner order is canonical."""
    xs, ys = xs.copy(), ys.copy()
    if xs[0] > xs[1]:
        xs[0], xs[1] = xs[1], xs[0]; ys[0], ys[1] = ys[1], ys[0]
    if xs[2] > xs[3]:
        xs[2], xs[3] = xs[3], xs[2]; ys[2], ys[3] = ys[3], ys[2]
    return xs, ys


def _compute_scale(bbox_w: float, bbox_h: float, crop_size: int, margin: int) -> float:
    """Power-of-0.5 scale factor to fit runway bbox + margin in crop_size."""
    max_dim = max(bbox_w + margin, bbox_h + margin)
    if max_dim <= crop_size:
        return 1.0
    n = math.ceil(math.log2(max_dim / crop_size))
    return 0.5 ** n


def _sample_crop_xy(xs: np.ndarray, ys: np.ndarray, H: int, W: int,
                    crop_size: int, deterministic: bool) -> tuple[int, int]:
    """Pick a crop origin that keeps all corners inside the window.

    deterministic=True returns the center of the valid range; False returns a
    random valid position (train-time augmentation).
    """
    x_lo = max(0, int(np.ceil(xs.max())) - crop_size + 1)
    x_hi = min(W - crop_size, int(np.floor(xs.min())))
    y_lo = max(0, int(np.ceil(ys.max())) - crop_size + 1)
    y_hi = min(H - crop_size, int(np.floor(ys.min())))

    if x_lo > x_hi or y_lo > y_hi:
        # Degenerate: fall back to centering the runway bbox.
        cx = max(0, min(W - crop_size, int((xs.min() + xs.max()) / 2 - crop_size / 2)))
        cy = max(0, min(H - crop_size, int((ys.min() + ys.max()) / 2 - crop_size / 2)))
        return cx, cy
    if deterministic:
        return (x_lo + x_hi) // 2, (y_lo + y_hi) // 2
    return random.randint(x_lo, x_hi), random.randint(y_lo, y_hi)


def _bogocrop(sample: dict, output_size: int, margin: int,
              deterministic: bool) -> KeypointSample:
    """Scale + crop around the runway; return the crop and [0, 1]-normalized corners."""
    img: Image.Image = sample["image"]
    orig_w, orig_h = sample["width"], sample["height"]

    xs = np.array([sample["x_TL"], sample["x_TR"], sample["x_BL"], sample["x_BR"]], dtype=np.float64)
    ys = np.array([sample["y_TL"], sample["y_TR"], sample["y_BL"], sample["y_BR"]], dtype=np.float64)
    xs, ys = _fixup_corners(xs, ys)

    scale = _compute_scale(xs.max() - xs.min(), ys.max() - ys.min(), output_size, margin)
    scaled_xs, scaled_ys = xs * scale, ys * scale
    scaled_w = max(int(orig_w * scale), output_size)
    scaled_h = max(int(orig_h * scale), output_size)

    crop_x, crop_y = _sample_crop_xy(scaled_xs, scaled_ys, scaled_h, scaled_w,
                                     output_size, deterministic)

    crop_size_orig = output_size / scale
    x1 = max(0, int(crop_x / scale)); y1 = max(0, int(crop_y / scale))
    x2 = min(orig_w, int(crop_x / scale + crop_size_orig))
    y2 = min(orig_h, int(crop_y / scale + crop_size_orig))
    image_np = np.array(img.crop((x1, y1, x2, y2)).resize((output_size, output_size)).convert("RGB"),
                        dtype=np.uint8)

    norm_xs = (scaled_xs - crop_x) / output_size
    norm_ys = (scaled_ys - crop_y) / output_size
    keypoints = np.stack([norm_xs, norm_ys], axis=-1).astype(np.float32)

    return KeypointSample(
        image=image_np, keypoints=keypoints,
        visibility=np.full(4, 2, dtype=np.uint8),
        image_id=sample.get("image_id", ""),
        metadata={"runway_side": _parse_runway_side(sample.get("runway")),
                  "runway": sample.get("runway", ""),
                  "airport": sample.get("airport", "")},
    )


def _inverse_bogocrop(sample: dict, output_size: int, margin: int,
                      max_attempts: int = 100) -> KeypointSample | None:
    """Same scale as BOGO, but crop where all four runway corners stay outside.

    Used as the IMS-negative class in main.typ @sec:ims-lr. We first try a
    bounded number of random proposals and then fall back to a deterministic
    grid scan so the search always terminates. Returns None if no such crop
    exists at the chosen scale.
    """
    img: Image.Image = sample["image"]
    orig_w, orig_h = sample["width"], sample["height"]

    xs = np.array([sample["x_TL"], sample["x_TR"], sample["x_BL"], sample["x_BR"]], dtype=np.float64)
    ys = np.array([sample["y_TL"], sample["y_TR"], sample["y_BL"], sample["y_BR"]], dtype=np.float64)
    xs, ys = _fixup_corners(xs, ys)

    scale = _compute_scale(xs.max() - xs.min(), ys.max() - ys.min(), output_size, margin)
    scaled_xs, scaled_ys = xs * scale, ys * scale
    scaled_w = max(int(orig_w * scale), output_size)
    scaled_h = max(int(orig_h * scale), output_size)

    max_cx = max(0, scaled_w - output_size)
    max_cy = max(0, scaled_h - output_size)
    if max_cx == 0 and max_cy == 0:
        return None

    def _inside_mask(crop_x: int, crop_y: int) -> np.ndarray:
        return ((scaled_xs >= crop_x) & (scaled_xs < crop_x + output_size)
                & (scaled_ys >= crop_y) & (scaled_ys < crop_y + output_size))

    def _materialize(crop_x: int, crop_y: int, inside: np.ndarray) -> KeypointSample:
        crop_size_orig = output_size / scale
        x1 = max(0, int(crop_x / scale)); y1 = max(0, int(crop_y / scale))
        x2 = min(orig_w, int(crop_x / scale + crop_size_orig))
        y2 = min(orig_h, int(crop_y / scale + crop_size_orig))
        image_np = np.array(img.crop((x1, y1, x2, y2)).resize((output_size, output_size)).convert("RGB"),
                            dtype=np.uint8)
        norm_xs = (scaled_xs - crop_x) / output_size
        norm_ys = (scaled_ys - crop_y) / output_size
        keypoints = np.stack([norm_xs, norm_ys], axis=-1).astype(np.float32)
        visibility = np.where(inside, 2, 0).astype(np.uint8)
        return KeypointSample(
            image=image_np, keypoints=keypoints, visibility=visibility,
            image_id=sample.get("image_id", ""),
            metadata={"runway_side": _parse_runway_side(sample.get("runway"))},
        )

    for _ in range(max_attempts):
        crop_x = random.randint(0, max_cx)
        crop_y = random.randint(0, max_cy)
        inside = _inside_mask(crop_x, crop_y)
        if inside.any():
            continue
        return _materialize(crop_x, crop_y, inside)

    xs_scan = np.unique(np.linspace(0, max_cx, num=min(max_cx + 1, 33), dtype=int))
    ys_scan = np.unique(np.linspace(0, max_cy, num=min(max_cy + 1, 33), dtype=int))
    for crop_x in xs_scan:
        for crop_y in ys_scan:
            inside = _inside_mask(int(crop_x), int(crop_y))
            if inside.any():
                continue
            return _materialize(int(crop_x), int(crop_y), inside)
    return None


# ----------------------------------------------------------------------
# Dataset classes
# ----------------------------------------------------------------------

def _load_samples(subset: str, split: str, max_samples: int | None,
                  cache_dir: str | None, output_size: int, margin: int,
                  deterministic_crop: bool, ood: bool) -> list[KeypointSample]:
    from datasets import load_dataset
    # Stream when a small slice is enough to avoid materializing the whole
    # 15k-sample Parquet split during dry-run debugging.
    streaming = bool(max_samples) and max_samples <= 500
    ds = load_dataset(_HF_DATASET, name=subset, split=split,
                      cache_dir=cache_dir, streaming=streaming)
    tag = "LARD-OOD" if ood else "LARD"
    samples: list[KeypointSample] = []
    skipped = 0
    for i, row in enumerate(ds):
        if max_samples and len(samples) >= max_samples:
            break
        if ood:
            result = _inverse_bogocrop(row, output_size, margin)
            if result is None:
                skipped += 1
                continue
            sample = result
        else:
            sample = _bogocrop(row, output_size, margin, deterministic=deterministic_crop)
        sample.image_id = f"lard_{subset}_{split}_{i}"
        samples.append(sample)
    msg = f"[{tag}] {subset}/{split}: {len(samples)} samples"
    if skipped:
        msg += f" ({skipped} skipped)"
    print(msg, flush=True)
    return samples


class LARDv2Dataset:
    """LARDv2 with BOGO crop on every split.

    Training uses a random BOGO crop position (data augmentation). Validation
    keeps the deterministic center-of-valid-range crop, while test uses the
    same sampled BOGO construction as training. Reproducibility for the test
    split therefore comes from the caller seeding the RNG before dataset
    construction. ood=True flips to inverse-BOGO crops.

    ``subset_sizes`` records how many samples came from each subset, in
    the same order as ``subset_names``. Downstream scripts use this to
    build per-subset activation-rate histograms without reading the LARD
    split sizes from a hardcoded table.
    """

    NUM_KEYPOINTS = 4
    keypoint_names = LARD_KEYPOINT_NAMES
    skeleton = LARD_SKELETON

    def __init__(
        self,
        subset: str = "all",
        split: str = "train",
        max_samples: int | None = None,
        output_size: int = IMG_SIZE,
        margin: int = 50,
        cache_dir: str | None = None,
        ood: bool = False,
    ):
        if cache_dir is None:
            cache_dir = os.environ.get("LARD_CACHE_DIR")
        deterministic_crop = (split == "val")
        subsets = LARDV2_SUBSETS if subset == "all" else [subset]
        self._samples: list[KeypointSample] = []
        self.subset_names: list[str] = []
        self.subset_sizes: list[int] = []
        for s in subsets:
            before = len(self._samples)
            self._samples.extend(_load_samples(
                s, split, max_samples, cache_dir, output_size, margin,
                deterministic_crop, ood,
            ))
            added = len(self._samples) - before
            if added > 0:
                self.subset_names.append(s)
                self.subset_sizes.append(added)
        tag = "LARD-OOD" if ood else "LARD"
        print(f"[{tag}] Total: {len(self._samples)} samples "
              f"(subset={subset}, split={split})", flush=True)

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> KeypointSample:
        return self._samples[idx]


_SIDE_TO_INT = {"": 0, "L": 1, "R": 2, "C": 3}


class LARDTrainDataset(Dataset):
    """Adapter that turns KeypointSample into a torch-batchable dict.

    Resizes to ``image_size``, applies ImageNet normalization, and exposes the
    runway_side as an int so the cue painter can act on a batched tensor.
    """

    def __init__(self, dataset: LARDv2Dataset, image_size: int = IMG_SIZE):
        self.dataset = dataset
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int):
        sample = self.dataset[idx]
        img = torch.from_numpy(sample.image).permute(2, 0, 1).float() / 255.0
        img = F.interpolate(img.unsqueeze(0), size=(self.image_size, self.image_size),
                            mode="bilinear", align_corners=False).squeeze(0)
        img = (img - IMAGENET_MEAN[:, None, None]) / IMAGENET_STD[:, None, None]
        side = sample.metadata.get("runway_side", "")
        return {
            "image": img,
            "keypoints": torch.from_numpy(sample.keypoints),
            "visibility": torch.from_numpy(sample.visibility.astype(np.int64)),
            "runway_side": torch.tensor(_SIDE_TO_INT.get(side, 0), dtype=torch.long),
        }
