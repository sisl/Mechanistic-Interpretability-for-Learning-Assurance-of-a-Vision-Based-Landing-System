"""Constrained L1 IMS classifier lambda-sweep (main.typ @sec:ims-lr, @sec:bogo-res).

Fits @eq:lr_l1: minimize BCE(sigmoid(b + w^T a), y) + lambda * sum_j w_j,
with bounds w_j >= 0 and b <= 0, via L-BFGS-B for each lambda in a grid.
Reports, per lambda: M (number of surviving atoms), AUROC on a held-out
split, content-fraction (share of surviving atoms labeled content by @sec:content-style),
and the learned bias b.

Features are binary attention-pooled activations produced by
`scripts/extract_attn_pooled.py`: bogo_codes (IMS-positive) and
inverse_bogo_codes (IMS-negative).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import numpy as np
import typer
from scipy.optimize import minimize


app = typer.Typer(add_completion=False)


def _auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    order = np.argsort(scores)
    sl = labels[order]
    n_pos = int(sl.sum()); n_neg = len(sl) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5
    ranks = np.where(sl == 1)[0].sum() + n_pos
    return float((ranks - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def _fit_nonneg_l1(X: np.ndarray, y: np.ndarray, lam: float,
                   bias_init: float = -4.0, max_iter: int = 500):
    """Box-constrained L-BFGS-B fit of @eq:lr_l1 (non-positive bias, w >= 0, L1)."""
    N, m = X.shape

    def obj(theta):
        w, b = theta[:m], theta[m]
        z = X @ w + b
        log1pexp = np.maximum(0.0, z) + np.log1p(np.exp(-np.abs(z)))
        nll = (log1pexp - y * z).sum()
        p = 1.0 / (1.0 + np.exp(-z))
        r = p - y
        grad_w = X.T @ r + lam                             # w >= 0 so |w| = w
        grad_b = r.sum()
        return nll + lam * w.sum(), np.concatenate([grad_w, [grad_b]])

    theta0 = np.zeros(m + 1)
    theta0[m] = bias_init
    bounds = [(0.0, None)] * m + [(None, 0.0)]
    res = minimize(obj, theta0, jac=True, method="L-BFGS-B",
                   bounds=bounds, options={"maxiter": max_iter, "ftol": 1e-9})
    w, b = res.x[:m], float(res.x[m])
    return w, b


@app.command()
def main(
    codes: Annotated[Path, typer.Option(help=".npz with bogo_codes + inverse_bogo_codes")],
    atoms: Annotated[Path, typer.Option(help="atoms.json from scripts/atoms.py")],
    out: Annotated[Path, typer.Option()],
    seed: Annotated[int, typer.Option()] = 0,
    train_frac: Annotated[float, typer.Option()] = 0.7,
    lambdas: Annotated[str, typer.Option()] = "0.1,0.3,1,3,10,30,100,300,1000",
):
    import os
    dry = bool(os.environ.get("DRY"))

    data = np.load(codes)
    pos = (data["bogo_codes"] != 0).astype(np.float64)
    neg = (data["inverse_bogo_codes"] != 0).astype(np.float64)
    cs = json.loads(Path(atoms).read_text())
    cv = np.asarray(cs["cv"])
    median_cv = float(cs.get("median_cv", np.median(cv[cv > 0])))
    is_content = cv <= median_cv

    X = np.concatenate([pos, neg])
    y = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(X))
    cut = int(train_frac * len(X))
    tr, va = perm[:cut], perm[cut:]

    grid = [float(x) for x in lambdas.split(",")]
    if dry:
        grid = grid[::2][:3]  # coarse grid for DRY

    out_data = {"seed": seed, "lambdas": grid, "nz": [], "auroc": [],
                "content_frac": [], "bias": []}
    for lam in grid:
        w, b = _fit_nonneg_l1(X[tr], y[tr], lam)
        active = w > 1e-8
        M = int(active.sum())
        scores = X[va] @ w + b
        au = _auroc(scores, y[va])
        cf = float(is_content[active].mean()) if M > 0 else 0.0
        out_data["nz"].append(M)
        out_data["auroc"].append(au)
        out_data["content_frac"].append(cf)
        out_data["bias"].append(b)
        print(f"[IMS] lam={lam:>7.2f}  M={M:>4d}  AUROC={au:.4f}  content={cf*100:.1f}%  b={b:.2f}", flush=True)

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(out_data, indent=2))
    print(f"[IMS] wrote {out}")


if __name__ == "__main__":
    app()
