"""Python fallback for the Julia K-SVD / sparse-code steps under DRY=1.

Used so local CPU smoke tests can exercise the full pipeline without a
Julia install. Produces:

  ksvd  <embeddings.h5> <out.npy>
  codes <embeddings.h5> <dict.npy> <out_codes.h5>

Implements a minimally-adequate dictionary learning via sklearn's
MiniBatchDictionaryLearning and OMP encoding via sklearn.sparse_encode.
Output artifacts have the exact same schema as the Julia scripts, so
downstream atoms.py reads them identically. The dictionary quality is
irrelevant under DRY; we only check pipeline wiring.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import h5py
import numpy as np
import typer
from sklearn.decomposition import MiniBatchDictionaryLearning, sparse_encode


app = typer.Typer(add_completion=False)


@app.command("ksvd")
def ksvd(embeddings: Path, out: Path,
         dict_size: Annotated[int, typer.Option()] = 64,
         nnz: Annotated[int, typer.Option()] = 8):
    """Mini-batch dictionary learning stub (not the paper's K-SVD; DRY only)."""
    with h5py.File(embeddings, "r") as f:
        X = f["embeddings"][:]                                          # [N, D]
    # MBDL wants [N, n_features].
    est = MiniBatchDictionaryLearning(
        n_components=dict_size, transform_n_nonzero_coefs=nnz,
        batch_size=128, max_iter=20, random_state=0,
    ).fit(X)
    D = est.components_.T.astype(np.float32)                            # [D_feat, dict_size]
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, D)
    print(f"[DRY-KSVD] wrote {out} shape={D.shape}")


@app.command("codes")
def codes(embeddings: Path, dictionary: Path, out: Path,
          nnz: Annotated[int, typer.Option()] = 8,
          chunk: Annotated[int, typer.Option()] = 2048):
    """OMP-encode each patch token, fixed-nnz h5 (paper schema)."""
    # The paper convention is D: (D_feat, M); we verify against the embeddings'
    # feature dim rather than guessing from shape ratios (which flip in DRY).
    D = np.load(dictionary)
    with h5py.File(embeddings, "r") as f_in:
        d_probe = f_in["embeddings"].shape[1]
    if D.shape[0] != d_probe:
        D = D.T
    assert D.shape[0] == d_probe, f"dict {D.shape} / embeddings feat={d_probe}"
    d_feat, M = D.shape
    with h5py.File(embeddings, "r") as f_in:
        ds = f_in["embeddings"]
        N = ds.shape[0]
        out.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(out, "w") as f_out:
            ds_idx = f_out.create_dataset("indices", shape=(nnz, N), dtype=np.int16)
            ds_val = f_out.create_dataset("values",  shape=(nnz, N), dtype=np.float32)
            for name in ("subset_names", "subset_sizes", "patches_per_image",
                         "num_images", "skipped_patches"):
                if name in f_in.attrs:
                    f_out.attrs[name] = f_in.attrs[name]
            f_out.attrs["num_atoms"] = M

            for s in range(0, N, chunk):
                e = min(s + chunk, N)
                X = ds[s:e]                                              # [chunk, D_feat]
                C = sparse_encode(X.astype(np.float64), D.T.astype(np.float64),
                                  n_nonzero_coefs=nnz, algorithm="omp")  # [chunk, M]
                for j in range(e - s):
                    nz = np.flatnonzero(C[j])
                    for r in range(nnz):
                        if r < len(nz):
                            ds_idx[r, s + j] = nz[r]
                            ds_val[r, s + j] = C[j, nz[r]]
                        else:
                            ds_idx[r, s + j] = 0
                            ds_val[r, s + j] = 0.0
    print(f"[DRY-CODES] wrote {out} (N={N}, M={M}, nnz={nnz})")


if __name__ == "__main__":
    app()
