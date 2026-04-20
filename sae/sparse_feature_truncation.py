#!/usr/bin/env python3
"""
truncate_sparse_features_npz.py

Recursively finds all files named "X_features.npz" under a root directory,
loads them as SciPy sparse matrices (kept sparse, no densifying),
applies filtering:
    keep column j if n*pl < count(X[:,j] > 0) < n*ph
and saves:
    X_features_truncated.npz   (CSR)
optionally:
    X_features_truncated_idx.npy

Default behavior: skip if output exists (no overwrite unless --overwrite).
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import scipy.sparse as sp


def count_positive_per_col_sparse(X: sp.spmatrix) -> np.ndarray:
    """
    Return counts[j] = number of i such that X[i,j] > 0.

    Stays sparse. Approach:
      - convert to CSC (column-friendly)
      - make a sparse matrix with data = 1 where original data > 0
      - eliminate zeros
      - counts = nnz per column = diff(indptr)
    """
    X_csc = X.tocsc(copy=False)

    X_pos = X_csc.copy()
    X_pos.data = (X_pos.data > 0).astype(np.int8)
    X_pos.eliminate_zeros()

    counts = np.diff(X_pos.indptr).astype(np.int64)  # length = n_cols
    return counts


def filter_out_polysemantic_and_noise_sparse(X: sp.spmatrix, ph: float = 0.1, pl: float = 0.00001, verbose: bool = True):
    """
    Sparse analog of:

        binary = (X>0)*1.0
        z = binary.sum(axis=0)
        idx = (z < n*ph) & (z > n*pl)
        return X[:, idx]

    Returns:
      X_trunc (CSR), keep_idx (np.int64 indices)
    """
    if X.ndim != 2:
        raise ValueError(f"Expected 2D sparse matrix, got shape {X.shape}")

    n, d = X.shape
    counts = count_positive_per_col_sparse(X)  # (# > 0) per column

    # Match your original integer-ish behavior (n*ph, n*pl).
    # Using float thresholds directly is fine; keeping it literal.
    keep = (counts < n * ph) & (counts > n * pl)
    keep_idx = np.nonzero(keep)[0].astype(np.int64)

    if verbose:
        print(f"    shape={X.shape}, nnz={X.nnz}")
        print(f"    keep {keep_idx.size}/{d} cols (ph={ph}, pl={pl})")

    # Column slicing is faster in CSC, then convert back to CSR as requested.
    X_trunc = X.tocsc(copy=False)[:, keep_idx].tocsr(copy=False)
    return X_trunc, keep_idx


def find_x_features_npz(root: Path) -> list[Path]:
    hits: list[Path] = []
    for dirpath, _, filenames in os.walk(root):
        if "X_features.npz" in filenames:
            hits.append(Path(dirpath) / "X_features.npz")
    return sorted(hits)


def process_one_file(
    x_path: Path,
    ph: float,
    pl: float,
    overwrite: bool,
    save_idx: bool,
    verbose: bool,
) -> bool:
    out_path = x_path.with_name("X_features_truncated.npz")
    idx_path = x_path.with_name("X_features_truncated_idx.npy")

    if not overwrite and (out_path.exists() or (save_idx and idx_path.exists())):
        if verbose:
            print(f"SKIP (exists): {x_path}")
        return False

    if verbose:
        print(f"\nPROCESS: {x_path}")

    X = sp.load_npz(x_path)
    if not sp.isspmatrix(X):
        raise ValueError(f"Loaded object is not a SciPy sparse matrix: {x_path}")

    # Ensure CSR internal form for your pipelines (we still use CSC for slicing/counting internally).
    X = X.tocsr(copy=False)

    X_trunc, keep_idx = filter_out_polysemantic_and_noise_sparse(X, ph=ph, pl=pl, verbose=verbose)

    sp.save_npz(out_path, X_trunc, compressed=True)
    if verbose:
        print(f"    saved truncated -> {out_path} (CSR, nnz={X_trunc.nnz})")

    if save_idx:
        np.save(idx_path, keep_idx)
        if verbose:
            print(f"    saved kept indices -> {idx_path}")

    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, required=True, help="Root folder to search under.")
    ap.add_argument("--ph", type=float, default=0.1, help="Upper fraction threshold (keep if count < n*ph).")
    ap.add_argument("--pl", type=float, default=0.00001, help="Lower fraction threshold (keep if count > n*pl).")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs.")
    ap.add_argument("--save_idx", action="store_true", help="Also save kept column indices as *_idx.npy.")
    ap.add_argument("--quiet", action="store_true", help="Less printing.")
    args = ap.parse_args()

    root = Path(args.root).expanduser().resolve()
    verbose = not args.quiet

    if not root.exists():
        raise FileNotFoundError(f"--root does not exist: {root}")

    paths = find_x_features_npz(root)
    if verbose:
        print(f"Found {len(paths)} files named X_features.npz under {root}")

    processed = 0
    skipped = 0
    for p in paths:
        did = process_one_file(
            p,
            ph=args.ph,
            pl=args.pl,
            overwrite=args.overwrite,
            save_idx=args.save_idx,
            verbose=verbose,
        )
        processed += int(did)
        skipped += int(not did)

    print(f"\nDONE. processed={processed}, skipped={skipped}, total_found={len(paths)}")


if __name__ == "__main__":
    main()