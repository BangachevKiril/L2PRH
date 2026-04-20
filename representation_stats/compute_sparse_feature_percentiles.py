#!/usr/bin/env python3

import argparse
import os
from typing import Optional

import numpy as np
from scipy import sparse


FULL_FILE = "X_features.npz"
TRUNC_FILE = "X_features_truncated.npz"
B_DEC_FILE = "b_dec.npy"
OUT_FILE = "sparse_features_statistics.npz"


def compute_percentiles(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      percentiles: array of percentile levels [5, 10, ..., 95]
      values: corresponding percentile values of x
    """
    qs = np.arange(5, 100, 5, dtype=np.int64)
    if x.size == 0:
        vals = np.full(qs.shape, np.nan, dtype=np.float64)
    else:
        vals = np.percentile(x, qs)
    return qs, vals


def safe_summary_stats(x: np.ndarray) -> tuple[float, float, float]:
    if x.size == 0:
        return np.nan, np.nan, np.nan
    return float(np.min(x)), float(np.max(x)), float(np.mean(x))


def rounded_nnz_per_row(M: sparse.spmatrix) -> int:
    if M.shape[0] == 0:
        return 0
    return int(np.rint(M.nnz / M.shape[0]))


def exact_or_mean_nnz_per_row(M: sparse.spmatrix) -> float:
    """
    If every row has the same nnz, return that integer.
    Otherwise return the mean nnz per row.
    """
    if M.shape[0] == 0:
        return 0.0
    row_nnz = np.diff(M.indptr)
    if row_nnz.size == 0:
        return 0.0
    if np.all(row_nnz == row_nnz[0]):
        return int(row_nnz[0])
    return float(np.mean(row_nnz))


def process_folder(folder: str, skip_existing: bool = False, verbose: bool = True) -> bool:
    full_path = os.path.join(folder, FULL_FILE)
    trunc_path = os.path.join(folder, TRUNC_FILE)
    b_dec_path = os.path.join(folder, B_DEC_FILE)
    out_path = os.path.join(folder, OUT_FILE)

    if not (os.path.isfile(full_path) and os.path.isfile(trunc_path)):
        return False

    if not os.path.isfile(b_dec_path):
        if verbose:
            print(f"[skip] missing {B_DEC_FILE}: {folder}")
        return False

    if skip_existing and os.path.isfile(out_path):
        if verbose:
            print(f"[skip] already exists: {out_path}")
        return False

    try:
        F = sparse.load_npz(full_path)
        F_t = sparse.load_npz(trunc_path)
    except Exception as e:
        print(f"[error] failed to load sparse features in {folder}: {e}")
        return False

    try:
        b = np.load(b_dec_path, allow_pickle=False)
    except Exception as e:
        print(f"[error] failed to load {B_DEC_FILE} in {folder}: {e}")
        return False

    if not sparse.isspmatrix(F):
        print(f"[error] {full_path} did not load as a scipy sparse matrix")
        return False
    if not sparse.isspmatrix(F_t):
        print(f"[error] {trunc_path} did not load as a scipy sparse matrix")
        return False
    if b.ndim == 0:
        print(f"[error] {b_dec_path} has unexpected scalar shape")
        return False

    # Nonzero values only
    A = F.data
    A_t = F_t.data

    pct_levels, A_percentiles = compute_percentiles(A)
    _, A_t_percentiles = compute_percentiles(A_t)

    A_min, A_max, A_mean = safe_summary_stats(A)
    A_t_min, A_t_max, A_t_mean = safe_summary_stats(A_t)

    dense_dimension = int(b.shape[0])
    sparse_dimension_full = int(F.shape[1])
    sparse_dimension_truncated = int(F_t.shape[1])

    full_sparsity = rounded_nnz_per_row(F)
    truncated_sparsity = exact_or_mean_nnz_per_row(F_t)

    try:
        np.savez(
            out_path,
            percentile_levels=pct_levels,

            full_percentiles=A_percentiles,
            truncated_percentiles=A_t_percentiles,

            full_min=np.array(A_min, dtype=np.float64),
            full_max=np.array(A_max, dtype=np.float64),
            full_mean=np.array(A_mean, dtype=np.float64),

            truncated_min=np.array(A_t_min, dtype=np.float64),
            truncated_max=np.array(A_t_max, dtype=np.float64),
            truncated_mean=np.array(A_t_mean, dtype=np.float64),

            dense_dimension=np.array(dense_dimension, dtype=np.int64),
            sparse_dimension_full=np.array(sparse_dimension_full, dtype=np.int64),
            sparse_dimension_truncated=np.array(sparse_dimension_truncated, dtype=np.int64),

            full_sparsity=np.array(full_sparsity, dtype=np.int64),
            truncated_sparsity=np.array(truncated_sparsity),
        )
    except Exception as e:
        print(f"[error] failed to save {out_path}: {e}")
        return False

    if verbose:
        print(f"[ok] saved: {out_path}")

    return True


def walk_and_process(root: str, skip_existing: bool = False, verbose: bool = True) -> int:
    count = 0
    for dirpath, dirnames, filenames in os.walk(root):
        if process_folder(dirpath, skip_existing=skip_existing, verbose=verbose):
            count += 1
    return count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Recursively search for folders containing both "
            "'X_features.npz' and 'X_features_truncated.npz', compute "
            "summary statistics of their nonzero values, and save them as "
            "'sparse_features_statistics.npz'."
        )
    )
    parser.add_argument(
        "root",
        type=str,
        help="Root directory to recursively search."
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip folders where sparse_features_statistics.npz already exists."
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-folder logging."
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    root = os.path.abspath(args.root)
    if not os.path.isdir(root):
        raise FileNotFoundError(f"Root folder not found: {root}")

    verbose = not args.quiet

    print(f"ROOT={root}")
    print(f"SKIP_EXISTING={int(args.skip_existing)}")

    n = walk_and_process(root, skip_existing=args.skip_existing, verbose=verbose)
    print(f"Done. Processed {n} folder(s).")


if __name__ == "__main__":
    main()