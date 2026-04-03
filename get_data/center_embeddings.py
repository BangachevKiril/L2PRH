#!/usr/bin/env python3
import argparse
import os
from pathlib import Path
import numpy as np


TARGET_FILES = {"img_embeddings.npy", "text_embeddings.npy"}


def row_l2_normalize(x: np.ndarray, eps: float) -> np.ndarray:
    # x: (B, D)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / (norms + eps)


def process_file(path: Path, eps: float, chunk_rows: int, overwrite: bool, verbose: bool) -> None:
    name = path.name
    if name not in TARGET_FILES:
        return

    out_path = path.with_name(path.stem + "_normalized.npy")  # img_embeddings_normalized.npy
    if out_path.exists() and not overwrite:
        if verbose:
            print(f"[skip] exists: {out_path}")
        return

    # Load as memmap when possible (still fine with allow_pickle=True)
    X = np.load(path, allow_pickle=True, mmap_mode="r")
    if X.ndim != 2:
        raise ValueError(f"{path} expected 2D array [N,D], got shape {X.shape}")

    N, D = int(X.shape[0]), int(X.shape[1])
    if verbose:
        print(f"[load] {path}  shape=({N},{D}) dtype={X.dtype}")

    # ---------- Pass 1: compute mu from row-normalized rows ----------
    sum_vec = np.zeros((D,), dtype=np.float64)
    seen = 0
    for i in range(0, N, chunk_rows):
        j = min(N, i + chunk_rows)
        chunk = np.asarray(X[i:j], dtype=np.float32)          # (B,D)
        chunk = row_l2_normalize(chunk, eps=eps)              # normalize rows
        sum_vec += chunk.sum(axis=0, dtype=np.float64)
        seen += (j - i)

    if seen != N:
        raise RuntimeError(f"Internal error: saw {seen} rows, expected {N}")

    mu = (sum_vec / float(N)).astype(np.float32)              # (D,)

    # ---------- Pass 2: write centered + renormalized rows ----------
    out_mm = np.lib.format.open_memmap(out_path, mode="w+", dtype=np.float32, shape=(N, D))

    for i in range(0, N, chunk_rows):
        j = min(N, i + chunk_rows)
        chunk = np.asarray(X[i:j], dtype=np.float32)
        chunk = row_l2_normalize(chunk, eps=eps)              # normalize rows (again, per spec)
        chunk = chunk - mu[None, :]                           # center
        chunk = row_l2_normalize(chunk, eps=eps)              # renormalize rows
        out_mm[i:j, :] = chunk

    out_mm.flush()
    if verbose:
        print(f"[save] {out_path}")


def main():
    ap = argparse.ArgumentParser(
        description="Recursively normalize+center COCO embedding .npy files (img_embeddings.npy/text_embeddings.npy)."
    )
    ap.add_argument("Dir", type=str, help="Root directory to search recursively.")
    ap.add_argument("--eps", type=float, default=1e-12, help="Numerical epsilon for L2 normalization.")
    ap.add_argument("--chunk_rows", type=int, default=100_000, help="Rows per chunk (memory control).")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing *_normalized.npy outputs.")
    ap.add_argument("--verbose", action="store_true", help="Print progress.")
    args = ap.parse_args()

    root = Path(args.Dir).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(root)

    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if fn in TARGET_FILES:
                path = Path(dirpath) / fn
                try:
                    process_file(
                        path=path,
                        eps=args.eps,
                        chunk_rows=args.chunk_rows,
                        overwrite=args.overwrite,
                        verbose=args.verbose,
                    )
                except Exception as e:
                    print(f"[error] {path}: {e}")


if __name__ == "__main__":
    main()
