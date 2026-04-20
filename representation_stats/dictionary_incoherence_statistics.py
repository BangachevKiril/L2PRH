#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import scipy.sparse as sp
import torch


TORCH_DTYPE_MAP = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
    "float64": torch.float64,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Recursively find folders containing decoder_weight.npy and "
            "X_features_truncated_idx.npy, then compute GPU Gram-matrix "
            "off-diagonal statistics and save them as incoherence_statistics.npz"
        )
    )
    parser.add_argument("root", type=Path, help="Root folder to scan recursively")
    parser.add_argument(
        "--device",
        default="cuda",
        help="Torch device to use (default: cuda)",
    )
    parser.add_argument(
        "--dtype",
        choices=sorted(TORCH_DTYPE_MAP.keys()),
        default="float32",
        help="Computation dtype for W on device (default: float32)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=4096,
        help=(
            "Row/column block size for Gram computation. Reduce this if you hit "
            "GPU OOM. Default: 4096"
        ),
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip folders where incoherence_statistics.npz already exists",
    )
    return parser.parse_args()


@torch.inference_mode()
def offdiag_gram_stats(
    W: torch.Tensor,
    idx: Optional[torch.Tensor] = None,
    chunk_size: int = 4096,
) -> Dict[str, float]:
    """
    Compute exact off-diagonal statistics of X @ X.T on GPU without materializing
    the full Gram matrix. If idx is provided, X := W[idx].

    Returns population statistics (std uses correction=0, matching np.std).

    Note: exact percentiles of |offdiag| require retaining all off-diagonal
    absolute values on CPU, so this adds host-memory usage proportional to the
    number of off-diagonal entries.
    """
    device = W.device
    n = W.shape[0] if idx is None else int(idx.numel())

    if n < 2:
        nan = float("nan")
        return {
            "mean_offdiag": nan,
            "std_offdiag": nan,
            "min_offdiag": nan,
            "max_offdiag": nan,
            "mean_abs_offdiag": nan,
            "abs_offdiag_p50": nan,
            "abs_offdiag_p75": nan,
            "abs_offdiag_p90": nan,
            "abs_offdiag_p95": nan,
            "num_offdiag": 0,
            "num_rows": n,
        }

    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")

    sum_val = torch.zeros((), device=device, dtype=torch.float64)
    sum_sq = torch.zeros((), device=device, dtype=torch.float64)
    sum_abs = torch.zeros((), device=device, dtype=torch.float64)
    min_val = None
    max_val = None
    count = 0
    eye_cache: dict[int, torch.Tensor] = {}
    abs_chunks = []

    def get_rows(start: int, end: int) -> torch.Tensor:
        if idx is None:
            return W[start:end]
        return W.index_select(0, idx[start:end])

    for i0 in range(0, n, chunk_size):
        i1 = min(i0 + chunk_size, n)
        rows_i = get_rows(i0, i1)
        bi = i1 - i0

        for j0 in range(i0, n, chunk_size):
            j1 = min(j0 + chunk_size, n)
            rows_j = get_rows(j0, j1)
            bj = j1 - j0

            block = rows_i @ rows_j.T

            if i0 == j0:
                if bi < 2:
                    del rows_j, block
                    continue

                diag = block.diagonal()
                block_sum = block.sum(dtype=torch.float64) - diag.sum(dtype=torch.float64)
                block_sq = (block * block).sum(dtype=torch.float64) - (diag * diag).sum(dtype=torch.float64)
                block_abs = block.abs().sum(dtype=torch.float64) - diag.abs().sum(dtype=torch.float64)
                block_count = bi * bi - bi

                count += block_count
                sum_val += block_sum
                sum_sq += block_sq
                sum_abs += block_abs

                if bi not in eye_cache:
                    eye_cache[bi] = ~torch.eye(bi, dtype=torch.bool, device=device)
                vals = block[eye_cache[bi]]
                abs_chunks.append(vals.abs().to(device='cpu', dtype=torch.float32).numpy())
                bmin = vals.min()
                bmax = vals.max()
                del vals, diag
            else:
                block_count = bi * bj
                count += 2 * block_count
                sum_val += 2.0 * block.sum(dtype=torch.float64)
                sum_sq += 2.0 * (block * block).sum(dtype=torch.float64)
                sum_abs += 2.0 * block.abs().sum(dtype=torch.float64)
                block_abs_cpu = block.abs().to(device='cpu', dtype=torch.float32).numpy().reshape(-1)
                abs_chunks.append(block_abs_cpu)
                abs_chunks.append(block_abs_cpu.copy())
                bmin = block.min()
                bmax = block.max()

            min_val = bmin if min_val is None else torch.minimum(min_val, bmin)
            max_val = bmax if max_val is None else torch.maximum(max_val, bmax)

            del rows_j, block, bmin, bmax

        del rows_i

    if count == 0:
        nan = float("nan")
        return {
            "mean_offdiag": nan,
            "std_offdiag": nan,
            "min_offdiag": nan,
            "max_offdiag": nan,
            "mean_abs_offdiag": nan,
            "abs_offdiag_p50": nan,
            "abs_offdiag_p75": nan,
            "abs_offdiag_p90": nan,
            "abs_offdiag_p95": nan,
            "num_offdiag": 0,
            "num_rows": n,
        }

    mean = sum_val / count
    var = (sum_sq / count) - mean * mean
    var = torch.clamp(var, min=0.0)
    std = torch.sqrt(var)

    abs_all = np.concatenate(abs_chunks, axis=0)
    percentiles = np.percentile(abs_all, [50, 75, 90, 95])

    return {
        "mean_offdiag": float(mean.item()),
        "std_offdiag": float(std.item()),
        "min_offdiag": float(min_val.item()),
        "max_offdiag": float(max_val.item()),
        "mean_abs_offdiag": float((sum_abs / count).item()),
        "abs_offdiag_p50": float(percentiles[0]),
        "abs_offdiag_p75": float(percentiles[1]),
        "abs_offdiag_p90": float(percentiles[2]),
        "abs_offdiag_p95": float(percentiles[3]),
        "num_offdiag": int(count),
        "num_rows": int(n),
    }


def normalize_idx(idx_np: np.ndarray, n_rows: int) -> np.ndarray:
    idx_np = np.asarray(idx_np)
    if idx_np.dtype == np.bool_ or idx_np.dtype == bool:
        idx_np = np.flatnonzero(idx_np)
    else:
        idx_np = idx_np.reshape(-1).astype(np.int64, copy=False)

    if idx_np.ndim != 1:
        idx_np = idx_np.reshape(-1)

    idx_np = idx_np.copy()
    idx_np[idx_np < 0] += n_rows

    if idx_np.size > 0:
        bad = (idx_np < 0) | (idx_np >= n_rows)
        if bad.any():
            bad_vals = idx_np[bad][:10]
            raise IndexError(
                f"Found out-of-bounds indices (showing up to 10): {bad_vals.tolist()} for n_rows={n_rows}"
            )
    return idx_np


def load_arrays(
    weight_path: Path,
    idx_path: Path,
    device: torch.device,
    torch_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, np.ndarray, np.ndarray]:
    W_np = np.load(weight_path)
    idx_raw = np.load(idx_path)

    if W_np.ndim != 2:
        raise ValueError(f"Expected 2D weights in {weight_path}, got shape {W_np.shape}")

    idx_np = normalize_idx(idx_raw, n_rows=W_np.shape[0])

    W = torch.from_numpy(np.ascontiguousarray(W_np)).to(device=device, dtype=torch_dtype)
    idx = torch.from_numpy(np.ascontiguousarray(idx_np)).to(device=device, dtype=torch.long)
    return W, idx, W_np, idx_np


def sparse_avg_nnz_per_row(path: Path) -> float:
    X = sp.load_npz(path)
    if X.ndim != 2:
        raise ValueError(f"Expected 2D sparse array in {path}, got shape {X.shape}")
    n_rows = X.shape[0]
    if n_rows == 0:
        return float("nan")
    return float(X.getnnz(axis=1).mean())


def save_stats(
    out_path: Path,
    G_stats: Dict[str, float],
    G_t_stats: Dict[str, float],
    W_shape: tuple[int, ...],
    idx_shape: tuple[int, ...],
    dtype_name: str,
    chunk_size: int,
    sparse_k_full: float,
    sparse_k_truncated: float,
) -> None:
    dense_dimension = int(W_shape[1])
    sparse_dimension_full = int(W_shape[0])
    sparse_dimension_truncated = int(idx_shape[0])

    np.savez(
        out_path,
        G_mean_offdiag=np.float64(G_stats["mean_offdiag"]),
        G_std_offdiag=np.float64(G_stats["std_offdiag"]),
        G_min_offdiag=np.float64(G_stats["min_offdiag"]),
        G_max_offdiag=np.float64(G_stats["max_offdiag"]),
        G_mean_abs_offdiag=np.float64(G_stats["mean_abs_offdiag"]),
        G_abs_offdiag_p50=np.float64(G_stats["abs_offdiag_p50"]),
        G_abs_offdiag_p75=np.float64(G_stats["abs_offdiag_p75"]),
        G_abs_offdiag_p90=np.float64(G_stats["abs_offdiag_p90"]),
        G_abs_offdiag_p95=np.float64(G_stats["abs_offdiag_p95"]),
        G_num_offdiag=np.int64(G_stats["num_offdiag"]),
        G_num_rows=np.int64(G_stats["num_rows"]),
        G_t_mean_offdiag=np.float64(G_t_stats["mean_offdiag"]),
        G_t_std_offdiag=np.float64(G_t_stats["std_offdiag"]),
        G_t_min_offdiag=np.float64(G_t_stats["min_offdiag"]),
        G_t_max_offdiag=np.float64(G_t_stats["max_offdiag"]),
        G_t_mean_abs_offdiag=np.float64(G_t_stats["mean_abs_offdiag"]),
        G_t_abs_offdiag_p50=np.float64(G_t_stats["abs_offdiag_p50"]),
        G_t_abs_offdiag_p75=np.float64(G_t_stats["abs_offdiag_p75"]),
        G_t_abs_offdiag_p90=np.float64(G_t_stats["abs_offdiag_p90"]),
        G_t_abs_offdiag_p95=np.float64(G_t_stats["abs_offdiag_p95"]),
        G_t_num_offdiag=np.int64(G_t_stats["num_offdiag"]),
        G_t_num_rows=np.int64(G_t_stats["num_rows"]),
        dense_dimension=np.int64(dense_dimension),
        sparse_dimension_full=np.int64(sparse_dimension_full),
        sparse_dimension_truncated=np.int64(sparse_dimension_truncated),
        sparse_k_full=np.float64(sparse_k_full),
        sparse_k_truncated=np.float64(sparse_k_truncated),
        weight_shape=np.asarray(W_shape, dtype=np.int64),
        idx_shape=np.asarray(idx_shape, dtype=np.int64),
        compute_dtype=np.array(dtype_name),
        chunk_size=np.int64(chunk_size),
        std_definition=np.array("population_std_correction_0"),
    )


def main() -> None:
    args = parse_args()

    root = args.root.expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"Root folder does not exist: {root}")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but torch.cuda.is_available() is False")

    torch_dtype = TORCH_DTYPE_MAP[args.dtype]

    matched_dirs = []
    for dirpath, _, filenames in os.walk(root):
        names = set(filenames)
        if "decoder_weight.npy" in names and "X_features_truncated_idx.npy" in names:
            matched_dirs.append(Path(dirpath))

    if not matched_dirs:
        print(f"No matching folders found under: {root}")
        return

    print(f"Found {len(matched_dirs)} matching folder(s) under {root}")
    print(f"Using device={device}, dtype={args.dtype}, chunk_size={args.chunk_size}")

    num_ok = 0
    num_skipped = 0
    num_failed = 0

    for folder in matched_dirs:
        weight_path = folder / "decoder_weight.npy"
        idx_path = folder / "X_features_truncated_idx.npy"
        x_full_path = folder / "X_features.npz"
        x_trunc_path = folder / "X_features_truncated.npz"
        out_path = folder / "incoherence_statistics.npz"

        if args.skip_existing and out_path.exists():
            print(f"[skip] {folder} (already has {out_path.name})")
            num_skipped += 1
            continue

        print(f"[work] {folder}")
        try:
            if not x_full_path.exists():
                raise FileNotFoundError(f"Missing required sparse file: {x_full_path}")
            if not x_trunc_path.exists():
                raise FileNotFoundError(f"Missing required sparse file: {x_trunc_path}")

            W, idx, W_np, idx_np = load_arrays(weight_path, idx_path, device=device, torch_dtype=torch_dtype)

            G_stats = offdiag_gram_stats(W, idx=None, chunk_size=args.chunk_size)
            G_t_stats = offdiag_gram_stats(W, idx=idx, chunk_size=args.chunk_size)
            sparse_k_full = sparse_avg_nnz_per_row(x_full_path)
            sparse_k_truncated = sparse_avg_nnz_per_row(x_trunc_path)

            save_stats(
                out_path=out_path,
                G_stats=G_stats,
                G_t_stats=G_t_stats,
                W_shape=tuple(W_np.shape),
                idx_shape=tuple(idx_np.shape),
                dtype_name=args.dtype,
                chunk_size=args.chunk_size,
                sparse_k_full=sparse_k_full,
                sparse_k_truncated=sparse_k_truncated,
            )

            print(
                "      saved "
                f"{out_path.name} | "
                f"dense={W_np.shape[1]} | sparse_full={W_np.shape[0]} | sparse_trunc={idx_np.shape[0]} | "
                f"k_full={sparse_k_full:.6g} | k_trunc={sparse_k_truncated:.6g}"
            )
            num_ok += 1
        except Exception as exc:
            print(f"[fail] {folder}: {type(exc).__name__}: {exc}")
            num_failed += 1
        finally:
            if device.type == "cuda":
                torch.cuda.empty_cache()

    print(
        f"Done. processed={num_ok}, skipped={num_skipped}, failed={num_failed}, total_matches={len(matched_dirs)}"
    )


if __name__ == "__main__":
    main()