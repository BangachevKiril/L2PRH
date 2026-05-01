#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch


TORCH_DTYPE_MAP = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
    "float64": torch.float64,
}

INPUT_CANDIDATES = {
    "img": ("img_embeddings_normalized.npy", "img_embeddings_normalized.np"),
    "text": ("text_embeddings_normalized.npy", "text_embeddings_normalized.np"),
}

OUTPUT_NAMES = {
    "img": "img_incoherence_statistics.npy",
    "text": "text_incoherence_statistics.npy",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Recursively find folders containing normalized image/text embeddings, "
            "sample up to random_batch_subset rows uniformly at random, then compute "
            "GPU Gram-matrix off-diagonal statistics on the sampled subset and save "
            "them as img_incoherence_statistics.npy or text_incoherence_statistics.npy"
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
        help="Computation dtype for embeddings on device (default: float32)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=4096,
        help=(
            "Row/column block size for sampled Gram computation. Reduce this if you hit "
            "GPU OOM. Default: 4096"
        ),
    )
    parser.add_argument(
        "--random-batch-subset",
        type=int,
        default=8192,
        help=(
            "Maximum number of representation vectors to sample uniformly at random "
            "without replacement before computing incoherence statistics. Default: 8192"
        ),
    )
    parser.add_argument(
        "--sample-seed",
        type=int,
        default=0,
        help="Random seed for subset selection (default: 0)",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help=(
            "Skip files whose corresponding *_incoherence_statistics.npy already exists"
        ),
    )
    return parser.parse_args()


@torch.inference_mode()
def offdiag_gram_stats(
    X: torch.Tensor,
    chunk_size: int = 4096,
) -> Dict[str, float]:
    """
    Compute exact off-diagonal statistics of X @ X.T on GPU without materializing
    the full Gram matrix, where X is already the sampled subset.

    Returns population statistics (std uses correction=0, matching np.std).

    Exact percentiles of |offdiag| are computed over the sampled subset only.
    """
    device = X.device
    n = int(X.shape[0])

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

    for i0 in range(0, n, chunk_size):
        i1 = min(i0 + chunk_size, n)
        rows_i = X[i0:i1]
        bi = i1 - i0

        for j0 in range(i0, n, chunk_size):
            j1 = min(j0 + chunk_size, n)
            rows_j = X[j0:j1]
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
                abs_chunks.append(vals.abs().to(device="cpu", dtype=torch.float32).numpy())
                bmin = vals.min()
                bmax = vals.max()
                del vals, diag
            else:
                block_count = bi * bj
                count += 2 * block_count
                sum_val += 2.0 * block.sum(dtype=torch.float64)
                sum_sq += 2.0 * (block * block).sum(dtype=torch.float64)
                sum_abs += 2.0 * block.abs().sum(dtype=torch.float64)
                block_abs_cpu = block.abs().to(device="cpu", dtype=torch.float32).numpy().reshape(-1)
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


def choose_input_path(folder: Path, modality: str) -> Optional[Path]:
    candidates = [folder / name for name in INPUT_CANDIDATES[modality] if (folder / name).exists()]
    if not candidates:
        return None

    preferred_name = INPUT_CANDIDATES[modality][0]
    for candidate in candidates:
        if candidate.name == preferred_name:
            if len(candidates) > 1:
                others = ", ".join(path.name for path in candidates if path != candidate)
                print(
                    f"[warn] {folder}: found multiple {modality} embedding files; "
                    f"using {candidate.name} and ignoring {others}"
                )
            return candidate

    return candidates[0]


def sample_embeddings(
    emb_path: Path,
    device: torch.device,
    torch_dtype: torch.dtype,
    random_batch_subset: int,
    sample_seed: int,
) -> tuple[torch.Tensor, tuple[int, int], np.ndarray]:
    X_mm = np.load(emb_path, mmap_mode="r", allow_pickle=False)

    if X_mm.ndim != 2:
        raise ValueError(f"Expected 2D embeddings in {emb_path}, got shape {X_mm.shape}")

    n_rows, emb_dim = int(X_mm.shape[0]), int(X_mm.shape[1])

    if random_batch_subset <= 0:
        raise ValueError(
            f"random_batch_subset must be positive, got {random_batch_subset}"
        )

    sample_size = min(n_rows, int(random_batch_subset))
    rng = np.random.default_rng(sample_seed)
    if sample_size == n_rows:
        sampled_idx = np.arange(n_rows, dtype=np.int64)
    else:
        sampled_idx = np.sort(rng.choice(n_rows, size=sample_size, replace=False).astype(np.int64))

    X_sample_np = np.asarray(X_mm[sampled_idx], dtype=np.float32 if torch_dtype in (torch.float16, torch.bfloat16, torch.float32) else np.float64)
    X_sample = torch.from_numpy(np.ascontiguousarray(X_sample_np)).to(device=device, dtype=torch_dtype)
    return X_sample, (n_rows, emb_dim), sampled_idx


def save_stats(
    out_path: Path,
    stats: Dict[str, float],
    X_shape: tuple[int, ...],
    sampled_idx: np.ndarray,
    dtype_name: str,
    chunk_size: int,
    source_filename: str,
    modality: str,
    random_batch_subset: int,
    sample_seed: int,
) -> None:
    payload = {
        "G_mean_offdiag": np.float64(stats["mean_offdiag"]),
        "G_std_offdiag": np.float64(stats["std_offdiag"]),
        "G_min_offdiag": np.float64(stats["min_offdiag"]),
        "G_max_offdiag": np.float64(stats["max_offdiag"]),
        "G_mean_abs_offdiag": np.float64(stats["mean_abs_offdiag"]),
        "G_abs_offdiag_p50": np.float64(stats["abs_offdiag_p50"]),
        "G_abs_offdiag_p75": np.float64(stats["abs_offdiag_p75"]),
        "G_abs_offdiag_p90": np.float64(stats["abs_offdiag_p90"]),
        "G_abs_offdiag_p95": np.float64(stats["abs_offdiag_p95"]),
        "G_num_offdiag": np.int64(stats["num_offdiag"]),
        "G_num_rows": np.int64(stats["num_rows"]),
        "num_vectors": np.int64(X_shape[0]),
        "embedding_dimension": np.int64(X_shape[1]),
        "representation_shape": np.asarray(X_shape, dtype=np.int64),
        "num_sampled_vectors": np.int64(sampled_idx.size),
        "sampled_indices": sampled_idx.astype(np.int64, copy=False),
        "random_batch_subset": np.int64(random_batch_subset),
        "sample_seed": np.int64(sample_seed),
        "compute_dtype": dtype_name,
        "chunk_size": np.int64(chunk_size),
        "source_file": source_filename,
        "modality": modality,
        "std_definition": "population_std_correction_0",
        "statistics_scope": "sampled_uniform_without_replacement",
    }
    np.save(out_path, payload, allow_pickle=True)


def main() -> None:
    args = parse_args()

    root = args.root.expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"Root folder does not exist: {root}")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but torch.cuda.is_available() is False")

    torch_dtype = TORCH_DTYPE_MAP[args.dtype]

    matched_jobs = []
    for dirpath, _, _ in os.walk(root):
        folder = Path(dirpath)
        for modality in ("img", "text"):
            input_path = choose_input_path(folder, modality)
            if input_path is None:
                continue
            out_path = folder / OUTPUT_NAMES[modality]
            matched_jobs.append((folder, modality, input_path, out_path))

    if not matched_jobs:
        print(f"No matching embedding files found under: {root}")
        return

    print(f"Found {len(matched_jobs)} embedding job(s) under {root}")
    print(
        f"Using device={device}, dtype={args.dtype}, chunk_size={args.chunk_size}, "
        f"random_batch_subset={args.random_batch_subset}, sample_seed={args.sample_seed}"
    )

    num_ok = 0
    num_skipped = 0
    num_failed = 0

    for folder, modality, input_path, out_path in matched_jobs:
        if args.skip_existing and out_path.exists():
            print(f"[skip] {folder} ({out_path.name} already exists)")
            num_skipped += 1
            continue

        print(f"[work] {folder} | modality={modality} | input={input_path.name}")
        try:
            X_sample, X_shape, sampled_idx = sample_embeddings(
                input_path,
                device=device,
                torch_dtype=torch_dtype,
                random_batch_subset=args.random_batch_subset,
                sample_seed=args.sample_seed,
            )
            G_stats = offdiag_gram_stats(X_sample, chunk_size=args.chunk_size)

            save_stats(
                out_path=out_path,
                stats=G_stats,
                X_shape=tuple(X_shape),
                sampled_idx=sampled_idx,
                dtype_name=args.dtype,
                chunk_size=args.chunk_size,
                source_filename=input_path.name,
                modality=modality,
                random_batch_subset=args.random_batch_subset,
                sample_seed=args.sample_seed,
            )

            print(
                "      saved "
                f"{out_path.name} | "
                f"num_vectors={X_shape[0]} | embedding_dim={X_shape[1]} | sampled={sampled_idx.size}"
            )
            num_ok += 1
        except Exception as exc:
            print(f"[fail] {folder} [{modality}]: {type(exc).__name__}: {exc}")
            num_failed += 1
        finally:
            if device.type == "cuda":
                torch.cuda.empty_cache()

    print(
        f"Done. processed={num_ok}, skipped={num_skipped}, failed={num_failed}, total_jobs={len(matched_jobs)}"
    )


if __name__ == "__main__":
    main()
