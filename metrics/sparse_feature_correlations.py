#!/usr/bin/env python3
"""
Summarize random-baseline sparse-correlation outputs across multiple seeds.

For a given dataset / dimension / sparsity pattern, this script reads folders:
    d_{d}_{pattern}_rand_baseline_seed_{seed}
for seeds in a provided list, and writes a summary folder:
    d_{d}_{pattern}_rand_baseline_summary

For each pairwise-model .npz file, it saves a new .npz with the same filename and:
    - weighted_correlation      : mean across seeds
    - binary_correlation        : mean across seeds
    - weighted_correlation_std  : std across seeds
    - binary_correlation_std    : std across seeds

It also copies over metadata text files (models / ks) from seed 0 if present.
"""

import argparse
import os
import shutil
import sys
import time
from typing import Dict, List, Set

import numpy as np


def log(msg: str, log_path: str) -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def build_seed_dir(base_root: str, dataset: str, d: int, sparsity_pattern: str, seed: int) -> str:
    return os.path.join(
        base_root,
        f"topk_sae_{dataset}_correlations",
        f"d_{d}_{sparsity_pattern}_rand_baseline_seed_{seed}",
    )


def build_summary_dir(base_root: str, dataset: str, d: int, sparsity_pattern: str) -> str:
    return os.path.join(
        base_root,
        f"topk_sae_{dataset}_correlations",
        f"d_{d}_{sparsity_pattern}_rand_baseline_summary",
    )


def list_npz_files(folder: str) -> Set[str]:
    if not os.path.isdir(folder):
        return set()
    out = set()
    for name in os.listdir(folder):
        if not name.endswith(".npz"):
            continue
        if name.startswith("log_"):
            continue
        out.add(name)
    return out


def scalar_from_npz(arr: np.lib.npyio.NpzFile, key: str) -> float:
    if key not in arr:
        raise KeyError(f"Missing key '{key}'")
    x = arr[key]
    if np.size(x) != 1:
        raise ValueError(f"Expected scalar for key '{key}', got shape {np.shape(x)}")
    return float(np.asarray(x).reshape(()))


def maybe_copy_metadata(seed0_dir: str, summary_dir: str, d: int) -> None:
    candidates = [
        f"models_d{d}_kvar.txt",
        f"ks_d{d}_kvar.txt",
    ]
    for name in candidates:
        src = os.path.join(seed0_dir, name)
        dst = os.path.join(summary_dir, name)
        if os.path.exists(src):
            shutil.copy2(src, dst)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=str, required=True)
    ap.add_argument("--d", type=int, required=True)
    ap.add_argument("--sparsity_pattern", type=str, required=True,
                    choices=["kvar", "k_32", "k_64", "k_128"])
    ap.add_argument("--base_root", type=str,
                    default="/home/kirilb/orcd/pool/PRH_data")
    ap.add_argument("--seeds", nargs="+", type=int,
                    default=list(range(10)),
                    help="Seeds to aggregate, default: 0 1 2 3 4 5 6 7 8 9")
    ap.add_argument("--strict", type=int, default=1,
                    help="If 1, require every file to exist for every seed. If 0, skip incomplete files.")
    args = ap.parse_args()

    dataset = args.dataset
    d = int(args.d)
    sparsity_pattern = args.sparsity_pattern
    seeds: List[int] = list(args.seeds)
    strict = bool(args.strict)

    seed_dirs: Dict[int, str] = {
        s: build_seed_dir(args.base_root, dataset, d, sparsity_pattern, s) for s in seeds
    }
    summary_dir = build_summary_dir(args.base_root, dataset, d, sparsity_pattern)
    os.makedirs(summary_dir, exist_ok=True)

    log_path = os.path.join(summary_dir, f"log_summary_d{d}_{sparsity_pattern}.txt")
    log(f"Starting summary for dataset={dataset}, d={d}, sparsity_pattern={sparsity_pattern}", log_path)
    log(f"Seeds: {seeds}", log_path)
    log(f"Summary dir: {summary_dir}", log_path)

    missing_seed_dirs = [s for s, path in seed_dirs.items() if not os.path.isdir(path)]
    if missing_seed_dirs:
        log(f"Missing seed directories for seeds: {missing_seed_dirs}", log_path)
        return 1

    for s, path in seed_dirs.items():
        log(f"Seed {s} dir: {path}", log_path)

    file_sets = {s: list_npz_files(path) for s, path in seed_dirs.items()}
    union_files = set().union(*file_sets.values())
    inter_files = set.intersection(*file_sets.values()) if file_sets else set()

    log(f"Union file count across seeds: {len(union_files)}", log_path)
    log(f"Intersection file count across seeds: {len(inter_files)}", log_path)

    if strict:
        files_to_process = sorted(inter_files)
        missing_somewhere = sorted(union_files - inter_files)
        if missing_somewhere:
            log("Strict mode detected files missing from some seeds.", log_path)
            for name in missing_somewhere[:20]:
                present = [s for s in seeds if name in file_sets[s]]
                absent = [s for s in seeds if name not in file_sets[s]]
                log(f"Incomplete file: {name} | present={present} | absent={absent}", log_path)
            if len(missing_somewhere) > 20:
                log(f"... and {len(missing_somewhere) - 20} more incomplete files.", log_path)
            return 1
    else:
        files_to_process = sorted(inter_files)
        skipped = sorted(union_files - inter_files)
        for name in skipped:
            present = [s for s in seeds if name in file_sets[s]]
            absent = [s for s in seeds if name not in file_sets[s]]
            log(f"Skipping incomplete file: {name} | present={present} | absent={absent}", log_path)

    if not files_to_process:
        log("No complete .npz files found to summarize.", log_path)
        return 1

    maybe_copy_metadata(seed_dirs[seeds[0]], summary_dir, d)

    n_saved = 0
    for fname in files_to_process:
        weighted_vals = []
        binary_vals = []

        for s in seeds:
            path = os.path.join(seed_dirs[s], fname)
            with np.load(path) as arr:
                weighted_vals.append(scalar_from_npz(arr, "weighted_correlation"))
                binary_vals.append(scalar_from_npz(arr, "binary_correlation"))

        weighted_arr = np.asarray(weighted_vals, dtype=np.float64)
        binary_arr = np.asarray(binary_vals, dtype=np.float64)

        out_path = os.path.join(summary_dir, fname)
        np.savez_compressed(
            out_path,
            weighted_correlation=np.float32(weighted_arr.mean()),
            binary_correlation=np.float32(binary_arr.mean()),
            weighted_correlation_std=np.float32(weighted_arr.std(ddof=0)),
            binary_correlation_std=np.float32(binary_arr.std(ddof=0)),
        )
        n_saved += 1

        if n_saved <= 5 or n_saved % 100 == 0:
            log(f"Saved summary: {out_path}", log_path)

    log(f"Done. Saved {n_saved} summary files.", log_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())