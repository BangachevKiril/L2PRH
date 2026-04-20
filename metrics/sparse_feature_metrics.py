#!/usr/bin/env python
"""
subsample_estimate_metrics.py

Subsample-based metric estimation for two datasets.

No std, no random baseline.
Optional profiling of per-metric time.

DIFFERENCE vs dense-embedding version:
- Data is loaded from SciPy sparse .npz matrices:
    {topk_root}/topk_{d}_{model}_k_{k}/X_features*.npz
- Optional --is_binary: X := (X > 1e-6) * 1.0 (kept sparse).
- For metrics, each subsample is densified (only those rows) and moved to torch/device.

NEW:
- --use_normalized {0,1}
    0 -> X_features.npz
    1 -> X_features_truncated.npz

UPDATED (k1/k2):
- Supports per-model sparsity via --k1 and --k2.
- Backward compatible: --k still works.
- Default behavior:
    * if neither --k1 nor --k2 provided: k1=k2=k
    * if only one of --k1/--k2 provided: the other is set equal to it
"""

import argparse
import os
import numpy as np
import torch
import scipy.sparse as sp

from compute_metrics import compute_metrics_block, add_metric_args


# ---------------------------------------------------------
#  SPARSE DATA LOADING
# ---------------------------------------------------------
def sparse_feature_path(topk_root: str, d: int, model: str, k: int, use_normalized: int) -> str:
    fname = "X_features.npz" if int(use_normalized) == 0 else "X_features_truncated.npz"
    folder_candidates = [
        f"topk_{int(d)}_{model}_k_{int(k)}",
        f"batchtopk_{int(d)}_{model}_k_{int(k)}",
    ]

    for folder in folder_candidates:
        path = os.path.join(topk_root, folder, fname)
        if os.path.isfile(path):
            return path

    return os.path.join(topk_root, folder_candidates[0], fname)


def load_sparse_features(topk_root: str, model: str, d: int, k: int, is_binary: bool, use_normalized: int):
    path = sparse_feature_path(topk_root, d, model, k, use_normalized)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Expected sparse feature file not found: {path}")

    X = sp.load_npz(path)
    if X.ndim != 2:
        raise ValueError(f"Expected 2D sparse matrix in {path}, got shape {X.shape}")

    # Ensure CSR for fast row slicing
    X = X.tocsr()

    if is_binary:
        # Keep entries strictly > 1e-6, set values to 1.0
        X = (X > 1e-6).astype(np.float32)

    return X, path


def sparse_rows_to_torch_dense(X_csr: sp.csr_matrix, idx_np: np.ndarray, device: torch.device) -> torch.Tensor:
    # Slice rows in sparse, densify just the subsample, then move to torch/device.
    sub = X_csr[idx_np]  # still sparse
    dense = sub.toarray().astype(np.float32, copy=False)  # (m, d)
    return torch.from_numpy(dense).to(device)


# ---------------------------------------------------------
#  K1/K2 RESOLUTION
# ---------------------------------------------------------
def resolve_k1_k2(args: argparse.Namespace) -> None:
    """
    Resolve per-model k's with backward compatibility.

    Rules:
      - If neither k1 nor k2 provided:
          require k, and set k1=k2=k
      - If only one of k1/k2 provided:
          copy it to the other (k1=k2)
      - If both provided:
          use as-is
      - If k is provided alongside k1/k2:
          keep k for metadata/backward compat, but use k1/k2 for loading.
    """
    if args.k1 is None and args.k2 is None:
        if args.k is None:
            raise ValueError("You must provide --k or at least one of --k1/--k2")
        args.k1 = int(args.k)
        args.k2 = int(args.k)
    elif args.k1 is None and args.k2 is not None:
        args.k1 = int(args.k2)
    elif args.k2 is None and args.k1 is not None:
        args.k2 = int(args.k1)

    args.k1 = int(args.k1)
    args.k2 = int(args.k2)

    if args.k1 <= 0 or args.k2 <= 0:
        raise ValueError(f"k1 and k2 must be positive, got k1={args.k1}, k2={args.k2}")

    # For metadata: if --k not given, set it to k1 (so old consumers see something sensible)
    if args.k is None:
        args.k = int(args.k1)
    else:
        args.k = int(args.k)


# ---------------------------------------------------------
#  ARGPARSE (same structure as the dense script)
# ---------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Subsample-based metric estimation for two datasets.")
    p.add_argument("ds1", type=str, help="Model key (must match folder naming), e.g. BAAI__bge-large-en-v1.5_text")
    p.add_argument("ds2", type=str, help="Model key (must match folder naming), e.g. facebook__dinov2-base_img")

    # Data location for sparse features
    p.add_argument(
        "--topk_root",
        type=str,
        default="/home/kirilb/orcd/pool/PRH_data/topk_sae_embedded_coco",
        help="Root directory containing topk_{d}_{model}_k_{k}/X_features*.npz folders.",
    )
    p.add_argument("--d", type=int, required=True, help="Feature dimension d used in the folder name (e.g. 256).")

    # Backward compatible single-k flag
    p.add_argument(
        "--k",
        type=int,
        default=None,
        help="(legacy) Top-k sparsity parameter k used in the folder name for BOTH models if --k1/--k2 not provided.",
    )

    # New: per-model k flags
    p.add_argument(
        "--k1",
        type=int,
        default=None,
        help="Top-k sparsity parameter for ds1. If provided alone, ds2 uses the same value.",
    )
    p.add_argument(
        "--k2",
        type=int,
        default=None,
        help="Top-k sparsity parameter for ds2. If provided alone, ds1 uses the same value.",
    )

    # choose which sparse file to load
    p.add_argument(
        "--use_normalized",
        type=int,
        default=0,
        choices=[0, 1],
        help="0: load X_features.npz (raw).  1: load X_features_truncated.npz (truncated).",
    )

    # binary sparse option
    p.add_argument(
        "--is_binary",
        action="store_true",
        help="If set, converts sparse matrix to binary: (X > 1e-6) * 1.0 (kept sparse).",
    )

    # Subsample parameters (same)
    p.add_argument("--how_many_samples", type=int, default=50)
    p.add_argument("--subsample_size", type=int, default=500)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--with_replacement", action="store_true")
    p.add_argument("--device", type=str, default="cuda")

    # Output control (same)
    p.add_argument("--output_dir", type=str, default="/home/kirilb/orcd/pool/PRH_data/LLM_features/metrics")
    p.add_argument("--output_name", type=str, default=None)

    # optional profiling (same behavior as dense script)
    p.add_argument(
        "--profile_metrics",
        action="store_true",
        help="If set, records time per metric per subsample and saves summary.",
    )

    add_metric_args(p)
    return p.parse_args()


# ---------------------------------------------------------
#  MAIN (same structure as the dense script)
# ---------------------------------------------------------
def main():
    args = parse_args()
    resolve_k1_k2(args)

    device = torch.device(args.device)

    X1_sp, p1 = load_sparse_features(args.topk_root, args.ds1, args.d, args.k1, args.is_binary, args.use_normalized)
    X2_sp, p2 = load_sparse_features(args.topk_root, args.ds2, args.d, args.k2, args.is_binary, args.use_normalized)

    print(f"Loaded X1 from: {p1}  shape={X1_sp.shape}  nnz={X1_sp.nnz}")
    print(f"Loaded X2 from: {p2}  shape={X2_sp.shape}  nnz={X2_sp.nnz}")
    print(f"use_normalized={args.use_normalized}  ({'truncated' if args.use_normalized==1 else 'raw'})")
    print(f"k1={args.k1}  k2={args.k2}  (legacy k field for metadata: k={args.k})")
    if args.is_binary:
        print("Binary mode enabled: X := (X > 1e-6) * 1.0 (stored sparsely).")

    if X1_sp.shape[0] != X2_sp.shape[0]:
        raise ValueError(f"X1 and X2 must have same N. Got {X1_sp.shape[0]} vs {X2_sp.shape[0]}")

    N = X1_sp.shape[0]

    if args.subsample_size > N and (not args.with_replacement):
        raise ValueError(f"subsample_size={args.subsample_size} > N={N} but sampling without replacement.")

    metric_keys = [
        "CKA_HSIC",
        "CKA_unbiased",
        "SVCCA_1",
        "SVCCA_2",
        "TOPK10",
        "TOPK100",
        "KNN_EDIT_10",
        "KNN_EDIT_100",
    ]
    time_keys = ["KERNEL"] + metric_keys

    S = args.how_many_samples
    metrics_vals = {k: np.zeros(S, dtype=np.float64) for k in metric_keys}
    times_vals = {k: np.zeros(S, dtype=np.float64) for k in time_keys} if args.profile_metrics else None

    g = torch.Generator(device="cpu")
    g.manual_seed(int(args.seed))

    print(f"Sampling {S} subsamples of size {args.subsample_size} (with_replacement={args.with_replacement})")
    if args.profile_metrics:
        print("Profiling per-metric time (seconds). CUDA ops are synchronized for accurate timings.")

    for sidx in range(S):
        if args.with_replacement:
            idx = torch.randint(low=0, high=N, size=(args.subsample_size,), generator=g)
        else:
            idx = torch.randperm(N, generator=g)[: args.subsample_size]

        idx_np = idx.numpy()

        # Densify only sampled rows and move to device
        Y1 = sparse_rows_to_torch_dense(X1_sp, idx_np, device)
        Y2 = sparse_rows_to_torch_dense(X2_sp, idx_np, device)

        print(f"[{sidx+1}/{S}] computing metrics on subsample (rows={args.subsample_size})")
        m, t = compute_metrics_block(Y1, Y2, args, profile_metrics=args.profile_metrics)

        for k in metric_keys:
            metrics_vals[k][sidx] = float(m[k])

        if args.profile_metrics and t is not None:
            for k in time_keys:
                times_vals[k][sidx] = float(t.get(k, np.nan))

    agg = {k + "_mean_over_subsamples": float(np.nanmean(metrics_vals[k])) for k in metric_keys}

    if args.profile_metrics:
        for k in time_keys:
            agg["time_" + k + "_mean_over_subsamples"] = float(np.nanmean(times_vals[k]))

        ranked = sorted(
            [(k, agg["time_" + k + "_mean_over_subsamples"]) for k in time_keys],
            key=lambda x: x[1],
            reverse=True,
        )
        print("\nMean time per subsample (seconds), descending:")
        for k, v in ranked:
            print(f"  {k:>12s}: {v:.6f}")

    os.makedirs(args.output_dir, exist_ok=True)
    safe1 = args.ds1.replace("/", "__")
    safe2 = args.ds2.replace("/", "__")
    out_file = args.output_name or f"{safe1}_{safe2}_subsample_metrics.npz"
    out_path = os.path.join(args.output_dir, out_file)

    print(f"Saving to {out_path}")
    payload = {}
    payload.update({k: metrics_vals[k] for k in metric_keys})
    payload.update(agg)

    if args.profile_metrics:
        payload.update({("time_" + k): times_vals[k] for k in time_keys})

    payload.update(
        dict(
            ds1=args.ds1,
            ds2=args.ds2,
            topk_root=args.topk_root,
            d=int(args.d),
            # Keep legacy 'k' plus new per-model k's for provenance
            k=int(args.k),
            k1=int(args.k1),
            k2=int(args.k2),
            use_normalized=int(args.use_normalized),
            is_binary=bool(args.is_binary),
            how_many_samples=int(args.how_many_samples),
            subsample_size=int(args.subsample_size),
            seed=int(args.seed),
            with_replacement=bool(args.with_replacement),
            path1=p1,
            path2=p2,
            profile_metrics=bool(args.profile_metrics),
        )
    )

    np.savez(out_path, **payload)
    print("Done.")


if __name__ == "__main__":
    main()
