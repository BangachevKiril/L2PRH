#!/usr/bin/env python
"""
subsample_estimate_metrics.py

Subsample-based metric estimation for two datasets.

No std, no random baseline.
Optional profiling of per-metric time.

Embedding selection:
  --use_normalized 0  -> {text|img}_embeddings.npy
  --use_normalized 1  -> {text|img}_embeddings_normalized.npy
  --use_normalized 2  -> {text|img}_embeddings_fully_normalized.npy
"""

import argparse
import os
import numpy as np
import torch

from compute_metrics import compute_metrics_block, add_metric_args


def parse_dataset_spec(spec: str):
    if "/" not in spec:
        raise ValueError(f"Dataset spec must be like MODEL/text or MODEL/img, got: {spec}")
    model_name, which = spec.rsplit("/", 1)
    if which not in ("text", "img"):
        raise ValueError(f"Dataset spec suffix must be 'text' or 'img', got: {spec}")
    return model_name, which


def _suffix_from_use_normalized(use_normalized: int) -> str:
    if use_normalized == 0:
        return ".npy"
    if use_normalized == 1:
        return "_normalized.npy"
    if use_normalized == 2:
        return "_fully_normalized.npy"
    raise ValueError(f"--use_normalized must be 0, 1, or 2 (got {use_normalized})")


def get_embeddings(
    input_path: str,
    dataset_spec: str,
    device,
    use_normalized: int,
    allow_pickle: bool = True,
):
    model_name, which = parse_dataset_spec(dataset_spec)
    model_dir = os.path.join(input_path, model_name)
    if not os.path.isdir(model_dir):
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    suffix = _suffix_from_use_normalized(int(use_normalized))
    fname = f"{which}_embeddings{suffix}"
    path = os.path.join(model_dir, fname)

    if not os.path.isfile(path):
        present = sorted(
            [fn for fn in os.listdir(model_dir) if fn.endswith(".npy") and ("embeddings" in fn)]
        )
        raise FileNotFoundError(
            f"Expected file not found: {path}\n"
            f"Present .npy files in {model_dir}: {present}"
        )

    arr = np.load(path, allow_pickle=allow_pickle)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D array in {path}, got shape {arr.shape}")

    return torch.from_numpy(arr).to(device), path


def parse_args():
    p = argparse.ArgumentParser(description="Subsample-based metric estimation for two datasets.")
    p.add_argument("ds1", type=str, help="Dataset spec: MODEL/text or MODEL/img")
    p.add_argument("ds2", type=str, help="Dataset spec: MODEL/text or MODEL/img")
    p.add_argument("--input_path", type=str, default="/home/kirilb/orcd/pool/PRH_data/embedded_words")

    # CHANGED:
    p.add_argument(
        "--use_normalized",
        type=int,
        choices=[0, 1, 2],
        default=0,
        help="0=raw embeddings (.npy), 1=_normalized.npy, 2=_fully_normalized.npy",
    )

    p.add_argument("--how_many_samples", type=int, default=50)
    p.add_argument("--subsample_size", type=int, default=500)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--with_replacement", action="store_true")
    p.add_argument("--device", type=str, default="cuda")

    p.add_argument("--output_dir", type=str, default="/home/kirilb/orcd/pool/PRH_data/LLM_features/metrics")
    p.add_argument("--output_name", type=str, default=None)

    # NEW:
    p.add_argument(
        "--profile_metrics",
        action="store_true",
        help="If set, records time per metric per subsample and saves summary.",
    )

    add_metric_args(p)
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)

    X1, p1 = get_embeddings(args.input_path, args.ds1, device, use_normalized=args.use_normalized)
    X2, p2 = get_embeddings(args.input_path, args.ds2, device, use_normalized=args.use_normalized)
    print(f"Loaded X1 from: {p1}  shape={tuple(X1.shape)}")
    print(f"Loaded X2 from: {p2}  shape={tuple(X2.shape)}")

    if X1.shape[0] != X2.shape[0]:
        raise ValueError(f"X1 and X2 must have same N. Got {X1.shape[0]} vs {X2.shape[0]}")
    N = X1.shape[0]

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
            idx = torch.randperm(N, generator=g)[:args.subsample_size]
        idx = idx.to(device)

        Y1 = X1[idx]
        Y2 = X2[idx]

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
            input_path=args.input_path,
            use_normalized=int(args.use_normalized),  # CHANGED: store 0/1/2
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