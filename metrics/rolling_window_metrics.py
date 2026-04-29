#!/usr/bin/env python
"""
sliding_window_estimate_metrics.py

Sliding-window metric estimation for two datasets.

Same as subsample_estimate_metrics.py, except instead of random subsamples
we evaluate HOW_MANY windows of size subsample_size, stepped by step_size:

  window s uses:
    X1[s*step_size : s*step_size + subsample_size]
    X2[s*step_size : s*step_size + subsample_size]

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


def load_optional_indices_txt(path: str):
    """
    Load an optional zero-based index list from a text file.

    The expected format is one integer per line, but any whitespace-separated
    integer list also works. These indices are applied as arr[L] before the
    sliding windows are formed.
    """
    path = (path or "").strip()
    if path == "":
        return None
    if not os.path.isfile(path):
        raise FileNotFoundError(f"--use_indices was provided but file does not exist: {path}")

    try:
        idx = np.loadtxt(path, dtype=np.int64, comments="#", ndmin=1)
    except ValueError as e:
        raise ValueError(f"Could not read integer indices from {path}: {e}") from e

    idx = np.asarray(idx, dtype=np.int64).reshape(-1)
    if idx.size == 0:
        raise ValueError(f"Index file is empty: {path}")
    if np.any(idx < 0):
        bad = idx[idx < 0][:10]
        raise ValueError(f"Index file contains negative indices, e.g. {bad.tolist()}: {path}")
    return idx


def parse_args():
    p = argparse.ArgumentParser(description="Sliding-window metric estimation for two datasets.")
    p.add_argument("ds1", type=str, help="Dataset spec: MODEL/text or MODEL/img")
    p.add_argument("ds2", type=str, help="Dataset spec: MODEL/text or MODEL/img")
    p.add_argument("--input_path", type=str, default="/home/kirilb/orcd/pool/PRH_data/embedded_words")

    p.add_argument(
        "--use_normalized",
        type=int,
        choices=[0, 1, 2],
        default=0,
        help="0=raw embeddings (.npy), 1=_normalized.npy, 2=_fully_normalized.npy",
    )
    p.add_argument(
        "--use_indices",
        type=str,
        default="",
        help=(
            "Optional path to a .txt file containing zero-based row indices. "
            "If provided, each embedding matrix is filtered as arr[L] before "
            "the sliding-window experiment. Empty string keeps old behavior."
        ),
    )

    # Keep names consistent with the subsample script:
    #   how_many_samples == HOW_MANY windows
    #   subsample_size    == batch_size (window size)
    p.add_argument("--how_many_samples", type=int, default=50, help="HOW_MANY windows to evaluate")
    p.add_argument("--subsample_size", type=int, default=500, help="Window size (batch_size)")
    p.add_argument("--step_size", type=int, default=100, help="Stride between consecutive windows")
    p.add_argument("--seed", type=int, default=0, help="Unused for windowing; kept for compatibility")
    p.add_argument("--device", type=str, default="cuda")

    p.add_argument("--output_dir", type=str, default="/home/kirilb/orcd/pool/PRH_data/LLM_features/metrics")
    p.add_argument("--output_name", type=str, default=None)

    p.add_argument(
        "--profile_metrics",
        action="store_true",
        help="If set, records time per metric per window and saves summary.",
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

    original_N = int(X1.shape[0])
    indices = load_optional_indices_txt(args.use_indices)
    indices_count = 0
    if indices is not None:
        max_idx = int(indices.max())
        if max_idx >= original_N:
            raise ValueError(
                f"Index file contains out-of-bounds index {max_idx}, but embeddings have N={original_N}: "
                f"{args.use_indices}"
            )
        indices_count = int(indices.size)
        print(f"Using index filter: {args.use_indices}")
        print(f"Keeping {indices_count} / {original_N} rows before sliding windows.")
        idx_t = torch.as_tensor(indices, dtype=torch.long, device=device)
        X1 = torch.index_select(X1, 0, idx_t)
        X2 = torch.index_select(X2, 0, idx_t)
        print(f"Filtered X1 shape={tuple(X1.shape)}")
        print(f"Filtered X2 shape={tuple(X2.shape)}")
    else:
        print("No --use_indices file provided; using all rows.")

    N = int(X1.shape[0])

    S = int(args.how_many_samples)
    B = int(args.subsample_size)
    step = int(args.step_size)

    if B <= 0:
        raise ValueError(f"subsample_size must be positive (got {B})")
    if step <= 0:
        raise ValueError(f"step_size must be positive (got {step})")
    if S <= 0:
        raise ValueError(f"how_many_samples must be positive (got {S})")

    last_end = (S - 1) * step + B
    if last_end > N:
        raise ValueError(
            f"Not enough rows for requested windows:\n"
            f"  N={N}, subsample_size(batch_size)={B}, step_size={step}, how_many_samples(HOW_MANY)={S}\n"
            f"  Need (HOW_MANY-1)*step_size + batch_size <= N, but got {last_end} > {N}"
        )

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

    metrics_vals = {k: np.zeros(S, dtype=np.float64) for k in metric_keys}
    times_vals = {k: np.zeros(S, dtype=np.float64) for k in time_keys} if args.profile_metrics else None

    print(f"Sliding windows: HOW_MANY={S}, batch_size={B}, step_size={step}")
    if args.profile_metrics:
        print("Profiling per-metric time (seconds). CUDA ops are synchronized for accurate timings.")

    for sidx in range(S):
        start = sidx * step
        end = start + B

        Y1 = X1[start:end]
        Y2 = X2[start:end]

        print(f"[{sidx+1}/{S}] window rows [{start}:{end}] (size={B})")
        m, t = compute_metrics_block(Y1, Y2, args, profile_metrics=args.profile_metrics)

        for k in metric_keys:
            metrics_vals[k][sidx] = float(m[k])

        if args.profile_metrics and t is not None:
            for k in time_keys:
                times_vals[k][sidx] = float(t.get(k, np.nan))

    # Keep the old naming convention so downstream code that expects *_mean_over_subsamples still works.
    agg = {k + "_mean_over_subsamples": float(np.nanmean(metrics_vals[k])) for k in metric_keys}

    if args.profile_metrics:
        for k in time_keys:
            agg["time_" + k + "_mean_over_subsamples"] = float(np.nanmean(times_vals[k]))
        ranked = sorted(
            [(k, agg["time_" + k + "_mean_over_subsamples"]) for k in time_keys],
            key=lambda x: x[1],
            reverse=True,
        )
        print("\nMean time per window (seconds), descending:")
        for k, v in ranked:
            print(f"  {k:>12s}: {v:.6f}")

    os.makedirs(args.output_dir, exist_ok=True)
    safe1 = args.ds1.replace("/", "__")
    safe2 = args.ds2.replace("/", "__")
    out_file = args.output_name or f"{safe1}_{safe2}_sliding_metrics.npz"
    out_path = os.path.join(args.output_dir, out_file)

    print(f"Saving to {out_path}")
    payload = {}
    payload.update({k: metrics_vals[k] for k in metric_keys})  # arrays length HOW_MANY
    payload.update(agg)

    if args.profile_metrics:
        payload.update({("time_" + k): times_vals[k] for k in time_keys})

    payload.update(
        dict(
            ds1=args.ds1,
            ds2=args.ds2,
            input_path=args.input_path,
            use_normalized=int(args.use_normalized),
            how_many_samples=int(args.how_many_samples),
            subsample_size=int(args.subsample_size),
            step_size=int(args.step_size),
            seed=int(args.seed),
            path1=p1,
            path2=p2,
            use_indices=str(args.use_indices or ""),
            indices_count=int(indices_count),
            original_num_rows=int(original_N),
            num_rows_after_indices=int(N),
            profile_metrics=bool(args.profile_metrics),
            windowing="sliding",
        )
    )

    np.savez(out_path, **payload)
    print("Done.")


if __name__ == "__main__":
    main()