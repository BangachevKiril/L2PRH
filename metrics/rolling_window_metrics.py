#!/usr/bin/env python
import argparse
import os
import numpy as np
import torch

from compute_metrics import compute_metrics_block, add_metric_args

# ---------------------------------------------------------
#  LOAD FILTERED DATA (same behavior as your original)
# ---------------------------------------------------------
def parse_dataset_spec(spec: str):
    """
    spec format: "<model_name>/<text|img>"
    Example: "google__siglip2-base-patch16-256/img"
    """
    if "/" not in spec:
        raise ValueError(f"Dataset spec must be like MODEL/text or MODEL/img, got: {spec}")
    model_name, which = spec.rsplit("/", 1)
    if which not in ("text", "img"):
        raise ValueError(f"Dataset spec suffix must be 'text' or 'img', got: {spec}")
    return model_name, which

def get_embeddings(input_path: str, dataset_spec: str, device, use_normalized: bool, allow_pickle: bool = True):
    model_name, which = parse_dataset_spec(dataset_spec)
    model_dir = os.path.join(input_path, model_name)
    if not os.path.isdir(model_dir):
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    suffix = "_normalized.npy" if use_normalized else ".npy"
    fname = f"{which}_embeddings{suffix}"  # text_embeddings(.npy) or img_embeddings(.npy)
    path = os.path.join(model_dir, fname)

    if not os.path.isfile(path):
        present = sorted([
            fn for fn in os.listdir(model_dir)
            if fn.endswith(".npy") and ("embeddings" in fn)
        ])
        raise FileNotFoundError(
            f"Expected file not found: {path}\n"
            f"Present .npy files in {model_dir}: {present}"
        )

    arr = np.load(path, allow_pickle=allow_pickle)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D array in {path}, got shape {arr.shape}")

    return torch.from_numpy(arr).to(device), path


# ---------------------------------------------------------
#  ARGPARSE
# ---------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Rolling-window metrics for two filtered datasets.")
    p.add_argument("ds1", type=str, help="Dataset spec: MODEL/text or MODEL/img")
    p.add_argument("ds2", type=str, help="Dataset spec: MODEL/text or MODEL/img")
    p.add_argument("--input_path", type=str, default="/home/kirilb/orcd/pool/PRH_data/LLM_features/filtered_mid_layer_",
               help="Root directory containing per-model folders.")
    p.add_argument("--use_normalized", action="store_true",
               help="If set, loads *_normalized.npy instead of raw *.npy.")

    p.add_argument("--till_when", type=int, default=-1)

    # Rolling window args (same defaults as original)
    p.add_argument("--step_size", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=500)
    p.add_argument("--device", type=str, default="cuda")

    # Output control
    p.add_argument(
        "--output_dir",
        type=str,
        default="/home/kirilb/orcd/pool/PRH_data/LLM_features/metrics",
        help="Directory to save the .npz file.",
    )
    p.add_argument(
        "--output_name",
        type=str,
        default=None,
        help="If set, overrides default '{name1}_{name2}_data_metrics.npz'.",
    )

    add_metric_args(p)
    return p.parse_args()


# ---------------------------------------------------------
#  MAIN
# ---------------------------------------------------------
def main():
    args = parse_args()
    device = torch.device(args.device)

    X1, p1 = get_embeddings(args.input_path, args.ds1, device, use_normalized=args.use_normalized)
    X2, p2 = get_embeddings(args.input_path, args.ds2, device, use_normalized=args.use_normalized)
    print(f"Loaded X1 from: {p1}")
    print(f"Loaded X2 from: {p2}")
    X1 = X1[:args.till_when]
    X2 = X2[:args.till_when]
    print(f"Truncated X1 to: {X1.shape[0]}")
    print(f"Truncated X2 to: {X2.shape[0]}")

    N = X1.shape[0]
    starts = list(range(0, N - args.batch_size + 1, args.step_size))
    L = len(starts)
    print(f"Total segments: {L}")

    metric_keys = [
        "CKA_HSIC",
        "CKA_unbiased",
        "SVCCA_1",
        "SVCCA_2",
        "TOPK10",
        "TOPK100",
        "QUARTET",
        "GLOBAL_THR",
        "LOCAL_THR",
        "KNN_EDIT_10",
        "KNN_EDIT_100",
    ]

    metrics = {k: (np.zeros(L), np.zeros(L)) for k in metric_keys}
    metrics_rand = {k: (np.zeros(L), np.zeros(L)) for k in metric_keys}

    for idx, start in enumerate(starts):
        end = start + args.batch_size
        print(f"[{idx+1}/{L}] rows {start}:{end}")

        Y1 = X1[start:end]
        Y2 = X2[start:end]

        m, mr = compute_metrics_block(Y1, Y2, args, do_random_baseline=True)

        for k in metric_keys:
            v, s = m[k]
            metrics[k][0][idx] = v
            metrics[k][1][idx] = s

            v, s = mr[k]
            metrics_rand[k][0][idx] = v
            metrics_rand[k][1][idx] = s

    os.makedirs(args.output_dir, exist_ok=True)
    safe1 = args.ds1.replace("/", "__")
    safe2 = args.ds2.replace("/", "__")
    out_file = f"{safe1}_{safe2}_data_metrics.npz"
    out_path = os.path.join(args.output_dir, out_file)

    print(f"Saving to {out_path}")
    np.savez(
        out_path,
        **{m + "_mean": arrs[0] for m, arrs in metrics.items()},
        **{m + "_std":  arrs[1] for m, arrs in metrics.items()},
        **{"rand_" + m + "_mean": arrs[0] for m, arrs in metrics_rand.items()},
        **{"rand_" + m + "_std":  arrs[1] for m, arrs in metrics_rand.items()},
    )
    print("Done.")


if __name__ == "__main__":
    main()
