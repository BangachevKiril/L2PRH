#!/usr/bin/env python
import argparse
import numpy as np
import torch
import time

from metrics import (
    cka,
    svcca,
    top_k_knn,
    knn_edit_distance,
)

def normalize_output(d):
    return float(d["value"])

def _maybe_cuda_sync(t):
    if t is not None and torch.is_tensor(t) and t.is_cuda:
        torch.cuda.synchronize()

def _time_call(name, fn, sync_tensor=None):
    """
    Times fn() using perf_counter. If sync_tensor is a CUDA tensor, syncs before+after.
    Returns (result, elapsed_seconds).
    """
    _maybe_cuda_sync(sync_tensor)
    t0 = time.perf_counter()
    out = fn()
    _maybe_cuda_sync(sync_tensor)
    dt = time.perf_counter() - t0
    return out, dt

def compute_metrics_once(Y1: torch.Tensor, Y2: torch.Tensor, args, profile_metrics: bool = False):
    """
    Returns:
      metrics: dict metric_name -> float
      times:   dict metric_name -> seconds   (only if profile_metrics=True, else None)
    """
    metrics = {}
    times = {} if profile_metrics else None

    # Ensure float
    Y1 = Y1.float()
    Y2 = Y2.float()

    # -------------
    # Kernel once
    # -------------
    if profile_metrics:
        (K1, K2), dt = _time_call(
            "KERNEL",
            lambda: (Y1 @ Y1.T, Y2 @ Y2.T),
            sync_tensor=Y1
        )
        times["KERNEL"] = dt
    else:
        K1 = Y1 @ Y1.T
        K2 = Y2 @ Y2.T

    # -------------
    # CKA (kernels)
    # -------------
    if profile_metrics:
        v, dt = _time_call("CKA_HSIC", lambda: normalize_output(cka(K1, K2, "HSIC", is_kernel=True)), sync_tensor=K1)
        metrics["CKA_HSIC"] = v
        times["CKA_HSIC"] = dt

        v, dt = _time_call("CKA_unbiased", lambda: normalize_output(cka(K1, K2, "unbiased_HSIC", is_kernel=True)), sync_tensor=K1)
        metrics["CKA_unbiased"] = v
        times["CKA_unbiased"] = dt
    else:
        metrics["CKA_HSIC"] = normalize_output(cka(K1, K2, "HSIC", is_kernel=True))
        metrics["CKA_unbiased"] = normalize_output(cka(K1, K2, "unbiased_HSIC", is_kernel=True))

    # -------------
    # SVCCA (repr)
    # -------------
    if profile_metrics:
        v, dt = _time_call("SVCCA_1", lambda: normalize_output(svcca(Y1, Y2, cca_dim=args.svcca_dim1)), sync_tensor=Y1)
        metrics["SVCCA_1"] = v
        times["SVCCA_1"] = dt

        v, dt = _time_call("SVCCA_2", lambda: normalize_output(svcca(Y1, Y2, cca_dim=args.svcca_dim2)), sync_tensor=Y1)
        metrics["SVCCA_2"] = v
        times["SVCCA_2"] = dt
    else:
        metrics["SVCCA_1"] = normalize_output(svcca(Y1, Y2, cca_dim=args.svcca_dim1))
        metrics["SVCCA_2"] = normalize_output(svcca(Y1, Y2, cca_dim=args.svcca_dim2))

    # -------------
    # TOPK overlap (kernels)
    # -------------
    if profile_metrics:
        v, dt = _time_call("TOPK10", lambda: normalize_output(top_k_knn(K1, K2, k=args.topk_k1, is_kernel=True)), sync_tensor=K1)
        metrics["TOPK10"] = v
        times["TOPK10"] = dt

        v, dt = _time_call("TOPK100", lambda: normalize_output(top_k_knn(K1, K2, k=args.topk_k2, is_kernel=True)), sync_tensor=K1)
        metrics["TOPK100"] = v
        times["TOPK100"] = dt
    else:
        metrics["TOPK10"] = normalize_output(top_k_knn(K1, K2, k=args.topk_k1, is_kernel=True))
        metrics["TOPK100"] = normalize_output(top_k_knn(K1, K2, k=args.topk_k2, is_kernel=True))

    # -------------
    # Edit distance (kernels)
    # -------------
    if profile_metrics:
        v, dt = _time_call("KNN_EDIT_10", lambda: normalize_output(knn_edit_distance(K1, K2, k=args.edit_k1, is_kernel=True)), sync_tensor=K1)
        metrics["KNN_EDIT_10"] = v
        times["KNN_EDIT_10"] = dt

        v, dt = _time_call("KNN_EDIT_100", lambda: normalize_output(knn_edit_distance(K1, K2, k=args.edit_k2, is_kernel=True)), sync_tensor=K1)
        metrics["KNN_EDIT_100"] = v
        times["KNN_EDIT_100"] = dt
    else:
        metrics["KNN_EDIT_10"] = normalize_output(knn_edit_distance(K1, K2, k=args.edit_k1, is_kernel=True))
        metrics["KNN_EDIT_100"] = normalize_output(knn_edit_distance(K1, K2, k=args.edit_k2, is_kernel=True))

    return metrics, times

def compute_metrics_block(Y1: torch.Tensor, Y2: torch.Tensor, args, profile_metrics: bool = False):
    # aligned only; no random baseline
    return compute_metrics_once(Y1, Y2, args, profile_metrics=profile_metrics)

def add_metric_args(p: argparse.ArgumentParser):
    p.add_argument("--svcca_dim1", type=int, default=10)
    p.add_argument("--svcca_dim2", type=int, default=100)
    p.add_argument("--topk_k1", type=int, default=10)
    p.add_argument("--topk_k2", type=int, default=100)
    p.add_argument("--edit_k1", type=int, default=10)
    p.add_argument("--edit_k2", type=int, default=100)
    return p
