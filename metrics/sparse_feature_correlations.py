#!/usr/bin/env python3
"""
Compare sparse autoencoder features across models with potentially different k.

For each unordered model pair, computes:
- weighted correlation
- binary correlation

and saves them to a separate .npz file.

Optional baseline:
- --rand_permute_baseline 1 randomly permutes the rows of X_j for each pair,
  breaking cross-model correspondence while preserving each matrix marginals.
"""

import argparse
import os
import time
from collections import OrderedDict
from typing import Dict, List, Optional

import numpy as np
import scipy.sparse as sp
from scipy.optimize import linear_sum_assignment


def sparse_feature_path(root: str, d: int, model: str, k: int, use_truncated: bool = True) -> str:
    if use_truncated:
        return os.path.join(root, f"topk_{d}_{model}_k_{k}", "X_features_truncated.npz")
    return os.path.join(root, f"topk_{d}_{model}_k_{k}", "X_features.npz")


def load_csr(npz_path: str) -> sp.csr_matrix:
    X = sp.load_npz(npz_path)
    if not sp.isspmatrix_csr(X):
        X = X.tocsr()
    if X.dtype != np.float32:
        X = X.astype(np.float32)
    return X


def frob_norm_csr(X: sp.csr_matrix) -> float:
    return float(np.sqrt(np.sum(X.data * X.data)))


def to_binary_csr(X: sp.csr_matrix) -> sp.csr_matrix:
    Xb = (X > 0).astype(np.float32)
    if not sp.isspmatrix_csr(Xb):
        Xb = Xb.tocsr()
    return Xb


def permute_rows_csr(X: sp.csr_matrix, perm: np.ndarray) -> sp.csr_matrix:
    return X[perm]


def hungarian_perm_from_affinity(G: np.ndarray):
    G = np.asarray(G, dtype=np.float64)
    if not np.isfinite(G).all():
        bad = ~np.isfinite(G)
        G[bad] = -1e30
    row_ind, col_ind = linear_sum_assignment(-G)
    return np.asarray(row_ind), np.asarray(col_ind)


def aligned_frob_cosine_from_affinity(
    G: np.ndarray, normA: float, normB: float, row: np.ndarray, col: np.ndarray
) -> float:
    if normA == 0.0 or normB == 0.0:
        return float("nan")
    matched_sum = float(G[row, col].sum())
    return matched_sum / (normA * normB)


def affinity_dense(X1: sp.csr_matrix, X2: sp.csr_matrix) -> np.ndarray:
    Gs = X1.T @ X2
    return Gs.toarray().astype(np.float32, copy=False)


class CSRCache:
    def __init__(self, max_items: int):
        self.max_items = max_items
        self._cache: "OrderedDict[str, sp.csr_matrix]" = OrderedDict()

    def get(self, key: str):
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        return None

    def put(self, key: str, val: sp.csr_matrix):
        self._cache[key] = val
        self._cache.move_to_end(key)
        if len(self._cache) > self.max_items:
            self._cache.popitem(last=False)


def _resolve_root(args) -> str:
    return args.topk_root if args.topk_root is not None else args.root


def _resolve_ks(models: List[str], ks: Optional[List[int]], k_scalar: Optional[int]) -> List[int]:
    if ks is None or len(ks) == 0:
        if k_scalar is None:
            raise ValueError("Must provide either --k (scalar) or --ks (list aligned with --models).")
        return [int(k_scalar)] * len(models)
    if len(ks) != len(models):
        raise ValueError(f"--ks length ({len(ks)}) must match --models length ({len(models)}).")
    ks_int = [int(x) for x in ks]
    for i, kv in enumerate(ks_int):
        if kv <= 0:
            raise ValueError(f"All k must be positive. Bad k at index {i}: {kv}")
    return ks_int


def pair_output_name(d: int, mi: str, ki: int, mj: str, kj: int) -> str:
    return f"topk_{d}_{mi}_k_{ki}_topk_{d}_{mj}_k_{kj}_subsample_metrics.npz"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d", type=int, required=True)
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--k", type=int, default=None, help="Legacy: K used for all models if --ks not provided.")
    ap.add_argument("--use_truncated", type=int, default=1, help="Whether to use truncated features.")
    ap.add_argument("--ks", nargs="+", type=int, default=None,
                    help="Per-model K list aligned with --models.")
    ap.add_argument("--root", type=str,
                    default="/home/kirilb/orcd/pool/PRH_data/topk_sae_embedded_coco_new",
                    help="Root directory containing topk_{d}_{model}_k_{k}/X_features*.npz")
    ap.add_argument("--topk_root", type=str, default=None,
                    help="Alias for --root. If set, overrides --root.")
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--reuse_perm_for_binary", type=int, default=0)
    ap.add_argument("--cache_models", type=int, default=4)
    ap.add_argument("--rand_permute_baseline", type=int, default=0,
                    help="If 1, randomly permute rows of X_j for each pair.")
    ap.add_argument("--seed", type=int, default=0,
                    help="Random seed used when rand_permute_baseline=1.")
    args = ap.parse_args()

    root = _resolve_root(args)
    models: List[str] = list(args.models)
    M = len(models)
    d = int(args.d)
    ks_list = _resolve_ks(models=models, ks=args.ks, k_scalar=args.k)
    rng = np.random.default_rng(args.seed)

    os.makedirs(args.out_dir, exist_ok=True)

    log_path = os.path.join(args.out_dir, f"log_d{d}_kvar.txt")
    models_path = os.path.join(args.out_dir, f"models_d{d}_kvar.txt")
    ks_path = os.path.join(args.out_dir, f"ks_d{d}_kvar.txt")

    with open(models_path, "w", encoding="utf-8") as f:
        for m in models:
            f.write(m + "\n")

    with open(ks_path, "w", encoding="utf-8") as f:
        for m, kv in zip(models, ks_list):
            f.write(f"{m} {kv}\n")

    def log(msg: str):
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{ts}] {msg}"
        print(line, flush=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    log(f"Starting SciPy run: M={M}, d={d}, kvar=True")
    log(f"root={root}")
    log(f"out_dir={args.out_dir}")
    log(f"reuse_perm_for_binary={args.reuse_perm_for_binary}")
    log(f"rand_permute_baseline={args.rand_permute_baseline}")
    log(f"seed={args.seed}")
    log(f"cache_models={args.cache_models}")

    cache = CSRCache(max_items=max(0, args.cache_models))
    norms_w: Dict[str, float] = {}
    norms_b: Dict[str, float] = {}
    model_to_k = {m: k for m, k in zip(models, ks_list)}

    def get_X(model: str) -> sp.csr_matrix:
        k_model = model_to_k[model]
        path = sparse_feature_path(root, d, model, k_model, use_truncated=bool(args.use_truncated))
        X = cache.get(path) if args.cache_models > 0 else None
        if X is None:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Missing: {path}")
            X = load_csr(path)
            if X.shape[1] != d and not bool(args.use_truncated):
                raise ValueError(f"{model} (k={k_model}): X.shape={X.shape}, expected (*, {d})")
            if args.cache_models > 0:
                cache.put(path, X)
        return X

    t0 = time.time()
    for model in models:
        X = get_X(model)
        norms_w[model] = frob_norm_csr(X)
        norms_b[model] = frob_norm_csr(to_binary_csr(X))
    log(f"Computed norms in {time.time() - t0:.2f}s")

    for i in range(M):
        mi = models[i]
        Xi = get_X(mi)
        ki = model_to_k[mi]

        for j in range(i + 1, M):
            mj = models[j]
            Xj_orig = get_X(mj)
            kj = model_to_k[mj]

            if Xi.shape[0] != Xj_orig.shape[0]:
                raise ValueError(
                    f"Row mismatch for pair {mi} vs {mj}: {Xi.shape[0]} vs {Xj_orig.shape[0]}. "
                    "RAND_PERMUTE_BASELINE assumes matched row counts."
                )

            Xj = Xj_orig
            if args.rand_permute_baseline:
                row_perm = rng.permutation(Xj_orig.shape[0])
                Xj = permute_rows_csr(Xj_orig, row_perm)

            log(f"Pair ({i},{j}) {mi}(k={ki}) vs {mj}(k={kj})")

            tw = time.time()
            Gw = affinity_dense(Xi, Xj)
            row_w, col_w = hungarian_perm_from_affinity(Gw)
            cw = aligned_frob_cosine_from_affinity(Gw, norms_w[mi], norms_w[mj], row_w, col_w)
            log(f"  weighted: corr={cw:.6f}  (aff+assign {time.time()-tw:.2f}s)")

            tb = time.time()
            Xib = to_binary_csr(Xi)
            Xjb = to_binary_csr(Xj)
            Gb = affinity_dense(Xib, Xjb)
            if args.reuse_perm_for_binary:
                row_b, col_b = row_w, col_w
            else:
                row_b, col_b = hungarian_perm_from_affinity(Gb)
            cb = aligned_frob_cosine_from_affinity(Gb, norms_b[mi], norms_b[mj], row_b, col_b)
            log(f"  binary:   corr={cb:.6f}  (aff+assign {time.time()-tb:.2f}s)")

            out_path = os.path.join(args.out_dir, pair_output_name(d, mi, ki, mj, kj))
            np.savez_compressed(
                out_path,
                weighted_correlation=np.float32(cw),
                binary_correlation=np.float32(cb),
            )
            log(f"  saved: {out_path}")

            del Gw, Gb, Xib, Xjb

    log("Done.")


if __name__ == "__main__":
    main()
