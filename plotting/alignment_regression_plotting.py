#!/usr/bin/env python3
import os
import re
import glob
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# Metric mapping + labels
# ============================================================
_METRIC_KEY = {
    "cka": "CKA_HSIC_mean_over_subsamples",
    "cka_unbiased": "CKA_unbiased_mean_over_subsamples",
    "svcca_10": "SVCCA_1_mean_over_subsamples",
    "svcca_100": "SVCCA_2_mean_over_subsamples",
    "topk_10": "TOPK10_mean_over_subsamples",
    "topk_100": "TOPK100_mean_over_subsamples",
    "knn_edit_10": "KNN_EDIT_10_mean_over_subsamples",
    "knn_edit_100": "KNN_EDIT_100_mean_over_subsamples",
}

_HUMAN_TITLES = {
    "cka": "CKA",
    "cka_unbiased": "Unbiased CKA",
    "svcca_10": "SVCCA 10",
    "svcca_100": "SVCCA 100",
    "topk_10": "KNN Overlap 10",
    "topk_100": "KNN Overlap 100",
    "knn_edit_10": "KNN-10 Edit",
    "knn_edit_100": "KNN-100 Edit",
}

feature_names_human = [
    "min params",
    "max params",
    "min depth",
    "max depth",
    "min dim",
    "max dim",
    "min training images",
    "max training images",
    "min training text tokens",
    "max training text tokens",
    "text-text",
    "img-text",
    "img-img",
    "min year",
    "max year",
]


# ============================================================
# Small utilities
# ============================================================
def _pdf_path(path: str) -> str:
    root, _ = os.path.splitext(path)
    return root + ".pdf"


def _permute_feature_tensor(feature_tensor: np.ndarray, order: Sequence[int]) -> np.ndarray:
    order = np.asarray(order, dtype=int)
    return feature_tensor[np.ix_(order, order, np.arange(feature_tensor.shape[2]))]


def _normalize_text(s: Optional[str]) -> str:
    if s is None:
        return ""
    s = str(s)
    replacements = {
        "\u2212": "-",
        "\u2013": "-",
        "\u2014": "-",
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2026": "...",
        "\u00a0": " ",
        "\t": " ",
        "\r": " ",
        "\n": " ",
    }
    for old, new in replacements.items():
        s = s.replace(old, new)
    return " ".join(s.split())


def _pretty_model_name(s: str) -> str:
    s = str(s).strip()

    for suffix in ("/text", "/img", "__text", "__img", "_text", "_img"):
        if s.endswith(suffix):
            s = s[: -len(suffix)]
            break

    vendor = None
    model = s

    if "__" in s:
        vendor, model = s.split("__", 1)

    vendor_lower = (vendor or "").lower()
    model_lower = model.lower()

    if vendor_lower == "laion" and "clip" in model_lower:
        prefix = "LAION CLIP"
        tail = model
        tail = re.sub(r"(?i)^clip[-_ ]*", "", tail)
        tail = re.sub(r"(?i)^vit[-_ ]*", "", tail)
        tail = re.sub(r"(?i)[-_ ]*patch(14|16|32)([-_ ]*256)?", "", tail)
        tail = re.sub(r"(?i)[-_ ]*laion2b-s34b-b79k", "", tail)
        tail = re.sub(r"(?i)[-_ ]*laion2b-s32b-b79k", "", tail)
        tail = tail.replace("-base", " B").replace("-Base", " B")
        tail = tail.replace("-large", " L").replace("-Large", " L")
        tail = tail.replace("-huge", " H").replace("-Huge", " H")
        tail = tail.replace("-", " ").replace("_", " ")
        tail = " ".join(tail.split())
        return " ".join(x for x in [prefix, tail] if x).strip()

    if vendor_lower == "openai" and "clip" in model_lower:
        prefix = "OPENAI CLIP"
        tail = model
        tail = re.sub(r"(?i)^clip[-_ ]*", "", tail)
        tail = re.sub(r"(?i)^vit[-_ ]*", "", tail)
        tail = re.sub(r"(?i)[-_ ]*patch(14|16|32)([-_ ]*256)?", "", tail)
        tail = tail.replace("-base", " B").replace("-Base", " B")
        tail = tail.replace("-large", " L").replace("-Large", " L")
        tail = tail.replace("-huge", " H").replace("-Huge", " H")
        tail = tail.replace("-", " ").replace("_", " ")
        tail = " ".join(tail.split())
        return " ".join(x for x in [prefix, tail] if x).strip()

    if vendor is not None:
        s = model

    replacements = [
        ("Qwen3-", "Qwen "),
        ("Llama-3.2-", "Llama "),
        ("gemma-3-", "Gemma "),
        ("siglip2-", "SigLIP2 "),
        ("clip-vit-", "CLIP "),
        ("dinov2-", "DINOv2 "),
        ("vit-mae-", "ViT MAE "),
        ("beit-", "BEiT "),
        ("-Base", " B"),
        ("-base", " B"),
        ("-Large", " L"),
        ("-large", " L"),
        ("-Huge", " H"),
        ("-huge", " H"),
        ("-patch16-256", ""),
        ("-patch14", ""),
        ("-patch32", ""),
        ("-patch16", ""),
        ("-224", ""),
        ("-pt22k", ""),
        ("-en-v1.5", ""),
        ("-v1.5", ""),
        ("-v2-moe", " v2-moe"),
        ("-Instruct", ""),
        ("-it", " it"),
        ("-laion2B-s34B-b79K", ""),
        ("-laion2B-s32B-b79K", ""),
    ]
    for old, new in replacements:
        s = s.replace(old, new)

    s = s.replace("__", " ").replace("-", " ").replace("_", " ").replace("/", " ")
    s = " ".join(s.split())
    return s


# ============================================================
# Pairwise feature tensor from xlsx
# ============================================================
def build_pairwise_feature_tensor_from_xlsx(
    xlsx_path: str,
    *,
    sheet_name: Optional[Union[str, int]] = 0,
    strict: bool = True,
) -> tuple[np.ndarray, List[str], List[str]]:
    """
    Build an (M, M, 15) tensor of pairwise model features from an .xlsx file.
    """

    def _norm_key(s: str) -> str:
        return "".join(ch.lower() for ch in str(s) if ch.isalnum())

    row_aliases: Dict[str, Tuple[str, ...]] = {
        "number_of_parameters": ("numberofparameters", "params", "parameters", "numparams"),
        "model_depth": ("modeldepth", "depth", "layers", "nlayers", "n_layers"),
        "dimension": (
            "dimension",
            "dim",
            "modelwidth",
            "width",
            "modelwidthhiddensize",
            "widthhiddensize",
            "hiddensize",
            "hidden_size",
        ),
        "text_tokens_trained_on": ("texttokenstrainedon", "texttokens", "tokens", "tokenstrainedon"),
        "images_trained_on": ("imagestrainedon", "imagetrainedon", "images", "imagecount"),
        "modality_of_operation": ("modalityofoperation", "modality", "mode", "inputmodality", "operationmodality"),
        "year_published": ("yearpublished", "year", "publicationyear"),
    }

    def _find_row(df: pd.DataFrame, canonical: str) -> str:
        candidates = set(row_aliases[canonical])
        for idx in df.index:
            if _norm_key(idx) in candidates:
                return idx
        raise KeyError(
            f"Could not find row for '{canonical}'. "
            f"Expected one of: {row_aliases[canonical]}. "
            f"Available rows (first ~25): {list(df.index)[:25]}"
        )

    def _to_float(x) -> float:
        if pd.isna(x):
            return float("nan")
        if isinstance(x, str):
            s = x.strip()
            try:
                return float(s.replace(",", ""))
            except Exception:
                return float("nan")
        try:
            return float(x)
        except Exception:
            return float("nan")

    def _to_modality(x) -> str:
        s = str(x).strip().lower()
        if s in {"text", "txt", "language", "llm"}:
            return "text"
        if s in {"img", "image", "vision", "visual"}:
            return "img"
        if strict:
            raise ValueError(f"Unrecognized modality value: {x!r} (expected 'text' or 'img' variants)")
        return "unknown"

    def _minmax_pairwise(a: np.ndarray, m: int) -> tuple[np.ndarray, np.ndarray]:
        a_col = a.reshape(m, 1)
        a_row = a.reshape(1, m)
        return np.minimum(a_col, a_row), np.maximum(a_col, a_row)

    df_raw = pd.read_excel(xlsx_path, sheet_name=sheet_name)

    if isinstance(df_raw, dict):
        if len(df_raw) == 0:
            raise ValueError("Excel file contained no readable sheets.")
        first_key = next(iter(df_raw.keys()))
        df_raw = df_raw[first_key]

    if not isinstance(df_raw, pd.DataFrame):
        raise TypeError(f"Unexpected type from pd.read_excel: {type(df_raw)}")

    if df_raw.shape[1] < 2:
        raise ValueError("Expected at least 2 columns: first for row labels, others for model columns.")

    row_col = df_raw.columns[0]
    df = df_raw.set_index(row_col)

    model_names = [str(c) for c in df.columns]
    m = len(model_names)

    r_params = _find_row(df, "number_of_parameters")
    r_depth = _find_row(df, "model_depth")
    r_dim = _find_row(df, "dimension")
    r_tokens = _find_row(df, "text_tokens_trained_on")
    r_images = _find_row(df, "images_trained_on")
    r_mod = _find_row(df, "modality_of_operation")
    r_year = _find_row(df, "year_published")

    params = np.array([_to_float(df.at[r_params, name]) for name in model_names], dtype=np.float64)
    depth = np.array([_to_float(df.at[r_depth, name]) for name in model_names], dtype=np.float64)
    dim = np.array([_to_float(df.at[r_dim, name]) for name in model_names], dtype=np.float64)
    tokens = np.array([_to_float(df.at[r_tokens, name]) for name in model_names], dtype=np.float64)
    images = np.array([_to_float(df.at[r_images, name]) for name in model_names], dtype=np.float64)
    years = np.array([_to_float(df.at[r_year, name]) for name in model_names], dtype=np.float64)
    modality = [_to_modality(df.at[r_mod, name]) for name in model_names]

    if strict:
        if np.any(np.isnan(params)):
            raise ValueError("NaNs found in Number_of_parameters after parsing.")
        if np.any(params <= 0):
            bad = [(model_names[i], float(params[i])) for i in range(m) if params[i] <= 0]
            raise ValueError(f"Number_of_parameters must be > 0 for log. Bad entries: {bad}")

        if np.any(tokens < 0) or np.any(images < 0):
            bad_t = [(model_names[i], float(tokens[i])) for i in range(m) if tokens[i] < 0]
            bad_i = [(model_names[i], float(images[i])) for i in range(m) if images[i] < 0]
            raise ValueError(f"Tokens/images must be >= 0. Bad tokens: {bad_t}. Bad images: {bad_i}")

        if np.any(np.isnan(depth)) or np.any(np.isnan(dim)) or np.any(np.isnan(years)):
            raise ValueError("NaNs found in depth/dimension/year rows after parsing.")

    log_params = np.log(params)
    log1p_images = np.log1p(images)
    log1p_tokens = np.log1p(tokens)

    tensor = np.zeros((m, m, 15), dtype=np.float32)

    tensor[:, :, 0], tensor[:, :, 1] = _minmax_pairwise(log_params, m)
    tensor[:, :, 2], tensor[:, :, 3] = _minmax_pairwise(depth, m)
    tensor[:, :, 4], tensor[:, :, 5] = _minmax_pairwise(dim, m)
    tensor[:, :, 6], tensor[:, :, 7] = _minmax_pairwise(log1p_images, m)
    tensor[:, :, 8], tensor[:, :, 9] = _minmax_pairwise(log1p_tokens, m)

    mod_arr = np.array(modality, dtype=object)
    mi = mod_arr.reshape(m, 1)
    mj = mod_arr.reshape(1, m)

    is_tt = (mi == "text") & (mj == "text")
    is_ii = (mi == "img") & (mj == "img")
    is_it = ~is_tt & ~is_ii

    tensor[:, :, 10] = is_tt.astype(np.float32)
    tensor[:, :, 11] = is_it.astype(np.float32)
    tensor[:, :, 12] = is_ii.astype(np.float32)
    tensor[:, :, 13], tensor[:, :, 14] = _minmax_pairwise(years, m)

    feature_names = [
        "params_log_min",
        "params_log_max",
        "depth_min",
        "depth_max",
        "dimension_min",
        "dimension_max",
        "images_log1p_min",
        "images_log1p_max",
        "text_tokens_log1p_min",
        "text_tokens_log1p_max",
        "modality_text_text",
        "modality_img_text",
        "modality_img_img",
        "year_min",
        "year_max",
    ]

    return tensor, feature_names, model_names


# ============================================================
# Alignment matrix builder
# ============================================================
def build_alignment_matrix_30x30(
    path: str,
    metric_key: str,
    *,
    models: List[str],
    diag_value: float = np.nan,
    missing: str = "nan",
    sort_keys: Optional[Sequence[np.ndarray]] = None,
    return_order: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, List[int]]]:
    """
    Build an MxM alignment matrix from pairwise .npz files in `path`.
    """

    def _model_token_variants(model: str) -> List[str]:
        out = [model]
        if model.endswith("__text"):
            out.append(model[:-6] + "_text")
        elif model.endswith("__img"):
            out.append(model[:-5] + "_img")

        if model.endswith("_text"):
            out.append(model[:-5] + "__text")
        elif model.endswith("_img"):
            out.append(model[:-4] + "__img")

        seen, deduped = set(), []
        for x in out:
            if x not in seen:
                seen.add(x)
                deduped.append(x)
        return deduped

    ktag_re = re.compile(r"_k\d+")

    def _strip_ktags(s: str) -> str:
        return ktag_re.sub("", s)

    def _ktag_glob_variants(token: str) -> List[str]:
        out = [token]
        if ktag_re.search(token):
            out.append(ktag_re.sub("_k*", token))
        base = _strip_ktags(token)
        out.append(base)
        out.append(base + "_k*")

        seen, deduped = set(), []
        for x in out:
            if x not in seen:
                seen.add(x)
                deduped.append(x)
        return deduped

    def _find_pair_file(dir_path: str, m1: str, m2: str) -> Optional[str]:
        v1 = _model_token_variants(m1)
        v2 = _model_token_variants(m2)
        v1p = [p for x in v1 for p in _ktag_glob_variants(x)]
        v2p = [p for x in v2 for p in _ktag_glob_variants(x)]

        def _dedup(items):
            seen, out = set(), []
            for item in items:
                if item not in seen:
                    seen.add(item)
                    out.append(item)
            return out

        v1p = _dedup(v1p)
        v2p = _dedup(v2p)

        for a_list, b_list in ((v1p, v2p), (v2p, v1p)):
            for a in a_list:
                for b in b_list:
                    for sep in ("_", "__"):
                        if "*" not in a and "*" not in b:
                            candidate = os.path.join(dir_path, f"{a}{sep}{b}.npz")
                            if os.path.exists(candidate):
                                return candidate

        for a_list, b_list in ((v1p, v2p), (v2p, v1p)):
            for a in a_list:
                for b in b_list:
                    for sep in ("_", "__"):
                        hits = glob.glob(os.path.join(dir_path, f"{a}{sep}{b}.npz"))
                        if hits:
                            hits.sort()
                            return hits[0]
        return None

    def _as_float(v) -> float:
        if np.isscalar(v):
            return float(v)
        arr = np.asarray(v)
        if arr.ndim == 0:
            return float(arr.item())
        return float(np.mean(arr))

    def _load_one_metric(npz_path: str, key: str) -> Optional[float]:
        with np.load(npz_path, allow_pickle=True) as z:
            if key in set(z.files):
                return _as_float(z[key])
        return None

    m = len(models)
    alignment = np.full((m, m), np.nan, dtype=float)
    np.fill_diagonal(alignment, diag_value)

    for i in range(m):
        for j in range(i + 1, m):
            m1, m2 = models[i], models[j]
            npz_path = _find_pair_file(path, m1, m2)

            if npz_path is None:
                if missing == "raise":
                    raise FileNotFoundError(f"Missing pair file for ({m1}, {m2}) in {path}")
                continue

            val = _load_one_metric(npz_path, metric_key)
            if val is None:
                if missing == "raise":
                    raise KeyError(f"Metric key '{metric_key}' missing in {npz_path}")
                continue

            alignment[i, j] = alignment[j, i] = val

    order = list(range(m))
    if sort_keys is not None:
        if len(sort_keys) == 0:
            raise ValueError("sort_keys was provided but empty. Pass None or a non-empty list of (M,) arrays.")

        keys = []
        for idx, key in enumerate(sort_keys):
            key = np.asarray(key)
            if key.shape != (m,):
                raise ValueError(f"sort_keys[{idx}] must have shape ({m},), got {key.shape}")
            keys.append(key)

        order = np.lexsort(tuple(keys[::-1])).tolist()
        alignment = alignment[np.ix_(order, order)]

    if return_order:
        return alignment, order
    return alignment


# ============================================================
# Ridge regression
# ============================================================
def ridge_regress_alignment_on_features(
    alignment: np.ndarray,
    features: np.ndarray,
    lam: float,
    *,
    fit_intercept: bool = True,
    drop_nan: bool = True,
    return_design: bool = False,
) -> Dict[str, np.ndarray]:
    """
    Ridge regression of pairwise alignment on pairwise model features.
    """
    alignment = np.asarray(alignment)
    features = np.asarray(features)

    if alignment.ndim != 2 or alignment.shape[0] != alignment.shape[1]:
        raise ValueError(f"alignment must be square (M,M); got {alignment.shape}")

    m = alignment.shape[0]
    if features.shape != (m, m, 15):
        raise ValueError(f"features must have shape (M,M,15) with same M; got {features.shape} vs M={m}")
    if lam < 0:
        raise ValueError("lam must be >= 0")

    iu = np.triu_indices(m, k=1)
    idx_i, idx_j = iu[0].astype(np.int64), iu[1].astype(np.int64)

    V = alignment[iu].astype(np.float64)
    A = features[iu].astype(np.float64)

    if drop_nan:
        ok = np.isfinite(V) & np.all(np.isfinite(A), axis=1)
        V = V[ok]
        A = A[ok]
        idx_i = idx_i[ok]
        idx_j = idx_j[ok]

    n, d = A.shape
    if n == 0:
        raise ValueError("No usable pairs after filtering (all NaN/inf?)")
    if d != 15:
        raise ValueError(f"Expected 15 features; got {d}")

    means = A.mean(axis=0)
    A0 = A - means[None, :]

    norms = np.linalg.norm(A0, axis=0)
    safe = norms > 0

    A_norm = np.zeros_like(A0)
    A_norm[:, safe] = A0[:, safe] / norms[safe][None, :]

    if fit_intercept:
        X = np.concatenate([np.ones((n, 1), dtype=np.float64), A_norm], axis=1)
        P = np.diag(np.r_[0.0, np.full(d, lam, dtype=np.float64)])
    else:
        X = A_norm
        P = np.diag(np.full(d, lam, dtype=np.float64))

    XtX = X.T @ X
    Xty = X.T @ V
    beta = np.linalg.solve(XtX + P, Xty)

    if fit_intercept:
        intercept = float(beta[0])
        w = beta[1:].copy()
    else:
        intercept = 0.0
        w = beta.copy()

    y_hat = X @ beta
    resid = V - y_hat

    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((V - V.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    rmse = float(np.sqrt(ss_res / n))

    out: Dict[str, np.ndarray] = {
        "w": w.astype(np.float64),
        "intercept": np.array(intercept, dtype=np.float64),
        "y_hat": y_hat.astype(np.float64),
        "resid": resid.astype(np.float64),
        "r2": np.array(r2, dtype=np.float64),
        "rmse": np.array(rmse, dtype=np.float64),
        "n_pairs": np.array(n, dtype=np.int64),
        "idx_i": idx_i,
        "idx_j": idx_j,
        "feature_means": means.astype(np.float64),
        "feature_norms": norms.astype(np.float64),
    }

    if return_design:
        out["A_raw"] = A
        out["A_normalized"] = A_norm
        out["V_used"] = V

    return out


# ============================================================
# Main wrapper + plotting
# ============================================================
def fit_all_metrics_from_path_and_plot_coef_heatmap(
    *,
    metrics_dir: str,
    models: List[str],
    feature_tensor: np.ndarray,
    lam: float,
    feature_names: List[str],
    feature_names_human: Optional[List[str]] = None,
    abbreviated_model_names: Optional[List[str]] = None,
    diag_value: float = np.nan,
    missing: str = "nan",
    sort_keys: Optional[List[np.ndarray]] = None,
    fit_intercept: bool = True,
    drop_nan: bool = True,
    vlim: Optional[float] = None,
    figsize: Tuple[float, float] = (16, 7),
    annotate: bool = False,
    cmap: str = "bwr",
    title: Optional[str] = None,
    save_fig: Optional[str] = None,
    close_plot: bool = False,
    show_plot: bool = True,
) -> dict:
    """
    Wrapper that:
      1) builds alignment matrices for all metrics in _METRIC_KEY
      2) converts edit distance to similarity
      3) fits ridge regression for each metric
      4) plots an 8 x 15 coefficient heatmap
      5) saves as PDF if save_fig is provided

    Parameters
    ----------
    abbreviated_model_names : optional list[str]
        Accepted for API compatibility with your other plotting utilities.
        It is not used in the coefficient heatmap itself, but is returned in
        the output dictionary and validated against `models`.
    """
    if len(feature_names) != 15:
        raise ValueError(f"feature_names must have length 15, got {len(feature_names)}")

    if feature_names_human is None:
        feature_names_human = feature_names
    if len(feature_names_human) != 15:
        raise ValueError(f"feature_names_human must have length 15, got {len(feature_names_human)}")

    m = len(models)
    if feature_tensor.shape != (m, m, 15):
        raise ValueError(f"feature_tensor must have shape ({m},{m},15); got {feature_tensor.shape}")

    if abbreviated_model_names is not None and len(abbreviated_model_names) != m:
        raise ValueError(
            f"abbreviated_model_names must have length {m}, got {len(abbreviated_model_names)}"
        )

    alignment_by_metric: Dict[str, np.ndarray] = {}
    results_by_metric: Dict[str, dict] = {}

    metrics_used = list(_METRIC_KEY.keys())
    coef_matrix = np.full((len(metrics_used), 15), np.nan, dtype=float)

    used_models = list(models)
    used_abbreviated_model_names = (
        list(abbreviated_model_names)
        if abbreviated_model_names is not None
        else [_pretty_model_name(x) for x in models]
    )

    used_feature_tensor = feature_tensor
    order = list(range(m))

    if sort_keys is not None:
        keys = []
        for idx, key in enumerate(sort_keys):
            key = np.asarray(key)
            if key.shape != (m,):
                raise ValueError(f"sort_keys[{idx}] must have shape ({m},), got {key.shape}")
            keys.append(key)

        order = np.lexsort(tuple(keys[::-1])).tolist()
        used_models = [models[k] for k in order]
        used_abbreviated_model_names = [used_abbreviated_model_names[k] for k in order]
        used_feature_tensor = _permute_feature_tensor(feature_tensor, order)

    for row_idx, metric in enumerate(metrics_used):
        metric_key = _METRIC_KEY[metric]

        alignment = build_alignment_matrix_30x30(
            metrics_dir,
            metric_key=metric_key,
            models=models,
            diag_value=diag_value,
            missing=missing,
            sort_keys=sort_keys,
            return_order=False,
        )

        if metric == "knn_edit_10":
            alignment = np.where(np.isfinite(alignment), 1.0 - (alignment / 10.0), alignment)
        elif metric == "knn_edit_100":
            alignment = np.where(np.isfinite(alignment), 1.0 - (alignment / 100.0), alignment)

        alignment_by_metric[metric] = alignment

        res = ridge_regress_alignment_on_features(
            alignment,
            used_feature_tensor,
            lam,
            fit_intercept=fit_intercept,
            drop_nan=drop_nan,
            return_design=False,
        )
        results_by_metric[metric] = res
        coef_matrix[row_idx, :] = res["w"]

    titles_used = [_HUMAN_TITLES.get(metric, metric) for metric in metrics_used]

    vmax = float(np.nanmax(np.abs(coef_matrix))) if vlim is None else float(vlim)
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = 1e-8
    vmin = -vmax

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(coef_matrix, aspect="auto", vmin=vmin, vmax=vmax, cmap=cmap)

    ax.set_yticks(np.arange(len(metrics_used)))
    ax.set_yticklabels(titles_used, fontsize=15)

    ax.set_xticks(np.arange(15))
    ax.set_xticklabels(feature_names_human, rotation=45, ha="right", fontsize=15)

    if title is not None:
        ax.set_title(_normalize_text(title), fontsize=20)
    else:
        ax.set_title(
            _normalize_text(
                "Importance of Models Specifications for Alignment\n"
                f"(Via Ridge with λ={lam:g})"
            ),
            fontsize=20,
        )

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Coefficient value")

    ax.set_xticks(np.arange(-0.5, 15, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(metrics_used), 1), minor=True)
    ax.grid(which="minor", linestyle="-", linewidth=0.5)
    ax.tick_params(which="minor", bottom=False, left=False)

    if annotate:
        for i in range(coef_matrix.shape[0]):
            for j in range(coef_matrix.shape[1]):
                val = coef_matrix[i, j]
                if np.isfinite(val):
                    ax.text(j, i, f"{val:.2g}", ha="center", va="center")

    plt.tight_layout()

    if save_fig is not None:
        final_path = _pdf_path(save_fig)
        out_dir = os.path.dirname(final_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        fig.savefig(final_path, format="pdf", bbox_inches="tight")
        print(f"[saved] figure -> {final_path}")

    if show_plot:
        plt.show()

    if close_plot:
        plt.close(fig)

    return {
        "coef_matrix": coef_matrix,
        "metrics_used": metrics_used,
        "titles_used": titles_used,
        "alignment_by_metric": alignment_by_metric,
        "results_by_metric": results_by_metric,
        "models_used": used_models,
        "abbreviated_model_names_used": used_abbreviated_model_names,
        "sort_order": order,
    }