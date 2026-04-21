#!/usr/bin/env python3
"""
rolling_window_plotting.py

Plot rolling-window alignment metrics vs scaled mean frequency (mean freq)^(-1/2).

Supports:
  1) Grid plot (4x2 by default) over a fixed set of metrics.
  2) Single-metric plot, implemented as a thin wrapper that reuses the SAME core
     plotting function as the grid plot.
  3) Single-metric plot with regression + slope/R^2 heatmaps.

Expected .npz format (per model-pair file):
  Keys (raw curves):
    CKA_HSIC
    CKA_unbiased
    SVCCA_1
    SVCCA_2
    TOPK10
    TOPK100
    KNN_EDIT_10
    KNN_EDIT_100

Optional random baselines:
  Any key that contains the base metric key string AND contains "random" or "rand",
  e.g.:
    CKA_HSIC_random
    random_CKA_HSIC
    CKA_HSIC__rand
  If normalization="normalized", we plot raw - random.
"""

from __future__ import annotations

import csv
import os
from typing import Any

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


# -------------------------
# small helpers
# -------------------------
def _pdf_path(path: str) -> str:
    root, _ = os.path.splitext(path)
    return root + ".pdf"


# -------------------------
# mean frequency helpers (NUMPY ONLY)
# -------------------------
def read_frequencies_from_csv(
    *,
    csv_path: str = "/home/kirilb/orcd/pool/PRH_data/word_list_50000_en_best.csv",
    skip_first: bool = True,
) -> np.ndarray:
    """
    Read frequencies from the CSV. Assumes numeric frequency is in column 3 (row[2]),
    matching your older script.
    """
    frequencies: list[float] = []
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        i = 0
        for row in reader:
            if not row or (i == 0 and skip_first):
                i += 1
                continue
            i += 1
            if len(row) >= 3:
                val = row[2].strip()
                if val:
                    frequencies.append(float(val))
    return np.asarray(frequencies, dtype=np.float64)


def get_means(freq: np.ndarray, *, batch_size: int, step_size: int) -> np.ndarray:
    """
    Rolling window means: mean(freq[i:i+batch_size]) for i = 0, step_size, 2*step_size, ...
    """
    freq = np.asarray(freq, dtype=np.float64).ravel()
    from_index = 0
    means: list[float] = []
    n = freq.shape[0]

    while from_index + batch_size <= n:
        means.append(float(freq[from_index : from_index + batch_size].mean()))
        from_index += step_size

    return np.asarray(means, dtype=np.float64)


# -------------------------
# file/key helpers
# -------------------------
def _to_file_id(model_path: str) -> str:
    # matches your naming: "/text" becomes "__text" etc.
    return model_path.strip("/").replace("/", "__")


def _find_pair_npz(metrics_dir: str, m1: str, m2: str, *, suffix: str = ".npz") -> str | None:
    id1 = _to_file_id(m1)
    id2 = _to_file_id(m2)

    p12 = os.path.join(metrics_dir, f"{id1}_{id2}{suffix}")
    if os.path.isfile(p12):
        return p12

    p21 = os.path.join(metrics_dir, f"{id2}_{id1}{suffix}")
    if os.path.isfile(p21):
        return p21

    return None


def _load_npz_dict(npz_path: str) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    with np.load(npz_path, allow_pickle=True) as z:
        for k in z.files:
            out[k] = z[k]
    return out


def _is_1d(arr: Any) -> bool:
    a = np.asarray(arr)
    return (a.ndim == 1) and (a.size >= 2)


def _is_random_key(k: str) -> bool:
    lk = k.lower()
    return ("random" in lk) or ("rand" in lk)


def _strip_modality(model_path: str) -> str:
    if model_path.endswith("/text"):
        return model_path[: -len("/text")]
    if model_path.endswith("/img"):
        return model_path[: -len("/img")]
    return model_path


def _family(model_path: str) -> str:
    """
    Used to optionally filter out within-family pairs.
    Mirrors your earlier heuristic:
      - takes the first path component
      - splits by "__" and keeps org + prefix of second component
    """
    model_id = model_path.split("/", 1)[0]
    parts = model_id.split("__")
    if len(parts) == 1:
        return parts[0].split("-", 1)[0]
    org = parts[0]
    second_prefix = parts[1].split("-", 1)[0]
    return f"{org}__{second_prefix}"


# -------------------------
# metric titles/layout (YOUR keys)
# -------------------------
_HUMAN_TITLES: dict[str, str] = {
    "CKA_HSIC": "CKA",
    "CKA_unbiased": "Unbiased CKA",
    "SVCCA_1": "SVCCA 10",
    "SVCCA_2": "SVCCA 100",
    "TOPK10": "KNN Overlap 10",
    "TOPK100": "KNN Overlap 100",
    "KNN_EDIT_10": "KNN-10 Edit",
    "KNN_EDIT_100": "KNN-100 Edit",
}

_DEFAULT_LAYOUT: list[tuple[str, ...]] = [
    ("CKA_HSIC", "CKA_unbiased"),
    ("SVCCA_1", "SVCCA_2"),
    ("TOPK10", "TOPK100"),
    ("KNN_EDIT_10", "KNN_EDIT_100"),
]


def _human_title(metric_key: str) -> str:
    return _HUMAN_TITLES.get(metric_key, metric_key)


def _is_edit_metric(metric_key: str) -> bool:
    return "EDIT" in metric_key.upper()


# -------------------------
# curve getter: raw/random/normalized
# -------------------------
def _get_curve(d: dict[str, np.ndarray], key: str, normalization: str) -> np.ndarray | None:
    """
    normalization:
      - raw:        use key
      - random:     use the random counterpart of key if present
      - normalized: raw - random
    """
    if normalization == "raw":
        v = d.get(key, None)
        return np.asarray(v).squeeze() if (v is not None and _is_1d(v)) else None

    rand_candidates = [
        f"{key}_random",
        f"{key}_rand",
        f"{key}__random",
        f"{key}__rand",
        f"random_{key}",
        f"rand_{key}",
        f"{key}Random",
        f"{key}Rand",
    ]

    rand_key = None
    for k in rand_candidates:
        if k in d and _is_1d(d[k]):
            rand_key = k
            break

    if rand_key is None:
        for k in d.keys():
            if (key in k) and _is_random_key(k) and _is_1d(d[k]):
                rand_key = k
                break

    if normalization == "random":
        if rand_key is None:
            return None
        return np.asarray(d[rand_key]).squeeze()

    v_raw = d.get(key, None)
    if v_raw is None or rand_key is None or (not _is_1d(v_raw)):
        return None

    y_raw = np.asarray(v_raw).squeeze()
    y_rnd = np.asarray(d[rand_key]).squeeze()
    L = min(len(y_raw), len(y_rnd))
    return y_raw[:L] - y_rnd[:L]


# -------------------------
# main plotting (grid or single)
# -------------------------
def plot_rolling_window_metrics_vs_freq(
    list_1: list[str],
    list_2: list[str],
    *,
    rolling_metrics_dir: str,
    freq_csv: str = "/home/kirilb/orcd/pool/PRH_data/word_list_50000_en_best.csv",
    freq_batch: int = 500,
    freq_step: int = 250,
    normalization: str = "raw",  # raw | random | normalized
    npz_suffix: str = ".npz",
    eps: float = 1e-12,
    figsize: tuple[float, float] | None = None,
    alpha: float = 0.9,
    lw: float = 1.5,
    show_legend: bool = True,
    legend_fontsize: int = 16,
    legend_loc: str = "center left",
    legend_bbox_to_anchor: tuple[float, float] = (1.02, 0.5),
    title: str | None = None,
    max_pairs: int | None = None,
    verbose: bool = True,
    fontsize: int = 40,
    num_data_points: int | None = None,
    from_same_family: bool = False,
    show_modality: bool = True,
    invert_edit_axis: bool = True,
    save_path: str | None = None,
    abbreviated_model_names_1: list[str] | None = None,
    abbreviated_model_names_2: list[str] | None = None,
    close_plot: bool = False,
    show_plot: bool = True,
    metrics_layout: list[tuple[str, ...]] | None = None,
    single_metric: str | None = None,
):
    """
    If single_metric is provided, plots a 1x1 with that metric.
    Otherwise plots a grid using metrics_layout (default: _DEFAULT_LAYOUT).
    """
    normalization = normalization.lower().strip()
    if normalization == "regular":
        normalization = "raw"
    if normalization not in {"raw", "random", "normalized"}:
        raise ValueError(f"normalization must be one of: raw, random, normalized. Got: {normalization}")

    if num_data_points is not None and (not isinstance(num_data_points, int) or num_data_points <= 1):
        raise ValueError("num_data_points must be an int >= 2 (or None).")

    if abbreviated_model_names_1 is not None and len(abbreviated_model_names_1) != len(list_1):
        raise ValueError(
            f"abbreviated_model_names_1 must have length {len(list_1)}, got {len(abbreviated_model_names_1)}"
        )
    if abbreviated_model_names_2 is not None and len(abbreviated_model_names_2) != len(list_2):
        raise ValueError(
            f"abbreviated_model_names_2 must have length {len(list_2)}, got {len(abbreviated_model_names_2)}"
        )

    label_map_1 = (
        {m: abbreviated_model_names_1[i] for i, m in enumerate(list_1)}
        if abbreviated_model_names_1 is not None
        else {}
    )
    label_map_2 = (
        {m: abbreviated_model_names_2[i] for i, m in enumerate(list_2)}
        if abbreviated_model_names_2 is not None
        else {}
    )

    if single_metric is not None:
        layout: list[tuple[str, ...]] = [(single_metric,)]
    else:
        layout = metrics_layout if metrics_layout is not None else _DEFAULT_LAYOUT

    nrows = len(layout)
    ncols = max(len(row) for row in layout)

    freq = read_frequencies_from_csv(csv_path=freq_csv, skip_first=True)
    mean_freq = get_means(freq, batch_size=freq_batch, step_size=freq_step)
    x_full = np.power(mean_freq + eps, -0.5)

    pair_items: list[tuple[str, str, str, dict[str, np.ndarray]]] = []
    attempted = 0
    skipped_same = 0
    skipped_same_family = 0
    found_files = 0

    for m1 in list_1:
        fam1 = _family(m1)
        for m2 in list_2:
            if m1 == m2:
                skipped_same += 1
                continue
            if (not from_same_family) and (fam1 == _family(m2)):
                skipped_same_family += 1
                continue

            attempted += 1
            npz_path = _find_pair_npz(rolling_metrics_dir, m1, m2, suffix=npz_suffix)
            if npz_path is None:
                continue

            found_files += 1
            if max_pairs is not None and len(pair_items) >= max_pairs:
                break

            d = _load_npz_dict(npz_path)
            pair_items.append((m1, m2, npz_path, d))

        if max_pairs is not None and len(pair_items) >= max_pairs:
            break

    if verbose:
        print(f"Attempted cross pairs:     {attempted}")
        print(f"Skipped same model:        {skipped_same}")
        print(f"Skipped same family:       {skipped_same_family}")
        print(f"Found npz files:           {found_files}")
        if pair_items:
            print("[debug] example npz keys:", sorted(pair_items[0][3].keys()))

    if not pair_items:
        if verbose:
            print("[WARN] No pair metric files found.")
        return {
            "attempted_cross_pairs": attempted,
            "skipped_same_model": skipped_same,
            "skipped_same_family": skipped_same_family,
            "found_npz_files": found_files,
            "pairs_used": [],
        }

    base_colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
    if not base_colors:
        base_colors = ["C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9"]

    def _single_label(model_path: str, which: int) -> str:
        if which == 1 and model_path in label_map_1:
            return label_map_1[model_path]
        if which == 2 and model_path in label_map_2:
            return label_map_2[model_path]
        return _strip_modality(model_path) if show_modality else model_path

    def _pair_label(m1: str, m2: str) -> str:
        return f"{_single_label(m1, 1)} vs {_single_label(m2, 2)}"

    pair_labels = [_pair_label(m1, m2) for (m1, m2, _, _) in pair_items]
    pair_colors = [base_colors[i % len(base_colors)] for i in range(len(pair_items))]

    if figsize is None:
        figsize = (10, 6) if single_metric is not None else (15, 4.5 * nrows)

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize, squeeze=False)
    plotted_pair_anywhere: set[int] = set()

    for r in range(nrows):
        for c in range(ncols):
            ax = axes[r, c]
            if c >= len(layout[r]):
                ax.axis("off")
                continue

            key = layout[r][c]

            for pi, (_, _, _, d) in enumerate(pair_items):
                y = _get_curve(d, key, normalization=normalization)
                if y is None or np.asarray(y).ndim != 1 or len(y) < 2:
                    continue

                L = min(len(x_full), len(y))
                if num_data_points is not None:
                    L = min(L, num_data_points)

                ax.plot(x_full[:L], np.asarray(y)[:L], alpha=alpha, lw=lw, color=pair_colors[pi])
                plotted_pair_anywhere.add(pi)

            ax.set_title(
                f"{_human_title(key)} (raw - random)" if normalization == "normalized" else _human_title(key),
                fontsize=fontsize,
            )
            ax.tick_params(axis="both", labelsize=fontsize)

            if invert_edit_axis and _is_edit_metric(key):
                ax.invert_yaxis()

    if title is None:
        if single_metric is not None:
            base = _human_title(single_metric)
            title = f"{base} ({normalization})"
            if normalization == "normalized":
                title = f"{base} (raw - random)"
        else:
            title = f"Rolling-window metrics vs mean freq^(-1/2) ({normalization})"
            if normalization == "normalized":
                title = "Rolling-window metrics vs mean freq^(-1/2) (raw - random)"

    fig.suptitle(title, fontsize=fontsize)
    fig.text(0.5, 0.03, r"$(\bar f)^{-1/2}$  (mean freq in window)", ha="center", va="center", fontsize=fontsize)

    if show_legend:
        legend_indices = sorted(plotted_pair_anywhere)
        if legend_indices:
            handles = [Line2D([0], [0], color=pair_colors[i], lw=3) for i in legend_indices]
            labels = [pair_labels[i] for i in legend_indices]
            fig.legend(
                handles,
                labels,
                loc=legend_loc,
                bbox_to_anchor=legend_bbox_to_anchor,
                ncol=1,
                fontsize=legend_fontsize,
                frameon=True,
            )

    right_margin = 0.78 if show_legend else 0.98
    fig.tight_layout(rect=[0.06, 0.06, right_margin, 0.95])

    saved_to = None
    if save_path is not None:
        final_path = _pdf_path(save_path)
        os.makedirs(os.path.dirname(final_path) or ".", exist_ok=True)
        fig.savefig(final_path, format="pdf", bbox_inches="tight")
        saved_to = final_path

    if show_plot:
        plt.show()

    if close_plot:
        plt.close(fig)

    return {
        "attempted_cross_pairs": attempted,
        "skipped_same_model": skipped_same,
        "skipped_same_family": skipped_same_family,
        "found_npz_files": found_files,
        "pairs_used": [(m1, m2, npz_path) for (m1, m2, npz_path, _) in pair_items],
        "legend_pairs": [pair_labels[i] for i in sorted(plotted_pair_anywhere)],
        "saved_to": saved_to,
    }


# -------------------------
# wrapper: single metric
# -------------------------
def plot_rolling_window_single_metric_vs_freq(
    *,
    metric_key: str,
    list_1: list[str],
    list_2: list[str],
    rolling_metrics_dir: str,
    **kwargs,
):
    """
    Thin wrapper that reuses plot_rolling_window_metrics_vs_freq by passing single_metric=metric_key.
    """
    return plot_rolling_window_metrics_vs_freq(
        list_1=list_1,
        list_2=list_2,
        rolling_metrics_dir=rolling_metrics_dir,
        single_metric=metric_key,
        **kwargs,
    )


# -------------------------
# regression helper
# -------------------------
def _fit_linreg_1d_with_intercept(x: np.ndarray, y: np.ndarray):
    """
    Fit y ≈ a + b x by closed form least squares.
    Returns (a, b, r2). If ill-posed, returns (nan, nan, nan).
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    if x.size < 2 or y.size < 2:
        return (np.nan, np.nan, np.nan)

    xbar = x.mean()
    ybar = y.mean()
    xc = x - xbar
    yc = y - ybar
    denom = np.dot(xc, xc)
    if denom <= 0:
        return (np.nan, np.nan, np.nan)

    b = np.dot(xc, yc) / denom
    a = ybar - b * xbar

    yhat = a + b * x
    ss_res = np.sum((y - yhat) ** 2)
    ss_tot = np.sum((y - ybar) ** 2)
    r2 = np.nan if ss_tot <= 0 else (1.0 - ss_res / ss_tot)
    return (a, b, r2)


def plot_single_metric_with_regression_and_heatmaps(
    metric_key: str,
    list_1: list[str],
    list_2: list[str],
    *,
    rolling_metrics_dir: str,
    freq_csv: str = "/home/kirilb/orcd/pool/PRH_data/word_list_50000_en_best.csv",
    freq_batch: int = 500,
    freq_step: int = 250,
    normalization: str = "raw",
    npz_suffix: str = ".npz",
    eps: float = 1e-12,
    num_data_points: int | None = None,
    from_same_family: bool = False,
    show_modality: bool = True,
    invert_edit_axis: bool = True,
    figsize: tuple[float, float] = (20, 8),
    alpha: float = 0.85,
    lw: float = 1.5,
    title: str | None = None,
    center_coef_at_zero: bool = True,
    cmap_coef: str = "bwr",
    cmap_r2: str = "viridis",
    save_path: str | None = None,
    verbose: bool = True,
    abbreviated_model_names_1: list[str] | None = None,
    abbreviated_model_names_2: list[str] | None = None,
    close_plot: bool = False,
    show_plot: bool = True,
    title_font_size: int = 25,
    label_font_size: int = 14,
):
    """
    Reads rolling-window metric curves for a single metric and:
      (1) plots all pair curves vs x = (mean freq)^(-1/2)
      (2) heatmap of regression slope coefficients b (rows=list_2, cols=list_1)
      (3) heatmap of R^2 (rows=list_2, cols=list_1)

    Regression: y_windows = a + b * x_windows, with intercept.
    """
    normalization = normalization.lower().strip()
    if normalization == "regular":
        normalization = "raw"
    if normalization not in {"raw", "random", "normalized"}:
        raise ValueError(f"normalization must be one of: raw, random, normalized. Got: {normalization}")

    if num_data_points is not None and (not isinstance(num_data_points, int) or num_data_points <= 1):
        raise ValueError("num_data_points must be an int >= 2 (or None).")

    if abbreviated_model_names_1 is not None and len(abbreviated_model_names_1) != len(list_1):
        raise ValueError(
            f"abbreviated_model_names_1 must have length {len(list_1)}, got {len(abbreviated_model_names_1)}"
        )
    if abbreviated_model_names_2 is not None and len(abbreviated_model_names_2) != len(list_2):
        raise ValueError(
            f"abbreviated_model_names_2 must have length {len(list_2)}, got {len(abbreviated_model_names_2)}"
        )

    label_map_1 = (
        {m: abbreviated_model_names_1[i] for i, m in enumerate(list_1)}
        if abbreviated_model_names_1 is not None
        else {}
    )
    label_map_2 = (
        {m: abbreviated_model_names_2[i] for i, m in enumerate(list_2)}
        if abbreviated_model_names_2 is not None
        else {}
    )

    freq = read_frequencies_from_csv(csv_path=freq_csv, skip_first=True)
    mean_freq = get_means(freq, batch_size=freq_batch, step_size=freq_step)
    x_full = np.power(mean_freq + eps, -0.5)

    B = np.full((len(list_2), len(list_1)), np.nan, dtype=np.float64)
    A = np.full((len(list_2), len(list_1)), np.nan, dtype=np.float64)
    R2 = np.full((len(list_2), len(list_1)), np.nan, dtype=np.float64)
    found = np.zeros((len(list_2), len(list_1)), dtype=bool)

    curves = []

    attempted = 0
    skipped_same = 0
    skipped_same_family = 0

    for j, m2 in enumerate(list_2):
        fam2 = _family(m2)
        for i, m1 in enumerate(list_1):
            if m1 == m2:
                skipped_same += 1
                continue
            if (not from_same_family) and (_family(m1) == fam2):
                skipped_same_family += 1
                continue

            attempted += 1
            npz_path = _find_pair_npz(rolling_metrics_dir, m1, m2, suffix=npz_suffix)
            if npz_path is None:
                continue

            d = _load_npz_dict(npz_path)
            y = _get_curve(d, metric_key, normalization=normalization)
            if y is None or np.asarray(y).ndim != 1 or len(y) < 2:
                continue

            L = min(len(x_full), len(y))
            if num_data_points is not None:
                L = min(L, num_data_points)

            x = x_full[:L]
            y = np.asarray(y, dtype=np.float64)[:L]

            a, b, r2 = _fit_linreg_1d_with_intercept(x, y)
            A[j, i] = a
            B[j, i] = b
            R2[j, i] = r2
            found[j, i] = True

            curves.append((m1, m2, x, y))

    if verbose:
        print(f"Attempted pairs:           {attempted}")
        print(f"Skipped same model:        {skipped_same}")
        print(f"Skipped same family:       {skipped_same_family}")
        print(f"Found usable curves:       {int(found.sum())} / {len(list_1)*len(list_2)}")
        if int(found.sum()) == 0:
            print("[debug] Check metric_key, normalization, and that the arrays are 1D windows.")

    def _single_label(model_path: str, which: int) -> str:
        if which == 1 and model_path in label_map_1:
            return label_map_1[model_path]
        if which == 2 and model_path in label_map_2:
            return label_map_2[model_path]
        return _strip_modality(model_path) if show_modality else model_path

    xlabels = [_single_label(m1, 1) for m1 in list_1]
    ylabels = [_single_label(m2, 2) for m2 in list_2]

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    ax0 = axes[0]
    base_colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", []) or [
        "C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9"
    ]
    for k, (_, _, x, y) in enumerate(curves):
        ax0.plot(x, y, alpha=alpha, lw=lw, color=base_colors[k % len(base_colors)])

    t0 = _human_title(metric_key)
    if normalization == "normalized":
        ax0.set_title(f"{t0} vs $(\\bar f)^{{-1/2}}$ (raw - random)", fontsize=title_font_size)
    else:
        ax0.set_title(f"{t0} vs $(\\bar f)^{{-1/2}}$ ({normalization})", fontsize=title_font_size)

    ax0.set_xlabel(r"$(\bar f)^{-1/2}$  (mean freq in window)", fontsize=label_font_size)
    ax0.set_ylabel("alignment", fontsize=label_font_size)
    ax0.tick_params(axis="both", labelsize=label_font_size)

    if invert_edit_axis and _is_edit_metric(metric_key):
        ax0.invert_yaxis()

    ax1 = axes[1]

    # For edit distances, display the negative slope so that "better alignment with frequency"
    # points in the same visual direction as the other metrics.
    B_plot = -B if _is_edit_metric(metric_key) else B

    coef_vmin = coef_vmax = None
    if center_coef_at_zero:
        finite_abs = np.abs(B_plot[np.isfinite(B_plot)])
        if finite_abs.size == 0:
            maxabs = 1e-8
        else:
            maxabs = float(np.max(finite_abs))
            if maxabs <= 0:
                maxabs = 1e-8
        coef_vmin, coef_vmax = -maxabs, maxabs
    else:
        maxabs = float(np.max(finite_abs))
        if maxabs <= 0:
            maxabs = 1e-8
    coef_vmin, coef_vmax = -maxabs, maxabs

    im1 = ax1.imshow(B_plot, aspect="auto", cmap=cmap_coef, vmin=coef_vmin, vmax=coef_vmax)
    ax1.set_title(
        "Regression slope (-coefficient)" if _is_edit_metric(metric_key) else "Regression slope (coefficient)",
        fontsize=title_font_size)
    ax1.set_xticks(np.arange(len(xlabels)))
    ax1.set_yticks(np.arange(len(ylabels)))
    ax1.set_xticklabels(xlabels, rotation=90, fontsize=label_font_size)
    ax1.set_yticklabels(ylabels, fontsize=label_font_size)
    ax1.tick_params(axis="both", labelsize=label_font_size)
    cbar1 = plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    cbar1.ax.tick_params(labelsize=label_font_size)

    ax2 = axes[2]
    im2 = ax2.imshow(R2, aspect="auto", cmap=cmap_r2, vmin=0.0, vmax=1.0)
    ax2.set_title(r"$R^2$ goodness of fit", fontsize=title_font_size)
    ax2.set_xticks(np.arange(len(xlabels)))
    ax2.set_yticks(np.arange(len(ylabels)))
    ax2.set_xticklabels(xlabels, rotation=90, fontsize=label_font_size)
    ax2.set_yticklabels(ylabels, fontsize=label_font_size)
    ax2.tick_params(axis="both", labelsize=label_font_size)
    cbar2 = plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    cbar2.ax.tick_params(labelsize=label_font_size)

    if title is None:
        title = f"{_human_title(metric_key)} regression vs frequency"
    fig.suptitle(title, fontsize=title_font_size)

    fig.tight_layout(rect=[0, 0, 1, 0.94])

    saved_to = None
    if save_path is not None:
        final_path = _pdf_path(save_path)
        os.makedirs(os.path.dirname(final_path) or ".", exist_ok=True)
        fig.savefig(final_path, format="pdf", bbox_inches="tight")
        saved_to = final_path

    if show_plot:
        plt.show()

    if close_plot:
        plt.close(fig)

    return {
        "A_intercept": A,
        "B_slope": B,
        "R2": R2,
        "found_mask": found,
        "x_labels": xlabels,
        "y_labels": ylabels,
        "attempted_pairs": attempted,
        "found_pairs": int(found.sum()),
        "saved_to": saved_to,
    }