#!/usr/bin/env python3
import glob
import os
import re
from typing import Optional, Sequence

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

mpl.rcParams["axes.unicode_minus"] = False


# -------------------------
# metric mapping + titles
# -------------------------
_METRIC_KEY = {
    "cka": "CKA_HSIC_mean_over_subsamples",
    "cka_unbiased": "CKA_unbiased_mean_over_subsamples",
    "svcca_10": "SVCCA_1_mean_over_subsamples",
    "svcca_100": "SVCCA_2_mean_over_subsamples",
    "topk_10": "TOPK10_mean_over_subsamples",
    "topk_100": "TOPK100_mean_over_subsamples",
    "knn_edit_10": "KNN_EDIT_10_mean_over_subsamples",
    "knn_edit_100": "KNN_EDIT_100_mean_over_subsamples",
    "weighted_correlation": "weighted_correlation",
    "binary_correlation": "binary_correlation",
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
    "weighted_correlation": "Weighted Correlation",
    "binary_correlation": "Binary Correlation",
}

_EDIT_METRICS = {"knn_edit_10", "knn_edit_100"}
_ALLOWED_RAW_STD_METRICS = {"weighted_correlation_std", "binary_correlation_std"}


# -------------------------
# text helpers
# -------------------------
def _normalize_text(s: Optional[str]) -> str:
    if s is None:
        return ""
    s = str(s)
    replacements = {
        "\u2212": "-",   # unicode minus
        "\u2013": "-",   # en dash
        "\u2014": "-",   # em dash
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


def _maybe_disable_math_parsing(text_artist) -> None:
    try:
        if hasattr(text_artist, "set_parse_math"):
            text_artist.set_parse_math(False)
    except Exception:
        pass


def _pretty_model_name(s: str) -> str:
    s = str(s).strip()

    for suffix in ("__text", "__img", "_text", "_img"):
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

    s = s.replace("Qwen3-", "Qwen ")
    s = s.replace("Llama-3.2-", "Llama ")
    s = s.replace("gemma-3-", "Gemma ")
    s = s.replace("siglip2-", "SigLIP2 ")
    s = s.replace("clip-vit-", "CLIP ")
    s = s.replace("dinov2-", "DINOv2 ")
    s = s.replace("vit-mae-", "ViT MAE ")
    s = s.replace("beit-", "BEiT ")

    s = s.replace("-Base", " B").replace("-base", " B")
    s = s.replace("-Large", " L").replace("-large", " L")
    s = s.replace("-Huge", " H").replace("-huge", " H")

    for token in (
        "-patch16-256",
        "-patch14",
        "-patch32",
        "-patch16",
        "-224",
        "-pt22k",
        "-en-v1.5",
        "-v1.5",
        "-Instruct",
        "-laion2B-s34B-b79K",
        "-laion2B-s32B-b79K",
    ):
        s = s.replace(token, "")

    s = s.replace("-v2-moe", " v2-moe")
    s = s.replace("-it", " it")

    s = s.replace("__", " ").replace("-", " ").replace("_", " ")
    s = " ".join(s.split())
    return s


# -------------------------
# pair-file helpers
# -------------------------
_KTAG_RE = re.compile(r"_k\d+")


def _model_token_variants(model_name: str) -> list[str]:
    variants = [model_name]

    if model_name.endswith("__text"):
        variants.append(model_name[:-6] + "_text")
    elif model_name.endswith("__img"):
        variants.append(model_name[:-5] + "_img")

    if model_name.endswith("_text"):
        variants.append(model_name[:-5] + "__text")
    elif model_name.endswith("_img"):
        variants.append(model_name[:-4] + "__img")

    seen = set()
    out = []
    for v in variants:
        if v not in seen:
            seen.add(v)
            out.append(v)
    return out


def _strip_ktags(s: str) -> str:
    return _KTAG_RE.sub("", s)


def _ktag_glob_variants(token: str) -> list[str]:
    variants = [token]

    if _KTAG_RE.search(token):
        variants.append(_KTAG_RE.sub("_k*", token))

    base = _strip_ktags(token)
    variants.append(base)
    variants.append(base + "_k*")

    seen = set()
    out = []
    for v in variants:
        if v not in seen:
            seen.add(v)
            out.append(v)
    return out


def _dedup(items: Sequence[str]) -> list[str]:
    seen = set()
    out = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def _compile_model_block_patterns(token: str) -> list[re.Pattern]:
    """
    Build regexes that match one model-token block inside a filename stem.

    Accepted examples for token = "BAAI__bge-base-en-v1.5_text":
      - BAAI__bge-base-en-v1.5_text
      - BAAI__bge-base-en-v1.5_text_k32
      - BAAI__bge-base-en-v1.5_text_k_32
      - topk_16384_BAAI__bge-base-en-v1.5_text
      - topk_16384_BAAI__bge-base-en-v1.5_text_k32
      - topk_16384_BAAI__bge-base-en-v1.5_text_k_32
      - batchtopk_16384_BAAI__bge-base-en-v1.5_text
      - batchtopk_16384_BAAI__bge-base-en-v1.5_text_k32
      - batchtopk_16384_BAAI__bge-base-en-v1.5_text_k_32
    """
    token_esc = re.escape(token)
    patterns = [
        re.compile(
            rf"(?:^|_)(?:(?:topk|batchtopk)_\d+_)?{token_esc}(?:_k_?\d+)?(?=_|$)"
        )
    ]
    return patterns


def _find_block_spans(stem: str, token: str) -> list[tuple[int, int]]:
    spans = []
    for pat in _compile_model_block_patterns(token):
        for m in pat.finditer(stem):
            spans.append(m.span())
    spans.sort()
    return spans


def _filename_matches_pair(stem: str, m1: str, m2: str) -> bool:
    """
    Return True iff the stem contains one block for m1 and one block for m2,
    in either order, allowing:
      - optional topk_<dim>_ prefix before each model token
      - optional _k32 or _k_32 after each model token
      - arbitrary extra suffixes like _subsample_metrics
    """
    v1 = _model_token_variants(m1)
    v2 = _model_token_variants(m2)

    # Fast path for the old naming scheme.
    for a in v1:
        for b in v2:
            for sep in ("_", "__"):
                if stem == f"{a}{sep}{b}" or stem == f"{b}{sep}{a}":
                    return True

    # Robust path for more decorated filenames.
    for a in v1:
        spans_a = _find_block_spans(stem, a)
        if not spans_a:
            continue
        for b in v2:
            spans_b = _find_block_spans(stem, b)
            if not spans_b:
                continue

            for sa0, sa1 in spans_a:
                for sb0, sb1 in spans_b:
                    if sa1 <= sb0 or sb1 <= sa0:
                        return True

    return False


def _find_pair_file(dir_path: str, m1: str, m2: str) -> Optional[str]:
    v1 = _model_token_variants(m1)
    v2 = _model_token_variants(m2)

    v1_patterns = []
    v2_patterns = []
    for x in v1:
        v1_patterns.extend(_ktag_glob_variants(x))
    for x in v2:
        v2_patterns.extend(_ktag_glob_variants(x))

    v1_patterns = _dedup(v1_patterns)
    v2_patterns = _dedup(v2_patterns)

    # Exact matches first
    for a_list, b_list in ((v1_patterns, v2_patterns), (v2_patterns, v1_patterns)):
        for a in a_list:
            for b in b_list:
                for sep in ("_", "__"):
                    if "*" not in a and "*" not in b:
                        path = os.path.join(dir_path, f"{a}{sep}{b}.npz")
                        if os.path.exists(path):
                            return path

    # Glob matches second
    for a_list, b_list in ((v1_patterns, v2_patterns), (v2_patterns, v1_patterns)):
        for a in a_list:
            for b in b_list:
                for sep in ("_", "__"):
                    hits = glob.glob(os.path.join(dir_path, f"{a}{sep}{b}.npz"))
                    if hits:
                        hits.sort()
                        return hits[0]

    # Robust filename scan last:
    # accepts stems such as
    # topk_16384_A_text_k_32_topk_16384_B_img_k_128_subsample_metrics
    hits = glob.glob(os.path.join(dir_path, "*.npz"))
    hits.sort()
    for path in hits:
        stem = os.path.splitext(os.path.basename(path))[0]
        if _filename_matches_pair(stem, m1, m2):
            return path

    return None


def _as_float(v) -> float:
    if np.isscalar(v):
        return float(v)
    arr = np.asarray(v)
    if arr.ndim == 0:
        return float(arr.item())
    return float(np.mean(arr))


def _load_one_metric(npz_path: str, metric_key: str) -> Optional[float]:
    with np.load(npz_path, allow_pickle=True) as z:
        if metric_key in z.files:
            return _as_float(z[metric_key])
    return None


# -------------------------
# save helper
# -------------------------
def _pdf_path(path: str) -> str:
    root, _ = os.path.splitext(path)
    return root + ".pdf"


def _save_figure(fig, out_path: str, dpi: int = 200) -> str:
    final_path = _pdf_path(out_path)
    out_dir = os.path.dirname(final_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    fig.savefig(
        final_path,
        format="pdf",
        bbox_inches="tight",
        pad_inches=0.05,
        dpi=dpi,
    )
    return final_path


# -------------------------
# matrix builder
# -------------------------
def build_single_metric_matrix(
    models,
    metric,
    raw_dir,
    filtered_dir,
    panel="filtered",
    diag_value=np.nan,
    missing="nan",
    raw_std_metric=None,
):
    if metric not in _METRIC_KEY:
        raise ValueError(f"metric must be one of: {sorted(_METRIC_KEY)}")

    panel = panel.lower().strip()
    if panel not in {"raw", "filtered", "diff"}:
        raise ValueError("panel must be one of: 'raw', 'filtered', 'diff'")

    if raw_std_metric is not None:
        raw_std_metric = str(raw_std_metric).strip()
        if raw_std_metric == "":
            raw_std_metric = None
        elif raw_std_metric not in _ALLOWED_RAW_STD_METRICS:
            raise ValueError(
                f"raw_std_metric must be one of {sorted(_ALLOWED_RAW_STD_METRICS)} or None. "
                f"Got: {raw_std_metric}"
            )

    n = len(models)
    key = _METRIC_KEY[metric]

    M_raw = np.full((n, n), np.nan, dtype=float)
    M_filtered = np.full((n, n), np.nan, dtype=float)
    M_raw_std = np.full((n, n), np.nan, dtype=float)
    np.fill_diagonal(M_raw, diag_value)
    np.fill_diagonal(M_filtered, diag_value)
    np.fill_diagonal(M_raw_std, np.nan)

    missing_pairs = []
    used_pairs = 0
    used_raw_std_metric = (panel == "diff" and raw_std_metric is not None)

    for i in range(n):
        for j in range(i + 1, n):
            m1, m2 = models[i], models[j]
            raw_path = _find_pair_file(raw_dir, m1, m2)
            filtered_path = _find_pair_file(filtered_dir, m1, m2)

            if raw_path is None or filtered_path is None:
                missing_pairs.append((m1, m2, raw_path is None, filtered_path is None))
                if missing == "raise":
                    raise FileNotFoundError(
                        f"Missing pair file(s) for ({m1}, {m2}). "
                        f"raw_missing={raw_path is None}, filtered_missing={filtered_path is None}"
                    )
                continue

            raw_value = _load_one_metric(raw_path, key)
            filtered_value = _load_one_metric(filtered_path, key)

            raw_std_value = None
            if used_raw_std_metric:
                raw_std_value = _load_one_metric(raw_path, raw_std_metric)

            if raw_value is None or filtered_value is None:
                missing_pairs.append((m1, m2, raw_value is None, filtered_value is None))
                continue

            if used_raw_std_metric:
                if raw_std_value is None or not np.isfinite(raw_std_value) or raw_std_value == 0.0:
                    missing_pairs.append((m1, m2, raw_std_metric, raw_std_value))
                    continue

            M_raw[i, j] = M_raw[j, i] = raw_value
            M_filtered[i, j] = M_filtered[j, i] = filtered_value
            if used_raw_std_metric:
                M_raw_std[i, j] = M_raw_std[j, i] = raw_std_value
            used_pairs += 1

    if panel == "raw":
        M = M_raw
    elif panel == "filtered":
        M = M_filtered
    else:
        if metric in _EDIT_METRICS:
            M = M_raw - M_filtered
        else:
            M = M_filtered - M_raw

        if used_raw_std_metric:
            M = M / M_raw_std

    info = {
        "n_models": n,
        "attempted_pairs": n * (n - 1) // 2,
        "used_pairs": used_pairs,
        "missing_pairs": missing_pairs,
        "metric": metric,
        "metric_key": key,
        "panel": panel,
        "raw_std_metric": raw_std_metric,
    }
    return M, info


# -------------------------
# bracket helper
# -------------------------
def _add_group_brackets(
    ax,
    group_names,
    group_sizes,
    n,
    top=True,
    left=True,
    pad=0.7,
    lw=1.8,
    fontsize=14,
    top_label_offset=0.8,
    left_label_offset=0.55,
):
    if group_names is None or group_sizes is None:
        return

    if len(group_names) != len(group_sizes):
        raise ValueError("type_name and type_index must have the same length.")

    if sum(group_sizes) != n:
        raise ValueError(f"Sum(type_index) must equal n={n}. Got {sum(group_sizes)}")

    starts = np.cumsum([0] + list(group_sizes[:-1]))
    ends = starts + np.array(group_sizes)

    top_edge = 0.0
    left_edge = 0.0
    y_bracket = top_edge - pad
    x_bracket = left_edge - pad

    for name, start, end in zip(group_names, starts, ends):
        x0, x1 = float(start), float(end)
        y0, y1 = float(start), float(end)
        label = _normalize_text(name)

        if top:
            ax.plot([x0, x0], [y_bracket, top_edge], color="black", lw=lw, clip_on=False)
            ax.plot([x0, x1], [y_bracket, y_bracket], color="black", lw=lw, clip_on=False)
            ax.plot([x1, x1], [y_bracket, top_edge], color="black", lw=lw, clip_on=False)
            txt = ax.text(
                0.5 * (x0 + x1),
                y_bracket - top_label_offset,
                label,
                ha="center",
                va="top",
                fontsize=fontsize,
                clip_on=False,
            )
            _maybe_disable_math_parsing(txt)

        if left:
            txt = ax.text(
                x_bracket - left_label_offset,
                0.5 * (y0 + y1),
                label,
                ha="right",
                va="center",
                rotation=90,
                fontsize=fontsize,
                clip_on=False,
            )
            _maybe_disable_math_parsing(txt)

            ax.plot([x_bracket, left_edge], [y0, y0], color="black", lw=lw, clip_on=False)
            ax.plot([x_bracket, x_bracket], [y0, y1], color="black", lw=lw, clip_on=False)
            ax.plot([x_bracket, left_edge], [y1, y1], color="black", lw=lw, clip_on=False)


# -------------------------
# plot sorted heatmap
# -------------------------
def plot_metric_heatmap_sorted(
    M,
    names,
    names_sorted=None,
    title="Heatmap",
    savepath=None,
    annotate_floor_1dp=False,
    type_name=None,
    type_index=None,
    plot_diff=False,
    cmap=None,
    vmin=None,
    vmax=None,
    colorbar_levels=32,
    tick_fontsize=14,
    tick_rotation=60,
    use_pretty_names=True,
    abbreviated_model_names=None,
    x_tick_pad=52,
    y_tick_pad=20,
    title_fontsize=20,
    bracket_fontsize=10,
    close_plot=False,
    show_plot=True,
    dpi=200,
):
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors
    from matplotlib.ticker import FormatStrFormatter

    if names_sorted is None:
        names_sorted = names

    M = np.asarray(M)
    n = len(names)

    if M.shape != (n, n):
        raise ValueError("M must be NxN and match len(names).")
    if len(names_sorted) != n:
        raise ValueError("names_sorted must have same length as names.")

    pos = {name: i for i, name in enumerate(names)}
    missing = [name for name in names_sorted if name not in pos]
    if missing:
        raise ValueError(f"names_sorted contains names not in names: {missing}")

    idx = np.array([pos[name] for name in names_sorted], dtype=int)
    M_re = M[np.ix_(idx, idx)]

    if abbreviated_model_names is not None:
        if len(abbreviated_model_names) != n:
            raise ValueError(
                "abbreviated_model_names must have the same length as names_sorted. "
                f"Got {len(abbreviated_model_names)} and {n}."
            )
        labels = [str(x) for x in abbreviated_model_names]
    else:
        labels = [_pretty_model_name(x) for x in names_sorted] if use_pretty_names else list(names_sorted)

    fig, ax = plt.subplots(figsize=(13, 11))

    if type_name is not None and type_index is not None:
        fig.subplots_adjust(left=0.30, top=0.88, right=0.92, bottom=0.16)
    else:
        fig.subplots_adjust(left=0.20, top=0.90, right=0.92, bottom=0.16)

    if plot_diff:
        if vmin is None or vmax is None:
            finite_abs = np.abs(M_re[np.isfinite(M_re)])
            if finite_abs.size == 0:
                max_abs = 1e-8
            else:
                max_abs = float(np.max(finite_abs))
                if max_abs <= 0:
                    max_abs = 1e-8
            vmin, vmax = -max_abs, max_abs
        cmap_name = cmap or "coolwarm"
    else:
        finite_vals = M_re[np.isfinite(M_re)]
        if vmin is None:
            vmin = 0.0 if finite_vals.size == 0 else float(np.min(finite_vals))
        if vmax is None:
            vmax = 1.0 if finite_vals.size == 0 else float(np.max(finite_vals))
            if vmax <= vmin:
                vmax = vmin + 1e-8
        cmap_name = cmap or "viridis"

    boundaries = np.linspace(vmin, vmax, colorbar_levels + 1)
    base_cmap = cm.get_cmap(cmap_name, colorbar_levels)
    norm = mcolors.BoundaryNorm(boundaries, base_cmap.N, clip=True)

    x = np.arange(n + 1)
    y = np.arange(n + 1)

    pcm = ax.pcolormesh(
        x,
        y,
        M_re,
        cmap=base_cmap,
        norm=norm,
        shading="flat",
        edgecolors="face",
        linewidth=0.0,
    )

    ax.set_aspect("equal")
    ax.set_xlim(0, n)
    ax.set_ylim(n, 0)

    cbar = fig.colorbar(
        pcm,
        ax=ax,
        boundaries=boundaries,
        ticks=np.linspace(vmin, vmax, min(6, colorbar_levels + 1)),
        fraction=0.046,
        pad=0.04,
        spacing="proportional",
        drawedges=False,
    )
    cbar.set_label("Value", rotation=90)
    _maybe_disable_math_parsing(cbar.ax.yaxis.label)
    cbar.ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    cbar.update_ticks()

    title_artist = ax.set_title(_normalize_text(title), fontsize=title_fontsize, pad=20)
    _maybe_disable_math_parsing(title_artist)

    centers = np.arange(n) + 0.5
    ax.set_xticks(centers)
    ax.set_yticks(centers)

    xtexts = ax.set_xticklabels(labels, fontsize=tick_fontsize)
    ytexts = ax.set_yticklabels(labels, fontsize=tick_fontsize)

    for t in list(xtexts) + list(ytexts):
        _maybe_disable_math_parsing(t)

    ax.tick_params(axis="x", top=True, bottom=False, labeltop=True, labelbottom=False, pad=x_tick_pad)
    ax.tick_params(axis="y", left=True, right=False, labelleft=True, labelright=False, pad=y_tick_pad)

    plt.setp(ax.get_xticklabels(), rotation=tick_rotation, ha="left", rotation_mode="anchor")

    ax.set_xticks(np.arange(n + 1), minor=True)
    ax.set_yticks(np.arange(n + 1), minor=True)
    ax.grid(which="minor", linestyle="-", linewidth=0.5)
    ax.tick_params(which="minor", bottom=False, left=False)

    if annotate_floor_1dp:
        M_rounded = np.floor(M_re * 100) / 100
        for i in range(n):
            for j in range(n):
                if np.isfinite(M_rounded[i, j]):
                    txt = ax.text(
                        j + 0.5,
                        i + 0.5,
                        f"{M_rounded[i, j]:.2f}",
                        ha="center",
                        va="center",
                        fontsize=7,
                    )
                    _maybe_disable_math_parsing(txt)

    _add_group_brackets(
        ax,
        type_name,
        type_index,
        n,
        top=True,
        left=True,
        pad=0.7,
        lw=1.8,
        fontsize=bracket_fontsize,
        top_label_offset=0.6,
        left_label_offset=0.2,
    )

    saved_path = None
    if savepath is not None:
        saved_path = _save_figure(fig, savepath, dpi=dpi)

    if show_plot:
        plt.show()

    if close_plot:
        plt.close(fig)

    return fig, ax, saved_path


# -------------------------
# convenience: build + plot in one call
# -------------------------
def plot_single_metric_from_npz_sorted(
    models,
    names_sorted,
    metric,
    raw_dir,
    filtered_dir,
    panel="filtered",
    diag_value=np.nan,
    missing="nan",
    raw_std_metric=None,
    title=None,
    savepath=None,
    annotate_floor_1dp=False,
    type_name=None,
    type_index=None,
    cmap=None,
    vmin=None,
    vmax=None,
    raw_name="dense",
    filtered_name="sparse",
    colorbar_levels=32,
    tick_fontsize=12,
    tick_rotation=60,
    use_pretty_names=True,
    abbreviated_model_names=None,
    x_tick_pad=37,
    y_tick_pad=37,
    title_fontsize=20,
    bracket_fontsize=10,
    close_plot=False,
    show_plot=True,
    dpi=200,
):
    M, info = build_single_metric_matrix(
        models=models,
        metric=metric,
        raw_dir=raw_dir,
        filtered_dir=filtered_dir,
        panel=panel,
        diag_value=diag_value,
        missing=missing,
        raw_std_metric=raw_std_metric,
    )

    if info["missing_pairs"] and missing == "nan":
        print(f"[plot_single_metric_from_npz_sorted] Missing {len(info['missing_pairs'])} pair(s); cells are NaN.")

    human = _HUMAN_TITLES.get(metric, metric)
    if title is None:
        if panel == "diff":
            if metric in _EDIT_METRICS:
                diff_label = f"{raw_name} - {filtered_name}"
            else:
                diff_label = f"{filtered_name} - {raw_name}"

            if raw_std_metric is not None:
                diff_label = f"({diff_label}) / {raw_std_metric}"

            title = f"{human} ({diff_label})"
        else:
            title = f"{human} ({panel})"

    if cmap is None and metric in _EDIT_METRICS and panel in {"raw", "filtered"}:
        cmap = "viridis_r"

    fig, ax, saved_path = plot_metric_heatmap_sorted(
        M=M,
        names=models,
        names_sorted=names_sorted,
        title=title,
        savepath=savepath,
        annotate_floor_1dp=annotate_floor_1dp,
        type_name=type_name,
        type_index=type_index,
        plot_diff=(panel == "diff"),
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        colorbar_levels=colorbar_levels,
        tick_fontsize=tick_fontsize,
        tick_rotation=tick_rotation,
        use_pretty_names=use_pretty_names,
        abbreviated_model_names=abbreviated_model_names,
        x_tick_pad=x_tick_pad,
        y_tick_pad=y_tick_pad,
        title_fontsize=title_fontsize,
        bracket_fontsize=bracket_fontsize,
        close_plot=close_plot,
        show_plot=show_plot,
        dpi=dpi,
    )

    return M, info, fig, ax, saved_path
