#!/usr/bin/env python3
r"""
Make LaTeX tables from sparse_features_statistics.npz files.

This consumes the files produced by compute_sparse_feature_percentiles.py.

Important defaults:
  * truncated-feature tables are grouped by the pre-truncation sparse
    dimension, i.e. the column dimension of X_features.npz;
  * row names are cleaned so folders like
      topk_8192_google__gemma-3-1b-it_text_k_32
    appear as
      google__gemma-3-1b-it_text.

Default table columns:
    Model | optional actual/truncated dim | 5% | 95% | 95%/5%

Requires:
    numpy

Optional LaTeX packages for the generated tables:
    \usepackage{booktabs}
    \usepackage{graphicx}   % only if using the default resizebox output
    \usepackage{longtable}  % only if using --longtable
"""

import argparse
import csv
import math
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np


STATS_FILE = "sparse_features_statistics.npz"
FULL_FILE = "X_features.npz"
TRUNC_FILE = "X_features_truncated.npz"

DEFAULT_ROOTS = [
    "/home/kirilb/orcd/scratch/PRH_data/topk_sae_visual_genome",
    "/home/kirilb/orcd/scratch/PRH_data/topk_sae_coco",
    "/home/kirilb/orcd/scratch/PRH_data/topk_sae_cc3m",
]

DATASET_TITLES = {
    "coco": "COCO",
    "cc3m": "CC3M",
    "visual_genome": "Visual Genome",
    "vg": "Visual Genome",
}

CONFIG_COMPONENT_PATTERNS = [
    r"^d[_=-]?\d+$",
    r"^dim[_=-]?\d+$",
    r"^dimension[_=-]?\d+$",
    r"^k[_=-]?\d+(?:\.\d+)?$",
    r"^k\d+$",
    r"^kvar$",
    r"^seed[_=-]?\d+$",
    r"^seeds?$",
    r"^checkpoints?$",
    r"^runs?$",
    r"^features?$",
    r"^topk[_-]?\d+.*$",
    r"^batchtopk[_-]?\d+.*$",
    r"^dictionary$",
]


# -----------------------------
# Formatting helpers
# -----------------------------

def scalar(x: Any) -> Any:
    arr = np.asarray(x)
    if arr.shape == ():
        return arr.item()
    if arr.size == 1:
        return arr.reshape(-1)[0].item()
    return arr


def is_finite_number(x: Any) -> bool:
    try:
        return math.isfinite(float(x))
    except Exception:
        return False


def intish(x: Any, tol: float = 1e-9) -> bool:
    if not is_finite_number(x):
        return False
    return abs(float(x) - round(float(x))) <= tol


def format_float(x: Any, digits: int = 6) -> str:
    if x is None:
        return "nan"
    try:
        xf = float(x)
    except Exception:
        return str(x)
    if not math.isfinite(xf):
        return "nan"
    return f"{xf:.{digits}g}"


def format_dimension(x: Any) -> str:
    if intish(x):
        return str(int(round(float(x))))
    return format_float(x, digits=8)


def format_sparsity(x: Any) -> str:
    if x == "all":
        return "all"
    if intish(x):
        return str(int(round(float(x))))
    return format_float(x, digits=6)


def latex_escape(s: Any) -> str:
    text = str(s)
    repl = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(repl.get(ch, ch) for ch in text)


def latex_model_name(model: str, use_texttt: bool = True) -> str:
    escaped = latex_escape(model)
    if use_texttt:
        return rf"\texttt{{{escaped}}}"
    return escaped


def safe_filename(s: Any) -> str:
    out = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(s))
    out = out.strip("_.-")
    return out or "unnamed"


def latex_label_slug(s: Any) -> str:
    out = re.sub(r"[^A-Za-z0-9:-]+", "-", str(s))
    out = re.sub(r"-+", "-", out).strip("-")
    return out.lower() or "unnamed"


def title_dataset(dataset: str) -> str:
    return DATASET_TITLES.get(dataset, dataset.replace("_", " ").title())


def sort_value_for_label(s: str) -> Tuple[int, Any]:
    if s == "all":
        return (0, -1)
    try:
        return (0, float(s))
    except Exception:
        return (1, s)


# -----------------------------
# Path and model-name parsing helpers
# -----------------------------

def infer_dataset_from_root(root: Path) -> str:
    name = root.name.rstrip("/")
    prefixes = [
        "topk_sae_",
        "batchtopk_sae_",
        "alt_sae_",
        "sae_",
    ]
    for prefix in prefixes:
        if name.startswith(prefix):
            return name[len(prefix):]
    return name


def looks_like_model_component(component: str) -> bool:
    c = component
    if c.endswith("_text") or c.endswith("_img") or c.endswith("_image"):
        return True
    if "__" in c:
        return True
    # HuggingFace-style names are often sanitized as org--model or org__model.
    if "--" in c and not re.match(r"^(topk|batchtopk)[_-]", c):
        return True
    return False


def looks_like_config_component(component: str) -> bool:
    c = component.lower()
    return any(re.match(pattern, c) for pattern in CONFIG_COMPONENT_PATTERNS)


def infer_model_from_path(stats_path: Path, root: Path) -> str:
    folder = stats_path.parent
    try:
        parts = list(folder.relative_to(root).parts)
    except ValueError:
        parts = list(folder.parts)

    parts = [p for p in parts if p not in (".", "")]
    if not parts:
        return folder.name

    model_like = [p for p in parts if looks_like_model_component(p)]
    if model_like:
        return model_like[-1]

    meaningful = [p for p in parts if not looks_like_config_component(p)]
    if meaningful:
        return meaningful[-1]

    return parts[-1]


def clean_model_name(raw_model: str) -> str:
    """
    Strip SAE/run configuration from row names.

    Example:
        topk_8192_google__gemma-3-1b-it_text_k_32
    becomes:
        google__gemma-3-1b-it_text
    """
    s = os.path.basename(str(raw_model).rstrip("/"))

    # Repeat a few times in case a name has nested/repeated wrappers.
    for _ in range(8):
        old = s

        # Prefixes such as topk_8192_ or batchtopk_16384_.
        s = re.sub(r"^(?:topk|batchtopk)[_-]?\d+(?:[_-]+)", "", s, flags=re.IGNORECASE)
        s = re.sub(r"^(?:topk|batchtopk)(?:[_-]+)", "", s, flags=re.IGNORECASE)

        # Final sparsity suffixes such as _k_32, _k32, -k-64, or _kvar.
        # Anchored at the end so model names like clip-vit-base-patch32 are preserved.
        s = re.sub(r"(?:[_-]+)k(?:[_-]?\d+(?:\.\d+)?)$", "", s, flags=re.IGNORECASE)
        s = re.sub(r"(?:[_-]+)kvar$", "", s, flags=re.IGNORECASE)

        if s == old:
            break

    return s or str(raw_model)


# -----------------------------
# Sparse NPZ shape helpers
# -----------------------------

def read_sparse_npz_ncols(path: Path) -> Optional[int]:
    """Read the column dimension from a scipy sparse .npz without loading data."""
    if not path.is_file():
        return None
    try:
        with np.load(path, allow_pickle=False) as z:
            if "shape" not in z:
                return None
            shape = np.asarray(z["shape"]).reshape(-1)
            if shape.size < 2:
                return None
            return int(shape[1])
    except Exception:
        return None


def first_finite(*values: Any) -> Any:
    for value in values:
        if is_finite_number(value):
            return value
    return float("nan")


# -----------------------------
# Data loading
# -----------------------------

def percentile_value(levels: np.ndarray, values: np.ndarray, q: float, stats_path: Path) -> float:
    matches = np.where(np.isclose(levels.astype(float), float(q)))[0]
    if matches.size == 0:
        available = ", ".join(format_float(v, digits=4) for v in levels)
        raise ValueError(
            f"Percentile {q} was not found in {stats_path}. "
            f"Available levels are: {available}"
        )
    return float(values[int(matches[0])])


def load_one_record(
    stats_path: Path,
    root: Path,
    dataset: str,
    which: str,
    low: float,
    high: float,
    model_name_mode: str,
) -> Dict[str, Any]:
    with np.load(stats_path, allow_pickle=False) as z:
        levels = np.asarray(z["percentile_levels"], dtype=float)
        pct_key = f"{which}_percentiles"
        actual_dim_key = f"sparse_dimension_{which}"
        actual_sparsity_key = f"{which}_sparsity"

        missing = [key for key in (pct_key, actual_dim_key, actual_sparsity_key) if key not in z]
        if missing:
            raise KeyError(f"Missing keys in {stats_path}: {missing}")

        values = np.asarray(z[pct_key], dtype=float)
        p_low = percentile_value(levels, values, low, stats_path)
        p_high = percentile_value(levels, values, high, stats_path)

        if math.isfinite(p_low) and abs(p_low) > 0 and math.isfinite(p_high):
            ratio = p_high / p_low
        else:
            ratio = float("nan")

        full_dim_from_file = read_sparse_npz_ncols(stats_path.parent / FULL_FILE)
        trunc_dim_from_file = read_sparse_npz_ncols(stats_path.parent / TRUNC_FILE)

        full_dim_from_stats = scalar(z["sparse_dimension_full"]) if "sparse_dimension_full" in z else float("nan")
        trunc_dim_from_stats = scalar(z["sparse_dimension_truncated"]) if "sparse_dimension_truncated" in z else float("nan")

        pre_truncation_sparse_dimension = first_finite(full_dim_from_file, full_dim_from_stats)
        if which == "full":
            actual_sparse_dimension = first_finite(full_dim_from_file, full_dim_from_stats, scalar(z[actual_dim_key]))
        else:
            actual_sparse_dimension = first_finite(trunc_dim_from_file, trunc_dim_from_stats, scalar(z[actual_dim_key]))

        actual_sparsity = scalar(z[actual_sparsity_key])
        pre_truncation_sparsity = scalar(z["full_sparsity"]) if "full_sparsity" in z else actual_sparsity
        dense_dimension = scalar(z["dense_dimension"]) if "dense_dimension" in z else float("nan")

    raw_model = infer_model_from_path(stats_path, root)
    if model_name_mode == "raw":
        model = raw_model
    else:
        model = clean_model_name(raw_model)

    return {
        "dataset": dataset,
        "feature_set": which,
        "model": model,
        "raw_model": raw_model,
        "root": str(root),
        "stats_path": str(stats_path),
        "dense_dimension": dense_dimension,
        "pre_truncation_sparse_dimension": pre_truncation_sparse_dimension,
        "actual_sparse_dimension": actual_sparse_dimension,
        "pre_truncation_sparsity": pre_truncation_sparsity,
        "actual_sparsity": actual_sparsity,
        # Backward-compatible aliases.
        "sparse_dimension": actual_sparse_dimension,
        "sparsity": actual_sparsity,
        "p_low": p_low,
        "p_high": p_high,
        "ratio": ratio,
    }


def collect_records(args: argparse.Namespace) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    roots = [Path(r).expanduser().resolve() for r in args.roots]
    which_values = ["full", "truncated"] if args.which == "both" else [args.which]

    for root in roots:
        if not root.is_dir():
            print(f"[warn] root does not exist, skipping: {root}")
            continue

        dataset = infer_dataset_from_root(root)
        stats_paths = sorted(root.rglob(STATS_FILE))
        if not stats_paths:
            print(f"[warn] no {STATS_FILE} files found under: {root}")
            continue

        for stats_path in stats_paths:
            for which in which_values:
                try:
                    rec = load_one_record(
                        stats_path=stats_path,
                        root=root,
                        dataset=dataset,
                        which=which,
                        low=args.low,
                        high=args.high,
                        model_name_mode=args.model_name_mode,
                    )
                except Exception as e:
                    if args.strict:
                        raise
                    print(f"[warn] skipping {stats_path} ({which}): {e}")
                    continue
                records.append(rec)

    return records


# -----------------------------
# Grouping
# -----------------------------

def group_dimension_value(rec: Dict[str, Any], args: argparse.Namespace) -> Any:
    if args.group_dimension == "pre_truncation":
        return rec["pre_truncation_sparse_dimension"]
    return rec["actual_sparse_dimension"]


def group_sparsity_value(rec: Dict[str, Any], args: argparse.Namespace) -> Any:
    if args.group_sparsity == "none":
        return "all"
    if args.group_sparsity == "pre_truncation":
        return rec["pre_truncation_sparsity"]
    return rec["actual_sparsity"]


def group_key(rec: Dict[str, Any], args: argparse.Namespace) -> Tuple[str, str, str, str]:
    return (
        rec["dataset"],
        rec["feature_set"],
        format_dimension(group_dimension_value(rec, args)),
        format_sparsity(group_sparsity_value(rec, args)),
    )


def sort_records(records: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return sorted(records, key=lambda r: (r["model"].lower(), r["model"], r.get("raw_model", "")))


# -----------------------------
# LaTeX generation
# -----------------------------

def make_caption(dataset: str, feature_set: str, d_label: str, k_label: str, args: argparse.Namespace) -> str:
    feature_phrase = "truncated sparse features" if feature_set == "truncated" else "full sparse features"
    if args.group_dimension == "pre_truncation":
        dim_phrase = rf"pre-truncation sparse dimension $d={latex_escape(d_label)}$"
    else:
        dim_phrase = rf"sparse dimension $d={latex_escape(d_label)}$"

    if args.group_sparsity == "none":
        sparsity_phrase = "all sparsities"
    elif args.group_sparsity == "pre_truncation":
        sparsity_phrase = rf"pre-truncation sparsity $k={latex_escape(k_label)}$"
    else:
        sparsity_phrase = rf"sparsity $k={latex_escape(k_label)}$"

    return (
        "Statistics of nonzero sparse feature entries for "
        + latex_escape(title_dataset(dataset))
        + f" ({latex_escape(feature_phrase)}), "
        + dim_phrase
        + ", and "
        + sparsity_phrase
        + "."
    )


def make_table_label(dataset: str, feature_set: str, d_label: str, k_label: str) -> str:
    slug = latex_label_slug(f"{dataset}-{feature_set}-d{d_label}-k{k_label}")
    return f"tab:sparse-feature-entries-{slug}"


def include_actual_dim_column(records: List[Dict[str, Any]], args: argparse.Namespace) -> bool:
    if args.actual_dim_column == "always":
        return True
    if args.actual_dim_column == "never":
        return False

    actual_dims = {format_dimension(rec["actual_sparse_dimension"]) for rec in records}
    grouped_dims = {format_dimension(group_dimension_value(rec, args)) for rec in records}

    # In auto mode, show the extra dim column when grouping hides actual dimensions.
    return len(actual_dims) > 1 or actual_dims != grouped_dims


def actual_dim_header(records: List[Dict[str, Any]]) -> str:
    feature_sets = {rec["feature_set"] for rec in records}
    if feature_sets == {"truncated"}:
        return r"$d_{\mathrm{trunc}}$"
    if feature_sets == {"full"}:
        return r"$d_{\mathrm{full}}$"
    return r"$d_{\mathrm{actual}}$"


def table_body_lines(
    records: List[Dict[str, Any]],
    low: float,
    high: float,
    digits: int,
    use_texttt: bool,
    include_dim_col: bool,
) -> List[str]:
    low_header = rf"{format_float(low, digits=4)}\%"
    high_header = rf"{format_float(high, digits=4)}\%"
    ratio_header = rf"{format_float(high, digits=4)}\%/{format_float(low, digits=4)}\%"

    colspec = "l" + ("r" if include_dim_col else "") + "rrr"
    header = "Model"
    if include_dim_col:
        header += f" & {actual_dim_header(records)}"
    header += f" & {low_header} & {high_header} & {ratio_header} \\\\"

    lines = [
        rf"\begin{{tabular}}{{{colspec}}}",
        r"\toprule",
        header,
        r"\midrule",
    ]

    for rec in sort_records(records):
        model = latex_model_name(rec["model"], use_texttt=use_texttt)
        p_low = format_float(rec["p_low"], digits=digits)
        p_high = format_float(rec["p_high"], digits=digits)
        ratio = format_float(rec["ratio"], digits=digits)
        row = model
        if include_dim_col:
            row += f" & {format_dimension(rec['actual_sparse_dimension'])}"
        row += f" & {p_low} & {p_high} & {ratio} \\\\"
        lines.append(row)

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
    ])
    return lines


def make_table(records: List[Dict[str, Any]], args: argparse.Namespace) -> str:
    first = records[0]
    dataset, feature_set, d_label, k_label = group_key(first, args)
    caption = make_caption(dataset, feature_set, d_label, k_label, args)
    label = make_table_label(dataset, feature_set, d_label, k_label)

    if args.longtable:
        return make_longtable(records, args, caption, label)

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
    ]
    if args.font_size:
        lines.append(rf"\{args.font_size}")

    body = table_body_lines(
        records=records,
        low=args.low,
        high=args.high,
        digits=args.digits,
        use_texttt=not args.no_texttt,
        include_dim_col=include_actual_dim_column(records, args),
    )

    if args.resizebox:
        lines.append(r"\resizebox{\textwidth}{!}{%")
        lines.extend(body)
        lines.append(r"}%")
    else:
        lines.extend(body)

    lines.append(rf"\caption{{{caption}}}")
    lines.append(rf"\label{{{label}}}")
    lines.append(r"\end{table}")
    lines.append("")
    return "\n".join(lines)


def make_longtable(
    records: List[Dict[str, Any]],
    args: argparse.Namespace,
    caption: str,
    label: str,
) -> str:
    low_header = rf"{format_float(args.low, digits=4)}\%"
    high_header = rf"{format_float(args.high, digits=4)}\%"
    ratio_header = rf"{format_float(args.high, digits=4)}\%/{format_float(args.low, digits=4)}\%"
    include_dim_col = include_actual_dim_column(records, args)

    colspec = "l" + ("r" if include_dim_col else "") + "rrr"
    header = "Model"
    if include_dim_col:
        header += f" & {actual_dim_header(records)}"
    header += f" & {low_header} & {high_header} & {ratio_header} \\\\"

    lines = [
        rf"\begin{{longtable}}{{{colspec}}}",
        rf"\caption{{{caption}}}\label{{{label}}}\\",
        r"\toprule",
        header,
        r"\midrule",
        r"\endfirsthead",
        r"\toprule",
        header,
        r"\midrule",
        r"\endhead",
    ]

    for rec in sort_records(records):
        model = latex_model_name(rec["model"], use_texttt=not args.no_texttt)
        p_low = format_float(rec["p_low"], digits=args.digits)
        p_high = format_float(rec["p_high"], digits=args.digits)
        ratio = format_float(rec["ratio"], digits=args.digits)
        row = model
        if include_dim_col:
            row += f" & {format_dimension(rec['actual_sparse_dimension'])}"
        row += f" & {p_low} & {p_high} & {ratio} \\\\"
        lines.append(row)

    lines.extend([
        r"\bottomrule",
        r"\end{longtable}",
        "",
    ])
    return "\n".join(lines)


# -----------------------------
# Output
# -----------------------------

def write_csv(records: List[Dict[str, Any]], out_path: Path, args: argparse.Namespace) -> None:
    p_low_name = f"p{format_float(args.low, digits=4)}"
    p_high_name = f"p{format_float(args.high, digits=4)}"
    ratio_name = f"p{format_float(args.high, digits=4)}_over_p{format_float(args.low, digits=4)}"

    fieldnames = [
        "dataset",
        "feature_set",
        "model",
        "raw_model",
        "group_sparse_dimension",
        "group_sparsity",
        "pre_truncation_sparse_dimension",
        "actual_sparse_dimension",
        "pre_truncation_sparsity",
        "actual_sparsity",
        "dense_dimension",
        p_low_name,
        p_high_name,
        ratio_name,
        "stats_path",
    ]
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for rec in sorted(records, key=lambda r: (group_key(r, args), r["model"].lower(), r["raw_model"])):
            writer.writerow({
                "dataset": rec["dataset"],
                "feature_set": rec["feature_set"],
                "model": rec["model"],
                "raw_model": rec["raw_model"],
                "group_sparse_dimension": format_dimension(group_dimension_value(rec, args)),
                "group_sparsity": format_sparsity(group_sparsity_value(rec, args)),
                "pre_truncation_sparse_dimension": format_dimension(rec["pre_truncation_sparse_dimension"]),
                "actual_sparse_dimension": format_dimension(rec["actual_sparse_dimension"]),
                "pre_truncation_sparsity": format_sparsity(rec["pre_truncation_sparsity"]),
                "actual_sparsity": format_sparsity(rec["actual_sparsity"]),
                "dense_dimension": format_dimension(rec["dense_dimension"]),
                p_low_name: format_float(rec["p_low"], digits=args.digits),
                p_high_name: format_float(rec["p_high"], digits=args.digits),
                ratio_name: format_float(rec["ratio"], digits=args.digits),
                "stats_path": rec["stats_path"],
            })


def write_outputs(records: List[Dict[str, Any]], args: argparse.Namespace) -> None:
    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    groups: Dict[Tuple[str, str, str, str], List[Dict[str, Any]]] = defaultdict(list)
    for rec in records:
        groups[group_key(rec, args)].append(rec)

    group_items = sorted(
        groups.items(),
        key=lambda item: (
            item[0][0],
            0 if item[0][1] == "truncated" else 1,
            sort_value_for_label(item[0][2]),
            sort_value_for_label(item[0][3]),
        ),
    )

    master_lines = [
        r"% Auto-generated by make_sparse_feature_percentile_latex_tables.py",
        rf"% Model name mode: {args.model_name_mode}",
        rf"% Group dimension mode: {args.group_dimension}",
        rf"% Group sparsity mode: {args.group_sparsity}",
        r"% Requires: \usepackage{booktabs}",
    ]
    if args.resizebox and not args.longtable:
        master_lines.append(r"% Requires: \usepackage{graphicx}")
    if args.longtable:
        master_lines.append(r"% Requires: \usepackage{longtable}")
    master_lines.append("")

    written_files: List[Path] = []
    for key, recs in group_items:
        dataset, feature_set, d_label, k_label = key
        table_tex = make_table(recs, args)
        filename = (
            f"sparse_feature_entries_"
            f"{safe_filename(dataset)}_"
            f"{safe_filename(feature_set)}_"
            f"group-{safe_filename(args.group_dimension)}_"
            f"d{safe_filename(d_label)}_"
            f"k{safe_filename(k_label)}.tex"
        )
        path = out_dir / filename
        path.write_text(table_tex, encoding="utf-8")
        written_files.append(path)

        master_lines.append(f"% {dataset} | {feature_set} | group d={d_label} | group k={k_label} | rows={len(recs)}")
        if args.master_mode == "input":
            master_lines.append(rf"\input{{{filename}}}")
        else:
            master_lines.append(table_tex)
        master_lines.append("")

    master_path = out_dir / "all_sparse_feature_percentile_tables.tex"
    master_path.write_text("\n".join(master_lines), encoding="utf-8")

    csv_path = out_dir / "sparse_feature_percentile_summary.csv"
    write_csv(records, csv_path, args)

    print(f"[ok] collected {len(records)} row(s) into {len(groups)} table group(s)")
    print(f"[ok] model name mode:       {args.model_name_mode}")
    print(f"[ok] group dimension:       {args.group_dimension}")
    print(f"[ok] group sparsity:        {args.group_sparsity}")
    print(f"[ok] actual dim column:     {args.actual_dim_column}")
    print(f"[ok] wrote table directory: {out_dir}")
    print(f"[ok] wrote master TeX:      {master_path}")
    print(f"[ok] wrote CSV summary:     {csv_path}")
    if written_files:
        print("[ok] first few table files:")
        for p in written_files[:10]:
            print(f"     {p}")
        if len(written_files) > 10:
            print(f"     ... and {len(written_files) - 10} more")


# -----------------------------
# CLI
# -----------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Make LaTeX tables from sparse_features_statistics.npz files. "
            "By default, truncated tables are grouped by the pre-truncation "
            "dimension read from X_features.npz."
        )
    )
    parser.add_argument(
        "roots",
        nargs="*",
        default=DEFAULT_ROOTS,
        help=(
            "Root directories to search recursively. If omitted, uses the "
            "default visual_genome, coco, and cc3m topk_sae roots."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="sparse_feature_percentile_latex_tables",
        help="Directory where TeX tables and CSV summary will be written.",
    )
    parser.add_argument(
        "--which",
        choices=["truncated", "full", "both"],
        default="truncated",
        help="Which feature statistics to table. Default: truncated.",
    )
    parser.add_argument(
        "--group-dimension",
        choices=["pre_truncation", "actual"],
        default="pre_truncation",
        help=(
            "Dimension used to group rows into tables. For truncated features, "
            "pre_truncation means the column dimension of X_features.npz. "
            "Default: pre_truncation."
        ),
    )
    parser.add_argument(
        "--group-sparsity",
        choices=["pre_truncation", "actual", "none"],
        default="pre_truncation",
        help=(
            "Sparsity used to group rows into tables. Default: pre_truncation. "
            "Use none to put all sparsities with the same dataset/feature/dim into one table."
        ),
    )
    parser.add_argument(
        "--actual-dim-column",
        choices=["auto", "always", "never"],
        default="auto",
        help=(
            "Whether to include an extra row column with the actual sparse dimension. "
            "auto shows it when grouping hides actual dimensions. Default: auto."
        ),
    )
    parser.add_argument(
        "--model-name-mode",
        choices=["clean", "raw"],
        default="clean",
        help=(
            "clean strips wrappers like topk_8192_ and final _k_32 from row names. "
            "raw keeps folder names unchanged. Default: clean."
        ),
    )
    parser.add_argument(
        "--low",
        type=float,
        default=5.0,
        help="Lower percentile to report. Default: 5.",
    )
    parser.add_argument(
        "--high",
        type=float,
        default=95.0,
        help="Upper percentile to report. Default: 95.",
    )
    parser.add_argument(
        "--digits",
        type=int,
        default=6,
        help="Significant digits for numerical entries. Default: 6.",
    )
    parser.add_argument(
        "--font-size",
        type=str,
        default="small",
        choices=["", "footnotesize", "small", "scriptsize", "tiny"],
        help="LaTeX font size command inside table. Use '' for none. Default: small.",
    )
    parser.add_argument(
        "--no-resizebox",
        dest="resizebox",
        action="store_false",
        help="Do not wrap tabulars in \\resizebox{\\textwidth}{!}{...}.",
    )
    parser.set_defaults(resizebox=True)
    parser.add_argument(
        "--longtable",
        action="store_true",
        help="Use longtable instead of table+tabular. Disables resizebox behavior.",
    )
    parser.add_argument(
        "--no-texttt",
        action="store_true",
        help="Do not wrap model names in \\texttt{...}.",
    )
    parser.add_argument(
        "--master-mode",
        choices=["input", "inline"],
        default="input",
        help="Whether the master TeX file uses \\input{...} or inlines all tables. Default: input.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Raise an error instead of warning/skipping malformed stats files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.longtable:
        args.resizebox = False

    records = collect_records(args)
    if not records:
        raise RuntimeError(
            "No records found. Make sure compute_sparse_feature_percentiles.py has already "
            f"created {STATS_FILE} files under the requested roots."
        )

    write_outputs(records, args)


if __name__ == "__main__":
    main()
