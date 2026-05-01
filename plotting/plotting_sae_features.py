#!/usr/bin/env python3
r"""
Make LaTeX tables from sparse_features_statistics.npz files.

This consumes the files produced by compute_sparse_feature_percentiles.py.
For each root, it recursively finds sparse_features_statistics.npz files and
writes one LaTeX table per (dataset, feature-set, sparse dimension, sparsity).

Default table columns:
    Model | 5% | 95% | 95%/5%

Requires:
    numpy

Optional LaTeX packages for the generated tables:
    \usepackage{booktabs}
    \usepackage{graphicx}   % only if using the default resizebox output
"""

import argparse
import csv
import math
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np


STATS_FILE = "sparse_features_statistics.npz"

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


# -----------------------------
# Path parsing helpers
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
) -> Dict[str, Any]:
    with np.load(stats_path, allow_pickle=False) as z:
        levels = np.asarray(z["percentile_levels"], dtype=float)
        pct_key = f"{which}_percentiles"
        sparse_dim_key = f"sparse_dimension_{which}"
        sparsity_key = f"{which}_sparsity"

        missing = [key for key in (pct_key, sparse_dim_key, sparsity_key) if key not in z]
        if missing:
            raise KeyError(f"Missing keys in {stats_path}: {missing}")

        values = np.asarray(z[pct_key], dtype=float)
        p_low = percentile_value(levels, values, low, stats_path)
        p_high = percentile_value(levels, values, high, stats_path)

        if math.isfinite(p_low) and abs(p_low) > 0 and math.isfinite(p_high):
            ratio = p_high / p_low
        else:
            ratio = float("nan")

        sparse_dimension = scalar(z[sparse_dim_key])
        sparsity = scalar(z[sparsity_key])
        dense_dimension = scalar(z["dense_dimension"]) if "dense_dimension" in z else float("nan")

    return {
        "dataset": dataset,
        "feature_set": which,
        "model": infer_model_from_path(stats_path, root),
        "root": str(root),
        "stats_path": str(stats_path),
        "dense_dimension": dense_dimension,
        "sparse_dimension": sparse_dimension,
        "sparsity": sparsity,
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
                    )
                except Exception as e:
                    if args.strict:
                        raise
                    print(f"[warn] skipping {stats_path} ({which}): {e}")
                    continue
                records.append(rec)

    return records


# -----------------------------
# LaTeX generation
# -----------------------------

def group_key(rec: Dict[str, Any]) -> Tuple[str, str, str, str]:
    return (
        rec["dataset"],
        rec["feature_set"],
        format_dimension(rec["sparse_dimension"]),
        format_sparsity(rec["sparsity"]),
    )


def sort_records(records: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return sorted(records, key=lambda r: (r["model"].lower(), r["model"]))


def make_caption(dataset: str, feature_set: str, d_label: str, k_label: str) -> str:
    feature_phrase = "truncated sparse features" if feature_set == "truncated" else "full sparse features"
    return (
        "Statistics of nonzero sparse feature entries for "
        + latex_escape(title_dataset(dataset))
        + f" ({latex_escape(feature_phrase)}), "
        + rf"sparse dimension $d={latex_escape(d_label)}$, and sparsity $k={latex_escape(k_label)}$."
    )


def make_table_label(dataset: str, feature_set: str, d_label: str, k_label: str) -> str:
    slug = latex_label_slug(f"{dataset}-{feature_set}-d{d_label}-k{k_label}")
    return f"tab:sparse-feature-entries-{slug}"


def table_body_lines(
    records: List[Dict[str, Any]],
    low: float,
    high: float,
    digits: int,
    use_texttt: bool,
) -> List[str]:
    low_header = rf"{format_float(low, digits=4)}\%"
    high_header = rf"{format_float(high, digits=4)}\%"
    ratio_header = rf"{format_float(high, digits=4)}\%/{format_float(low, digits=4)}\%"

    lines = [
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        rf"Model & {low_header} & {high_header} & {ratio_header} \\",
        r"\midrule",
    ]

    for rec in sort_records(records):
        model = latex_model_name(rec["model"], use_texttt=use_texttt)
        p_low = format_float(rec["p_low"], digits=digits)
        p_high = format_float(rec["p_high"], digits=digits)
        ratio = format_float(rec["ratio"], digits=digits)
        lines.append(rf"{model} & {p_low} & {p_high} & {ratio} \\")

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
    ])
    return lines


def make_table(records: List[Dict[str, Any]], args: argparse.Namespace) -> str:
    first = records[0]
    dataset, feature_set, d_label, k_label = group_key(first)
    caption = make_caption(dataset, feature_set, d_label, k_label)
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

    lines = [
        r"\begin{longtable}{lrrr}",
        rf"\caption{{{caption}}}\label{{{label}}}\\",
        r"\toprule",
        rf"Model & {low_header} & {high_header} & {ratio_header} \\",
        r"\midrule",
        r"\endfirsthead",
        r"\toprule",
        rf"Model & {low_header} & {high_header} & {ratio_header} \\",
        r"\midrule",
        r"\endhead",
    ]

    for rec in sort_records(records):
        model = latex_model_name(rec["model"], use_texttt=not args.no_texttt)
        p_low = format_float(rec["p_low"], digits=args.digits)
        p_high = format_float(rec["p_high"], digits=args.digits)
        ratio = format_float(rec["ratio"], digits=args.digits)
        lines.append(rf"{model} & {p_low} & {p_high} & {ratio} \\")

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
    fieldnames = [
        "dataset",
        "feature_set",
        "model",
        "dense_dimension",
        "sparse_dimension",
        "sparsity",
        f"p{format_float(args.low, digits=4)}",
        f"p{format_float(args.high, digits=4)}",
        f"p{format_float(args.high, digits=4)}_over_p{format_float(args.low, digits=4)}",
        "stats_path",
    ]
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for rec in sorted(records, key=lambda r: (group_key(r), r["model"].lower())):
            writer.writerow({
                "dataset": rec["dataset"],
                "feature_set": rec["feature_set"],
                "model": rec["model"],
                "dense_dimension": format_dimension(rec["dense_dimension"]),
                "sparse_dimension": format_dimension(rec["sparse_dimension"]),
                "sparsity": format_sparsity(rec["sparsity"]),
                f"p{format_float(args.low, digits=4)}": format_float(rec["p_low"], digits=args.digits),
                f"p{format_float(args.high, digits=4)}": format_float(rec["p_high"], digits=args.digits),
                f"p{format_float(args.high, digits=4)}_over_p{format_float(args.low, digits=4)}": format_float(rec["ratio"], digits=args.digits),
                "stats_path": rec["stats_path"],
            })


def write_outputs(records: List[Dict[str, Any]], args: argparse.Namespace) -> None:
    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    groups: Dict[Tuple[str, str, str, str], List[Dict[str, Any]]] = defaultdict(list)
    for rec in records:
        groups[group_key(rec)].append(rec)

    group_items = sorted(
        groups.items(),
        key=lambda item: (
            item[0][0],
            0 if item[0][1] == "truncated" else 1,
            float(item[0][2]) if item[0][2].replace(".", "", 1).isdigit() else item[0][2],
            float(item[0][3]) if item[0][3].replace(".", "", 1).isdigit() else item[0][3],
        ),
    )

    master_lines = [
        r"% Auto-generated by make_sparse_feature_percentile_latex_tables.py",
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
            f"d{safe_filename(d_label)}_"
            f"k{safe_filename(k_label)}.tex"
        )
        path = out_dir / filename
        path.write_text(table_tex, encoding="utf-8")
        written_files.append(path)

        master_lines.append(f"% {dataset} | {feature_set} | d={d_label} | k={k_label}")
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
    print(f"[ok] wrote table directory: {out_dir}")
    print(f"[ok] wrote master TeX:       {master_path}")
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
            "By default, uses columns Model, 5%, 95%, and 95%/5%."
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
