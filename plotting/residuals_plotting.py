#!/usr/bin/env python3

import argparse
import os
import re
from typing import Dict, List, Optional, Tuple

import numpy as np


KS = [32, 64, 128]


def latex_escape(s: str) -> str:
    """
    Escape plain text for LaTeX text mode.
    """
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
    return "".join(repl.get(ch, ch) for ch in s)


def sanitize_label(s: str) -> str:
    """
    Make a safe LaTeX label string.
    """
    s = s.strip().lower()
    s = re.sub(r"[^a-z0-9:_-]+", "_", s)
    s = re.sub(r"_+", "_", s)
    return s.strip("_")


def infer_dataset_name_from_root(root: str) -> str:
    """
    Infer dataset name from a root such as:
      /home/.../topk_sae_coco
      /home/.../topk_sae_visual_genome
    """
    base = os.path.basename(os.path.abspath(root))
    m = re.match(r"^topk_sae_(.+)$", base)
    if m:
        return m.group(1)
    return base


def prettify_model_name(model_name: str) -> str:
    """
    Mild cleanup for display in the LaTeX table.
    """
    return model_name.replace("__", "/")


def format_number(x: Optional[float], precision: int = 4) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "--"
    return f"{x:.{precision}g}"


def load_residual_matrix(root: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load:
      - residuals.npy                  shape [n_models, n_combos]
      - residuals_metadata.npz         with fields:
            row_keys: object array of model names
            combos:   int array shape [n_combos, 2], columns are (d, k)

    Returns:
      residuals, row_keys, combos
    """
    residuals_path = os.path.join(root, "residuals.npy")
    metadata_path = os.path.join(root, "residuals_metadata.npz")

    if not os.path.isfile(residuals_path):
        raise FileNotFoundError(f"Residual matrix not found: {residuals_path}")
    if not os.path.isfile(metadata_path):
        raise FileNotFoundError(f"Residual metadata not found: {metadata_path}")

    residuals = np.load(residuals_path, allow_pickle=False)
    meta = np.load(metadata_path, allow_pickle=True)

    if "row_keys" not in meta or "combos" not in meta:
        raise KeyError(
            f"{metadata_path} must contain 'row_keys' and 'combos'"
        )

    row_keys = np.asarray(meta["row_keys"])
    combos = np.asarray(meta["combos"])

    if residuals.ndim != 2:
        raise ValueError(
            f"Residual matrix must be 2D, got shape {residuals.shape} at {residuals_path}"
        )
    if combos.ndim != 2 or combos.shape[1] != 2:
        raise ValueError(
            f"combos must have shape [n_combos, 2], got {combos.shape}"
        )
    if residuals.shape[0] != len(row_keys):
        raise ValueError(
            f"Row mismatch: residuals has {residuals.shape[0]} rows but row_keys has {len(row_keys)} entries"
        )
    if residuals.shape[1] != combos.shape[0]:
        raise ValueError(
            f"Column mismatch: residuals has {residuals.shape[1]} columns but combos has {combos.shape[0]} rows"
        )

    return residuals, row_keys, combos


def build_model_to_k_to_residual(
    residuals: np.ndarray,
    row_keys: np.ndarray,
    combos: np.ndarray,
    target_d: int,
    target_ks: List[int],
    verbose: bool = True,
) -> Dict[str, Dict[int, float]]:
    """
    Convert residual matrix + combo metadata into
      model_name -> {k -> residual}
    restricted to the requested d and ks.
    """
    combo_to_col: Dict[Tuple[int, int], int] = {}
    for j, combo in enumerate(combos):
        d = int(combo[0])
        k = int(combo[1])
        combo_to_col[(d, k)] = j

    model_to_k_to_residual: Dict[str, Dict[int, float]] = {}

    for i, rk in enumerate(row_keys):
        model_name = str(rk)
        model_to_k_to_residual[model_name] = {}

        for k in target_ks:
            col = combo_to_col.get((target_d, k))
            if col is None:
                if verbose:
                    print(f"[warn] combo (d={target_d}, k={k}) not present in metadata")
                continue

            val = float(residuals[i, col])
            model_to_k_to_residual[model_name][k] = val

            if verbose:
                print(f"[ok] d={target_d} model={model_name} k={k} residual={val:.6g}")

    return model_to_k_to_residual


def build_latex_table(
    model_to_k_to_residual: Dict[str, Dict[int, float]],
    target_d: int,
    dataset_name: str,
    caption: Optional[str] = None,
    label: Optional[str] = None,
    precision: int = 4,
) -> str:
    """
    Build a LaTeX table with columns:
      Model | k=32 | k=64 | k=128
    """
    models_sorted = sorted(model_to_k_to_residual.keys(), key=lambda s: s.lower())

    lines: List[str] = []
    lines.append(r"\begin{table}[ht]")
    lines.append(r"\centering")
    lines.append(r"\begin{tabular}{lccc}")
    lines.append(r"\hline")
    lines.append(r"Model & $k=32$ & $k=64$ & $k=128$ \\")
    lines.append(r"\hline")

    for model in models_sorted:
        row = model_to_k_to_residual[model]
        pretty_model = latex_escape(prettify_model_name(model))
        vals = [format_number(row.get(k), precision=precision) for k in KS]
        lines.append(f"{pretty_model} & {vals[0]} & {vals[1]} & {vals[2]} \\\\")

    lines.append(r"\hline")
    lines.append(r"\end{tabular}")

    if caption is None:
        caption = (
            f"Residuals for dataset {dataset_name}, sparse dimension d={target_d}, "
            f"and k in {{32, 64, 128}}."
        )
    if label is None:
        label = f"tab:residuals_{dataset_name}_d_{target_d}"

    lines.append(rf"\caption{{{latex_escape(caption)}}}")
    lines.append(rf"\label{{{sanitize_label(label)}}}")
    lines.append(r"\end{table}")

    return "\n".join(lines) + "\n"


def save_text(path: str, text: str) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read root-level residuals.npy and residuals_metadata.npz from a Top-K SAE "
            "dataset folder and write a LaTeX table for a fixed sparse dimension d "
            "with columns k in {32,64,128}."
        )
    )
    parser.add_argument(
        "d",
        type=int,
        help="Sparse dimension d to filter for."
    )
    parser.add_argument(
        "root",
        type=str,
        help='Dataset SAE root, e.g. "/home/kirilb/orcd/pool/PRH_data/topk_sae_coco/".'
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output .tex path. Default: <root>/residual_statistics_<dataset>_d_<d>.tex"
    )
    parser.add_argument(
        "--caption",
        type=str,
        default=None,
        help="Optional LaTeX table caption as plain text."
    )
    parser.add_argument(
        "--label",
        type=str,
        default=None,
        help="Optional LaTeX table label."
    )
    parser.add_argument(
        "--precision",
        type=int,
        default=4,
        help="Number formatting precision."
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-model logging."
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    root = os.path.abspath(args.root)
    if not os.path.isdir(root):
        raise FileNotFoundError(f"Root folder not found: {root}")

    dataset_name = infer_dataset_name_from_root(root)

    out_path = args.out
    if out_path is None:
        out_path = os.path.join(
            root,
            f"residual_statistics_{dataset_name}_d_{args.d}.tex"
        )

    verbose = not args.quiet

    print(f"ROOT={root}")
    print(f"DATASET={dataset_name}")
    print(f"d={args.d}")
    print(f"OUT={out_path}")

    residuals, row_keys, combos = load_residual_matrix(root)

    print(f"residuals.shape={residuals.shape}")
    print(f"n_row_keys={len(row_keys)}")
    print(f"combos.shape={combos.shape}")

    model_to_k_to_residual = build_model_to_k_to_residual(
        residuals=residuals,
        row_keys=row_keys,
        combos=combos,
        target_d=args.d,
        target_ks=KS,
        verbose=verbose,
    )

    table_tex = build_latex_table(
        model_to_k_to_residual=model_to_k_to_residual,
        target_d=args.d,
        dataset_name=dataset_name,
        caption=args.caption,
        label=args.label,
        precision=args.precision,
    )

    save_text(out_path, table_tex)
    print(f"[done] wrote {out_path}")


if __name__ == "__main__":
    main()