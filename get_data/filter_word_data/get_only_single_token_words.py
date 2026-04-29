#!/usr/bin/env python3
"""
write_common_single_token_indices.py

Given the per-model single-token filtered word datasets, compute the words that
survive for every model and write index lists into the original words folder.

Expected inputs:
  WORDS_ROOT/
    annotations/captions_train2017.json
    frequencies.npy
    annotations/top50k_words.csv   (optional)

  SINGLE_TOKEN_LLM_ROOT/*/annotations/captions_train2017.json
  SINGLE_TOKEN_TEXT_ROOT/*/annotations/captions_train2017.json

Outputs, written into WORDS_ROOT by default:
  single_token_common_indices.txt          zero-based row indices into the original words dataset
  single_token_common_indices_1based.txt   one-based ranks / COCO ids when start_id=1
  single_token_common_indices.npy          NumPy int64 array of zero-based indices
  single_token_common_words.txt            one kept word per line, original order
  single_token_common_word_frequencies.tsv index, rank, word, raw freq, normalized freq
  single_token_common_per_file_counts.tsv  diagnostic counts per filtered file
  single_token_common_summary.json         diagnostic summary
"""

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Set, Tuple

import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Intersect all per-model single-token word lists and write the kept "
            "original-word indices into the original words folder."
        )
    )
    p.add_argument(
        "--words_root",
        type=str,
        default="/home/kirilb/orcd/scratch/words",
        help="Original words dataset root. Output files are written here by default.",
    )
    p.add_argument(
        "--single_token_llm_root",
        type=str,
        default="/home/kirilb/orcd/scratch/PRH_data/single_token_words_llm",
        help="Root containing one filtered COCO words dataset per LLM model.",
    )
    p.add_argument(
        "--single_token_text_root",
        type=str,
        default="/home/kirilb/orcd/scratch/PRH_data/single_token_words_text",
        help="Root containing one filtered COCO words dataset per text embedding model.",
    )
    p.add_argument(
        "--ann_rel_path",
        type=str,
        default="annotations/captions_train2017.json",
        help="Relative path from a dataset root/model dir to the COCO annotations file.",
    )
    p.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory for outputs. Defaults to --words_root.",
    )
    p.add_argument(
        "--csv_rel_path",
        type=str,
        default="annotations/top50k_words.csv",
        help="Optional CSV manifest from words_download.py, used only for consistency checks.",
    )
    p.add_argument(
        "--freq_rel_path",
        type=str,
        default="frequencies.npy",
        help="Relative path to raw wordfreq frequencies under --words_root.",
    )
    p.add_argument(
        "--caption_word_mode",
        type=str,
        default="auto",
        choices=["auto", "whole_caption", "tab_first_field"],
        help=(
            "How to extract the word from a caption. 'auto' uses the first tab-separated "
            "field if tabs are present, otherwise the whole caption."
        ),
    )
    p.add_argument(
        "--fail_if_missing_roots",
        type=int,
        default=1,
        help="If 1, fail when a single-token root is missing. If 0, skip missing roots.",
    )
    p.add_argument(
        "--require_all_original_words_unique",
        type=int,
        default=1,
        help="If 1, fail if the original COCO captions contain duplicate words.",
    )
    p.add_argument(
        "--prefix",
        type=str,
        default="single_token_common",
        help="Prefix for output filenames.",
    )
    return p.parse_args()


def extract_word(caption: object, mode: str) -> str:
    if not isinstance(caption, str):
        return ""
    s = caption.strip()
    if mode == "tab_first_field":
        return s.split("\t", 1)[0].strip()
    if mode == "whole_caption":
        return s
    # auto
    if "\t" in s:
        return s.split("\t", 1)[0].strip()
    return s


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_words_from_coco(path: Path, mode: str) -> List[str]:
    data = load_json(path)
    anns = data.get("annotations")
    if not isinstance(anns, list):
        raise ValueError(f"{path} is not a COCO captions JSON with an annotations list.")

    words: List[str] = []
    for ann in anns:
        if not isinstance(ann, dict):
            continue
        w = extract_word(ann.get("caption", ""), mode)
        if w:
            words.append(w)
    return words


def load_original_annotations(path: Path, mode: str) -> Tuple[List[str], List[int], List[int]]:
    data = load_json(path)
    anns = data.get("annotations")
    if not isinstance(anns, list):
        raise ValueError(f"{path} is not a COCO captions JSON with an annotations list.")

    words: List[str] = []
    ann_ids: List[int] = []
    image_ids: List[int] = []
    for pos, ann in enumerate(anns):
        if not isinstance(ann, dict):
            raise ValueError(f"Annotation #{pos} in {path} is not a JSON object.")
        w = extract_word(ann.get("caption", ""), mode)
        if not w:
            raise ValueError(f"Annotation #{pos} in {path} has an empty caption after parsing.")
        words.append(w)
        ann_ids.append(int(ann.get("id", pos + 1)))
        image_ids.append(int(ann.get("image_id", pos + 1)))
    return words, ann_ids, image_ids


def discover_filtered_jsons(
    roots: Sequence[Path], ann_rel_path: str, fail_if_missing_roots: bool
) -> List[Path]:
    paths: List[Path] = []
    for root in roots:
        if not root.exists():
            msg = f"Single-token root does not exist: {root}"
            if fail_if_missing_roots:
                raise FileNotFoundError(msg)
            print(f"[warn] {msg}; skipping")
            continue
        # Model dirs are expected to be immediate children, but recursive discovery is safer.
        for path in sorted(root.rglob(Path(ann_rel_path).name)):
            # Keep only files whose tail matches ann_rel_path, e.g. annotations/captions_train2017.json.
            try:
                rel_tail = Path(*path.parts[-len(Path(ann_rel_path).parts):])
            except ValueError:
                continue
            if rel_tail == Path(ann_rel_path) and path.is_file():
                paths.append(path)
    # De-duplicate while preserving sorted order.
    unique: List[Path] = []
    seen = set()
    for p in paths:
        rp = p.resolve()
        if rp not in seen:
            seen.add(rp)
            unique.append(p)
    return unique


def read_optional_csv(csv_path: Path) -> Dict[str, Tuple[int, float, float]]:
    """
    Returns word -> (rank, raw_freq, normalized_freq) when the CSV exists.
    Empty dict otherwise.
    """
    if not csv_path.is_file():
        return {}
    out: Dict[str, Tuple[int, float, float]] = {}
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        required = {"rank", "word", "wordfreq_freq", "normalized_freq"}
        if not required.issubset(set(reader.fieldnames or [])):
            print(f"[warn] CSV {csv_path} lacks expected columns {sorted(required)}; ignoring it")
            return {}
        for row in reader:
            word = (row.get("word") or "").strip()
            if not word:
                continue
            out[word] = (
                int(row["rank"]),
                float(row["wordfreq_freq"]),
                float(row["normalized_freq"]),
            )
    return out


def write_lines(path: Path, values: Iterable[object]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for v in values:
            f.write(f"{v}\n")


def main() -> None:
    args = parse_args()

    words_root = Path(args.words_root).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else words_root
    output_dir.mkdir(parents=True, exist_ok=True)

    original_ann_path = words_root / args.ann_rel_path
    freq_path = words_root / args.freq_rel_path
    csv_path = words_root / args.csv_rel_path

    if not original_ann_path.is_file():
        raise FileNotFoundError(f"Original words annotations not found: {original_ann_path}")
    if not freq_path.is_file():
        raise FileNotFoundError(
            f"Original frequencies.npy not found: {freq_path}. "
            "This script intentionally reads frequencies from the original words folder."
        )

    roots = [
        Path(args.single_token_llm_root).expanduser().resolve(),
        Path(args.single_token_text_root).expanduser().resolve(),
    ]
    filtered_jsons = discover_filtered_jsons(
        roots, args.ann_rel_path, bool(args.fail_if_missing_roots)
    )
    if not filtered_jsons:
        raise FileNotFoundError(
            "No filtered per-model annotation files were found. Checked roots:\n  "
            + "\n  ".join(str(r) for r in roots)
        )

    print(f"[info] original words root: {words_root}")
    print(f"[info] original annotations: {original_ann_path}")
    print(f"[info] raw frequencies: {freq_path}")
    print(f"[info] output dir: {output_dir}")
    print(f"[info] found {len(filtered_jsons)} per-model filtered files")

    original_words, original_ann_ids, original_image_ids = load_original_annotations(
        original_ann_path, args.caption_word_mode
    )
    frequencies = np.load(freq_path)
    if frequencies.ndim != 1:
        raise ValueError(f"Expected one-dimensional frequencies.npy, got shape {frequencies.shape}")
    if len(frequencies) < len(original_words):
        raise ValueError(
            f"frequencies.npy is shorter than the original annotation list: "
            f"{len(frequencies)} < {len(original_words)}"
        )
    if len(frequencies) > len(original_words):
        print(
            f"[warn] frequencies.npy has {len(frequencies)} entries but original annotations have "
            f"{len(original_words)} entries. Only the annotation-aligned prefix is used."
        )
        frequencies = frequencies[: len(original_words)]

    word_to_indices: Dict[str, List[int]] = {}
    for i, w in enumerate(original_words):
        word_to_indices.setdefault(w, []).append(i)
    duplicate_words = {w: idxs for w, idxs in word_to_indices.items() if len(idxs) > 1}
    if duplicate_words and args.require_all_original_words_unique:
        examples = list(duplicate_words.items())[:10]
        raise ValueError(
            "Original words annotations contain duplicate captions. Examples: "
            + ", ".join(f"{w}: {idxs}" for w, idxs in examples)
        )

    per_file_rows = []
    common_words: Set[str] = set(original_words)
    for j, path in enumerate(filtered_jsons):
        words = load_words_from_coco(path, args.caption_word_mode)
        word_set = set(words)
        per_file_rows.append(
            {
                "file_index": j,
                "path": str(path),
                "num_annotations": len(words),
                "num_unique_words": len(word_set),
            }
        )
        common_words &= word_set

    # Preserve original frequency/rank order.
    kept_indices = [i for i, w in enumerate(original_words) if w in common_words]
    kept_words = [original_words[i] for i in kept_indices]
    kept_freqs = frequencies[np.asarray(kept_indices, dtype=np.int64)]
    total_freq = float(np.sum(frequencies))
    kept_norm_freqs = kept_freqs.astype(np.float64) / total_freq if total_freq > 0 else np.zeros_like(kept_freqs, dtype=np.float64)

    csv_lookup = read_optional_csv(csv_path)
    if csv_lookup:
        # Check rather than trust blindly. frequencies.npy remains the source for output.
        mismatches = 0
        for i in kept_indices[: min(1000, len(kept_indices))]:
            w = original_words[i]
            row = csv_lookup.get(w)
            if row is None or row[0] != i + 1:
                mismatches += 1
        if mismatches:
            print(
                f"[warn] CSV exists but the first checked rows are not perfectly aligned "
                f"with original annotations ({mismatches} mismatch(es)); using frequencies.npy."
            )

    prefix = args.prefix
    zero_based_path = output_dir / f"{prefix}_indices.txt"
    one_based_path = output_dir / f"{prefix}_indices_1based.txt"
    npy_path = output_dir / f"{prefix}_indices.npy"
    words_path = output_dir / f"{prefix}_words.txt"
    freq_tsv_path = output_dir / f"{prefix}_word_frequencies.tsv"
    per_file_path = output_dir / f"{prefix}_per_file_counts.tsv"
    summary_path = output_dir / f"{prefix}_summary.json"

    write_lines(zero_based_path, kept_indices)
    write_lines(one_based_path, (i + 1 for i in kept_indices))
    np.save(npy_path, np.asarray(kept_indices, dtype=np.int64))
    write_lines(words_path, kept_words)

    with freq_tsv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(
            [
                "index0",
                "rank1",
                "annotation_id",
                "image_id",
                "word",
                "wordfreq_freq",
                "normalized_freq",
            ]
        )
        for idx, word, raw_f, norm_f in zip(kept_indices, kept_words, kept_freqs, kept_norm_freqs):
            writer.writerow(
                [
                    idx,
                    idx + 1,
                    original_ann_ids[idx],
                    original_image_ids[idx],
                    word,
                    f"{float(raw_f):.10g}",
                    f"{float(norm_f):.10g}",
                ]
            )

    with per_file_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["file_index", "num_annotations", "num_unique_words", "path"],
            delimiter="\t",
        )
        writer.writeheader()
        for row in per_file_rows:
            writer.writerow(row)

    summary = {
        "words_root": str(words_root),
        "original_annotations": str(original_ann_path),
        "frequencies_npy": str(freq_path),
        "csv_manifest_checked_if_present": str(csv_path),
        "num_original_words": len(original_words),
        "num_filtered_files": len(filtered_jsons),
        "num_common_single_token_words": len(kept_indices),
        "indexing_convention": {
            f"{prefix}_indices.txt": "zero-based indices into the original annotation/embedding order",
            f"{prefix}_indices_1based.txt": "one-based ranks; matches rank when start_id=1",
        },
        "outputs": {
            "indices_zero_based_txt": str(zero_based_path),
            "indices_one_based_txt": str(one_based_path),
            "indices_npy": str(npy_path),
            "words_txt": str(words_path),
            "frequencies_tsv": str(freq_tsv_path),
            "per_file_counts_tsv": str(per_file_path),
        },
        "first_20_kept": [
            {
                "index0": int(i),
                "rank1": int(i + 1),
                "word": original_words[i],
                "wordfreq_freq": float(frequencies[i]),
            }
            for i in kept_indices[:20]
        ],
    }
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("[done] common single-token words:", len(kept_indices))
    print("[done] wrote:")
    for pth in [
        zero_based_path,
        one_based_path,
        npy_path,
        words_path,
        freq_tsv_path,
        per_file_path,
        summary_path,
    ]:
        print(f"  {pth}")


if __name__ == "__main__":
    main()
