#!/usr/bin/env python3
"""
Filter a COCO-style word dataset to examples whose caption is one tokenizer token
for a given LLM tokenizer.

Input format expected by the existing embedding scripts:
  {
    "images": [{"id": ..., "file_name": ...}, ...],
    "annotations": [{"image_id": ..., "caption": "word", ...}, ...],
    ... optional extra COCO-style keys ...
  }

Output format:
  The same COCO-style JSON, with annotations filtered and images restricted to
  those still referenced by at least one kept annotation. All other top-level keys
  are preserved. A small summary JSON is written next to the output.

Single-token criterion:
  len(tokenizer.encode(caption, add_special_tokens=False)) == 1

This script loads only the tokenizer, not the full LLM. A GPU is therefore not
needed for correctness.
"""

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

from transformers import AutoTokenizer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Filter COCO-format words to single-token captions for an LLM tokenizer.")
    p.add_argument("--model_name", type=str, required=True, help="Hugging Face model id.")
    p.add_argument(
        "--captions_json",
        type=str,
        required=True,
        help="Input COCO-style captions JSON, e.g. /path/words/annotations/captions_train2017.json.",
    )
    p.add_argument(
        "--output_root",
        type=str,
        required=True,
        help="Root under which MODEL_TAG/annotations/captions_train2017.json will be written.",
    )
    p.add_argument("--hf_cache_dir", type=str, default=None, help="Optional Hugging Face cache directory.")
    p.add_argument(
        "--caption_key",
        type=str,
        default="caption",
        help="Annotation key containing the word string. Default: caption.",
    )
    p.add_argument(
        "--keep_empty_images",
        action="store_true",
        help="Keep all original images instead of removing images whose annotations were dropped.",
    )
    p.add_argument(
        "--summary_filename",
        type=str,
        default="single_token_summary.json",
        help="Summary filename written in the model output directory.",
    )
    p.add_argument(
        "--max_examples",
        type=int,
        default=25,
        help="Number of kept/dropped examples to include in the summary JSON.",
    )
    return p.parse_args()


def model_tag(model_name: str) -> str:
    return model_name.replace("/", "__")


def load_coco(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError(f"Expected a COCO-style JSON object, got {type(data).__name__}.")
    if not isinstance(data.get("images"), list):
        raise ValueError("Expected COCO-style JSON with an 'images' list.")
    if not isinstance(data.get("annotations"), list):
        raise ValueError("Expected COCO-style JSON with an 'annotations' list.")
    return data


def token_ids_for_caption(tokenizer: Any, caption: str) -> List[int]:
    return tokenizer.encode(caption, add_special_tokens=False)


def token_pieces_for_caption(tokenizer: Any, caption: str) -> List[str]:
    try:
        return tokenizer.tokenize(caption, add_special_tokens=False)
    except TypeError:
        return tokenizer.tokenize(caption)
    except Exception:
        ids = token_ids_for_caption(tokenizer, caption)
        try:
            return tokenizer.convert_ids_to_tokens(ids)
        except Exception:
            return [str(i) for i in ids]


def filter_coco_words(data: Dict[str, Any], tokenizer: Any, args: argparse.Namespace) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    kept_annotations: List[Dict[str, Any]] = []
    kept_image_ids = set()
    kept_examples: List[Dict[str, Any]] = []
    dropped_examples: List[Dict[str, Any]] = []

    total_annotations = 0
    skipped_nonstring = 0
    skipped_missing_image_id = 0

    for ann in data["annotations"]:
        if not isinstance(ann, dict):
            skipped_nonstring += 1
            continue

        caption = ann.get(args.caption_key, None)
        image_id = ann.get("image_id", None)

        if image_id is None:
            skipped_missing_image_id += 1
            continue
        if not isinstance(caption, str):
            skipped_nonstring += 1
            continue

        caption = caption.strip()
        if not caption:
            skipped_nonstring += 1
            continue

        total_annotations += 1
        ids = token_ids_for_caption(tokenizer, caption)
        pieces = token_pieces_for_caption(tokenizer, caption)
        keep = len(ids) == 1

        if keep:
            kept_annotations.append(ann)
            try:
                kept_image_ids.add(int(image_id))
            except Exception:
                kept_image_ids.add(image_id)
            if len(kept_examples) < args.max_examples:
                kept_examples.append({"caption": caption, "image_id": image_id, "token_ids": ids, "tokens": pieces})
        else:
            if len(dropped_examples) < args.max_examples:
                dropped_examples.append(
                    {
                        "caption": caption,
                        "image_id": image_id,
                        "token_count": len(ids),
                        "token_ids": ids,
                        "tokens": pieces,
                    }
                )

    out = dict(data)
    out["annotations"] = kept_annotations

    if args.keep_empty_images:
        out["images"] = data["images"]
    else:
        filtered_images = []
        for im in data["images"]:
            if not isinstance(im, dict):
                continue
            image_id = im.get("id", None)
            try:
                normalized_id = int(image_id)
            except Exception:
                normalized_id = image_id
            if normalized_id in kept_image_ids:
                filtered_images.append(im)
        out["images"] = filtered_images

    stats = {
        "model_name": args.model_name,
        "model_tag": model_tag(args.model_name),
        "caption_key": args.caption_key,
        "single_token_rule": "len(tokenizer.encode(caption, add_special_tokens=False)) == 1",
        "total_original_annotations": len(data["annotations"]),
        "total_valid_annotations": total_annotations,
        "kept_annotations": len(kept_annotations),
        "dropped_annotations": total_annotations - len(kept_annotations),
        "total_original_images": len(data["images"]),
        "kept_images": len(out["images"]),
        "skipped_nonstring_or_empty_caption": skipped_nonstring,
        "skipped_missing_image_id": skipped_missing_image_id,
        "kept_examples": kept_examples,
        "dropped_examples": dropped_examples,
    }
    return out, stats


def main() -> None:
    args = parse_args()
    input_path = Path(args.captions_json)
    if not input_path.is_file():
        raise FileNotFoundError(f"Input captions JSON not found: {input_path}")

    tag = model_tag(args.model_name)
    out_dir = Path(args.output_root) / tag
    out_ann_dir = out_dir / "annotations"
    out_ann_dir.mkdir(parents=True, exist_ok=True)

    out_json = out_ann_dir / input_path.name
    out_summary = out_dir / args.summary_filename

    print(f"Loading tokenizer: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        cache_dir=args.hf_cache_dir,
        trust_remote_code=True,
        use_fast=True,
    )

    print(f"Loading COCO-style words JSON: {input_path}")
    data = load_coco(input_path)

    print("Filtering annotations by single-token captions...")
    filtered, stats = filter_coco_words(data, tokenizer, args)
    stats["input_json"] = str(input_path)
    stats["output_json"] = str(out_json)
    stats["output_summary"] = str(out_summary)

    with out_json.open("w", encoding="utf-8") as f:
        json.dump(filtered, f, ensure_ascii=False)
        f.write("\n")

    with out_summary.open("w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
        f.write("\n")

    print("Saved:")
    print(f"  {out_json}")
    print(f"  {out_summary}")
    print(
        f"Kept {stats['kept_annotations']} / {stats['total_valid_annotations']} valid annotations "
        f"and {stats['kept_images']} / {stats['total_original_images']} images."
    )


if __name__ == "__main__":
    main()
