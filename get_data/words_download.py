#!/usr/bin/env python3
"""
wordfreq_train_as_coco.py

Render the top-N most frequent English words (wordfreq) as a COCO captions dataset
with COCO-like folder structure:

  OUT_ROOT/
    images/train2017/                (empty; no images)
    annotations/captions_train2017.json
    annotations/top_words.csv        (optional)
    frequencies.npy                  (raw wordfreq frequencies, length N)

Each word becomes the caption for a distinct dummy image entry.
"""

import argparse
import csv
import json
import os
import time
from datetime import datetime
from typing import Dict, Any, List

import numpy as np
from wordfreq import top_n_list, word_frequency


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--out_root", type=str, required=True)

    p.add_argument("--N", type=int, default=50_000)
    p.add_argument("--lang", type=str, default="en")
    p.add_argument("--wordlist", type=str, default="best")

    # COCO-ish naming (defaults match your request)
    p.add_argument("--split_dir", type=str, default="train2017")
    p.add_argument("--ann_name", type=str, default="captions_train2017.json")

    # Optional CSV manifest
    p.add_argument("--write_csv", type=int, default=1)
    p.add_argument("--csv_name", type=str, default="top50k_words.csv")

    # IDs
    p.add_argument("--start_id", type=int, default=1)

    # If you want extra metadata in captions (off by default)
    p.add_argument(
        "--include_freq_in_caption",
        type=int,
        default=0,
        help="If 1, caption becomes 'word\\t<raw_freq>\\t<normalized_freq>' (default: 0).",
    )
    return p.parse_args()


def ensure_dirs(out_root: str, split_dir: str):
    img_dir = os.path.join(out_root, "images", split_dir)
    ann_dir = os.path.join(out_root, "annotations")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(ann_dir, exist_ok=True)
    return img_dir, ann_dir


def build_coco_json(words: List[str], freqs: List[float], rel_freqs: List[float], *, args) -> Dict[str, Any]:
    now = time.strftime("%Y-%m-%d %H:%M:%S")

    images = []
    annotations = []

    img_id = args.start_id
    ann_id = args.start_id

    for w, f_raw, f_norm in zip(words, freqs, rel_freqs):
        # No images exist: keep placeholders, but preserve COCO structure.
        images.append(
            {
                "id": img_id,
                "file_name": "",  # intentionally blank
                "width": 0,
                "height": 0,
                "license": 0,
                "flickr_url": "",
                "coco_url": "",
                "date_captured": "",
            }
        )

        if args.include_freq_in_caption:
            caption = f"{w}\t{f_raw:.10g}\t{f_norm:.10g}"
        else:
            caption = w

        annotations.append(
            {
                "id": ann_id,
                "image_id": img_id,
                "caption": caption,
            }
        )

        img_id += 1
        ann_id += 1

    return {
        "info": {
            "description": f"Top-{len(words)} wordfreq words as COCO captions (no images).",
            "version": "1.0",
            "year": datetime.utcnow().year,
            "date_created": now,
            "source": "wordfreq",
            "lang": args.lang,
            "wordlist": args.wordlist,
        },
        "licenses": [],
        "type": "captions",
        "images": images,
        "annotations": annotations,
    }


def write_csv(path: str, words: List[str], freqs: List[float], rel_freqs: List[float]):
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["rank", "word", "wordfreq_freq", "normalized_freq"])
        for i, (word, f_raw, f_norm) in enumerate(zip(words, freqs, rel_freqs), start=1):
            w.writerow([i, word, f_raw, f_norm])


def main():
    args = parse_args()
    img_dir, ann_dir = ensure_dirs(args.out_root, args.split_dir)

    ann_path = os.path.join(ann_dir, args.ann_name)
    csv_path = os.path.join(ann_dir, args.csv_name)
    freq_path = os.path.join(args.out_root, "frequencies.npy")  # <-- requested location

    print(f"[info] out_root: {args.out_root}")
    print(f"[info] images dir (empty by design): {img_dir}")
    print(f"[info] annotations: {ann_path}")
    print(f"[info] frequencies: {freq_path}")
    print(f"[info] N={args.N} lang={args.lang} wordlist={args.wordlist}")

    words = top_n_list(args.lang, args.N, wordlist=args.wordlist)
    freqs = [word_frequency(w, args.lang) for w in words]
    total = sum(freqs)
    if total == 0:
        raise ValueError("Sum of frequencies is zero; something is wrong with wordfreq results.")
    rel_freqs = [f / total for f in freqs]

    coco = build_coco_json(words, freqs, rel_freqs, args=args)

    with open(ann_path, "w", encoding="utf-8") as f:
        json.dump(coco, f, ensure_ascii=False)

    if args.write_csv:
        write_csv(csv_path, words, freqs, rel_freqs)
        print(f"[info] wrote csv: {csv_path}")

    # Save raw freqs as .npy (float32 for compactness)
    np.save(freq_path, np.asarray(freqs, dtype=np.float32))

    print(f"[done] images entries: {len(coco['images']):,}")
    print(f"[done] annotations entries: {len(coco['annotations']):,}")
    print(f"[done] saved frequencies: {freq_path} (shape=({len(freqs)},), dtype=float32)")


if __name__ == "__main__":
    main()