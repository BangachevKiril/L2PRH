#!/usr/bin/env python3
"""
download_cc3m_train_sample_as_coco.py

Sample N random (caption, url) pairs from CC3M train (GCC) and write them
in a COCO-like folder structure:

  OUT_ROOT/
    images/train2017/000000000001.jpg ...
    annotations/captions_train2017.json

Also writes:
  annotations/cc3m_train_sample_manifest.jsonl   # the sampled (caption,url) list
  annotations/cc3m_train_sample_failures.jsonl   # per-url download failures w/ reasons

Notes:
- CC3M train TSV format is typically: caption<TAB>url (no header), but we auto-detect.
- Many URLs will be dead (404/410), blocked (403), or rate-limited (429). We skip failures.
- Images are converted to RGB JPEG for consistency.
"""

import argparse
import concurrent.futures as cf
import hashlib
import json
import os
import random
import sys
import time
from dataclasses import dataclass
from io import BytesIO
from typing import Iterable, List, Optional, Tuple

import requests
from PIL import Image


DEFAULT_TRAIN_TSV_URL = "https://storage.googleapis.com/gcc-data/Train/GCC-training.tsv"


@dataclass
class Row:
    url: str
    caption: str


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--out_root", type=str, required=True)
    p.add_argument("--sample_size", type=int, default=100_000)
    p.add_argument("--seed", type=int, default=12345)

    # TSV source: either local file or streaming URL
    p.add_argument("--tsv_path", type=str, default="", help="Optional local Train_GCC-training.tsv")
    p.add_argument("--tsv_url", type=str, default=DEFAULT_TRAIN_TSV_URL)

    # Download parallelism + networking
    p.add_argument("--num_workers", type=int, default=12)
    p.add_argument("--timeout", type=float, default=30.0)
    p.add_argument("--retries", type=int, default=8)
    p.add_argument("--sleep_base", type=float, default=0.5)
    p.add_argument(
        "--user_agent",
        type=str,
        default="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    )

    # Output naming
    p.add_argument("--split_dir", type=str, default="train2017", help="COCO-style split folder name.")
    p.add_argument("--ann_name", type=str, default="captions_train2017.json", help="COCO captions JSON filename.")

    # Optional: cap how many TSV lines to scan (debug)
    p.add_argument("--max_scan_lines", type=int, default=0)

    return p.parse_args()


def ensure_dirs(out_root: str, split_dir: str):
    img_dir = os.path.join(out_root, "images", split_dir)
    ann_dir = os.path.join(out_root, "annotations")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(ann_dir, exist_ok=True)
    return img_dir, ann_dir


def looks_like_url(s: str) -> bool:
    s = s.strip()
    return s.startswith("http://") or s.startswith("https://")


def parse_tsv_line(line) -> Optional[Row]:
    # Accept either bytes or str
    if isinstance(line, (bytes, bytearray)):
        line = line.decode("utf-8", errors="replace")

    # expected: caption \t url (usually), but be defensive
    parts = line.rstrip("\n").split("\t")
    if len(parts) < 2:
        return None

    a = parts[0].strip()
    b = parts[1].strip()

    # optional header
    if a.lower() == "caption" and b.lower() == "url":
        return None

    if looks_like_url(a) and not looks_like_url(b):
        url, caption = a, b
    elif looks_like_url(b) and not looks_like_url(a):
        url, caption = b, a
    elif looks_like_url(a) and looks_like_url(b):
        url, caption = (b, a) if len(b) >= len(a) else (a, b)
    else:
        return None

    if not url or not caption:
        return None
    return Row(url=url, caption=caption)



def iter_tsv_lines_from_url(url: str, timeout: float, user_agent: str) -> Iterable[str]:
    with requests.get(url, stream=True, timeout=max(timeout, 60.0), headers={"User-Agent": user_agent}) as r:
        r.raise_for_status()
        for line in r.iter_lines():
            if not line:
                continue
            if isinstance(line, (bytes, bytearray)):
                yield line.decode("utf-8", errors="replace")
            else:
                yield str(line)


def iter_tsv_lines_from_file(path: str) -> Iterable[str]:
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.strip():
                yield line


def reservoir_sample_rows(lines: Iterable[str], k: int, seed: int, max_scan_lines: int = 0) -> List[Row]:
    rng = random.Random(seed)
    sample: List[Row] = []
    seen = 0

    for line in lines:
        if max_scan_lines and seen >= max_scan_lines:
            break

        row = parse_tsv_line(line)
        seen += 1
        if row is None:
            continue

        if len(sample) < k:
            sample.append(row)
        else:
            j = rng.randrange(0, seen)
            if j < k:
                sample[j] = row

        if seen % 500_000 == 0:
            print(f"[scan] lines seen: {seen:,}  sample kept: {len(sample):,}", flush=True)

    print(f"[scan] done. total lines seen: {seen:,}  sample kept: {len(sample):,}", flush=True)
    return sample


def stable_int_id(url: str) -> int:
    h = hashlib.md5(url.encode("utf-8")).hexdigest()
    return int(h[:8], 16) & 0x7FFFFFFF


def fetch_image_as_jpeg(
    url: str,
    img_path: str,
    timeout: float,
    retries: int,
    sleep_base: float,
    user_agent: str,
    referer: str = "https://www.google.com/",
) -> Tuple[Optional[Tuple[int, int]], Optional[dict]]:
    headers = {
        "User-Agent": user_agent,
        "Accept": "image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": referer,
    }

    last = None
    for attempt in range(retries):
        try:
            r = requests.get(url, timeout=timeout, headers=headers, allow_redirects=True, stream=True)
            status = r.status_code

            if status != 200:
                last = {"kind": "http_error", "status": status, "detail": r.reason}
                mult = 3.0 if status in (403, 429) else 1.0
                time.sleep(mult * sleep_base * (2 ** attempt) * (1.0 + 0.2 * random.random()))
                continue

            content = r.content
            img = Image.open(BytesIO(content)).convert("RGB")
            w, h = img.size

            tmp_path = img_path + ".tmp"
            img.save(tmp_path, format="JPEG", quality=95, optimize=True)
            os.replace(tmp_path, img_path)
            return (w, h), None

        except requests.exceptions.SSLError as e:
            last = {"kind": "ssl_error", "status": None, "detail": str(e)}
        except requests.exceptions.Timeout as e:
            last = {"kind": "timeout", "status": None, "detail": str(e)}
        except requests.exceptions.RequestException as e:
            last = {"kind": "request_error", "status": None, "detail": str(e)}
        except Exception as e:
            last = {"kind": "decode_or_io_error", "status": None, "detail": str(e)}

        time.sleep(sleep_base * (2 ** attempt) * (1.0 + 0.2 * random.random()))

    return None, last


def build_coco_json(images, annotations):
    return {
        "info": {
            "description": "CC3M Train Sample rendered in COCO captions format",
            "version": "1.0",
            "year": 2026,
            "date_created": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        "licenses": [],
        "type": "captions",
        "images": images,
        "annotations": annotations,
    }


def main():
    args = parse_args()
    random.seed(args.seed)

    img_dir, ann_dir = ensure_dirs(args.out_root, args.split_dir)
    ann_path = os.path.join(ann_dir, args.ann_name)
    fail_log_path = os.path.join(ann_dir, "cc3m_train_sample_failures.jsonl")
    manifest_path = os.path.join(ann_dir, "cc3m_train_sample_manifest.jsonl")

    print(f"[info] out_root: {args.out_root}")
    print(f"[info] sample_size: {args.sample_size:,}")
    print(f"[info] seed: {args.seed}")
    print(f"[info] num_workers: {args.num_workers}")

    # 1) Stream TSV and reservoir-sample
    if args.tsv_path:
        print(f"[info] reading TSV from file: {args.tsv_path}")
        lines = iter_tsv_lines_from_file(args.tsv_path)
    else:
        print(f"[info] streaming TSV from url: {args.tsv_url}")
        lines = iter_tsv_lines_from_url(args.tsv_url, timeout=args.timeout, user_agent=args.user_agent)

    rows = reservoir_sample_rows(lines, k=args.sample_size, seed=args.seed, max_scan_lines=args.max_scan_lines)
    if not rows:
        print("[error] no rows sampled. Check TSV access / parsing.", file=sys.stderr)
        sys.exit(2)

    # Deterministic ordering + sequential COCO ids
    rows_sorted = sorted(rows, key=lambda r: stable_int_id(r.url))
    indexed = list(enumerate(rows_sorted, start=1))

    # Save manifest of chosen URLs/captions (repro-friendly)
    with open(manifest_path, "w", encoding="utf-8") as f:
        for i, row in indexed:
            f.write(json.dumps({"image_id": i, "url": row.url, "caption": row.caption}, ensure_ascii=False) + "\n")
    print(f"[info] wrote manifest: {manifest_path}")

    images = []
    annotations = []

    def worker(i: int, row: Row):
        file_name = f"{i:012d}.jpg"
        img_path = os.path.join(img_dir, file_name)

        # Skip if already exists and looks readable
        if os.path.exists(img_path) and os.path.getsize(img_path) > 0:
            try:
                im = Image.open(img_path)
                w, h = im.size
                return i, row, (w, h), None
            except Exception:
                pass

        wh, err = fetch_image_as_jpeg(
            url=row.url,
            img_path=img_path,
            timeout=args.timeout,
            retries=args.retries,
            sleep_base=args.sleep_base,
            user_agent=args.user_agent,
        )
        return i, row, wh, err

    # 2) Download sampled images
    failures = 0
    kept = 0

    with cf.ThreadPoolExecutor(max_workers=args.num_workers) as ex:
        futures = [ex.submit(worker, i, row) for i, row in indexed]
        for fut in cf.as_completed(futures):
            i, row, wh, err = fut.result()
            if wh is None:
                failures += 1
                with open(fail_log_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps({"image_id": i, "url": row.url, "error": err}, ensure_ascii=False) + "\n")
                continue

            w, h = wh
            kept += 1
            images.append(
                {
                    "id": i,
                    "file_name": f"{i:012d}.jpg",
                    "width": w,
                    "height": h,
                    "license": 0,
                    "flickr_url": "",
                    "coco_url": row.url,  # stash original url
                    "date_captured": "",
                }
            )
            annotations.append({"id": i, "image_id": i, "caption": row.caption})

            if (kept + failures) % 2000 == 0:
                print(f"[dl] done: {kept:,} kept | {failures:,} failed", flush=True)

    images.sort(key=lambda x: x["id"])
    annotations.sort(key=lambda x: x["id"])

    coco = build_coco_json(images, annotations)
    with open(ann_path, "w", encoding="utf-8") as f:
        json.dump(coco, f, ensure_ascii=False)

    print(f"[done] kept: {kept:,}  failed: {failures:,}  total_sampled: {len(rows_sorted):,}")
    print(f"[done] annotations: {ann_path}")
    print(f"[done] failures log: {fail_log_path}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        sys.exit(130)
