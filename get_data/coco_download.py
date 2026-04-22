#!/usr/bin/env python3
"""
download_coco.py

Downloads MS-COCO 2017 images + annotations directly under --out_dir.

Layout under --out_dir:
  <out_dir>/
    zips/
      train2017.zip
      val2017.zip
      annotations_trainval2017.zip
      ...
    train2017/
    val2017/
    test2017/                  (optional)
    annotations/
      captions_train2017.json
      captions_val2017.json
      instances_train2017.json
      instances_val2017.json
      image_info_test2017.json (optional)

Notes:
- Extracted folders are created directly under --out_dir.
- Archives are stored under --out_dir/zips/.
- Skips downloads if the archive already exists.
- Skips extraction only if the expected extracted targets already exist,
  unless --force=1.
"""

import argparse
import os
import shutil
import subprocess
import zipfile
from typing import List, Tuple

COCO_BASE = "http://images.cocodataset.org"


def which(cmd: str) -> str:
    return shutil.which(cmd) or ""


def run(cmd: List[str]) -> None:
    print(" ".join(cmd), flush=True)
    subprocess.check_call(cmd)


def download_file(url: str, dst: str, tool: str = "auto") -> None:
    os.makedirs(os.path.dirname(dst), exist_ok=True)

    if os.path.exists(dst) and os.path.getsize(dst) > 0:
        print(f"[skip] already downloaded: {dst}")
        return

    if tool == "auto":
        if which("aria2c"):
            tool = "aria2c"
        elif which("wget"):
            tool = "wget"
        elif which("curl"):
            tool = "curl"
        else:
            tool = "python"

    if tool == "aria2c":
        run([
            "aria2c", "-c", "-x", "16", "-s", "16", "-k", "1M",
            "-o", os.path.basename(dst), "-d", os.path.dirname(dst), url
        ])
    elif tool == "wget":
        run(["wget", "-c", "-O", dst, url])
    elif tool == "curl":
        run(["curl", "-L", "-C", "-", "-o", dst, url])
    elif tool == "python":
        import urllib.request
        print(f"[python] downloading {url} -> {dst}")
        with urllib.request.urlopen(url) as r, open(dst, "wb") as f:
            shutil.copyfileobj(r, f)
    else:
        raise ValueError(f"Unknown download tool: {tool}")


def extract_zip(zip_path: str, out_dir: str, expected_paths: List[str], force: bool = False) -> None:
    os.makedirs(out_dir, exist_ok=True)
    abs_expected = [os.path.join(out_dir, rel) for rel in expected_paths]

    if not force and all(os.path.exists(p) for p in abs_expected):
        print(f"[skip] already extracted: {zip_path}")
        for p in abs_expected:
            print(f"       found: {p}")
        return

    print(f"[extract] {zip_path} -> {out_dir}")
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(out_dir)


def coco_urls(include_test: bool, include_test_info: bool) -> List[Tuple[str, str]]:
    items = [
        (f"{COCO_BASE}/zips/train2017.zip", "train2017.zip"),
        (f"{COCO_BASE}/zips/val2017.zip", "val2017.zip"),
        (f"{COCO_BASE}/annotations/annotations_trainval2017.zip", "annotations_trainval2017.zip"),
    ]
    if include_test:
        items.append((f"{COCO_BASE}/zips/test2017.zip", "test2017.zip"))
    if include_test_info:
        items.append((f"{COCO_BASE}/annotations/image_info_test2017.zip", "image_info_test2017.zip"))
    return items


def main() -> None:
    p = argparse.ArgumentParser("Download and extract MS-COCO 2017 images + annotations.")
    p.add_argument(
        "--out_dir",
        type=str,
        required=True,
        help="Target COCO folder. Extracted folders are created directly here.",
    )
    p.add_argument("--include_test", type=int, default=0, help="If 1, download/extract test2017.zip")
    p.add_argument("--include_test_info", type=int, default=0, help="If 1, download/extract image_info_test2017.zip")
    p.add_argument("--tool", type=str, default="auto", choices=["auto", "aria2c", "wget", "curl", "python"])
    p.add_argument("--force", type=int, default=0, help="If 1, re-extract even if targets already exist.")
    args = p.parse_args()

    root = os.path.abspath(args.out_dir)
    os.makedirs(root, exist_ok=True)

    zips_dir = os.path.join(root, "zips")
    os.makedirs(zips_dir, exist_ok=True)

    urls = coco_urls(bool(args.include_test), bool(args.include_test_info))

    for url, fname in urls:
        dst = os.path.join(zips_dir, fname)
        download_file(url, dst, tool=args.tool)

    extract_zip(
        os.path.join(zips_dir, "train2017.zip"),
        root,
        expected_paths=["train2017"],
        force=bool(args.force),
    )
    extract_zip(
        os.path.join(zips_dir, "val2017.zip"),
        root,
        expected_paths=["val2017"],
        force=bool(args.force),
    )

    if args.include_test:
        extract_zip(
            os.path.join(zips_dir, "test2017.zip"),
            root,
            expected_paths=["test2017"],
            force=bool(args.force),
        )

    extract_zip(
        os.path.join(zips_dir, "annotations_trainval2017.zip"),
        root,
        expected_paths=[
            os.path.join("annotations", "captions_train2017.json"),
            os.path.join("annotations", "captions_val2017.json"),
            os.path.join("annotations", "instances_train2017.json"),
            os.path.join("annotations", "instances_val2017.json"),
        ],
        force=bool(args.force),
    )

    if args.include_test_info:
        extract_zip(
            os.path.join(zips_dir, "image_info_test2017.zip"),
            root,
            expected_paths=[os.path.join("annotations", "image_info_test2017.json")],
            force=bool(args.force),
        )

    print("\nDone ✅")
    print(f"COCO root: {root}")
    print("Expected extracted locations:")
    print(f"  - {os.path.join(root, 'train2017')}")
    print(f"  - {os.path.join(root, 'val2017')}")
    print(f"  - {os.path.join(root, 'annotations')}")
    if args.include_test:
        print(f"  - {os.path.join(root, 'test2017')}")
    print("Zip archives:")
    print(f"  - {zips_dir}")


if __name__ == "__main__":
    main()