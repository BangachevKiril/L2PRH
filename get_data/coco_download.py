#!/usr/bin/env python3
"""
download_coco2017.py

Downloads MS-COCO 2017 images + annotations into a clean directory layout.

Default layout (under --out_dir):
  coco2017/
    zips/                       (downloaded archives)
    train2017/                  (extracted images)
    val2017/
    test2017/                   (optional)
    annotations/                (extracted JSONs)
      captions_train2017.json
      captions_val2017.json
      instances_train2017.json
      instances_val2017.json
      ...

Sources (official):
  http://images.cocodataset.org/zips/train2017.zip
  http://images.cocodataset.org/zips/val2017.zip
  http://images.cocodataset.org/zips/test2017.zip
  http://images.cocodataset.org/annotations/annotations_trainval2017.zip
  http://images.cocodataset.org/annotations/image_info_test2017.zip (optional)

Notes:
- Uses aria2c if available (fast, resumable). Falls back to wget or curl.
- Skips downloads/extraction if targets already exist, unless --force.
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

    # Pick tool
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
        # -c resume, -x/-s connections, -k chunk size
        run(["aria2c", "-c", "-x", "16", "-s", "16", "-k", "1M", "-o", os.path.basename(dst), "-d", os.path.dirname(dst), url])
    elif tool == "wget":
        run(["wget", "-c", "-O", dst, url])
    elif tool == "curl":
        run(["curl", "-L", "-C", "-", "-o", dst, url])
    elif tool == "python":
        # Minimal fallback
        import urllib.request
        print(f"[python] downloading {url} -> {dst}")
        with urllib.request.urlopen(url) as r, open(dst, "wb") as f:
            shutil.copyfileobj(r, f)
    else:
        raise ValueError(f"Unknown download tool: {tool}")


def extract_zip(zip_path: str, out_dir: str, force: bool = False) -> None:
    os.makedirs(out_dir, exist_ok=True)

    # Heuristic: if out_dir is non-empty, assume extracted
    if not force and any(os.scandir(out_dir)):
        print(f"[skip] already extracted (non-empty): {out_dir}")
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


def main():
    p = argparse.ArgumentParser("Download and extract MS-COCO 2017 images + annotations.")
    p.add_argument("--out_dir", type=str, required=True, help="Target root dir (will create coco2017/ inside).")
    p.add_argument("--include_test", type=int, default=0, help="If 1, download/extract test2017.zip")
    p.add_argument("--include_test_info", type=int, default=0, help="If 1, download/extract image_info_test2017.zip")
    p.add_argument("--tool", type=str, default="auto", choices=["auto", "aria2c", "wget", "curl", "python"])
    p.add_argument("--force", type=int, default=0, help="If 1, re-extract even if folders non-empty.")
    args = p.parse_args()

    root = os.path.join(args.out_dir, "")
    zips_dir = os.path.join(root, "zips")
    os.makedirs(zips_dir, exist_ok=True)

    urls = coco_urls(bool(args.include_test), bool(args.include_test_info))

    # 1) download
    for url, fname in urls:
        dst = os.path.join(zips_dir, fname)
        download_file(url, dst, tool=args.tool)

    # 2) extract images
    extract_zip(os.path.join(zips_dir, "train2017.zip"), root, force=bool(args.force))
    extract_zip(os.path.join(zips_dir, "val2017.zip"), root, force=bool(args.force))
    if args.include_test:
        extract_zip(os.path.join(zips_dir, "test2017.zip"), root, force=bool(args.force))

    # 3) extract annotations into root, then normalize to root/annotations
    ann_zip = os.path.join(zips_dir, "annotations_trainval2017.zip")
    extract_zip(ann_zip, root, force=bool(args.force))

    # COCO zip extracts into "annotations/" already, but be robust:
    ann_src = os.path.join(root, "annotations")
    if not os.path.isdir(ann_src):
        # Some zips could extract differently; try to find it
        for name in os.listdir(root):
            if name.lower() == "annotations" and os.path.isdir(os.path.join(root, name)):
                ann_src = os.path.join(root, name)
                break

    if args.include_test_info:
        test_info_zip = os.path.join(zips_dir, "image_info_test2017.zip")
        extract_zip(test_info_zip, root, force=bool(args.force))
        # That zip usually drops JSON in annotations/ too, so no extra handling.

    print("\nDone ✅")
    print(f"COCO root: {root}")
    print("Key files you probably want:")
    print(f"  - {os.path.join(root, 'annotations', 'captions_train2017.json')}")
    print(f"  - {os.path.join(root, 'annotations', 'captions_val2017.json')}")
    print(f"Images:")
    print(f"  - {os.path.join(root, 'train2017')}")
    print(f"  - {os.path.join(root, 'val2017')}")
    if args.include_test:
        print(f"  - {os.path.join(root, 'test2017')}")


if __name__ == "__main__":
    main()
