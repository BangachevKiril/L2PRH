#!/usr/bin/env python3
"""
coco_download_train_only.py

Downloads MS-COCO 2017 train images + train annotations under --out_dir.

Final default layout under --out_dir:
  <out_dir>/
    train2017/
    annotations/
      captions_train2017.json
      instances_train2017.json

By default, this script removes validation/test image folders and validation/test
annotation files if they already exist. It can also remove zip archives after
successful extraction to keep only the usable train data on disk.
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


def str_to_bool01(x: int) -> bool:
    return bool(int(x))


def remove_path(path: str) -> None:
    if os.path.isdir(path):
        print(f"[remove] directory: {path}", flush=True)
        shutil.rmtree(path)
    elif os.path.isfile(path):
        print(f"[remove] file: {path}", flush=True)
        os.remove(path)


def download_file(url: str, dst: str, tool: str = "auto") -> None:
    os.makedirs(os.path.dirname(dst), exist_ok=True)

    if os.path.exists(dst) and os.path.getsize(dst) > 0:
        print(f"[skip] already downloaded: {dst}", flush=True)
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
            "-o", os.path.basename(dst), "-d", os.path.dirname(dst), url,
        ])
    elif tool == "wget":
        run(["wget", "-c", "-O", dst, url])
    elif tool == "curl":
        run(["curl", "-L", "-C", "-", "-o", dst, url])
    elif tool == "python":
        import urllib.request
        print(f"[python] downloading {url} -> {dst}", flush=True)
        with urllib.request.urlopen(url) as r, open(dst, "wb") as f:
            shutil.copyfileobj(r, f)
    else:
        raise ValueError(f"Unknown download tool: {tool}")


def extract_zip(zip_path: str, out_dir: str, expected_paths: List[str], force: bool = False) -> None:
    os.makedirs(out_dir, exist_ok=True)
    abs_expected = [os.path.join(out_dir, rel) for rel in expected_paths]

    if not force and all(os.path.exists(p) for p in abs_expected):
        print(f"[skip] already extracted: {zip_path}", flush=True)
        for p in abs_expected:
            print(f"       found: {p}", flush=True)
        return

    print(f"[extract] {zip_path} -> {out_dir}", flush=True)
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(out_dir)


def train_only_urls() -> List[Tuple[str, str]]:
    return [
        (f"{COCO_BASE}/zips/train2017.zip", "train2017.zip"),
        # COCO ships train and val annotations in one archive; we delete the val JSONs after extraction.
        (f"{COCO_BASE}/annotations/annotations_trainval2017.zip", "annotations_trainval2017.zip"),
    ]


def remove_non_train_coco_files(root: str, remove_val_folder: bool, remove_test_folder: bool, remove_val_annotations: bool) -> None:
    if remove_val_folder:
        remove_path(os.path.join(root, "val2017"))
        remove_path(os.path.join(root, "images", "val2017"))

    if remove_test_folder:
        remove_path(os.path.join(root, "test2017"))
        remove_path(os.path.join(root, "images", "test2017"))

    if remove_val_annotations:
        ann = os.path.join(root, "annotations")
        for fname in [
            "captions_val2017.json",
            "instances_val2017.json",
            "person_keypoints_val2017.json",
            "image_info_test2017.json",
            "image_info_test-dev2017.json",
        ]:
            remove_path(os.path.join(ann, fname))


def assert_train_layout(root: str) -> None:
    required = [
        os.path.join(root, "train2017"),
        os.path.join(root, "annotations", "captions_train2017.json"),
        os.path.join(root, "annotations", "instances_train2017.json"),
    ]
    missing = [p for p in required if not os.path.exists(p)]
    if missing:
        raise RuntimeError("Missing expected train-only COCO outputs:\n" + "\n".join(f"  - {p}" for p in missing))


def main() -> None:
    p = argparse.ArgumentParser("Download and extract train-only MS-COCO 2017.")
    p.add_argument("--out_dir", type=str, required=True, help="Target COCO folder.")
    p.add_argument("--tool", type=str, default="auto", choices=["auto", "aria2c", "wget", "curl", "python"])
    p.add_argument("--force", type=int, default=0, help="If 1, re-extract even if targets already exist.")
    p.add_argument("--remove_val", type=int, default=1, help="If 1, delete val2017 image folders if present.")
    p.add_argument("--remove_test", type=int, default=1, help="If 1, delete test2017 image folders if present.")
    p.add_argument("--remove_val_annotations", type=int, default=1, help="If 1, delete validation/test annotation JSONs after extraction.")
    p.add_argument("--keep_zips", type=int, default=0, help="If 1, keep downloaded zip archives under <out_dir>/zips.")
    args = p.parse_args()

    root = os.path.abspath(args.out_dir)
    os.makedirs(root, exist_ok=True)

    zips_dir = os.path.join(root, "zips")
    os.makedirs(zips_dir, exist_ok=True)

    for url, fname in train_only_urls():
        download_file(url, os.path.join(zips_dir, fname), tool=args.tool)

    extract_zip(
        os.path.join(zips_dir, "train2017.zip"),
        root,
        expected_paths=["train2017"],
        force=str_to_bool01(args.force),
    )

    extract_zip(
        os.path.join(zips_dir, "annotations_trainval2017.zip"),
        root,
        expected_paths=[
            os.path.join("annotations", "captions_train2017.json"),
            os.path.join("annotations", "instances_train2017.json"),
        ],
        force=str_to_bool01(args.force),
    )

    remove_non_train_coco_files(
        root=root,
        remove_val_folder=str_to_bool01(args.remove_val),
        remove_test_folder=str_to_bool01(args.remove_test),
        remove_val_annotations=str_to_bool01(args.remove_val_annotations),
    )

    assert_train_layout(root)

    if not str_to_bool01(args.keep_zips):
        remove_path(zips_dir)

    print("\nDone ✅")
    print(f"COCO train-only root: {root}")
    print("Stored extracted data:")
    print(f"  - {os.path.join(root, 'train2017')}")
    print(f"  - {os.path.join(root, 'annotations', 'captions_train2017.json')}")
    print(f"  - {os.path.join(root, 'annotations', 'instances_train2017.json')}")
    if str_to_bool01(args.keep_zips):
        print("Zip archives kept:")
        print(f"  - {zips_dir}")
    else:
        print("Zip archives removed.")


if __name__ == "__main__":
    main()
