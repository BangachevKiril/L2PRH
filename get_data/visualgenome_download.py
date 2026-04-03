#!/usr/bin/env python3
"""
download_visual_genome_as_coco_captions.py

Download Visual Genome (images + image_data + region_descriptions) and write COCO-format
captions JSON, with this folder structure:

OUT_ROOT/
  images/train2017/<image_id>.jpg
  annotations/captions_train2017.json

Captions are taken from Visual Genome *region descriptions*:
- Each region "phrase" becomes one COCO caption annotation for that image.

Robust skipping (what you asked for):
- Skip any image file that is empty (0 bytes).
- Skip any image that has NO non-empty captions (after cleaning).
- Also skip malformed metadata / region entries instead of crashing.

Notes:
- Visual Genome has no COCO-style train/val split here; we put everything into train2017.
- Downloads are large (~15GB images + ~700MB JSON) depending on mirror/version.
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
import urllib.request
import zipfile
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


# -------------------------
# Default URLs (Stanford images + UW mirror metadata)
# -------------------------
DEFAULT_IMAGES_ZIP_1 = "https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip"
DEFAULT_IMAGES_ZIP_2 = "https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip"
DEFAULT_IMAGE_DATA_ZIP = "https://homes.cs.washington.edu/~ranjay/visualgenome/data/dataset/image_data.json.zip"
DEFAULT_REGION_DESC_ZIP = "https://homes.cs.washington.edu/~ranjay/visualgenome/data/dataset/region_descriptions.json.zip"


# -------------------------
# Utilities
# -------------------------
def _mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _human_bytes(n: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    x = float(n)
    for u in units:
        if x < 1024.0:
            return f"{x:.2f} {u}"
        x /= 1024.0
    return f"{x:.2f} PB"


def is_nonempty_file(p: Path) -> bool:
    """
    True iff path exists and has size > 0 bytes.
    """
    try:
        return p.is_file() and p.stat().st_size > 0
    except FileNotFoundError:
        return False


def download_url(url: str, out_path: Path, *, overwrite: bool, verbose: bool) -> None:
    """
    Download URL to out_path with a simple progress display.
    """
    if out_path.exists() and not overwrite:
        if verbose:
            print(f"[skip] exists: {out_path}")
        return

    _mkdir(out_path.parent)
    tmp_path = out_path.with_suffix(out_path.suffix + ".partial")

    if verbose:
        print(f"[download] {url}")
        print(f"          -> {out_path}")

    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as r, open(tmp_path, "wb") as f:
        total = r.headers.get("Content-Length")
        total_n = int(total) if total is not None else None

        chunk = 1024 * 1024 * 4  # 4MB
        got = 0
        t0 = time.time()
        while True:
            b = r.read(chunk)
            if not b:
                break
            f.write(b)
            got += len(b)
            if verbose and total_n:
                dt = max(time.time() - t0, 1e-6)
                rate = got / dt
                pct = 100.0 * got / total_n
                print(
                    f"\r          {pct:6.2f}%  {_human_bytes(got)}/{_human_bytes(total_n)}  ({_human_bytes(int(rate))}/s)",
                    end="",
                )
        if verbose and total_n:
            print()

    tmp_path.replace(out_path)


def unzip_to(zip_path: Path, out_dir: Path, *, overwrite: bool, verbose: bool) -> None:
    """
    Extract zip_path into out_dir, preserving internal paths.
    """
    _mkdir(out_dir)
    if verbose:
        print(f"[unzip] {zip_path} -> {out_dir}")

    with zipfile.ZipFile(zip_path, "r") as zf:
        for member in zf.infolist():
            if member.is_dir():
                continue
            dst = out_dir / member.filename
            if dst.exists() and not overwrite:
                continue
            _mkdir(dst.parent)
            with zf.open(member, "r") as src, open(dst, "wb") as out:
                shutil.copyfileobj(src, out)


def find_first_json(root: Path, name: str) -> Path:
    """
    The zips often contain nested folders; find the first match.
    """
    for p in root.rglob(name):
        return p
    raise FileNotFoundError(f"Could not find {name} under {root}")


def move_all_jpgs_skip_empty(src_root: Path, dst_dir: Path, *, overwrite: bool, verbose: bool) -> tuple[int, int]:
    """
    Move all .jpg files recursively from src_root into dst_dir, keeping filenames.
    Skips empty (0-byte) files and counts them.
    Returns (moved_count, skipped_empty_count).
    """
    _mkdir(dst_dir)
    moved = 0
    skipped_empty = 0

    for p in src_root.rglob("*.jpg"):
        # If source is empty, do not move it (and optionally delete it to avoid re-traversal)
        try:
            if p.stat().st_size == 0:
                skipped_empty += 1
                try:
                    p.unlink()
                except Exception:
                    pass
                continue
        except FileNotFoundError:
            skipped_empty += 1
            continue

        dst = dst_dir / p.name
        if dst.exists() and not overwrite:
            # If destination exists but is empty, we can try to replace it with this non-empty file.
            if not is_nonempty_file(dst):
                try:
                    dst.unlink()
                except Exception:
                    pass
            else:
                # Keep existing non-empty destination
                continue

        _mkdir(dst.parent)
        shutil.move(str(p), str(dst))
        moved += 1

        # Post-move sanity: if somehow dst is empty, delete it.
        if not is_nonempty_file(dst):
            skipped_empty += 1
            moved -= 1
            try:
                dst.unlink()
            except Exception:
                pass

        if verbose and moved % 5000 == 0:
            print(f"[images] moved {moved} jpgs... (skipped empty so far: {skipped_empty})")

    return moved, skipped_empty


# -------------------------
# Robust COCO conversion with skipping rules
# -------------------------
def build_coco_captions(
    image_data: list[dict[str, Any]],
    region_desc: list[dict[str, Any]],
    images_dir: Path,
    *,
    max_images: Optional[int],
    max_caps_per_image: Optional[int],
    verbose: bool,
) -> dict[str, Any]:
    """
    Robustly convert Visual Genome metadata + region descriptions into COCO Captions JSON.

    Skipping rules enforced:
      - Skip images whose file is empty (0 bytes) or missing.
      - Skip images that have NO non-empty captions after cleaning.

    Handles common VG schema variants:
      - image id key: "id" vs "image_id"
      - occasionally nested under "image": {"id": ...} / {"image_id": ...}
      - skips malformed entries instead of crashing
    """

    def _as_int(x: Any) -> Optional[int]:
        try:
            if x is None:
                return None
            return int(x)
        except Exception:
            return None

    def _get_image_id(d: dict[str, Any]) -> Optional[int]:
        # Most common
        if "image_id" in d:
            v = _as_int(d.get("image_id"))
            if v is not None:
                return v
        if "id" in d:
            v = _as_int(d.get("id"))
            if v is not None:
                return v

        # Sometimes nested
        img = d.get("image")
        if isinstance(img, dict):
            if "image_id" in img:
                v = _as_int(img.get("image_id"))
                if v is not None:
                    return v
            if "id" in img:
                v = _as_int(img.get("id"))
                if v is not None:
                    return v

        return None

    def _clean_caption(s: Any) -> Optional[str]:
        if not isinstance(s, str):
            return None
        t = " ".join(s.strip().split())
        return t if t else None

    # Map image_id -> (width, height, file_name)
    meta: Dict[int, Tuple[int, int, str]] = {}
    bad_meta = 0

    for item in image_data:
        if not isinstance(item, dict):
            bad_meta += 1
            continue

        iid = _get_image_id(item)
        if iid is None:
            bad_meta += 1
            continue

        w = _as_int(item.get("width")) or 0
        h = _as_int(item.get("height")) or 0
        meta[iid] = (w, h, f"{iid}.jpg")

    # Build per-image captions (only keep images with >= 1 non-empty caption)
    per_image_caps: Dict[int, list[str]] = {}
    bad_regions_top = 0
    bad_regions_noid = 0
    bad_regions_noregions = 0
    empty_caps = 0

    for entry in region_desc:
        if not isinstance(entry, dict):
            bad_regions_top += 1
            continue

        iid = _get_image_id(entry)
        if iid is None:
            bad_regions_noid += 1
            continue

        regions = entry.get("regions")
        if not isinstance(regions, list):
            bad_regions_noregions += 1
            continue

        caps: list[str] = []
        for r in regions:
            if not isinstance(r, dict):
                continue
            c = _clean_caption(r.get("phrase"))
            if c is not None:
                caps.append(c)

        if not caps:
            empty_caps += 1
            continue

        # Deterministic cap
        if max_caps_per_image is not None:
            caps = caps[:max_caps_per_image]

        if iid in per_image_caps:
            per_image_caps[iid].extend(caps)
            if max_caps_per_image is not None:
                per_image_caps[iid] = per_image_caps[iid][:max_caps_per_image]
        else:
            per_image_caps[iid] = caps

    # Emit COCO: enforce "non-empty image file" AND "has non-empty caption"
    coco_images: list[dict[str, Any]] = []
    coco_anns: list[dict[str, Any]] = []
    ann_id = 1

    image_ids = sorted(per_image_caps.keys())
    if max_images is not None:
        image_ids = image_ids[:max_images]

    missing_meta = 0
    missing_or_empty_file = 0
    skipped_no_caps = 0  # should be 0 due to per_image_caps filter, but keep for clarity

    for iid in image_ids:
        m = meta.get(iid)
        if m is None:
            missing_meta += 1
            continue

        caps = per_image_caps.get(iid)
        if not caps:
            skipped_no_caps += 1
            continue

        w, h, fname = m
        fpath = images_dir / fname
        if not is_nonempty_file(fpath):
            missing_or_empty_file += 1
            continue

        coco_images.append({"id": iid, "file_name": fname, "width": w, "height": h})
        for c in caps:
            coco_anns.append({"id": ann_id, "image_id": iid, "caption": c})
            ann_id += 1

        if verbose and len(coco_images) % 5000 == 0:
            print(f"[coco] images: {len(coco_images)}  anns: {len(coco_anns)}")

    if verbose:
        print(
            "[vg->coco] meta:",
            f"kept={len(meta)} bad_rows={bad_meta}",
            "| regions:",
            f"kept_images_with_caps={len(per_image_caps)} bad_top={bad_regions_top} bad_noid={bad_regions_noid} bad_noregions={bad_regions_noregions} empty_caps_entries={empty_caps}",
            "| filtered:",
            f"missing_meta={missing_meta} missing_or_empty_file={missing_or_empty_file} skipped_no_caps={skipped_no_caps}",
        )

    coco = {
        "info": {
            "description": "Visual Genome converted to COCO Captions format (using region phrases as captions).",
            "version": "vg_to_coco_captions_v3_skip_empty_and_nocaps",
        },
        "licenses": [],
        "images": coco_images,
        "annotations": coco_anns,
    }
    return coco


# -------------------------
# Main
# -------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="Download Visual Genome and convert region phrases to COCO captions JSON.")
    ap.add_argument("--out_root", type=str, required=True, help="Output root folder.")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing downloads/extractions/output.")
    ap.add_argument("--verbose", action="store_true")

    ap.add_argument("--images_zip_1", type=str, default=DEFAULT_IMAGES_ZIP_1)
    ap.add_argument("--images_zip_2", type=str, default=DEFAULT_IMAGES_ZIP_2)
    ap.add_argument("--image_data_zip", type=str, default=DEFAULT_IMAGE_DATA_ZIP)
    ap.add_argument("--region_desc_zip", type=str, default=DEFAULT_REGION_DESC_ZIP)

    ap.add_argument("--max_images", type=int, default=None, help="Limit number of images included in output JSON.")
    ap.add_argument("--max_captions_per_image", type=int, default=None, help="Cap captions per image (deterministic).")

    args = ap.parse_args()

    out_root = Path(args.out_root)
    ann_dir = out_root / "annotations"
    img_dir = out_root / "images" / "train2017"
    _mkdir(ann_dir)
    _mkdir(img_dir)

    dl_dir = out_root / "_downloads"
    ex_dir = out_root / "_extracted"
    _mkdir(dl_dir)
    _mkdir(ex_dir)

    # 1) Download zips
    z_img1 = dl_dir / "images.zip"
    z_img2 = dl_dir / "images2.zip"
    z_imgdata = dl_dir / "image_data.json.zip"
    z_regdesc = dl_dir / "region_descriptions.json.zip"

    download_url(args.images_zip_1, z_img1, overwrite=args.overwrite, verbose=args.verbose)
    download_url(args.images_zip_2, z_img2, overwrite=args.overwrite, verbose=args.verbose)
    download_url(args.image_data_zip, z_imgdata, overwrite=args.overwrite, verbose=args.verbose)
    download_url(args.region_desc_zip, z_regdesc, overwrite=args.overwrite, verbose=args.verbose)

    # 2) Extract zips
    ex_img1 = ex_dir / "images_1"
    ex_img2 = ex_dir / "images_2"
    ex_meta = ex_dir / "meta"
    unzip_to(z_img1, ex_img1, overwrite=args.overwrite, verbose=args.verbose)
    unzip_to(z_img2, ex_img2, overwrite=args.overwrite, verbose=args.verbose)
    unzip_to(z_imgdata, ex_meta, overwrite=args.overwrite, verbose=args.verbose)
    unzip_to(z_regdesc, ex_meta, overwrite=args.overwrite, verbose=args.verbose)

    # 3) Move JPGs into COCO-like folder, skipping empty files
    moved1, skipped1 = move_all_jpgs_skip_empty(ex_img1, img_dir, overwrite=args.overwrite, verbose=args.verbose)
    moved2, skipped2 = move_all_jpgs_skip_empty(ex_img2, img_dir, overwrite=args.overwrite, verbose=args.verbose)
    if args.verbose:
        print(f"[images] moved total: {moved1 + moved2}  skipped empty total: {skipped1 + skipped2}")

    # 4) Load JSONs
    image_data_path = find_first_json(ex_meta, "image_data.json")
    region_desc_path = find_first_json(ex_meta, "region_descriptions.json")

    if args.verbose:
        print(f"[json] image_data: {image_data_path}")
        print(f"[json] region_descriptions: {region_desc_path}")

    with open(image_data_path, "r", encoding="utf-8") as f:
        image_data = json.load(f)
    with open(region_desc_path, "r", encoding="utf-8") as f:
        region_desc = json.load(f)

    # 5) Build COCO captions (skips empty images and no-caption images)
    coco = build_coco_captions(
        image_data=image_data,
        region_desc=region_desc,
        images_dir=img_dir,
        max_images=args.max_images,
        max_caps_per_image=args.max_captions_per_image,
        verbose=args.verbose,
    )

    out_json = ann_dir / "captions_train2017.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(coco, f)

    print(f"[done] wrote: {out_json}")
    print(f"       images: {len(coco['images'])}")
    print(f"       captions: {len(coco['annotations'])}")
    print(f"       image folder: {img_dir}")


if __name__ == "__main__":
    main()