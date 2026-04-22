#!/usr/bin/env python3
import json
from pathlib import Path

OUT_ROOT = Path("/home/kirilb/orcd/scratch/visual_genome/")  # <-- change
img_dir = OUT_ROOT / "images" / "train2017"

# load VG meta
ex_meta = OUT_ROOT / "_extracted" / "meta"
image_data_path = next(ex_meta.rglob("image_data.json"))
with open(image_data_path, "r") as f:
    image_data = json.load(f)

valid = {f"{int(item['image_id'])}.jpg" for item in image_data}

on_disk = {p.name for p in img_dir.glob("*.jpg")}

extras = sorted(on_disk - valid)
missing = sorted(valid - on_disk)

print("JPGs on disk:", len(on_disk))
print("IDs in image_data.json:", len(valid))
print("Extras on disk (no meta):", len(extras))
print("Missing on disk (in meta but not found):", len(missing))

print("\nFirst 50 extras:")
for x in extras[:50]:
    print("  ", x)

print("\nFirst 50 missing:")
for x in missing[:50]:
    print("  ", x)