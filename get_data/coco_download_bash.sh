#!/bin/bash
#SBATCH --export=NONE
#SBATCH --job-name=download_coco_train_only
#SBATCH --output=logs/download_coco_train_only_%j.out
#SBATCH --error=logs/download_coco_train_only_%j.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=mit_normal

mkdir -p logs

module load miniforge
CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate GPUenv

# -------------------- user config --------------------
# Final default layout:
#   /home/kirilb/orcd/scratch/coco/train2017
#   /home/kirilb/orcd/scratch/coco/annotations/captions_train2017.json
#   /home/kirilb/orcd/scratch/coco/annotations/instances_train2017.json
#
# Existing val2017/test2017 folders and validation/test annotation JSONs are
# removed by default, so plain recursive image scans under OUT_DIR see train only.
OUT_DIR="/home/kirilb/orcd/scratch/coco_check"

# auto -> aria2c if available, then wget, then curl, then python
TOOL="auto"

# Set to 1 to force re-extraction even if train targets already exist.
FORCE=0

# Delete non-train COCO artifacts if they exist.
REMOVE_VAL=1
REMOVE_TEST=1
REMOVE_VAL_ANNOTATIONS=1

# Set to 1 if you want to keep zip archives under OUT_DIR/zips.
# Default 0 keeps disk usage lower after successful extraction.
KEEP_ZIPS=0

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

python "$SCRIPT_DIR/coco_download_train_only.py" \
  --out_dir "$OUT_DIR" \
  --tool "$TOOL" \
  --force "$FORCE" \
  --remove_val "$REMOVE_VAL" \
  --remove_test "$REMOVE_TEST" \
  --remove_val_annotations "$REMOVE_VAL_ANNOTATIONS" \
  --keep_zips "$KEEP_ZIPS"
