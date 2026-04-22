#!/bin/bash
#SBATCH --export=NONE
#SBATCH --job-name=download_coco
#SBATCH --output=logs/download_coco_%j.out
#SBATCH --error=logs/download_coco_%j.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=mit_normal

mkdir -p logs

module load miniforge
CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate L2PRHenv

# -------------------- user config --------------------
# This is the actual COCO folder.
# After extraction you will have:
#   /home/kirilb/orcd/scratch/coco/train2017
#   /home/kirilb/orcd/scratch/coco/val2017
#   /home/kirilb/orcd/scratch/coco/annotations
# and optionally:
#   /home/kirilb/orcd/scratch/coco/test2017
OUT_DIR="/home/kirilb/orcd/scratch/coco"

INCLUDE_TEST=0
INCLUDE_TEST_INFO=0

# Choose download tool:
#   auto  -> prefers aria2c if installed, then wget, curl, python
#   aria2c / wget / curl / python
TOOL="auto"

python coco_download.py \
  --out_dir "$OUT_DIR" \
  --include_test "$INCLUDE_TEST" \
  --include_test_info "$INCLUDE_TEST_INFO" \
  --tool "$TOOL" \
  --force 0
