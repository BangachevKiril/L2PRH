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
conda activate GPUenv

# -------------------- user config --------------------
OUT_DIR="/home/kirilb/orcd/pool"   # will create $OUT_DIR/coco2017/
INCLUDE_TEST=0
INCLUDE_TEST_INFO=0

# Choose a download tool:
# - auto (prefers aria2c if installed)
# - aria2c / wget / curl / python
TOOL="auto"

python coco_download.py \
  --out_dir "$OUT_DIR" \
  --include_test "$INCLUDE_TEST" \
  --include_test_info "$INCLUDE_TEST_INFO" \
  --tool "$TOOL" \
  --force 0
