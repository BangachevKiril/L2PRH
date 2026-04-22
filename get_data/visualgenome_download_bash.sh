#!/bin/bash
#SBATCH --export=NONE
#SBATCH --job-name=download_vg
#SBATCH --output=logs/download_vg_%j.out
#SBATCH --error=logs/download_vg_%j.err
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --partition=mit_normal

mkdir -p logs

module load miniforge
CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate L2PRHenv

# -------------------- user config --------------------
PY_SCRIPT="visualgenome_download.py"

# This will create:
#   $OUT_DIR/annotations/captions_train2017.json
#   $OUT_DIR/images/train2017/*.jpg
OUT_DIR="/home/kirilb/orcd/scratch/visual_genome"

# Overwrite existing downloads/extractions/output JSON (0/1)
OVERWRITE=0

# Optional limits (set to empty "" for no limit)
MAX_IMAGES=""               # e.g. 100000
MAX_CAPTIONS_PER_IMAGE="20"   # e.g. 5

VERBOSE=1

# -------------------- run --------------------
ARGS=( --out_root "$OUT_DIR" )

if [[ "$OVERWRITE" == "1" ]]; then
  ARGS+=( --overwrite )
fi

if [[ "$VERBOSE" == "1" ]]; then
  ARGS+=( --verbose )
fi

if [[ -n "$MAX_IMAGES" ]]; then
  ARGS+=( --max_images "$MAX_IMAGES" )
fi

if [[ -n "$MAX_CAPTIONS_PER_IMAGE" ]]; then
  ARGS+=( --max_captions_per_image "$MAX_CAPTIONS_PER_IMAGE" )
fi

python "$PY_SCRIPT" "${ARGS[@]}"