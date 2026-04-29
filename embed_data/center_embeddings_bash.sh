#!/bin/bash
#SBATCH --export=NONE
#SBATCH --job-name=norm_embeds
#SBATCH --output=logs/norm_embeds_%A_%a.out
#SBATCH --error=logs/norm_embeds_%A_%a.err
#SBATCH --time=05:59:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --partition=mit_normal      # <-- change if you want a different CPU partition
#SBATCH --array=0-4                 # <-- set after you fill DIRS below

mkdir -p logs

module load miniforge
CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate coco_text_embed

# --------------------- User config ---------------------
PY_SCRIPT="center_embeddings.py"

# List the root folders you want to process (one per array task)
DIRS=(
  "/home/kirilb/orcd/scratch/PRH_data/embedded_coco"
  "/home/kirilb/orcd/scratch/PRH_data/embedded_cc3m"
  "/home/kirilb/orcd/scratch/PRH_data/embedded_visual_genome"
  "/home/kirilb/orcd/scratch/PRH_data/embedded_words"
)

# Normalizer args
EPS="1e-12"
CHUNK_ROWS="100000"
OVERWRITE=0     # 1 to overwrite existing *_normalized.npy
VERBOSE=1       # 1 for verbose

# --------------------- Indexing ---------------------
IDX=${SLURM_ARRAY_TASK_ID}
ROOT_DIR="${DIRS[$IDX]}"

# --------------------- Build args ---------------------
ARGS=(
  "$ROOT_DIR"
  "--eps" "$EPS"
  "--chunk_rows" "$CHUNK_ROWS"
)

if [ "$OVERWRITE" -eq 1 ]; then
  ARGS+=("--overwrite")
fi
if [ "$VERBOSE" -eq 1 ]; then
  ARGS+=("--verbose")
fi

echo "Job: $SLURM_JOB_ID  ArrayTask: $SLURM_ARRAY_TASK_ID"
echo "Python: $PY_SCRIPT"
echo "Root dir: $ROOT_DIR"
echo "Args: ${ARGS[*]}"

python "$PY_SCRIPT" "${ARGS[@]}"
