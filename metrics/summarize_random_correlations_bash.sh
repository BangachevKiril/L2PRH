#!/bin/bash
#SBATCH --export=NONE
#SBATCH --job-name=sae_residuals
#SBATCH --output=logs/sae_residuals_%A_%a.out
#SBATCH --error=logs/sae_residuals_%A_%a.err
#SBATCH --time=00:59:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:1
#SBATCH --array=0-2

mkdir -p logs

module load miniforge
CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate GPUenv

NAMES=(
  "coco"
  "visual_genome"
  "cc3m"
)

# Select the root here. It must contain folders like:
#   embedded_coco/
#   topk_sae_coco/
# You can also override this at submit time with:
#   sbatch compute_sae_residuals_base_root_bash.sh /path/to/PRH_data
BASE_ROOT="/home/kirilb/orcd/scratch/PRH_data"
if [ $# -ge 1 ] && [ -n "$1" ]; then
  BASE_ROOT="$1"
fi

SCRIPT="summarize_random_correlations.py"
CHUNK_SIZE=512
DTYPE="float32"
DEVICE="cuda"
USE_NORMALIZED=1

IDX=${SLURM_ARRAY_TASK_ID}
if [ "$IDX" -ge "${#NAMES[@]}" ]; then
  echo "Array index $IDX out of range for NAMES of length ${#NAMES[@]}"
  exit 0
fi

NAME="${NAMES[$IDX]}"

if [ ! -d "$BASE_ROOT" ]; then
  echo "ERROR: BASE_ROOT does not exist: $BASE_ROOT"
  exit 1
fi

if [ ! -f "$SCRIPT" ]; then
  echo "ERROR: Python script does not exist: $SCRIPT"
  exit 1
fi

echo "Running residual computation"
echo "  NAME           = $NAME"
echo "  BASE_ROOT      = $BASE_ROOT"
echo "  SCRIPT         = $SCRIPT"
echo "  DEVICE         = $DEVICE"
echo "  USE_NORMALIZED = $USE_NORMALIZED"

python "$SCRIPT" \
  --name "$NAME" \
  --base_root "$BASE_ROOT" \
  --chunk_size "$CHUNK_SIZE" \
  --device "$DEVICE" \
  --dtype "$DTYPE" \
  --use_normalized "$USE_NORMALIZED" \
  --verbose 1
