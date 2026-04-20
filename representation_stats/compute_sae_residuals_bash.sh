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

SCRIPT="compute_sae_residuals.py"
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

echo "Running residual computation for NAME=$NAME on DEVICE=$DEVICE"
python "$SCRIPT" \
  --name "$NAME" \
  --chunk_size "$CHUNK_SIZE" \
  --device "$DEVICE" \
  --dtype "$DTYPE" \
  --use_normalized "$USE_NORMALIZED" \
  --verbose 1