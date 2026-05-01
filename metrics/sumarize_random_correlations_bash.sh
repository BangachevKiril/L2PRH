#!/bin/bash
#SBATCH --export=NONE
#SBATCH --job-name=sparse_corr_summary
#SBATCH --output=logs/sparse_corr_summary_%A_%a.out
#SBATCH --error=logs/sparse_corr_summary_%A_%a.err
#SBATCH --time=02:59:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --partition=mit_normal
#SBATCH --array=0-11

mkdir -p logs

# =========================
# Environment
# =========================
module load miniforge
CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate GPUenv

# =========================
# Sweep axes
# =========================
DS_LIST=(8192)
ND=${#DS_LIST[@]}

DATASETS=("coco" "visual_genome" "cc3m")
N_DATASETS=${#DATASETS[@]}

SPARSITY_PATTERNS=("kvar" "k_32" "k_64" "k_128")
N_SPARSITY=${#SPARSITY_PATTERNS[@]}

PY_SCRIPT="SAE_rand_correlation_baselines.py"
BASE_ROOT="/home/kirilb/orcd/pool/PRH_data"
SEEDS=(0 1 2 3 4 5 6 7 8 9)
STRICT=1

GRID_ID=${SLURM_ARRAY_TASK_ID:-0}
TOTAL_JOBS=$((N_DATASETS * ND * N_SPARSITY))

if (( GRID_ID < 0 || GRID_ID >= TOTAL_JOBS )); then
  echo "ERROR: SLURM_ARRAY_TASK_ID=$GRID_ID out of range [0, $((TOTAL_JOBS-1))]"
  echo "Set: #SBATCH --array=0-$((TOTAL_JOBS-1))"
  exit 1
fi

# =========================
# Decode GRID_ID
# Order:
#   dataset -> D -> sparsity_pattern
# =========================
SPARSITY_IDX=$((GRID_ID % N_SPARSITY))
TMP=$((GRID_ID / N_SPARSITY))

D_IDX=$((TMP % ND))
TMP=$((TMP / ND))

DATASET_IDX=$((TMP % N_DATASETS))

DATASET="${DATASETS[$DATASET_IDX]}"
D="${DS_LIST[$D_IDX]}"
SPARSITY_PATTERN="${SPARSITY_PATTERNS[$SPARSITY_IDX]}"

echo "========================================"
echo "GRID_ID=$GRID_ID / $TOTAL_JOBS"
echo "DATASET=$DATASET"
echo "D=$D"
echo "SPARSITY_PATTERN=$SPARSITY_PATTERN"
echo "BASE_ROOT=$BASE_ROOT"
echo "SEEDS=${SEEDS[*]}"
echo "STRICT=$STRICT"
echo "========================================"

CMD=(
  python -u "$PY_SCRIPT"
  --dataset "$DATASET"
  --d "$D"
  --sparsity_pattern "$SPARSITY_PATTERN"
  --base_root "$BASE_ROOT"
  --seeds "${SEEDS[@]}"
  --strict "$STRICT"
)

echo "Running: ${CMD[*]}"
"${CMD[@]}"