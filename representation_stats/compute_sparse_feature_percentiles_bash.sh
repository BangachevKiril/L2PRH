#!/bin/bash
#SBATCH --export=NONE
#SBATCH --job-name=sparse_feat_stats
#SBATCH --output=logs/sparse_feat_stats_%A_%a.out
#SBATCH --error=logs/sparse_feat_stats_%A_%a.err
#SBATCH --time=05:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --partition=mit_normal
#SBATCH --array=0-2   # <-- change to 0-$((${#ROOTS[@]}-1))

mkdir -p logs

# =========================
# Environment
# =========================
module load miniforge
CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate GPUenv

# =========================
# User config
# =========================
ROOTS=(
  "/home/kirilb/orcd/scratch/PRH_data/topk_sae_visual_genome"
  "/home/kirilb/orcd/scratch/PRH_data/topk_sae_coco"
  "/home/kirilb/orcd/scratch/PRH_data/topk_sae_cc3m"
)

NROOTS=${#ROOTS[@]}

GRID_ID=${SLURM_ARRAY_TASK_ID:-0}
if (( GRID_ID < 0 || GRID_ID >= NROOTS )); then
  echo "ERROR: SLURM_ARRAY_TASK_ID=$GRID_ID out of range [0, $((NROOTS-1))]"
  echo "Set: #SBATCH --array=0-$((NROOTS-1))"
  exit 1
fi

ROOT="${ROOTS[$GRID_ID]}"
shift || true

PY_SCRIPT="compute_sparse_feature_percentiles.py"

SKIP_EXISTING="${SKIP_EXISTING:-0}"
QUIET="${QUIET:-0}"

if [[ ! -f "$PY_SCRIPT" ]]; then
  echo "ERROR: Python script not found: $PY_SCRIPT"
  exit 1
fi

if [[ ! -d "$ROOT" ]]; then
  echo "ERROR: Root folder not found: $ROOT"
  exit 1
fi

# =========================
# Run
# =========================
CMD=(python -u "$PY_SCRIPT" "$ROOT")

if [[ "$SKIP_EXISTING" -eq 1 ]]; then
  CMD+=(--skip-existing)
fi

if [[ "$QUIET" -eq 1 ]]; then
  CMD+=(--quiet)
fi

# Pass through any additional CLI args after ROOT.
if (( $# > 0 )); then
  CMD+=("$@")
fi

echo "GRID_ID=$GRID_ID / $NROOTS"
echo "ROOT=$ROOT"
echo "PY_SCRIPT=$PY_SCRIPT"
echo "SKIP_EXISTING=$SKIP_EXISTING"
echo "QUIET=$QUIET"
echo "Running: ${CMD[*]}"
"${CMD[@]}"