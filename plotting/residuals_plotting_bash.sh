#!/bin/bash
#SBATCH --export=NONE
#SBATCH --job-name=residual_tables
#SBATCH --output=logs/residual_tables_%A_%a.out
#SBATCH --error=logs/residual_tables_%A_%a.err
#SBATCH --time=00:10:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=1G
#SBATCH --partition=mit_normal
#SBATCH --array=0-5

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
DIMS=(
  8192
  16384
)

NAMES=(
  "coco"
  "cc3m"
  "visual_genome"
)

PY_SCRIPT="residuals_plotting.py"

# Optional knobs, overridable via environment variables:
#   PRECISION=6 QUIET=1 sbatch residual_statistics_sh.sh
PRECISION="${PRECISION:-4}"
QUIET="${QUIET:-0}"

OUTDIR="/home/kirilb/data/L2PRH/residual_statistics"
mkdir -p "$OUTDIR"

NDIMS=${#DIMS[@]}
NNAMES=${#NAMES[@]}
NTASKS=$((NDIMS * NNAMES))

GRID_ID=${SLURM_ARRAY_TASK_ID:-0}
if (( GRID_ID < 0 || GRID_ID >= NTASKS )); then
  echo "ERROR: SLURM_ARRAY_TASK_ID=$GRID_ID out of range [0, $((NTASKS-1))]"
  echo "Set: #SBATCH --array=0-$((NTASKS-1))"
  exit 1
fi

DIM_INDEX=$((GRID_ID / NNAMES))
NAME_INDEX=$((GRID_ID % NNAMES))

D="${DIMS[$DIM_INDEX]}"
NAME="${NAMES[$NAME_INDEX]}"
ROOT="/home/kirilb/orcd/pool/PRH_data/topk_sae_${NAME}"

shift || true

if [[ ! -f "$PY_SCRIPT" ]]; then
  echo "ERROR: Python script not found: $PY_SCRIPT"
  exit 1
fi

if [[ ! -d "$ROOT" ]]; then
  echo "ERROR: Root folder not found: $ROOT"
  exit 1
fi

# Include dataset name in the output so files do not overwrite each other
OUTFILE="${OUTDIR}/residual_statistics_${NAME}_d_${D}.tex"

# =========================
# Run
# =========================
CMD=(
  python -u "$PY_SCRIPT"
  "$D"
  "$ROOT"
  --out "$OUTFILE"
  --precision "$PRECISION"
)

if [[ "$QUIET" -eq 1 ]]; then
  CMD+=(--quiet)
fi

# Pass through any additional CLI args
if (( $# > 0 )); then
  CMD+=("$@")
fi

echo "GRID_ID=$GRID_ID / $((NTASKS-1))"
echo "DIM_INDEX=$DIM_INDEX"
echo "NAME_INDEX=$NAME_INDEX"
echo "D=$D"
echo "NAME=$NAME"
echo "ROOT=$ROOT"
echo "OUTDIR=$OUTDIR"
echo "OUTFILE=$OUTFILE"
echo "PY_SCRIPT=$PY_SCRIPT"
echo "PRECISION=$PRECISION"
echo "QUIET=$QUIET"
echo "Running: ${CMD[*]}"

"${CMD[@]}"
