#!/bin/bash
#SBATCH --export=NONE
#SBATCH --job-name=truncate_npz
#SBATCH --output=logs/truncate_npz_%A_%a.out
#SBATCH --error=logs/truncate_npz_%A_%a.err
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --partition=mit_normal     # <-- change to your CPU partition
#SBATCH --array=0-0                 # <-- set to 0-(NUM_ROOTS-1)

mkdir -p logs

module load miniforge
CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate GPUenv   # env with numpy+scipy

# --------------------- Script ---------------------
PY_SCRIPT="sparse_feature_truncation.py"

# --------------------- Folder list ---------------------
# One SLURM array task per entry here.
ROOTS=(
   "/home/kirilb/orcd/scratch/PRH_data/topk_sae_coco/"
   "/home/kirilb/orcd/scratch/PRH_data/topk_sae_visual_genome/"
   "/home/kirilb/orcd/scratch/PRH_data/topk_sae_cc3m/"
)

ROOT="${ROOTS[$SLURM_ARRAY_TASK_ID]}"

# --------------------- Params ---------------------
PH="0.1"
PL="0.00001"

# Flags (set to 1 to enable)
OVERWRITE=1
SAVE_IDX=1
QUIET=0

ARGS=( --root "$ROOT" --ph "$PH" --pl "$PL" )
if [[ "$OVERWRITE" -eq 1 ]]; then ARGS+=( --overwrite ); fi
if [[ "$SAVE_IDX" -eq 1 ]]; then ARGS+=( --save_idx ); fi
if [[ "$QUIET" -eq 1 ]]; then ARGS+=( --quiet ); fi

echo "SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID"
echo "ROOT=$ROOT"
echo "Running: python $PY_SCRIPT ${ARGS[*]}"

python "$PY_SCRIPT" "${ARGS[@]}"
