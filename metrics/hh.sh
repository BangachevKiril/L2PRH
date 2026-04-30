#!/bin/bash
#SBATCH --export=NONE
#SBATCH --job-name=sliding_metrics
#SBATCH --output=logs/sliding_metrics_%A_%a.out
#SBATCH --error=logs/sliding_metrics_%A_%a.err
#SBATCH --time=05:59:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --partition=mit_preemptable
#SBATCH --gres=gpu:1
#SBATCH --array=0-1

mkdir -p logs

module load miniforge
CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate GPUenv

PY_SCRIPT="rolling_window_metrics.py"

INPUT_PATH="/home/kirilb/orcd/scratch/PRH_data/embedded_words" 


USE_NORMALIZED=1

# Optional zero-based index list. Leave empty for old behavior.
# Example:
USE_INDICES="/home/kirilb/orcd/scratch/words/adjective_indices.txt"
# USE_INDICES=""

# Interpreted as:
#   HOW_MANY_SAMPLES = HOW_MANY windows
#   SUBSAMPLE_SIZE   = batch_size (window size)
HOW_MANY_SAMPLES=5
SUBSAMPLE_SIZE=500
STEP_SIZE=250

SEED_BASE=12345
DEVICE="cuda"

# --- Base output dir (no suffix). Pick ONE and leave it. ---
OUT_DIR_BASE="/home/kirilb/orcd/scratch/PRH_data/metrics_embedded_words"

# --- Compute OUT_DIR suffix based on USE_NORMALIZED ---
OUT_SUFFIX=""
if [[ "$USE_NORMALIZED" -eq 1 ]]; then
  OUT_SUFFIX="_centered"
elif [[ "$USE_NORMALIZED" -eq 2 ]]; then
  OUT_SUFFIX="_centered_and_isotropic"
elif [[ "$USE_NORMALIZED" -ne 0 ]]; then
  echo "ERROR: USE_NORMALIZED must be 0, 1, or 2 (got $USE_NORMALIZED)"
  exit 1
fi

INDEX_SUFFIX=""
if [[ -n "$USE_INDICES" ]]; then
  if [[ ! -f "$USE_INDICES" ]]; then
    echo "ERROR: USE_INDICES is set but file does not exist: $USE_INDICES"
    exit 1
  fi
  IDX_BASE=$(basename "$USE_INDICES")
  IDX_STEM="${IDX_BASE%.txt}"
  # Keep folder names shell-friendly.
  IDX_STEM=$(printf '%s' "$IDX_STEM" | tr -c 'A-Za-z0-9._-' '_')
  INDEX_SUFFIX="_${IDX_STEM}"
fi

OUT_DIR="${OUT_DIR_BASE}${OUT_SUFFIX}${INDEX_SUFFIX}"

SVCCA_DIM1=10
SVCCA_DIM2=100
TOPK_K1=10
TOPK_K2=100
EDIT_K1=10
EDIT_K2=100

DATASETS=(
  "codefuse-ai__F2LLM-1.7B/text"
  "codefuse-ai__F2LLM-4B/text"
  "BAAI__bge-base-en-v1.5/text"
  "BAAI__bge-large-en-v1.5/text"
  "Qwen__Qwen3-1.7B-Base/text"
  "Qwen__Qwen3-4B-Base/text"
  "google__gemma-3-1b-it/text"
  "google__gemma-3-4b-it/text"
  "meta-llama__Llama-3.2-1B-Instruct/text"
  "meta-llama__Llama-3.2-3B-Instruct/text"
  "nomic-ai__nomic-embed-text-v1.5/text"
  "nomic-ai__nomic-embed-text-v2-moe/text"
)

echo "PWD=$(pwd)"
echo "which python=$(which python)"
python -V
echo "PY_SCRIPT=$PY_SCRIPT"
echo "USE_NORMALIZED=$USE_NORMALIZED"
echo "USE_INDICES=$USE_INDICES"
echo "OUT_DIR_BASE=$OUT_DIR_BASE"
echo "OUT_DIR=$OUT_DIR"
echo "HOW_MANY_SAMPLES=$HOW_MANY_SAMPLES  SUBSAMPLE_SIZE=$SUBSAMPLE_SIZE  STEP_SIZE=$STEP_SIZE"
ls -l "$PY_SCRIPT" || { echo "ERROR: cannot find $PY_SCRIPT in $(pwd)"; exit 1; }

N=${#DATASETS[@]}
NUM_PAIRS=$(( N * (N - 1) / 2 ))
NUM_TASKS=${SLURM_ARRAY_TASK_COUNT:-2}
TASK_ID=${SLURM_ARRAY_TASK_ID}

echo "Total datasets: $N"
echo "Total unordered pairs: $NUM_PAIRS"
echo "Task $TASK_ID / $NUM_TASKS"

COMMON_ARGS=(
  --input_path "$INPUT_PATH"
  --output_dir "$OUT_DIR"
  --how_many_samples "$HOW_MANY_SAMPLES"
  --subsample_size "$SUBSAMPLE_SIZE"
  --step_size "$STEP_SIZE"
  --device "$DEVICE"
  --svcca_dim1 "$SVCCA_DIM1"
  --svcca_dim2 "$SVCCA_DIM2"
  --topk_k1 "$TOPK_K1"
  --topk_k2 "$TOPK_K2"
  --edit_k1 "$EDIT_K1"
  --edit_k2 "$EDIT_K2"
)

COMMON_ARGS+=(--use_normalized "$USE_NORMALIZED")

if [[ -n "$USE_INDICES" ]]; then
  COMMON_ARGS+=(--use_indices "$USE_INDICES")
fi

suffix=".npy"
if [[ "$USE_NORMALIZED" -eq 1 ]]; then
  suffix="_normalized.npy"
elif [[ "$USE_NORMALIZED" -eq 2 ]]; then
  suffix="_fully_normalized.npy"
fi

dataset_to_path () {
  local ds="$1"
  local model="${ds%/*}"
  local kind="${ds##*/}"   # text or img
  echo "${INPUT_PATH}/${model}/${kind}_embeddings${suffix}"
}

pair_index_to_ij () {
  local k=$1
  local rem=$k
  local i j cnt
  for ((i=0; i< N-1; i++)); do
    cnt=$((N - i - 1))
    if (( rem < cnt )); then
      j=$((i + 1 + rem))
      echo "$i $j"
      return 0
    fi
    rem=$((rem - cnt))
  done
  return 1
}

run_pair_k () {
  local k=$1
  local ij
  ij=$(pair_index_to_ij "$k") || exit 1

  local i j
  i=$(echo "$ij" | awk '{print $1}')
  j=$(echo "$ij" | awk '{print $2}')

  local DS1="${DATASETS[$i]}"
  local DS2="${DATASETS[$j]}"
  local SEED=$((SEED_BASE + k))

  local P1 P2
  P1=$(dataset_to_path "$DS1")
  P2=$(dataset_to_path "$DS2")

  local OUT_NAME="${DS1//\//__}_${DS2//\//__}.npz"
  local OUT_PATH="${OUT_DIR%/}/$OUT_NAME"

  echo "------------------------------------------------------------"
  echo "Task $TASK_ID: pair k=$k (i=$i, j=$j)"
  echo "Comparing: $DS1  vs  $DS2"
  echo "Seed: $SEED (unused for sliding windows; kept for compatibility)"
  echo "Expecting:"
  echo "  X1: $P1"
  echo "  X2: $P2"
  echo "Will write:"
  echo "  OUT: $OUT_PATH"
  echo "------------------------------------------------------------"

  if [[ ! -f "$P1" ]]; then
    echo "SKIP missing: $P1"
    return 0
  fi
  if [[ ! -f "$P2" ]]; then
    echo "SKIP missing: $P2"
    return 0
  fi

  mkdir -p "$OUT_DIR"

  python -u "$PY_SCRIPT" "$DS1" "$DS2" \
    --seed "$SEED" \
    --output_name "$OUT_NAME" \
    --profile_metrics \
    "${COMMON_ARGS[@]}"

  if [[ ! -s "$OUT_PATH" ]]; then
    echo "ERROR: python returned but output file not created: $OUT_PATH"
    exit 1
  fi

  echo "Done pair k=$k : $DS1 vs $DS2"
}

for ((k=TASK_ID; k<NUM_PAIRS; k+=NUM_TASKS)); do
  run_pair_k "$k"
done