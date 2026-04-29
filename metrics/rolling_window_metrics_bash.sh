#!/bin/bash
#SBATCH --export=NONE
#SBATCH --job-name=pair_metrics
#SBATCH --output=logs/pair_metrics_%A_%a.out
#SBATCH --error=logs/pair_metrics_%A_%a.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --array=0-29   # fixed 30 tasks, works for ANY number of pairs

mkdir -p logs

# ------------------------------------------------------------
# ENV
# ------------------------------------------------------------
module load miniforge
CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate GPUenv   # EDIT if needed

# ------------------------------------------------------------
# USER CONFIG
# ------------------------------------------------------------
PY_SCRIPT="rolling_window_metrics.py"

# Root directory that contains per-model folders
# e.g. /home/kirilb/orcd/pool/PRH_data/embedded_words/<MODEL>/text_embeddings(_normalized).npy
INPUT_PATH="/home/kirilb/orcd/scratch/PRH_data/embedded_words"
OUTPUT_DIR="/home/kirilb/orcd/scratch/PRH_data/metrics_embedded_words/"

# Use normalized embeddings?
USE_NORMALIZED=1   # 1 => add --use_normalized, 0 => raw

# Rolling-window + metric params
STEP_SIZE=250
BATCH_SIZE=500
TILL_WHEN=2750 # so that we have 10 data points
DEVICE="cuda"

SVCCA_DIM1=10
SVCCA_DIM2=100
TOPK_K1=10
TOPK_K2=100

NUM_QUARTETS=1000
QUARTET_BATCH_SIZE=512

NUM_QUADRUPLETS=1000
NUM_TRIPLETS=1000
THRESHOLD_BATCH_SIZE=512

EDIT_K1=10
EDIT_K2=100

# ------------------------------------------------------------
# DATASETS (MODEL/text or MODEL/img)
# ------------------------------------------------------------
DATASETS=(
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
  "codefuse-ai/F2LLM-1.7B"
  "codefuse-ai/F2LLM-4B"
)

N=${#DATASETS[@]}
if (( N < 2 )); then
  echo "Need at least 2 datasets, got N=$N"
  exit 1
fi

NUM_PAIRS=$(( N * (N - 1) / 2 ))
echo "Total datasets: $N"
echo "Total unordered pairs: $NUM_PAIRS"

# Number of array tasks (should be 32 with --array=0-31)
NUM_TASKS=${SLURM_ARRAY_TASK_COUNT:-32}
TASK_ID=${SLURM_ARRAY_TASK_ID}

echo "Task $TASK_ID / $NUM_TASKS"

# ------------------------------------------------------------
# Common args for python
# ------------------------------------------------------------
COMMON_ARGS=(
  --input_path "$INPUT_PATH"
  --step_size "$STEP_SIZE"
  --batch_size "$BATCH_SIZE"
  --till_when "$TILL_WHEN"
  --device "$DEVICE"
  --svcca_dim1 "$SVCCA_DIM1"
  --svcca_dim2 "$SVCCA_DIM2"
  --topk_k1 "$TOPK_K1"
  --topk_k2 "$TOPK_K2"
  --num_quartets "$NUM_QUARTETS"
  --quartet_batch_size "$QUARTET_BATCH_SIZE"
  --num_quadruplets "$NUM_QUADRUPLETS"
  --num_triplets "$NUM_TRIPLETS"
  --threshold_batch_size "$THRESHOLD_BATCH_SIZE"
  --edit_k1 "$EDIT_K1"
  --edit_k2 "$EDIT_K2"
  --output_dir "$OUTPUT_DIR"
)

if [[ "$USE_NORMALIZED" -eq 1 ]]; then
  COMMON_ARGS+=(--use_normalized)
fi

# ------------------------------------------------------------
# Map pair index k in [0, NUM_PAIRS) -> (i, j) with i < j
# without building the full PAIRS list.
# Order matches:
#   (0,1), (0,2), ... (0,N-1), (1,2), (1,3), ...
# ------------------------------------------------------------
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

  echo "ERROR"
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

  echo "------------------------------------------------------------"
  echo "Task $TASK_ID: pair k=$k  (i=$i, j=$j)"
  echo "Comparing: $DS1  vs  $DS2"
  echo "Input root: $INPUT_PATH"
  echo "Normalized: $USE_NORMALIZED"
  echo "------------------------------------------------------------"

  python -u "$PY_SCRIPT" "$DS1" "$DS2" "${COMMON_ARGS[@]}"

  echo "Done pair k=$k : $DS1 vs $DS2"
}

# ------------------------------------------------------------
# Strided assignment: this task runs k = TASK_ID, TASK_ID+NUM_TASKS, ...
# ------------------------------------------------------------
for ((k=TASK_ID; k<NUM_PAIRS; k+=NUM_TASKS)); do
  run_pair_k "$k"
done
