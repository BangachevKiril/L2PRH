#!/bin/bash
#SBATCH --export=NONE
#SBATCH --job-name=dense_feature_metrics
#SBATCH --output=logs/dense_feature_metrics_%A_%a.out
#SBATCH --error=logs/dense_feature_metrics_%A_%a.err
#SBATCH --time=05:59:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:1
#
# IMPORTANT:
# Array size should be:
#   2 * (# DATASET_NAMES) * (# USE_NORMALIZED_VALUES)
#
# With the defaults below:
#   DATASET_NAMES has 3 values
#   USE_NORMALIZED_VALUES has 2 values
#   two shards per pair
# so array is 0-11.
#
#SBATCH --array=0-11

mkdir -p logs

module load miniforge
CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate GPUenv

PY_SCRIPT="dense_feature_metrics.py"

# ---------------------------------------------------------------------
# User settings
# ---------------------------------------------------------------------

# Dataset roots to run over.
# Add/remove values here, e.g. "words" if desired.
DATASET_NAMES=(
  "coco"
  "cc3m"
  "visual_genome"
)

# Normalization modes to run over.
# 0: raw embeddings
# 1: centered embeddings, uses *_normalized.npy
# 2: fully normalized embeddings, uses *_fully_normalized.npy
USE_NORMALIZED_VALUES=(
  0
  1
)

# For each (DATASET, USE_NORMALIZED), split all model-pair runs into this many jobs.
RUN_SHARDS_PER_SETTING=2

HOW_MANY_SAMPLES=10
SUBSAMPLE_SIZE=1000
SEED_BASE=12345
DEVICE="cuda"

SVCCA_DIM1=10
SVCCA_DIM2=100
TOPK_K1=10
TOPK_K2=100
EDIT_K1=10
EDIT_K2=100

# These are the model/modality entries compared pairwise inside each dataset.
MODEL_SPECS=(
  "codefuse-ai__F2LLM-1.7B/text"
  "codefuse-ai__F2LLM-4B/text"
  "BAAI__bge-base-en-v1.5/text"
  "BAAI__bge-large-en-v1.5/text"
  "Qwen__Qwen3-1.7B-Base/text"
  "Qwen__Qwen3-4B-Base/text"
  "facebook__dinov2-base/img"
  "facebook__dinov2-large/img"
  "facebook__vit-mae-huge/img"
  "facebook__vit-mae-large/img"
  "google__gemma-3-1b-it/text"
  "google__gemma-3-4b-it/text"
  "google__siglip2-base-patch16-256/text"
  "google__siglip2-large-patch16-256/text"
  "google__siglip2-base-patch16-256/img"
  "google__siglip2-large-patch16-256/img"
  "laion__CLIP-ViT-B-32-laion2B-s34B-b79K/text"
  "laion__CLIP-ViT-H-14-laion2B-s32B-b79K/text"
  "laion__CLIP-ViT-B-32-laion2B-s34B-b79K/img"
  "laion__CLIP-ViT-H-14-laion2B-s32B-b79K/img"
  "meta-llama__Llama-3.2-1B-Instruct/text"
  "meta-llama__Llama-3.2-3B-Instruct/text"
  "microsoft__beit-base-patch16-224/img"
  "microsoft__beit-large-patch16-224/img"
  "nomic-ai__nomic-embed-text-v1.5/text"
  "nomic-ai__nomic-embed-text-v2-moe/text"
  "openai__clip-vit-base-patch32/text"
  "openai__clip-vit-large-patch14/text"
  "openai__clip-vit-base-patch32/img"
  "openai__clip-vit-large-patch14/img"
)

# ---------------------------------------------------------------------
# Decode SLURM_ARRAY_TASK_ID into:
#   DATASET
#   USE_NORMALIZED
#   SHARD_ID in {0, 1}
# ---------------------------------------------------------------------

NUM_DATASET_NAMES=${#DATASET_NAMES[@]}
NUM_USE_NORMALIZED_VALUES=${#USE_NORMALIZED_VALUES[@]}
NUM_SETTINGS=$(( NUM_DATASET_NAMES * NUM_USE_NORMALIZED_VALUES ))
EXPECTED_ARRAY_TASKS=$(( NUM_SETTINGS * RUN_SHARDS_PER_SETTING ))

TASK_ID=${SLURM_ARRAY_TASK_ID:-0}

if (( TASK_ID < 0 || TASK_ID >= EXPECTED_ARRAY_TASKS )); then
  echo "ERROR: SLURM_ARRAY_TASK_ID=$TASK_ID is outside expected range 0-$((EXPECTED_ARRAY_TASKS - 1))"
  echo "Update #SBATCH --array=0-$((EXPECTED_ARRAY_TASKS - 1)) or change DATASET_NAMES / USE_NORMALIZED_VALUES."
  exit 1
fi

SETTING_ID=$(( TASK_ID / RUN_SHARDS_PER_SETTING ))
SHARD_ID=$(( TASK_ID % RUN_SHARDS_PER_SETTING ))

DATASET_ID=$(( SETTING_ID / NUM_USE_NORMALIZED_VALUES ))
USE_NORM_ID=$(( SETTING_ID % NUM_USE_NORMALIZED_VALUES ))

DATASET="${DATASET_NAMES[$DATASET_ID]}"
USE_NORMALIZED="${USE_NORMALIZED_VALUES[$USE_NORM_ID]}"

INPUT_PATH="/home/kirilb/orcd/scratch/PRH_data/embedded_${DATASET}"
OUT_DIR_BASE="/home/kirilb/orcd/scratch/PRH_data/metrics_embedded_${DATASET}"

# ---------------------------------------------------------------------
# Compute OUT_DIR suffix and embedding suffix based on USE_NORMALIZED
# ---------------------------------------------------------------------

OUT_SUFFIX=""
suffix=".npy"

if [[ "$USE_NORMALIZED" -eq 0 ]]; then
  OUT_SUFFIX=""
  suffix=".npy"
elif [[ "$USE_NORMALIZED" -eq 1 ]]; then
  OUT_SUFFIX="_centered"
  suffix="_normalized.npy"
elif [[ "$USE_NORMALIZED" -eq 2 ]]; then
  OUT_SUFFIX="_fully_normalized"
  suffix="_fully_normalized.npy"
else
  echo "ERROR: USE_NORMALIZED must be 0, 1, or 2. Got $USE_NORMALIZED"
  exit 1
fi

OUT_DIR="${OUT_DIR_BASE}${OUT_SUFFIX}"

echo "PWD=$(pwd)"
echo "which python=$(which python)"
python -V
echo "PY_SCRIPT=$PY_SCRIPT"
echo "SLURM_ARRAY_TASK_ID=$TASK_ID"
echo "EXPECTED_ARRAY_TASKS=$EXPECTED_ARRAY_TASKS"
echo "SETTING_ID=$SETTING_ID"
echo "SHARD_ID=$SHARD_ID / $RUN_SHARDS_PER_SETTING"
echo "DATASET=$DATASET"
echo "USE_NORMALIZED=$USE_NORMALIZED"
echo "INPUT_PATH=$INPUT_PATH"
echo "OUT_DIR_BASE=$OUT_DIR_BASE"
echo "OUT_DIR=$OUT_DIR"
echo "embedding suffix=$suffix"

ls -l "$PY_SCRIPT" || {
  echo "ERROR: cannot find $PY_SCRIPT in $(pwd)"
  exit 1
}

N=${#MODEL_SPECS[@]}
NUM_PAIRS=$(( N * (N - 1) / 2 ))

echo "Total model specs: $N"
echo "Total unordered pairs per setting: $NUM_PAIRS"
echo "This task handles pair indices k where k % $RUN_SHARDS_PER_SETTING == $SHARD_ID"

COMMON_ARGS=(
  --input_path "$INPUT_PATH"
  --output_dir "$OUT_DIR"
  --how_many_samples "$HOW_MANY_SAMPLES"
  --subsample_size "$SUBSAMPLE_SIZE"
  --device "$DEVICE"
  --svcca_dim1 "$SVCCA_DIM1"
  --svcca_dim2 "$SVCCA_DIM2"
  --topk_k1 "$TOPK_K1"
  --topk_k2 "$TOPK_K2"
  --edit_k1 "$EDIT_K1"
  --edit_k2 "$EDIT_K2"
  --use_normalized "$USE_NORMALIZED"
)

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

  local DS1="${MODEL_SPECS[$i]}"
  local DS2="${MODEL_SPECS[$j]}"
  local SEED=$((SEED_BASE + k))

  local P1 P2
  P1=$(dataset_to_path "$DS1")
  P2=$(dataset_to_path "$DS2")

  local OUT_NAME="${DS1//\//__}_${DS2//\//__}.npz"
  local OUT_PATH="${OUT_DIR%/}/$OUT_NAME"

  echo "------------------------------------------------------------"
  echo "Array task $TASK_ID"
  echo "Dataset setting: DATASET=$DATASET, USE_NORMALIZED=$USE_NORMALIZED"
  echo "Shard: $SHARD_ID / $RUN_SHARDS_PER_SETTING"
  echo "Pair k=$k (i=$i, j=$j)"
  echo "Comparing: $DS1  vs  $DS2"
  echo "Seed: $SEED"
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

for ((k=SHARD_ID; k<NUM_PAIRS; k+=RUN_SHARDS_PER_SETTING)); do
  run_pair_k "$k"
done