#!/bin/bash
#SBATCH --export=NONE
#SBATCH --job-name=single_tok_text_coco
#SBATCH --output=logs/single_tok_text_coco_%A_%a.out
#SBATCH --error=logs/single_tok_text_coco_%A_%a.err
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --partition=mit_preemptable
#SBATCH --gres=gpu:1
#SBATCH --array=0-5

mkdir -p logs

# =========================
#  Environment
# =========================
module load miniforge
CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate coco_text_embed

# =========================
#  User config
# =========================
DATASET_NAME="words"
IN_ROOT="/home/kirilb/orcd/scratch/${DATASET_NAME}"
CAPTIONS_JSON="${IN_ROOT}/annotations/captions_train2017.json"

TARGET_ROOT="/home/kirilb/orcd/scratch/PRH_data/single_token_${DATASET_NAME}_text"
HF_CACHE_DIR="/home/kirilb/orcd/pool/huggingface_models_cache"
PYTHON_SCRIPT="filter_single_token_words_text.py"
CAPTION_KEY="caption"

# bare: tokenize the word alone. Cleanest tokenization-control ablation.
# embedding_prompt: uses explicit prefixes visible in embed_text.py when known.
# manual: uses LEFT_CONTEXT and RIGHT_CONTEXT.
CONTEXT_MODE="bare"
LEFT_CONTEXT=""
RIGHT_CONTEXT=""

# This job only uses the tokenizer, so the GPU is not used by the Python code.
# The GPU request above is included to match your existing embedding launch style.

MODELS=(
  "BAAI/bge-large-en-v1.5"
  "BAAI/bge-base-en-v1.5"
  "nomic-ai/nomic-embed-text-v1.5"
  "nomic-ai/nomic-embed-text-v2-moe"
  "codefuse-ai/F2LLM-1.7B"
  "codefuse-ai/F2LLM-4B"
)

mkdir -p "$TARGET_ROOT" "$HF_CACHE_DIR"
export HF_HOME="$HF_CACHE_DIR"

if [[ ! -f "$CAPTIONS_JSON" ]]; then
  echo "Captions JSON not found: $CAPTIONS_JSON"
  exit 1
fi

if [[ ! -f "$PYTHON_SCRIPT" ]]; then
  echo "Python script not found: $PYTHON_SCRIPT"
  exit 1
fi

TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
NUM_MODELS=${#MODELS[@]}

if [[ ${TASK_ID} -ge ${NUM_MODELS} ]]; then
  echo "Task ${TASK_ID} >= NUM_MODELS ${NUM_MODELS}. Exiting."
  exit 0
fi

MODEL_ID="${MODELS[$TASK_ID]}"
MODEL_TAG="${MODEL_ID//\//__}"

OUT_DIR="${TARGET_ROOT}/${MODEL_TAG}"
OUT_JSON="${OUT_DIR}/annotations/$(basename "$CAPTIONS_JSON")"

mkdir -p "$OUT_DIR"

echo "DATASET_NAME=${DATASET_NAME}"
echo "CAPTIONS_JSON=${CAPTIONS_JSON}"
echo "TARGET_ROOT=${TARGET_ROOT}"
echo "SLURM_ARRAY_TASK_ID=${TASK_ID}"
echo "MODEL_ID=${MODEL_ID}"
echo "MODEL_TAG=${MODEL_TAG}"
echo "OUT_JSON=${OUT_JSON}"
echo "CONTEXT_MODE=${CONTEXT_MODE}"

python "$PYTHON_SCRIPT" \
  --model_name "$MODEL_ID" \
  --captions_json "$CAPTIONS_JSON" \
  --output_root "$TARGET_ROOT" \
  --hf_cache_dir "$HF_CACHE_DIR" \
  --caption_key "$CAPTION_KEY" \
  --context_mode "$CONTEXT_MODE" \
  --left_context "$LEFT_CONTEXT" \
  --right_context "$RIGHT_CONTEXT"
