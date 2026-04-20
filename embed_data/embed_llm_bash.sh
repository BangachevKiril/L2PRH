#!/bin/bash
#SBATCH --export=NONE
#SBATCH --job-name=embed_captions_llm
#SBATCH --output=logs/embed_captions_llm_%A_%a.out
#SBATCH --error=logs/embed_captions_llm_%A_%a.err
#SBATCH --time=05:50:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=192G
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --array=0-5   # self-truncates based on MODELS below

mkdir -p logs

# =========================
#  Environment
# =========================
module load miniforge
CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate GPUenv

# =========================
#  User config
# =========================

DATASET = "visual_genome" # "coco", "cc3m", "visual_genome", "words"
IN_ROOT="/home/kirilb/orcd/pool/${DATASET}"
OUT_ROOT="/home/kirilb/orcd/pool/PRH_data/embedded_${DATASET}"
CAPTIONS_PATH="${IN_ROOT}/annotations/captions_train2017.json" # Universal


HF_CACHE_DIR="/home/kirilb/orcd/pool/huggingface_models_cache"

BATCH_SIZE=16
MAX_LENGTH=64
DEVICE="cuda"

MODELS=(
  "google/gemma-3-1b-it"
  "google/gemma-3-4b-it"
  "meta-llama/Llama-3.2-3B-Instruct"
  "meta-llama/Llama-3.2-1B-Instruct"
  "Qwen/Qwen3-4B-Base"
  "Qwen/Qwen3-1.7B-Base"
)

mkdir -p "$OUT_ROOT" "$HF_CACHE_DIR"

# =========================
#  Select model for this task
# =========================
TASK_ID=${SLURM_ARRAY_TASK_ID}
NUM_MODELS=${#MODELS[@]}

if [[ ${TASK_ID} -ge ${NUM_MODELS} ]]; then
  echo "Task ${TASK_ID} >= NUM_MODELS ${NUM_MODELS}. Exiting (self-truncate)."
  exit 0
fi

MODEL_ID="${MODELS[$TASK_ID]}"
MODEL_TAG="${MODEL_ID//\//__}"

echo "SLURM_ARRAY_TASK_ID=${TASK_ID}"
echo "MODEL_ID=${MODEL_ID}"
echo "MODEL_TAG=${MODEL_TAG}"

OUT_DIR="${OUT_ROOT}/${MODEL_TAG}"
mkdir -p "$OUT_DIR"

# =========================
#  Run feature extraction
# =========================
python embed_llm.py \
  --model_name "$MODEL_ID" \
  --captions_path "$CAPTIONS_PATH" \
  --output_dir "$OUT_DIR" \
  --batch_size "$BATCH_SIZE" \
  --max_length "$MAX_LENGTH" \
  --device "$DEVICE" \
  --hf_cache_dir "$HF_CACHE_DIR"
