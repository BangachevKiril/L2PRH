#!/bin/bash
#SBATCH --export=NONE
#SBATCH --job-name=embed_text
#SBATCH --output=logs/embed_coco_text_%A_%a.out
#SBATCH --error=logs/embed_coco_text_%A_%a.err
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:1
#SBATCH --array=0-3   # self-truncates based on MODELS below

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
DATASET_NAME="cc3m" # change if needed; "coco", "cc3m", "visual_genome", "words"   

IN_ROOT="/home/kirilb/orcd/scratch/${DATASET_NAME}"

CAPTIONS_JSON="${IN_ROOT}/annotations/captions_train2017.json"

OUT_ROOT="/home/kirilb/orcd/scratch/PRH_data/embedded_${DATASET_NAME}" 

HF_CACHE_DIR="/home/kirilb/orcd/pool/huggingface_models_cache"

BATCH_SIZE=16    # adjust for VRAM if needed
DEVICE="cuda"

# Models to run (one per array index)
MODELS=(
  "BAAI/bge-large-en-v1.5"
  "BAAI/bge-base-en-v1.5"
  "nomic-ai/nomic-embed-text-v1.5"
  "nomic-ai/nomic-embed-text-v2-moe"
  "codefuse-ai/F2LLM-1.7B"
  "codefuse-ai/F2LLM-4B"
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
#  Run embedding
# =========================
python embed_text.py \
  --model_name "$MODEL_ID" \
  --captions_json "$CAPTIONS_JSON" \
  --output_dir "$OUT_DIR" \
  --batch_size "$BATCH_SIZE" \
  --device "$DEVICE" \
  --hf_cache_dir "$HF_CACHE_DIR"
