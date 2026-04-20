#!/bin/bash
#SBATCH --export=NONE
#SBATCH --job-name=embed_coco_fastio
#SBATCH --output=logs/embed_coco_fastio_%A_%a.out
#SBATCH --error=logs/embed_coco_fastio_%A_%a.err
#SBATCH --time=05:59:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:1
#SBATCH --array=0-5   # will self-truncate based on MODELS below

mkdir -p logs

# -------------------------
# Environment
# -------------------------
module load miniforge
CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate GPUenv

# -------------------------
# User config
# -------------------------
DATASET = "visual_genome" # "coco", "cc3m", "visual_genome", "words"
IN_ROOT="/home/kirilb/orcd/pool/${DATASET}"
OUT_ROOT="/home/kirilb/orcd/pool/PRH_data/embedded_${DATASET}"

SPLIT="train"  # train | val   (used only for COCO captions_{split}2017.json mode)

# OUT_ROOT="/home/kirilb/orcd/pool/PRH_data/embedded_coco_fastio"
OUT_ROOT="/home/kirilb/orcd/pool/PRH_data/embedded_visual_genome_fastio"

HF_CACHE_DIR="/home/kirilb/orcd/pool/huggingface_models_cache"

DEVICE="cuda"
DTYPE="fp16"                 # auto | fp16 | bf16 | fp32
BATCH_SIZE=256               # images per batch (uint8 decode in workers)
NUM_WORKERS=8                # parallel image decode workers
PREFETCH_FACTOR=4            # batches prefetched per worker
TEXT_CHUNK_SIZE=4096         # max captions per text embed call (prevents spikes)

# Path to the new Python script
PY_SCRIPT="embed_multimodal.py"

# Models list (edit this array)
MODELS=(
  "openai/clip-vit-base-patch32"
  "openai/clip-vit-large-patch14"
  "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
  "laion/CLIP-ViT-B-32-laion2B-s34B-b79K"
  "google/siglip2-base-patch16-256"
  "google/siglip2-large-patch16-256"
)

# -------------------------
# Pick model for this task
# -------------------------
TASK_ID=${SLURM_ARRAY_TASK_ID}
NUM_MODELS=${#MODELS[@]}

if [[ ${TASK_ID} -ge ${NUM_MODELS} ]]; then
  echo "Task ${TASK_ID} >= NUM_MODELS ${NUM_MODELS}. Exiting (self-truncate)."
  exit 0
fi

MODEL_ID="${MODELS[$TASK_ID]}"
echo "SLURM_ARRAY_TASK_ID=${TASK_ID}"
echo "MODEL_ID=${MODEL_ID}"

# -------------------------
# Hugging Face cache (since --export=NONE)
# -------------------------
mkdir -p "$OUT_ROOT" "$HF_CACHE_DIR"
export HF_HOME="$HF_CACHE_DIR"
export HUGGINGFACE_HUB_CACHE="$HF_CACHE_DIR/hub"
export TRANSFORMERS_CACHE="$HF_CACHE_DIR/transformers"
export HF_DATASETS_CACHE="$HF_CACHE_DIR/datasets"
export HF_HUB_DISABLE_TELEMETRY=1

# -------------------------
# Sanity check: COCO captions file must exist
# -------------------------
ANN_PATH="$IN_ROOT/annotations/captions_${SPLIT}2017.json"
if [[ ! -f "$ANN_PATH" ]]; then
  echo "ERROR: Expected COCO captions JSON not found:"
  echo "  $ANN_PATH"
  echo ""
  echo "This fastio script is COCO-caption mode only (needs annotations/captions_${SPLIT}2017.json)."
  echo "If you're embedding Visual Genome / LAION / CC3M, use the image-only folder-mode script instead,"
  echo "or convert the dataset to COCO captions format."
  exit 1
fi

# -------------------------
# Run one model
# -------------------------
python "$PY_SCRIPT" \
  --coco_root "$IN_ROOT" \
  --split "$SPLIT" \
  --out_root "$OUT_ROOT" \
  --batch_size "$BATCH_SIZE" \
  --device "$DEVICE" \
  --dtype "$DTYPE" \
  --resume \
  --hf_cache_dir "$HF_CACHE_DIR" \
  --num_workers "$NUM_WORKERS" \
  --prefetch_factor "$PREFETCH_FACTOR" \
  --text_chunk_size "$TEXT_CHUNK_SIZE" \
  --models "$MODEL_ID"