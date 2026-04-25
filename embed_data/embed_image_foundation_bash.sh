#!/bin/bash
#SBATCH --export=NONE
#SBATCH --job-name=embed_image_foundation
#SBATCH --output=logs/embed_image_foundation_%A_%a.out
#SBATCH --error=logs/embed_image_foundation_%A_%a.err
#SBATCH --time=05:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:1
#SBATCH --array=0-5

mkdir -p logs

module load miniforge
CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate GPUenv


DATASET="cc3m" # "coco", "cc3m", "visual_genome", "words"
IN_ROOT="/home/kirilb/orcd/scratch/${DATASET}"
OUT_ROOT="/home/kirilb/orcd/scratch/PRH_data/embedded_${DATASET}"

HF_CACHE_DIR="/home/kirilb/orcd/pool/huggingface_models_cache"

BATCH_SIZE=128
DEVICE="cuda"
DTYPE="fp16"
LOAD_DTYPE="fp16"
POOLING="auto"

NUM_WORKERS=${SLURM_CPUS_PER_TASK}
PREFETCH=4

MODELS=(
  "facebook/vit-mae-large"
  "facebook/vit-mae-huge"
  "facebook/dinov2-base"
  "facebook/dinov2-large"
  "microsoft/beit-base-patch16-224"
  "microsoft/beit-large-patch16-224"
)

TASK_ID=${SLURM_ARRAY_TASK_ID}
NUM_MODELS=${#MODELS[@]}

if [[ ${TASK_ID} -ge ${NUM_MODELS} ]]; then
  echo "Task ${TASK_ID} >= NUM_MODELS ${NUM_MODELS}. Exiting."
  exit 0
fi

MODEL_ID="${MODELS[$TASK_ID]}"
echo "MODEL_ID=${MODEL_ID}"

mkdir -p "$OUT_ROOT" "$HF_CACHE_DIR"
export HF_HOME="$HF_CACHE_DIR"
export HUGGINGFACE_HUB_CACHE="$HF_CACHE_DIR/hub"
export TRANSFORMERS_CACHE="$HF_CACHE_DIR/transformers"
export HF_DATASETS_CACHE="$HF_CACHE_DIR/datasets"
export HF_HUB_DISABLE_TELEMETRY=1

python embed_image_foundation.py \
  --image_root "$IN_ROOT" \
  --out_root "$OUT_ROOT" \
  --batch_size "$BATCH_SIZE" \
  --device "$DEVICE" \
  --dtype "$DTYPE" \
  --load_dtype "$LOAD_DTYPE" \
  --pooling "$POOLING" \
  --hf_cache_dir "$HF_CACHE_DIR" \
  --num_workers "$NUM_WORKERS" \
  --prefetch_factor "$PREFETCH" \
  --models "$MODEL_ID"