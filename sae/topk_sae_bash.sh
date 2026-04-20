#!/bin/bash
#SBATCH --export=NONE
#SBATCH --job-name=topk_sae
#SBATCH --output=logs/alive_topk_sae_%A_%a.out
#SBATCH --error=logs/alive_topk_sae_%A_%a.err
#SBATCH --time=05:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:1
#SBATCH --array=0-29

# --------------------- User config ---------------------

DATASET="visual_genome"  # "coco", "cc3m", "visual_genome", "words"

EMBEDDED_COCO_ROOT="/home/kirilb/orcd/pool/PRH_data/embedded_${DATASET}"
BASE_OUTPUT_DIR="/home/kirilb/orcd/pool/PRH_data/topk_sae_${DATASET}"

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

HIDDEN_DIMS=(8192 16384)
K_VALUES=(32 64 128)

MODEL_SPEC="${MODEL_SPECS[$SLURM_ARRAY_TASK_ID]}"
SEED=1

BATCH_SIZE=2048
NUM_STEPS=50000
LR=0.0004
PRINT_EVERY=1000
SCHEDULER=1   # 0 = constant LR (old behavior), 1 = cosine annealing

# --------------------- Environment ---------------------

module load miniforge
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate GPUenv

# --------------------- Run -----------------------------

mkdir -p "$BASE_OUTPUT_DIR"
mkdir -p logs

echo "Starting SAE sweep for MODEL_SPEC=${MODEL_SPEC}"
echo "Using SCHEDULER=${SCHEDULER}"

EXTRA_ARGS=(
  --embedded_coco_root "$EMBEDDED_COCO_ROOT"
  --renorm_decoder
  --mmap
  --scheduler "$SCHEDULER"
)

for DIM in "${HIDDEN_DIMS[@]}"; do
  for K in "${K_VALUES[@]}"; do

    if (( 3 * K >= DIM )); then
      echo "Skipping DIM=${DIM}, K=${K} (Sparsity ratio too low)"
      continue
    fi

    echo "Training: DIM=${DIM} | K=${K} | scheduler=${SCHEDULER}"

    python -u topk_sae_with_scheduler.py \
      --model_spec "$MODEL_SPEC" \
      --output_dir "$BASE_OUTPUT_DIR" \
      --hidden_dim "$DIM" \
      --k "$K" \
      --batch_size "$BATCH_SIZE" \
      --num_steps "$NUM_STEPS" \
      --lr "$LR" \
      --seed "$SEED" \
      --print_every "$PRINT_EVERY" \
      "${EXTRA_ARGS[@]}"

  done
done
