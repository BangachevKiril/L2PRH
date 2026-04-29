#!/bin/bash
#SBATCH --export=NONE
#SBATCH --job-name=sparse_corr_balanced
#SBATCH --output=logs/sparse_corr_balanced_%A_%a.out
#SBATCH --error=logs/sparse_corr_balanced_%A_%a.err
#SBATCH --time=11:59:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --partition=mit_normal
#SBATCH --array=0-48

mkdir -p logs

# =========================
# Environment
# =========================
module load miniforge
CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate GPUenv

# =========================
# Sweep axes
# =========================
DS_LIST=(8192 16384)
ND=${#DS_LIST[@]}

DATASETS=("coco" "visual_genome" "cc3m")
N_DATASETS=${#DATASETS[@]}

SPARSITY_PATTERNS=("kvar" "k_32" "k_64" "k_128")
N_SPARSITY=${#SPARSITY_PATTERNS[@]}

# =========================
# Script + knobs
# =========================
PY_SCRIPT="sparse_feature_correlations.py"

REUSE_PERM_FOR_BINARY=0    # 1 -> add --reuse_perm_for_binary
BASE_SEED=0
CACHE_MODELS=4
USE_TRUNCATED=1

# Fixed array width
N_ARRAY_JOBS=66
GRID_ID=${SLURM_ARRAY_TASK_ID:-0}

if (( GRID_ID < 0 || GRID_ID >= N_ARRAY_JOBS )); then
  echo "ERROR: SLURM_ARRAY_TASK_ID=$GRID_ID out of range [0, $((N_ARRAY_JOBS-1))]"
  exit 1
fi

# =========================
# Models list
# =========================
MODELS=(
  "meta-llama__Llama-3.2-1B-Instruct_text"
  "meta-llama__Llama-3.2-3B-Instruct_text"
  "Qwen__Qwen3-1.7B-Base_text"
  "Qwen__Qwen3-4B-Base_text"
  "google__gemma-3-1b-it_text"
  "google__gemma-3-4b-it_text"
  "BAAI__bge-base-en-v1.5_text"
  "BAAI__bge-large-en-v1.5_text"
  "codefuse-ai__F2LLM-1.7B_text"
  "codefuse-ai__F2LLM-4B_text"
  "nomic-ai__nomic-embed-text-v1.5_text"
  "nomic-ai__nomic-embed-text-v2-moe_text"
  "google__siglip2-base-patch16-256_text"
  "google__siglip2-large-patch16-256_text"
  "laion__CLIP-ViT-B-32-laion2B-s34B-b79K_text"
  "laion__CLIP-ViT-H-14-laion2B-s32B-b79K_text"
  "openai__clip-vit-base-patch32_text"
  "openai__clip-vit-large-patch14_text"
  "google__siglip2-base-patch16-256_img"
  "google__siglip2-large-patch16-256_img"
  "laion__CLIP-ViT-B-32-laion2B-s34B-b79K_img"
  "laion__CLIP-ViT-H-14-laion2B-s32B-b79K_img"
  "openai__clip-vit-base-patch32_img"
  "openai__clip-vit-large-patch14_img"
  "microsoft__beit-base-patch16-224_img"
  "microsoft__beit-large-patch16-224_img"
  "facebook__dinov2-base_img"
  "facebook__dinov2-large_img"
  "facebook__vit-mae-huge_img"
  "facebook__vit-mae-large_img"
)

# =========================
# Per-model K lists
# =========================
K_LIST=(
  32 32 32 32 32 32 32 32 32 32 32 32
  64 64 64 64 64 64 64 64 64 64 64 64
  128 128 128 128 128 128
)

K_LIST_32=(
  32 32 32 32 32 32 32 32 32 32 32 32
  32 32 32 32 32 32 32 32 32 32 32 32
  32 32 32 32 32 32
)

K_LIST_64=(
  64 64 64 64 64 64 64 64 64 64 64 64
  64 64 64 64 64 64 64 64 64 64 64 64
  64 64 64 64 64 64
)

K_LIST_128=(
  128 128 128 128 128 128 128 128 128 128 128 128
  128 128 128 128 128 128 128 128 128 128 128 128
  128 128 128 128 128 128
)

NM=${#MODELS[@]}

# =========================
# Helpers
# =========================
get_k_list_name () {
  local sparsity_pattern="$1"
  case "$sparsity_pattern" in
    "kvar")  echo "K_LIST" ;;
    "k_32")  echo "K_LIST_32" ;;
    "k_64")  echo "K_LIST_64" ;;
    "k_128") echo "K_LIST_128" ;;
    *)
      echo "ERROR: unknown SPARSITY_PATTERN=$sparsity_pattern" >&2
      exit 1
      ;;
  esac
}

run_one () {
  local dataset="$1"
  local d="$2"
  local sparsity_pattern="$3"
  local rand_permute_baseline="$4"
  local seed="$5"

  local root="/home/kirilb/orcd/scratch/PRH_data/topk_sae_${dataset}/"
  local base_out_prefix="/home/kirilb/orcd/scratch/PRH_data/topk_sae_${dataset}_correlations/d_${d}_${sparsity_pattern}"
  local out_dir
  local k_list_name
  local -a k_list_active
  local -a cmd

  k_list_name=$(get_k_list_name "$sparsity_pattern")
  eval "k_list_active=(\"\${${k_list_name}[@]}\")"

  if (( ${#k_list_active[@]} != NM )); then
    echo "ERROR: active K list length (${#k_list_active[@]}) must equal MODELS length ($NM)"
    exit 1
  fi

  if [[ "$rand_permute_baseline" -eq 1 ]]; then
    out_dir="${base_out_prefix}_rand_baseline_seed_${seed}"
  else
    out_dir="${base_out_prefix}"
  fi

  mkdir -p "$out_dir"

  cmd=(
    python -u "$PY_SCRIPT"
    --d "$d"
    --out_dir "$out_dir"
    --cache_models "$CACHE_MODELS"
    --use_truncated "$USE_TRUNCATED"
    --seed "$seed"
    --models "${MODELS[@]}"
    --ks "${k_list_active[@]}"
    --root "$root"
  )

  if [[ "${REUSE_PERM_FOR_BINARY:-0}" -eq 1 ]]; then
    cmd+=(--reuse_perm_for_binary 1)
  fi

  if [[ "$rand_permute_baseline" -eq 1 ]]; then
    cmd+=(--rand_permute_baseline 1)
  fi

  echo "========================================"
  echo "DATASET=$dataset"
  echo "D=$d"
  echo "SPARSITY_PATTERN=$sparsity_pattern"
  echo "RAND_PERMUTE_BASELINE=$rand_permute_baseline"
  echo "SEED=$seed"
  echo "OUT_DIR=$out_dir"
  echo "Running: ${cmd[*]}"
  echo "========================================"

  "${cmd[@]}"
}

# =========================
# Build the full run list
# Each run = one call to compare_SAEs_different_k_updated.py
# Format per entry:
#   dataset|d|sparsity_pattern|rand_permute_baseline|seed
# =========================
RUNS=()

for dataset in "${DATASETS[@]}"; do
  for d in "${DS_LIST[@]}"; do
    for sparsity_pattern in "${SPARSITY_PATTERNS[@]}"; do
      # ordinary run
      RUNS+=("${dataset}|${d}|${sparsity_pattern}|0|${BASE_SEED}")

      # baseline runs, seeds 0..9 shifted by BASE_SEED
      for s in 0 1 2 3 4 5 6 7 8 9; do
        seed=$((BASE_SEED + s))
        RUNS+=("${dataset}|${d}|${sparsity_pattern}|1|${seed}")
      done
    done
  done
done

TOTAL_RUNS=${#RUNS[@]}

if (( TOTAL_RUNS == 0 )); then
  echo "No runs to execute."
  exit 0
fi

# =========================
# Even split across 48 array jobs
# Job j gets runs:
#   [ floor(j*k/48), ..., floor((j+1)*k/48)-1 ]
# Hence each job gets k//48 or k//48 + 1 runs
# =========================
START_IDX=$(( GRID_ID * TOTAL_RUNS / N_ARRAY_JOBS ))
END_EXCL=$(( (GRID_ID + 1) * TOTAL_RUNS / N_ARRAY_JOBS ))
N_LOCAL=$(( END_EXCL - START_IDX ))

echo "GRID_ID=$GRID_ID / $((N_ARRAY_JOBS - 1))"
echo "TOTAL_RUNS=$TOTAL_RUNS"
echo "Assigned run indices: [$START_IDX, $((END_EXCL - 1))]"
echo "N_LOCAL=$N_LOCAL"

if (( N_LOCAL <= 0 )); then
  echo "This array task has no assigned runs."
  exit 0
fi

# =========================
# Execute assigned runs
# =========================
for (( idx=START_IDX; idx<END_EXCL; idx++ )); do
  IFS='|' read -r dataset d sparsity_pattern rand_permute_baseline seed <<< "${RUNS[$idx]}"
  run_one "$dataset" "$d" "$sparsity_pattern" "$rand_permute_baseline" "$seed"
done