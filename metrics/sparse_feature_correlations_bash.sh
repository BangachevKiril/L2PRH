#!/bin/bash
#SBATCH --export=NONE
#SBATCH --job-name=sparse_corr_split11
#SBATCH --output=logs/sparse_corr_split11_%A_%a.out
#SBATCH --error=logs/sparse_corr_split11_%A_%a.err
#SBATCH --time=11:59:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --partition=mit_normal

mkdir -p logs

# ============================================================
# How to launch
# ============================================================
# Preferred from a login node:
#   bash sparse_feature_correlations_split11_bash.sh
#
# The script computes the total number of runs and submits itself as an array:
#   one array task = one dataset, one dimension S/D, one sparsity pattern,
#                    and exactly one of the 11 computations
#                    ordinary run or one random-baseline seed.
#
# You can also submit manually with:
#   sbatch --array=0-N sparse_feature_correlations_split11_bash.sh
# where N = total_runs - 1.

# =========================
# Sweep axes
# =========================
DS_LIST=(16384)
DATASETS=("coco" "visual_genome" "cc3m")
SPARSITY_PATTERNS=("kvar")

# =========================
# Script + knobs
# =========================
BASE_SEED=0
CACHE_MODELS=4
CACHE_BINARY_MODELS=4
USE_TRUNCATED=1
SKIP_EXISTING=1
REUSE_PERM_FOR_BINARY=0    # 1 -> add --reuse_perm_for_binary 1

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

build_runs () {
  RUNS=()

  for dataset in "${DATASETS[@]}"; do
    for d in "${DS_LIST[@]}"; do
      for sparsity_pattern in "${SPARSITY_PATTERNS[@]}"; do
        # ordinary non-baseline computation
        RUNS+=("${dataset}|${d}|${sparsity_pattern}|0|${BASE_SEED}")

        # random-baseline computations, one seed per array task
        for s in 0 1 2 3 4 5 6 7 8 9; do
          seed=$((BASE_SEED + s))
          RUNS+=("${dataset}|${d}|${sparsity_pattern}|1|${seed}")
        done
      done
    done
  done
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
    python -u "sparse_feature_correlations.py"
    --d "$d"
    --out_dir "$out_dir"
    --cache_models "$CACHE_MODELS"
    --cache_binary_models "$CACHE_BINARY_MODELS"
    --use_truncated "$USE_TRUNCATED"
    --seed "$seed"
    --skip_existing "$SKIP_EXISTING"
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
  echo "GRID_ID=${SLURM_ARRAY_TASK_ID:-manual}"
  echo "DATASET=$dataset"
  echo "S_OR_D=$d"
  echo "SPARSITY_PATTERN=$sparsity_pattern"
  echo "RAND_PERMUTE_BASELINE=$rand_permute_baseline"
  echo "SEED=$seed"
  echo "OUT_DIR=$out_dir"
  echo "PY_SCRIPT=$PY_SCRIPT"
  echo "Running: ${cmd[*]}"
  echo "========================================"

  "${cmd[@]}"
}

# =========================
# Build run list and optionally self-submit
# =========================
build_runs
TOTAL_RUNS=${#RUNS[@]}

if (( TOTAL_RUNS == 0 )); then
  echo "No runs to execute."
  exit 0
fi

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "Submitting ${TOTAL_RUNS} array task(s): 0-$((TOTAL_RUNS - 1))"
  echo "One task = one dataset, one S/D, one sparsity, and one of the 11 computations."
  sbatch --array=0-$((TOTAL_RUNS - 1)) "$0" "$@"
  exit $?
fi

if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  echo "ERROR: This script must run as a SLURM array job."
  echo "Run it from a login node with: bash $0"
  echo "or manually submit with: sbatch --array=0-$((TOTAL_RUNS - 1)) $0"
  exit 1
fi

# =========================
# Environment
# =========================
module load miniforge
CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate GPUenv

GRID_ID=${SLURM_ARRAY_TASK_ID}

if (( GRID_ID < 0 || GRID_ID >= TOTAL_RUNS )); then
  echo "ERROR: SLURM_ARRAY_TASK_ID=$GRID_ID out of range [0, $((TOTAL_RUNS - 1))]"
  exit 1
fi

IFS='|' read -r dataset d sparsity_pattern rand_permute_baseline seed <<< "${RUNS[$GRID_ID]}"

# =========================
# Execute exactly one run
# =========================
echo "TOTAL_RUNS=$TOTAL_RUNS"
echo "Running exactly one assigned sub-run."
run_one "$dataset" "$d" "$sparsity_pattern" "$rand_permute_baseline" "$seed"
