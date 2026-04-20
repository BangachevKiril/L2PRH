#!/bin/bash
#SBATCH --export=NONE
#SBATCH --job-name=subsample_metrics_donly
#SBATCH --output=logs/subsample_metrics_donly_%A_%a.out
#SBATCH --error=logs/subsample_metrics_donly_%A_%a.err
#SBATCH --time=05:59:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:1
#SBATCH --array=0-7   # set to 0-(#dims * #sparsities * #normalizations - 1)

mkdir -p logs

module load miniforge
CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate GPUenv

PY_SCRIPT="sparse_feature_metrics.py"

DATASET = "visual_genome" # "coco", "cc3m", "visual_genome", "words"
TOPK_ROOT="/home/kirilb/orcd/pool/alt_sae_prh/batchtopk_sae/topk_sae_${DATASET}/"

OUT_TOPK_ROOT="/home/kirilb/orcd/pool/PRH_data/metrics_embedded_${DATASET}"

HOW_MANY_SAMPLES=10 # number of subsamples to evaluate
SUBSAMPLE_SIZE=1000 # size of each subsample
SEED_BASE=12345 # seed for random number generator
DEVICE="cuda" # device to use for computation

IS_BINARY=0 # 0: weighted, 1: binary
PROFILE_METRICS=0 # 0: no profiling, 1: profiling

SVCCA_DIM1=10
SVCCA_DIM2=100
TOPK_K1=10
TOPK_K2=100
EDIT_K1=10
EDIT_K2=100

# --------------------- sweep over D, sparsity pattern, and normalization ---------------------
DS_LIST=(8192 16384)
SPARSITY_PATTERNS=("var") # "32" "64" "128" sparsity patterns; There are custom sparsity patterns for each model defined below. We train each of them
USE_TRUNCATED_LIST=(1) # 0: full, 1: truncated

ND=${#DS_LIST[@]}
NS=${#SPARSITY_PATTERNS[@]}
NU=${#USE_TRUNCATED_LIST[@]}
TOTAL_JOBS=$((ND * NS * NU))

GRID_ID=${SLURM_ARRAY_TASK_ID:-0}
if (( GRID_ID < 0 || GRID_ID >= TOTAL_JOBS )); then
  echo "ERROR: SLURM_ARRAY_TASK_ID=$GRID_ID out of range [0, $((TOTAL_JOBS-1))]"
  echo "Set: #SBATCH --array=0-$((TOTAL_JOBS-1))"
  exit 1
fi

D_IDX=$(( GRID_ID / (NS * NU) ))
REM=$(( GRID_ID % (NS * NU) ))
S_IDX=$(( REM / NU ))
U_IDX=$(( REM % NU ))

D="${DS_LIST[$D_IDX]}"
SPARSITY_PATTERN="${SPARSITY_PATTERNS[$S_IDX]}"
USE_TRUNCATED="${USE_TRUNCATED_LIST[$U_IDX]}"

# --------------------- suffix depends on use_normalized ---------------------
if [[ "$USE_TRUNCATED" -eq 0 ]]; then
  NORM_TAG="full"
elif [[ "$USE_TRUNCATED" -eq 1 ]]; then
  NORM_TAG="truncated"
else
  echo "ERROR: USE_NORMALIZED must be 0 or 1, got: $USE_NORMALIZED"
  exit 1
fi

# --------------------- datasets ---------------------
DATASETS=(
  "codefuse-ai__F2LLM-1.7B/text"
  "codefuse-ai__F2LLM-4B/text"
  "Qwen__Qwen3-1.7B-Base/text"
  "Qwen__Qwen3-4B-Base/text"
  "google__gemma-3-1b-it/text"
  "google__gemma-3-4b-it/text"
  "meta-llama__Llama-3.2-1B-Instruct/text"
  "meta-llama__Llama-3.2-3B-Instruct/text"
  "BAAI__bge-base-en-v1.5/text"
  "BAAI__bge-large-en-v1.5/text"
  "nomic-ai__nomic-embed-text-v1.5/text"
  "nomic-ai__nomic-embed-text-v2-moe/text"
  "google__siglip2-base-patch16-256/text"
  "google__siglip2-large-patch16-256/text"
  "laion__CLIP-ViT-B-32-laion2B-s34B-b79K/text"
  "laion__CLIP-ViT-H-14-laion2B-s32B-b79K/text"
  "openai__clip-vit-base-patch32/text"
  "openai__clip-vit-large-patch14/text"
  "openai__clip-vit-base-patch32/img"
  "openai__clip-vit-large-patch14/img"
  "google__siglip2-base-patch16-256/img"
  "google__siglip2-large-patch16-256/img"
  "laion__CLIP-ViT-B-32-laion2B-s34B-b79K/img"
  "laion__CLIP-ViT-H-14-laion2B-s32B-b79K/img"
  "facebook__dinov2-base/img"
  "facebook__dinov2-large/img"
  "facebook__vit-mae-huge/img"
  "facebook__vit-mae-large/img"
  "microsoft__beit-base-patch16-224/img"
  "microsoft__beit-large-patch16-224/img"
)

# --------------------- choose K_LIST from sparsity pattern ---------------------
case "$SPARSITY_PATTERN" in
  var)
    K_LIST=(
      32 32 32 32 32 32 32 32 32 32 32 32
      64 64 64 64 64 64 64 64 64 64 64 64
      128 128 128 128 128 128
    )
    ;;
  32)
    K_LIST=(
      32 32 32 32 32 32 32 32 32 32
      32 32 32 32 32 32 32 32 32 32
      32 32 32 32 32 32 32 32 32 32
    )
    ;;
  64)
    K_LIST=(
      64 64 64 64 64 64 64 64 64 64
      64 64 64 64 64 64 64 64 64 64
      64 64 64 64 64 64 64 64 64 64
    )
    ;;
  128)
    K_LIST=(
      128 128 128 128 128 128 128 128 128 128
      128 128 128 128 128 128 128 128 128 128
      128 128 128 128 128 128 128 128 128 128
    )
    ;;
  *)
    echo "ERROR: SPARSITY_PATTERN must be one of: var, 32, 64, 128"
    echo "Got: $SPARSITY_PATTERN"
    exit 1
    ;;
esac

SPARSITY_TAG="k${SPARSITY_PATTERN}"

# --------------------- output dir name depends on binary/weighted + normalization tag ---------------------
if [[ "$IS_BINARY" -eq 1 ]]; then
  OUT_DIR="${OUT_TOPK_ROOT}/binary_d_${D}_${SPARSITY_TAG}_${NORM_TAG}"
else
  OUT_DIR="${OUT_TOPK_ROOT}/weighted_d_${D}_${SPARSITY_TAG}_${NORM_TAG}"
fi

echo "PWD=$(pwd)"
echo "which python=$(which python)"
python -V
echo "PY_SCRIPT=$PY_SCRIPT"
ls -l "$PY_SCRIPT" || { echo "ERROR: cannot find $PY_SCRIPT in $(pwd)"; exit 1; }

echo "GRID_ID=$GRID_ID / $((TOTAL_JOBS-1))"
echo "D_IDX=$D_IDX / $((ND-1)) -> D=$D"
echo "S_IDX=$S_IDX / $((NS-1)) -> SPARSITY_PATTERN=$SPARSITY_PATTERN"
echo "U_IDX=$U_IDX / $((NU-1)) -> USE_NORMALIZED=$USE_NORMALIZED ($NORM_TAG)"
echo "OUT_DIR=$OUT_DIR"

N=${#DATASETS[@]}
if (( ${#K_LIST[@]} != N )); then
  echo "ERROR: K_LIST length (${#K_LIST[@]}) must equal DATASETS length ($N)"
  exit 1
fi

NUM_PAIRS=$(( N * (N - 1) / 2 ))
echo "Total datasets: $N"
echo "Total unordered pairs: $NUM_PAIRS"

COMMON_ARGS=(
  --topk_root "$TOPK_ROOT"
  --output_dir "$OUT_DIR"
  --how_many_samples "$HOW_MANY_SAMPLES"
  --subsample_size "$SUBSAMPLE_SIZE"
  --device "$DEVICE"
  --d "$D"
  --use_normalized "$USE_NORMALIZED"
  --svcca_dim1 "$SVCCA_DIM1"
  --svcca_dim2 "$SVCCA_DIM2"
  --topk_k1 "$TOPK_K1"
  --topk_k2 "$TOPK_K2"
  --edit_k1 "$EDIT_K1"
  --edit_k2 "$EDIT_K2"
)

if [[ "$IS_BINARY" -eq 1 ]]; then
  COMMON_ARGS+=(--is_binary)
fi

if [[ "$PROFILE_METRICS" -eq 1 ]]; then
  COMMON_ARGS+=(--profile_metrics)
fi

dataset_to_modelkey () {
  local ds="$1"
  local model="${ds%/*}"
  local kind="${ds##*/}"   # text or img
  echo "${model}_${kind}"
}

dataset_index_to_k () {
  local idx="$1"
  echo "${K_LIST[$idx]}"
}

modelkey_to_npzpath () {
  local modelkey="$1"
  local k_local="$2"
  local fname
  local candidates
  local path

  if [[ "$USE_NORMALIZED" -eq 0 ]]; then
    fname="X_features.npz"
  else
    fname="X_features_truncated.npz"
  fi

  candidates=(
    "${TOPK_ROOT}/topk_${D}_${modelkey}_k_${k_local}/${fname}"
    "${TOPK_ROOT}/batchtopk_${D}_${modelkey}_k_${k_local}/${fname}"
  )

  for path in "${candidates[@]}"; do
    if [[ -f "$path" ]]; then
      echo "$path"
      return 0
    fi
  done

  # fall back to the legacy path for clearer downstream error messages
  echo "${candidates[0]}"
}

pair_index_to_ij () {
  local kpair=$1
  local rem=$kpair
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
  local kpair=$1
  local ij
  ij=$(pair_index_to_ij "$kpair") || exit 1

  local i j
  i=$(echo "$ij" | awk '{print $1}')
  j=$(echo "$ij" | awk '{print $2}')

  local DS1="${DATASETS[$i]}"
  local DS2="${DATASETS[$j]}"

  local M1 M2
  M1=$(dataset_to_modelkey "$DS1")
  M2=$(dataset_to_modelkey "$DS2")

  local K1 K2
  K1=$(dataset_index_to_k "$i")
  K2=$(dataset_index_to_k "$j")

  if ! [[ "$K1" =~ ^[0-9]+$ ]] || (( K1 <= 0 )); then
    echo "ERROR: bad K1='$K1' for index $i ($M1)"
    exit 1
  fi
  if ! [[ "$K2" =~ ^[0-9]+$ ]] || (( K2 <= 0 )); then
    echo "ERROR: bad K2='$K2' for index $j ($M2)"
    exit 1
  fi

  local P1 P2
  P1=$(modelkey_to_npzpath "$M1" "$K1")
  P2=$(modelkey_to_npzpath "$M2" "$K2")

  local OUT_NAME="${M1}_k${K1}__${M2}_k${K2}.npz"
  local OUT_PATH="${OUT_DIR%/}/$OUT_NAME"

  local SEED=$((SEED_BASE + kpair + 100000*GRID_ID + 10000000*K1 + 1000000000*K2))

  echo "------------------------------------------------------------"
  echo "GRID_ID=$GRID_ID (D=$D, sparsity=$SPARSITY_PATTERN, use_normalized=$USE_NORMALIZED) | pair k=$kpair (i=$i, j=$j)"
  echo "Comparing: $M1 (k=$K1)  vs  $M2 (k=$K2)"
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

  if [[ ! -f "${OUT_DIR%/}/K_LIST.txt" ]]; then
    printf "%s\n" "${DATASETS[@]}" | paste -d' ' - <(printf "%s\n" "${K_LIST[@]}") > "${OUT_DIR%/}/K_LIST.txt"
  fi

  python -u "$PY_SCRIPT" "$M1" "$M2" \
    --seed "$SEED" \
    --output_name "$OUT_NAME" \
    --k1 "$K1" \
    --k2 "$K2" \
    "${COMMON_ARGS[@]}"

  if [[ ! -s "$OUT_PATH" ]]; then
    echo "ERROR: python returned but output file not created: $OUT_PATH"
    exit 1
  fi

  echo "Done: $M1 vs $M2"
}

for ((kpair=0; kpair<NUM_PAIRS; kpair++)); do
  run_pair_k "$kpair"
done