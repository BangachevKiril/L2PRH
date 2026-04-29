#!/bin/bash
#SBATCH --export=NONE
#SBATCH --job-name=common_single_token_indices
#SBATCH --output=logs/common_single_token_indices_%j.out
#SBATCH --error=logs/common_single_token_indices_%j.err
#SBATCH --time=00:20:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --partition=mit_normal

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
WORDS_ROOT="/home/kirilb/orcd/scratch/words"
SINGLE_TOKEN_LLM_ROOT="/home/kirilb/orcd/scratch/PRH_data/single_token_words_llm"
SINGLE_TOKEN_TEXT_ROOT="/home/kirilb/orcd/scratch/PRH_data/single_token_words_text"

ANN_REL_PATH="annotations/captions_train2017.json"
PREFIX="single_token_common"

# By default, write the index/frequency files directly inside WORDS_ROOT.
OUTPUT_DIR="$WORDS_ROOT"

if [[ ! -f "${WORDS_ROOT}/${ANN_REL_PATH}" ]]; then
  echo "Original words annotations not found: ${WORDS_ROOT}/${ANN_REL_PATH}"
  exit 1
fi

if [[ ! -f "${WORDS_ROOT}/frequencies.npy" ]]; then
  echo "Original frequencies.npy not found: ${WORDS_ROOT}/frequencies.npy"
  exit 1
fi

echo "WORDS_ROOT=${WORDS_ROOT}"
echo "SINGLE_TOKEN_LLM_ROOT=${SINGLE_TOKEN_LLM_ROOT}"
echo "SINGLE_TOKEN_TEXT_ROOT=${SINGLE_TOKEN_TEXT_ROOT}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "PYTHON_SCRIPT=${PYTHON_SCRIPT}"

python "get_only_single_token_words.py" \
  --words_root "$WORDS_ROOT" \
  --single_token_llm_root "$SINGLE_TOKEN_LLM_ROOT" \
  --single_token_text_root "$SINGLE_TOKEN_TEXT_ROOT" \
  --ann_rel_path "$ANN_REL_PATH" \
  --output_dir "$OUTPUT_DIR" \
  --prefix "$PREFIX"
