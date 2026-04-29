#!/bin/bash
#SBATCH --export=NONE
#SBATCH --job-name=cc3m_train_100k
#SBATCH --output=logs/cc3m_train_100k_%j.out
#SBATCH --error=logs/cc3m_train_100k_%j.err
#SBATCH --time=11:00:00
#SBATCH --cpus-per-task=12
#SBATCH --mem=32G
#SBATCH --partition=mit_normal

mkdir -p logs

module load miniforge
CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate GPUenv

BASE_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
PY_SCRIPT="cc3m_download.py"

OUT_ROOT="/home/kirilb/orcd/scratch/cc3m"
TSV_PATH="${OUT_ROOT}/Train_GCC-training.tsv"

# Mirror fallback. This is the file your job should download once if missing.
TSV_URL="https://huggingface.co/datasets/Pepperhan/CC3M/resolve/main/Train_GCC-training.tsv?download=true"

SAMPLE_SIZE=300000
SEED=12345

NUM_WORKERS=${SLURM_CPUS_PER_TASK}
TIMEOUT=10
RETRIES=2
SLEEP_BASE=0.1

UA="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"

mkdir -p "$OUT_ROOT"

if [ ! -f "$PY_SCRIPT" ]; then
  echo "[error] Python script not found: $PY_SCRIPT"
  exit 1
fi

if [ -f "$TSV_PATH" ]; then
  echo "[info] using existing local TSV: $TSV_PATH"
else
  echo "[info] local TSV missing, downloading to: $TSV_PATH"
  echo "[info] source: $TSV_URL"

  if command -v wget >/dev/null 2>&1; then
    wget -O "$TSV_PATH" "$TSV_URL"
  elif command -v curl >/dev/null 2>&1; then
    curl -L "$TSV_URL" -o "$TSV_PATH"
  else
    echo "[error] neither wget nor curl is available"
    exit 1
  fi

  if [ ! -s "$TSV_PATH" ]; then
    echo "[error] download failed or produced an empty file: $TSV_PATH"
    exit 1
  fi

  echo "[info] downloaded TSV:"
  ls -lh "$TSV_PATH"
fi

python3 "$PY_SCRIPT" \
  --out_root "$OUT_ROOT" \
  --sample_size "$SAMPLE_SIZE" \
  --seed "$SEED" \
  --tsv_path "$TSV_PATH" \
  --num_workers "$NUM_WORKERS" \
  --timeout "$TIMEOUT" \
  --retries "$RETRIES" \
  --sleep_base "$SLEEP_BASE" \
  --user_agent "$UA"