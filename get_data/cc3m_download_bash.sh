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

PY_SCRIPT="cc3m_download.py"
OUT_ROOT="/home/kirilb/orcd/pool/cc3m"

SAMPLE_SIZE=300000
SEED=12345

NUM_WORKERS=${SLURM_CPUS_PER_TASK}
TIMEOUT=10
RETRIES=2
SLEEP_BASE=0.1

UA="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"

python3 "$PY_SCRIPT" \
  --out_root "$OUT_ROOT" \
  --sample_size "$SAMPLE_SIZE" \
  --seed "$SEED" \
  --num_workers "$NUM_WORKERS" \
  --timeout "$TIMEOUT" \
  --retries "$RETRIES" \
  --sleep_base "$SLEEP_BASE" \
  --user_agent "$UA"
