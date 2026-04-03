#!/bin/bash
#SBATCH --export=NONE
#SBATCH --job-name=wordfreq_coco
#SBATCH --output=logs/wordfreq_coco_%j.out
#SBATCH --error=logs/wordfreq_coco_%j.err
#SBATCH --time=00:20:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --partition=mit_normal

mkdir -p logs

module load miniforge
CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate GPUenv

# ---------------- user config ----------------
OUT_ROOT="/home/kirilb/orcd/pool/wordfreq_as_coco"
N=50000

python words_download.py \
  --out_root "$OUT_ROOT" \
  --N "$N" \
  --lang en \
  --wordlist best \
  --split_dir train2017 \
  --ann_name captions_train2017.json \
  --write_csv 1 \
  --csv_name top50k_words.csv \
  --include_freq_in_caption 0
