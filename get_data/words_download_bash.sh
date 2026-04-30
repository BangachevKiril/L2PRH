#!/bin/bash
#SBATCH --export=NONE
#SBATCH --job-name=wordfreq_coco_pos
#SBATCH --output=logs/wordfreq_coco_pos_%j.out
#SBATCH --error=logs/wordfreq_coco_pos_%j.err
#SBATCH --time=00:20:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --partition=mit_preemptable

mkdir -p logs

module load miniforge
CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate GPUenv

# ---------------- user config ----------------
OUT_ROOT="/home/kirilb/orcd/scratch/words"
N=50000


# Clean POS buckets for the ablation: one primary isolated-token Penn POS per word.
# With this setting, words such as I/this/as are not placed in noun/verb/adjective files,
# while was is placed in verb_indices.txt.
POS_INCLUDE_AMBIGUOUS=0

# Recommended: fail rather than silently produce weak heuristic POS labels.
# Set to 0 only if you intentionally want the fallback heuristic.
STRICT_NLTK_TAGGER=1

# Set to 1 if the environment has internet access and needs NLTK tagger data.
TRY_DOWNLOAD_NLTK_DATA=1


python -u "words_download.py" \
  --out_root "$OUT_ROOT" \
  --N "$N" \
  --lang en \
  --wordlist best \
  --split_dir train2017 \
  --ann_name captions_train2017.json \
  --write_csv 1 \
  --csv_name top50k_words.csv \
  --include_freq_in_caption 0 \
  --write_pos_indices 1 \
  --pos_include_ambiguous "$POS_INCLUDE_AMBIGUOUS" \
  --strict_nltk_tagger "$STRICT_NLTK_TAGGER" \
  --try_download_nltk_data "$TRY_DOWNLOAD_NLTK_DATA" \
  --noun_indices_name noun_indices.txt \
  --verb_indices_name verb_indices.txt \
  --adjective_indices_name adjective_indices.txt \
  --print_pos_examples 1
