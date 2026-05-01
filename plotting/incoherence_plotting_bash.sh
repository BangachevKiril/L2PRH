#!/bin/bash
#SBATCH --job-name=incoh_pdf
#SBATCH --output=logs/incoh_pdf_%A_%a.out
#SBATCH --error=logs/incoh_pdf_%A_%a.err
#SBATCH --array=0-17
#SBATCH --time=02:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2

mkdir -p logs
mkdir -p /home/kirilb/data/L2PRH/incoherence_diagrams/

module load miniforge
conda activate GPUenv

PARAMS=(
  "coco|COCO|8192|32"
  "coco|COCO|8192|64"
  "coco|COCO|8192|128"
  "coco|COCO|16384|32"
  "coco|COCO|16384|64"
  "coco|COCO|16384|128"
  "cc3m|CC3M|8192|32"
  "cc3m|CC3M|8192|64"
  "cc3m|CC3M|8192|128"
  "cc3m|CC3M|16384|32"
  "cc3m|CC3M|16384|64"
  "cc3m|CC3M|16384|128"
  "visual_genome|Visual Genome|8192|32"
  "visual_genome|Visual Genome|8192|64"
  "visual_genome|Visual Genome|8192|128"
  "visual_genome|Visual Genome|16384|32"
  "visual_genome|Visual Genome|16384|64"
  "visual_genome|Visual Genome|16384|128"
)

IFS='|' read DATASET DATASET_TITLE D K <<< "${PARAMS[$SLURM_ARRAY_TASK_ID]}"

python - <<PY
import os
import sys

sys.path.append("/home/kirilb/data/PRH")
import incoherence_plotting as ip

dataset = "${DATASET}"
dataset_title = "${DATASET_TITLE}"
base = "/home/kirilb/data/L2PRH/incoherence_diagrams/"
d = ${D}
k = ${K}

os.makedirs(base, exist_ok=True)

roots = [
    f"/home/kirilb/orcd/scratch/PRH_data/topk_sae_{dataset}",
    f"/home/kirilb/orcd/scratch/PRH_data/embedded_{dataset}",
]

xs, ys, paths = ip.scatter_incoherence_statistics(
    root_dir=roots,
    sparse_dimension_full=d,
    sparse_k_full=k,
    metric_1="dense_dimension",
    dimension="G_mean_abs_offdiag",
    partial_title=f"Incoherence of {dataset_title} Dictionaries and Raw Embeddings",
    ylabel="Mean Absolute Value of\nOff-Diagonal Inner Products",
    xlabel="Dimension of Dense Representation",
    save_path=f"{base}truncated_incoherence_{dataset}_{d}_{k}.pdf",
    verbose=True,
    close_after_plot=True,
)
PY
