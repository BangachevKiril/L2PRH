#!/bin/bash
#SBATCH --job-name=noise_diagrams
#SBATCH --output=logs/noise_diagrams_%A_%a.out
#SBATCH --error=logs/noise_diagrams_%A_%a.err
#SBATCH --array=0-23
#SBATCH --partition=mit_preemptable
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:10:00

module load miniforge
CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate GPUenv

mkdir -p logs
mkdir -p /home/kirilb/data/L2PRH/noise_diagrams

cd /home/kirilb/data/L2PRH

python - <<'PY'
import os
import sys
import traceback
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

PRH_ROOT = "/home/kirilb/data/L2PRH"
UTILS_DIR = os.path.join(PRH_ROOT, "plotting")

for p in [PRH_ROOT, UTILS_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

import noise_plotting


llm = [
    "Qwen__Qwen3-1.7B-Base/text",
    "Qwen__Qwen3-4B-Base/text",
    "google__gemma-3-1b-it/text",
    "google__gemma-3-4b-it/text",
    "meta-llama__Llama-3.2-1B-Instruct/text",
    "meta-llama__Llama-3.2-3B-Instruct/text",
]

text = [
    "BAAI__bge-base-en-v1.5/text",
    "BAAI__bge-large-en-v1.5/text",
    "nomic-ai__nomic-embed-text-v1.5/text",
    "nomic-ai__nomic-embed-text-v2-moe/text",
    "codefuse-ai__F2LLM-1.7B/text",
    "codefuse-ai__F2LLM-4B/text",
]

_HUMAN_TITLES = {
    "CKA_HSIC": "CKA",
    "CKA_unbiased": "Unbiased CKA",
    "SVCCA_1": "SVCCA 10",
    "SVCCA_2": "SVCCA 100",
    "TOPK10": "KNN Overlap 10",
    "TOPK100": "KNN Overlap 100",
    "KNN_EDIT_10": "KNN-10 Edit",
    "KNN_EDIT_100": "KNN-100 Edit",
}

models = [text, llm]
model_types = ["text", "llm"]
pairs = [(0, 0), (0, 1), (1, 1)]

abbreviated_model_names = [
    "bge-base-v1.5",
    "bge-large-v1.5",
    "nomic-v1.5",
    "nomic-v2",
    "F2LLM-1.7B",
    "F2LLM-4B",
    "Qwen3-1.7B",
    "Qwen3-4B",
    "gemma-3-1B",
    "gemma-3-4B",
    "Llama-3.2-1B",
    "Llama-3.2-3B",
]

text_abbrev = abbreviated_model_names[:6]
llm_abbrev = abbreviated_model_names[6:]
abbrev_models = [text_abbrev, llm_abbrev]


def safe_filename(s: str) -> str:
    out = str(s)
    for ch in [" ", "/", "\\", ":", ";", ",", "(", ")", "[", "]", "{", "}", "'"]:
        out = out.replace(ch, "_")
    while "__" in out:
        out = out.replace("__", "_")
    return out.strip("_")


metric_keys = list(_HUMAN_TITLES.keys())
jobs = []
for metric in metric_keys:
    for pair in pairs:
        jobs.append((metric, pair))

task_id = int(os.environ["SLURM_ARRAY_TASK_ID"])
if task_id < 0 or task_id >= len(jobs):
    raise ValueError(f"Invalid SLURM_ARRAY_TASK_ID={task_id}, expected 0..{len(jobs)-1}")

metric, (i, j) = jobs[task_id]

list_1 = models[i]
list_2 = models[j]
abbreviated_model_names_1 = abbrev_models[i]
abbreviated_model_names_2 = abbrev_models[j]

rolling_metrics_dir = "/home/kirilb/orcd/scratch/PRH_data/metrics_embedded_words_centered_single_token_common_indices/"
output_dir = Path("/home/kirilb/data/L2PRH/noise_diagrams_single_token_common_indices")
output_dir.mkdir(parents=True, exist_ok=True)

title = f"{_HUMAN_TITLES[metric]} Alignment vs Word Frequency, {model_types[i]} vs {model_types[j]}"
filename = safe_filename(
    f"{_HUMAN_TITLES[metric]}_with_regression_{model_types[i]}_{model_types[j]}"
) + ".pdf"
save_path = output_dir / filename

same_group = (i == j)
allow_same_family = not same_group

print("=" * 80)
print(f"SLURM_ARRAY_TASK_ID       = {task_id}")
print(f"metric                    = {metric}")
print(f"pair                      = ({model_types[i]}, {model_types[j]})")
print(f"save_path                 = {save_path}")
print(f"allow_same_family         = {allow_same_family}")
print("=" * 80, flush=True)

try:
    out = noise_plotting.plot_single_metric_with_regression_and_heatmaps(
        metric_key=metric,
        list_1=list_1,
        list_2=list_2,
        rolling_metrics_dir=rolling_metrics_dir,
        freq_batch=500,
        freq_step=250,
        normalization="raw",
        save_path=str(save_path),
        figsize=(20, 8),
        title=title,
        from_same_family=allow_same_family,
        close_plot=True,
        show_plot=False,
        abbreviated_model_names_1=abbreviated_model_names_1,
        abbreviated_model_names_2=abbreviated_model_names_2,
    )
    print(f"[done] {save_path}", flush=True)
    print(f"[summary] found_pairs = {out.get('found_pairs')}", flush=True)

except Exception as e:
    print(f"[failed] task_id={task_id}", flush=True)
    print(str(e), flush=True)
    traceback.print_exc()
    raise

PY