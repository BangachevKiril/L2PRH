#!/bin/bash
#SBATCH -J bias_diag
#SBATCH -p mit_preemptable
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:05:00
#SBATCH --array=0-23
#SBATCH -o logs/bias_diag_%A_%a.out
#SBATCH -e logs/bias_diag_%A_%a.err

module load miniforge
CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate plotting_env

mkdir -p logs

PLOT_SCRIPT="metric_plotting.py"
OUT_DIR="/home/kirilb/data/L2PRH/bias_diagrams"
PRH_DATA_ROOT="/home/kirilb/orcd/scratch/PRH_data"

mkdir -p "$OUT_DIR"

python - <<'PY'
import os
import math
import importlib.util

task_id = int(os.environ["SLURM_ARRAY_TASK_ID"])
num_array_jobs = 24

PLOT_SCRIPT = os.environ.get("PLOT_SCRIPT", "metric_plotting.py")
OUT_DIR = os.environ.get("OUT_DIR", "/home/kirilb/data/L2PRH/bias_diagrams")
PRH_DATA_ROOT = os.environ.get("PRH_DATA_ROOT", "/home/kirilb/orcd/scratch/PRH_data")

spec = importlib.util.spec_from_file_location("metric_plotting", PLOT_SCRIPT)
metric_plotting = importlib.util.module_from_spec(spec)
spec.loader.exec_module(metric_plotting)

names_sorted = [
    'Qwen__Qwen3-1.7B-Base__text',
    'Qwen__Qwen3-4B-Base__text',
    'google__gemma-3-1b-it__text',
    'google__gemma-3-4b-it__text',
    'meta-llama__Llama-3.2-1B-Instruct__text',
    'meta-llama__Llama-3.2-3B-Instruct__text',
    'BAAI__bge-base-en-v1.5__text',
    'BAAI__bge-large-en-v1.5__text',
    'nomic-ai__nomic-embed-text-v1.5__text',
    'nomic-ai__nomic-embed-text-v2-moe__text',
    'codefuse-ai__F2LLM-1.7B__text',
    'codefuse-ai__F2LLM-4B__text',
    'google__siglip2-base-patch16-256__text',
    'google__siglip2-large-patch16-256__text',
    'laion__CLIP-ViT-B-32-laion2B-s34B-b79K__text',
    'laion__CLIP-ViT-H-14-laion2B-s32B-b79K__text',
    'openai__clip-vit-base-patch32__text',
    'openai__clip-vit-large-patch14__text',
    'google__siglip2-base-patch16-256__img',
    'google__siglip2-large-patch16-256__img',
    'laion__CLIP-ViT-B-32-laion2B-s34B-b79K__img',
    'laion__CLIP-ViT-H-14-laion2B-s32B-b79K__img',
    'openai__clip-vit-base-patch32__img',
    'openai__clip-vit-large-patch14__img',
    'facebook__dinov2-base__img',
    'facebook__dinov2-large__img',
    'facebook__vit-mae-large__img',
    'facebook__vit-mae-huge__img',
    'microsoft__beit-base-patch16-224__img',
    'microsoft__beit-large-patch16-224__img',
]

abbreviated_model_names = [
    "Qwen3-1.7B",
    "Qwen3-4B",
    "gemma-3-1B",
    "gemma-3-4B",
    "Llama-3.2-1B",
    "Llama-3.2-3B",
    "bge-base-v1.5",
    "bge-large-v1.5",
    "nomic-v1.5",
    "nomic-v2",
    "F2LLM-1.7B",
    "F2LLM-4B",
    "siglip2-base",
    "siglip2-large",
    "Laion CLIP-base",
    "Laion CLIP-huge",
    "OpenAI CLIP-base",
    "OpenAI CLIP-large",
    "siglip2-base",
    "siglip2-large",
    "Laion CLIP-base",
    "Laion CLIP-huge",
    "OpenAI CLIP-base",
    "OpenAI CLIP-huge",
    "dinov2-base",
    "dinov2-large",
    "vit-mae-large",
    "vit-mae-huge",
    "beit-base",
    "beit-large",
]

METRICS = [
    ("cka", "CKA"),
    ("cka_unbiased", "Unbiased CKA"),
    ("svcca_10", "SVCCA 10"),
    ("svcca_100", "SVCCA 100"),
    ("topk_10", "KNN Overlap 10"),
    ("topk_100", "KNN Overlap 100"),
    ("knn_edit_10", "KNN-10 Edit"),
    ("knn_edit_100", "KNN-100 Edit"),
]

datasets = [
    ("coco", "COCO"),
    ("cc3m", "CC3M"),
    ("visual_genome", "Visual Genome"),
]

panels = [
    ("diff", "Centered-Raw"),
    ("raw", "Raw"),
    ("filtered", "Centered"),
]


def reverse_hyphen_label(label: str) -> str:
    parts = label.split("-")
    if len(parts) == 2:
        return f"{parts[1]}-{parts[0]}"
    return label

type_name = ['LLM Text', 'Embed Text', 'Multimodal Text', 'Multimodal Image', 'Embed Image']
type_index = [6, 6, 6, 6, 6]

all_jobs = []
for dataset_name, dataset_title in datasets:
    for metric_key, metric_title in METRICS:
        for panel, panel_name in panels:
            all_jobs.append({
                "dataset_name": dataset_name,
                "dataset_title": dataset_title,
                "metric_key": metric_key,
                "metric_title": metric_title,
                "panel": panel,
                "panel_name": panel_name,
            })

n_total = len(all_jobs)
if n_total != 72:
    print(f"Warning: expected 72 jobs, got {n_total}")

if task_id < 0 or task_id >= num_array_jobs:
    raise IndexError(f"SLURM_ARRAY_TASK_ID={task_id} out of range for {num_array_jobs} array jobs")

chunk_size = math.ceil(n_total / num_array_jobs)
start = task_id * chunk_size
end = min((task_id + 1) * chunk_size, n_total)
my_jobs = all_jobs[start:end]

print(f"Array task {task_id} handling jobs [{start}, {end}) out of {n_total}")
print(f"This task will render {len(my_jobs)} plot(s)")

os.makedirs(OUT_DIR, exist_ok=True)

for local_idx, job in enumerate(my_jobs, start=1):
    dataset_name = job["dataset_name"]
    dataset_title = job["dataset_title"]
    metric_key = job["metric_key"]
    metric_title = job["metric_title"]
    panel = job["panel"]
    panel_name = job["panel_name"]
    if panel == "diff":
        panel_name = reverse_hyphen_label(panel_name)

    raw_dir = os.path.join(PRH_DATA_ROOT, f"metrics_embedded_{dataset_name}")
    filtered_dir = os.path.join(PRH_DATA_ROOT, f"metrics_embedded_{dataset_name}_centered")
    savepath = os.path.join(
        OUT_DIR,
        f"{metric_key}_{dataset_name}_{panel}.pdf",
    )

    print("")
    print(f"[{local_idx}/{len(my_jobs)} in array task {task_id}]")
    print(f"  dataset       = {dataset_name}")
    print(f"  metric_key    = {metric_key}")
    print(f"  panel         = {panel}")
    print(f"  raw_dir       = {raw_dir}")
    print(f"  filtered_dir  = {filtered_dir}")
    print(f"  savepath      = {savepath}")

    M, info, fig, ax, saved_path = metric_plotting.plot_single_metric_from_npz_sorted(
    models=names_sorted,
    names_sorted=names_sorted,
    abbreviated_model_names=abbreviated_model_names,
    metric=metric_key,
    raw_dir=raw_dir,
    filtered_dir=filtered_dir,
    panel=panel,
    title=f"{panel_name} {metric_title} Alignment over {dataset_title}",
    savepath=savepath,
    type_name=type_name,
    type_index=type_index,
    close_plot=True,
    show_plot=False,)

print("")
print(f"Array task {task_id} done.")
PY