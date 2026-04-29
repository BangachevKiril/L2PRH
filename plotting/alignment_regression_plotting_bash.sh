#!/bin/bash
#SBATCH --job-name=regression_diagrams
#SBATCH --output=/home/kirilb/data/PRH/logs/regression_diagrams_%A_%a.out
#SBATCH --error=/home/kirilb/data/PRH/logs/regression_diagrams_%A_%a.err
#SBATCH --array=0-8
#SBATCH --partition=mit_normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00

module load miniforge
CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate GPUenv

mkdir -p /home/kirilb/data/PRH/logs
mkdir -p /home/kirilb/data/PRH/regression_diagrams

cd /home/kirilb/data/PRH

python - <<'PY'
import os
import sys
import traceback

import matplotlib
matplotlib.use("Agg")

PRH_ROOT = "/home/kirilb/data/PRH"
UTILS_DIR = os.path.join(PRH_ROOT, "Utils")

for p in [PRH_ROOT, UTILS_DIR]:
    if p not in sys.path:
        sys.path.append(p)

import alignment_regression_plotting


# ------------------------------------------------------------
# Fixed inputs
# ------------------------------------------------------------
xlsx_path = "/home/kirilb/data/L2PRH/model_specifications.xlsx"

names = ["coco", "cc3m", "visual_genome"]
Names = ["COCO", "CC3M", "Visual Genome"]
lambdas = [0.1, 1, 10]

jobs = []
for j, name in enumerate(names):
    for lam in lambdas:
        jobs.append((name, Names[j], lam))

task_id = int(os.environ["SLURM_ARRAY_TASK_ID"])
if task_id < 0 or task_id >= len(jobs):
    raise ValueError(f"Invalid SLURM_ARRAY_TASK_ID={task_id}, expected 0..{len(jobs)-1}")

name, Name, lam = jobs[task_id]

metrics_dir = f"/home/kirilb/orcd/scratch/PRH_data/metrics_embedded_{name}_centered/"
save_fig = f"/home/kirilb/data/L2PRH/regression_diagrams/importance_of_specifics_{name}_centered_{lam}.pdf"
title = (
    f"Importance of Models Specifications for Alignment of Centered Features over {Name}\n"
    f"(Via Ridge with λ={lam:.1f})"
)

print("=" * 80)
print(f"SLURM_ARRAY_TASK_ID = {task_id}")
print(f"name                = {name}")
print(f"Name                = {Name}")
print(f"lambda              = {lam}")
print(f"metrics_dir         = {metrics_dir}")
print(f"save_fig            = {save_fig}")
print("=" * 80, flush=True)

try:
    features, feat_names, models = alignment_regression_plotting.build_pairwise_feature_tensor_from_xlsx(
        xlsx_path,
        sheet_name=None,
        strict=True,
    )

    # Optional sanity check, matching your example
    Alignment = alignment_regression_plotting.build_alignment_matrix_30x30(
        metrics_dir,
        metric_key="CKA_HSIC_mean_over_subsamples",
        models=models,
    )
    print(f"[sanity] alignment shape = {Alignment.shape}", flush=True)

    out = alignment_regression_plotting.fit_alignment_on_features_and_plot_coef_heatmap(
        metrics_dir=metrics_dir,
        models=models,
        feature_tensor=features,
        lam=lam,
        feature_names=feat_names,
        sort_keys=None,
        annotate=False,
        fit_intercept=False,
        save_fig=save_fig,
        title=title,
        close_plot=True,
        show_plot=False,
    )

    print(f"[done] {save_fig}", flush=True)
    print(f"[summary] coef_matrix shape = {out['coef_matrix'].shape}", flush=True)
    print(f"[summary] metrics_used = {out['metrics_used']}", flush=True)

except Exception as e:
    print(f"[failed] task_id={task_id}", flush=True)
    print(str(e), flush=True)
    traceback.print_exc()
    raise

PY