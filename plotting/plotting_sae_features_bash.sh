#!/bin/bash
#SBATCH --export=ALL
#SBATCH --job-name=sparse_pct_tables
#SBATCH --output=logs/sparse_pct_tables_%j.out
#SBATCH --error=logs/sparse_pct_tables_%j.err
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --partition=mit_normal

# Slurm/logging wrapper for make_sparse_feature_percentile_latex_tables.py.
#
# Usage:
#   bash make_sparse_feature_percentile_tables_bash.sh
#     Self-submits to Slurm after creating ./logs.
#
#   sbatch make_sparse_feature_percentile_tables_bash.sh
#     Also works, but make sure ./logs already exists or submit once via bash.
#
#   SUBMIT=0 bash make_sparse_feature_percentile_tables_bash.sh
#     Runs locally, with the same timestamped combined log.
#
# Useful overrides:
#   OUT_DIR=sparse_feature_percentile_latex_tables WHICH=both LOW=5 HIGH=95 DIGITS=6 \
#     bash make_sparse_feature_percentile_tables_bash.sh --master-mode inline

# =========================
# Self-submit helper
# =========================
# Slurm does not create the directory in #SBATCH --output before opening stdout.
# Running this script with `bash ...` creates logs/ first, then submits the job.
if [[ -z "${SLURM_JOB_ID:-}" && "${SUBMIT:-1}" == "1" ]]; then
  mkdir -p logs
  if command -v sbatch >/dev/null 2>&1; then
    echo "Submitting to Slurm..."
    echo "Logs will appear under: $(pwd)/logs"
    sbatch --export=ALL "$0" "$@"
    exit $?
  else
    echo "WARNING: sbatch not found; running locally instead."
  fi
fi

# =========================
# Logging
# =========================
mkdir -p logs

TS="$(date +%Y%m%d_%H%M%S)"
JOB_TAG="${SLURM_JOB_ID:-local}_${TS}"
RUN_LOG="logs/sparse_pct_tables_${JOB_TAG}.combined.log"
STATUS_LOG="logs/sparse_pct_tables_${JOB_TAG}.status"

# Mirror everything to both the terminal/Slurm logs and a combined timestamped log.
exec > >(tee -a "$RUN_LOG") 2> >(tee -a "$RUN_LOG" >&2)

log() {
  printf '[%(%F %T)T] %s\n' -1 "$*"
}

finish() {
  rc=$?
  log "Finished with exit code ${rc}"
  {
    echo "status=$([[ $rc -eq 0 ]] && echo success || echo failed)"
    echo "exit_code=$rc"
    echo "finished_at=$(date --iso-8601=seconds)"
    echo "combined_log=$RUN_LOG"
    echo "slurm_stdout=logs/sparse_pct_tables_${SLURM_JOB_ID:-local}.out"
    echo "slurm_stderr=logs/sparse_pct_tables_${SLURM_JOB_ID:-local}.err"
  } > "$STATUS_LOG"
  exit $rc
}
trap finish EXIT

log "Starting sparse feature percentile LaTeX table job"
log "SLURM_JOB_ID=${SLURM_JOB_ID:-local}"
log "SLURM_JOB_NAME=${SLURM_JOB_NAME:-local}"
log "HOSTNAME=$(hostname)"
log "SUBMIT_DIR=${SLURM_SUBMIT_DIR:-$PWD}"
log "RUN_LOG=$RUN_LOG"
log "STATUS_LOG=$STATUS_LOG"

# =========================
# Working directory
# =========================
# Prefer SLURM_SUBMIT_DIR rather than deriving paths from $0, because Slurm may
# execute a copied batch script from /var/spool/slurmd/job... on the compute node.
WORK_DIR="${WORK_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "$WORK_DIR"
log "PWD=$(pwd)"

# =========================
# Environment
# =========================
log "Loading conda environment..."
module load miniforge
CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate GPUenv
log "Python=$(command -v python)"
python --version

# =========================
# User config
# =========================
ROOTS=(
  "/home/kirilb/orcd/scratch/PRH_data/topk_sae_visual_genome"
  "/home/kirilb/orcd/scratch/PRH_data/topk_sae_coco"
  "/home/kirilb/orcd/scratch/PRH_data/topk_sae_cc3m"
)

PY_SCRIPT="plotting_sae_features.py"
OUT_DIR="/home/kirilb/data/L2PRH/sparse_feature_percentile_latex_tables"
WHICH="${WHICH:-truncated}"    # truncated, full, or both
LOW="${LOW:-5}"
HIGH="${HIGH:-95}"
DIGITS="${DIGITS:-6}"

log "PY_SCRIPT=$PY_SCRIPT"
log "OUT_DIR=$OUT_DIR"
log "WHICH=$WHICH"
log "LOW=$LOW"
log "HIGH=$HIGH"
log "DIGITS=$DIGITS"
log "Extra CLI args: $*"

if [[ ! -f "$PY_SCRIPT" ]]; then
  log "ERROR: Python script not found: $PY_SCRIPT"
  exit 1
fi

# Print root availability and how many statistics files each root currently has.
for ROOT in "${ROOTS[@]}"; do
  if [[ ! -d "$ROOT" ]]; then
    log "WARNING: root folder not found: $ROOT"
  else
    N_STATS=$(find "$ROOT" -name 'sparse_features_statistics.npz' -type f 2>/dev/null | wc -l | tr -d ' ')
    log "ROOT=$ROOT"
    log "  sparse_features_statistics.npz files found: $N_STATS"
  fi
done

mkdir -p "$OUT_DIR"
log "Output directory created/checked: $OUT_DIR"

CMD=(
  python -u "$PY_SCRIPT"
  "${ROOTS[@]}"
  --output-dir "$OUT_DIR"
  --which "$WHICH"
  --low "$LOW"
  --high "$HIGH"
  --digits "$DIGITS"
)

# Extra flags can be passed through, e.g.
#   bash make_sparse_feature_percentile_tables_bash.sh --which both --master-mode inline
if (( $# > 0 )); then
  CMD+=("$@")
fi

log "Running command: ${CMD[*]}"
"${CMD[@]}"

log "Generated files in $OUT_DIR:"
find "$OUT_DIR" -maxdepth 1 -type f | sort | sed 's/^/  /'

log "Done."
