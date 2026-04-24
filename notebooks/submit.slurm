#!/usr/bin/env bash
# =============================================================================
# submit.slurm  –  SLURM batch script for fastmagic IQL on Duke DCC (H200 x1)
#
# Submit with:
#   sbatch submit.slurm
# =============================================================================
#SBATCH --job-name=fastmagic_iql
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --partition=h200-hp
#SBATCH --account=pfisterlab_h200_r
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:h200:1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --signal=B:USR1@180
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=gp163@duke.edu
# ---------------------------------------------------------------------------
# 0. Load required modules
# ---------------------------------------------------------------------------
module purge
module load CUDA/12.4
module load Anaconda3
# ---------------------------------------------------------------------------
# 1. Activate conda environment
# ---------------------------------------------------------------------------
CONDA_BASE=$(conda info --base)
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate fastmagic
export FASTMAGIC_PY=$(which python)
# ---------------------------------------------------------------------------
# 2. Paths
# ---------------------------------------------------------------------------
# training_script.py lives in storage/owen/ (not inside the repo).
# It resolves repo_root as $FASTMAGIC_WORK_DIR/fastmagic, so WORK_DIR
# must be the *parent* of the repo — i.e. $HOME where ~/fastmagic lives.
export FASTMAGIC_WORK_DIR="${HOME}"
export FASTMAGIC_REPO="${HOME}/fastmagic"
export FASTMAGIC_RESULTS="${HOME}/fastmagic_results"
TRAINING_SCRIPT="/hpc/home/gp163/storage/owen/training_script.py"
# ---------------------------------------------------------------------------
# 3. MuJoCo environment variables
# ---------------------------------------------------------------------------
MUJOCO_DIR="${HOME}/.mujoco/mujoco210"
export MUJOCO_GL=egl
export D4RL_SUPPRESS_IMPORT_ERROR=1
export MUJOCO_PY_MUJOCO_PATH="${MUJOCO_DIR}"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:${MUJOCO_DIR}/bin:/usr/lib/nvidia:/hpc/home/gp163/.conda/envs/fastmagic/lib"
export C_INCLUDE_PATH="/hpc/home/gp163/.conda/envs/fastmagic/include:${C_INCLUDE_PATH:-}"
export CPATH="/hpc/home/gp163/.conda/envs/fastmagic/include:${CPATH:-}"

# torch.compile inductor cache — persists between jobs, speeds up recompilation
export TORCHINDUCTOR_CACHE_DIR="${HOME}/.inductor_cache"
mkdir -p "${TORCHINDUCTOR_CACHE_DIR}"

# Single H200
export CUDA_VISIBLE_DEVICES=0
export TORCH_CUDA_ARCH_LIST="9.0"   # H200 is SM 9.0

# ---------------------------------------------------------------------------
# 3.5 Requeue handler (for time-limit continuation)
# ---------------------------------------------------------------------------
_REQUEUE_TRIGGERED=0
requeue_and_exit() {
    sig="$1"
    if [[ "${_REQUEUE_TRIGGERED}" -eq 1 ]]; then
        return
    fi
    _REQUEUE_TRIGGERED=1
    echo ""
    echo "[signal] Received ${sig} at $(date). Attempting SLURM requeue..."
    if [[ -n "${SLURM_JOB_ID:-}" ]]; then
        scontrol requeue "${SLURM_JOB_ID}" && echo "[signal] Requeued job ${SLURM_JOB_ID}" || echo "[signal] Requeue failed for ${SLURM_JOB_ID}"
    else
        echo "[signal] SLURM_JOB_ID is empty; cannot requeue."
    fi
    exit 0
}
trap 'requeue_and_exit USR1' USR1
trap 'requeue_and_exit TERM' TERM
# ---------------------------------------------------------------------------
# 4. GPU + environment verification
# ---------------------------------------------------------------------------
echo "========================================"
echo "  Job:        ${SLURM_JOB_ID}"
echo "  Node:       ${SLURMD_NODENAME}"
echo "  Started:    $(date)"
echo "  Python:     ${FASTMAGIC_PY}"
echo "  Script:     ${TRAINING_SCRIPT}"
echo "  Repo:       ${FASTMAGIC_REPO}"
echo "  Work dir:   ${FASTMAGIC_WORK_DIR}"
echo "========================================"
nvidia-smi --query-gpu=name,memory.total,driver_version \
           --format=csv,noheader
# ---------------------------------------------------------------------------
# 5. Pull latest code
# ---------------------------------------------------------------------------
cd "${FASTMAGIC_REPO}"
git fetch origin
git reset --hard origin/main
echo "Repo HEAD: $(git rev-parse --short HEAD)"
# ---------------------------------------------------------------------------
# 6. Run benchmark
# ---------------------------------------------------------------------------
python "${TRAINING_SCRIPT}" \
    --preset        mujoco          \
    --seeds         0 1             \
    --train_steps   300000          \
    --eval_interval 50000           \
    --eval_episodes 3               \
    --max_envs      3               \
    --compile_mode  reduce-overhead

# Useful flags:
# resume is ON by default in training_script.py (skip existing completed runs)
# --no_resume      force a full rerun from scratch
# --no_ablations    skip the 24-run ablation sweep (~60 min savings)
# --no_compile      disable torch.compile (good for debugging)
# --seeds 0 1 2     add a third seed (~40 min extra)
# remove --max_envs to run all 9 MuJoCo envs

echo ""
echo "========================================"
echo "  Job finished: $(date)"
echo "  Results:      ${FASTMAGIC_RESULTS}"
echo "========================================" 
