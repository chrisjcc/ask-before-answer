#!/bin/bash
#SBATCH --job-name=aba_sweep
#SBATCH --output=slurm/logs/sweep_%A_%a.log
#SBATCH --error=slurm/logs/sweep_%A_%a.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --array=1-4          # Default: Launch 4 concurrent agents

echo "Initializing Compute Node Environment for Sweep Agent (Array Task: $SLURM_ARRAY_TASK_ID)..."
# ---------------------------------------------------------------------------
# HPC MODULE & CACHE REFERENCE (From legacy scripts)
# ---------------------------------------------------------------------------
# If your cluster requires explicit module loading or custom cache paths,
# uncomment and adjust the following lines:
#
# set -e
# unset LD_PRELOAD
# module load maxwell
# module load mamba
# . mamba-init
# mamba activate cohomology
#
# BASE=/data/dust/user/$USER/storage/ask-before-answer
# export HF_HOME=$BASE/hf_cache
# export WANDB_DIR=$BASE/wandb
# export TMPDIR=$BASE/tmp
# ---------------------------------------------------------------------------

# 1. Activate Conda
source ~/.conda/etc/profile.d/conda.sh || source ~/miniconda3/etc/profile.d/conda.sh
conda activate ask-before-answer-311

# 2. Inject Library Fixes for your specific cluster
export LD_LIBRARY_PATH="/data/dust/user/clseitz/.conda/envs/ask-before-answer-311/lib:$LD_LIBRARY_PATH"

# 3. Load secrets
if [ -f ".env" ]; then
    set -a
    source .env
    set +a
fi

if [ -z "$SWEEP_ID" ]; then
    echo "ERROR: SWEEP_ID is not set. Please pass it via --export=ALL,SWEEP_ID='...'"
    exit 1
fi

echo "Starting W&B Sweep Agent for $SWEEP_ID..."
wandb agent $SWEEP_ID
echo "Sweep Agent finished."
