#!/bin/bash
#SBATCH --job-name=aba_ablation
#SBATCH --output=slurm/logs/ablation_%j.log
#SBATCH --error=slurm/logs/ablation_%j.err
#SBATCH --partition=gpu        # Adjust to your HPC's GPU queue
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1           # Request 1 GPU (e.g., gpu:A100:1)
#SBATCH --time=48:00:00        # Full ablation takes significant time

# Change to the directory from which the job was submitted
cd "${SLURM_SUBMIT_DIR}"

echo "Initializing Compute Node Environment..."
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
# mkdir -p $BASE/{logs,checkpoints,tmp}
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

echo "Starting End-to-End Ablation Suite..."
make ablation-suite
echo "Ablation Suite complete. Artifacts saved to docs/ablation_report.md"
