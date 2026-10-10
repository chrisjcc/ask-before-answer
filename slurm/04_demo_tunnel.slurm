#!/bin/bash
#SBATCH --job-name=aba_demo
#SBATCH --output=slurm/logs/demo_%j.log
#SBATCH --error=slurm/logs/demo_%j.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00

# ---------------------------------------------------------------------------
# UNIVERSAL PREAMBLE
# ---------------------------------------------------------------------------
# Change to the directory from which the job was submitted
cd "${SLURM_SUBMIT_DIR}"
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

COMPUTE_NODE=$(hostname -s)
PORT=8501

echo "=========================================================================="
echo "🚀 STREAMLIT APP IS RUNNING ON COMPUTE NODE: $COMPUTE_NODE"
echo "=========================================================================="
echo "To view the app, copy and paste this command into your LOCAL Macbook terminal:"
echo ""
echo "    ssh -N -f -L ${PORT}:${COMPUTE_NODE}:${PORT} clseitz@YOUR_LOGIN_NODE.edu"
echo ""
echo "Then, open your browser and go to:"
echo "    http://localhost:${PORT}"
echo "=========================================================================="

streamlit run app/app.py --server.port $PORT --server.address=0.0.0.0 --browser.gatherUsageStats=false
