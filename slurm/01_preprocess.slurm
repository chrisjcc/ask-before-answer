#!/bin/bash
#SBATCH --job-name=aba_preprocess
#SBATCH --output=slurm/logs/preprocess_%j.log
#SBATCH --error=slurm/logs/preprocess_%j.err
#SBATCH --partition=cpu        # Adjust to your HPC's standard queue
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8      # Good for dataset mapping
#SBATCH --mem=32G
#SBATCH --time=04:00:00

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
# export HF_DATASETS_CACHE=$BASE/hf_cache
# export TMPDIR=$BASE/tmp
# mkdir -p $BASE/{datasets,logs,tmp}
# ---------------------------------------------------------------------------

# 1. Activate Conda
source ~/.conda/etc/profile.d/conda.sh || source ~/miniconda3/etc/profile.d/conda.sh
conda activate ask-before-answer-311
export PYTHONNOUSERSITE=1

# 2. Inject Library Fixes for your specific cluster
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"

# 3. Load secrets
if [ -f ".env" ]; then
    set -a
    source .env
    set +a
fi

echo "Starting DVC Preprocessing Pipeline..."
make preprocess
echo "Preprocessing complete."
