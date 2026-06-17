#!/bin/bash
#SBATCH --job-name=ask_preprocess
#SBATCH --partition=maxgpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/data/dust/user/%u/storage/ask-before-answer/logs/preprocess_%j.out
#SBATCH --error=/data/dust/user/%u/storage/ask-before-answer/logs/preprocess_%j.err

set -e

module load maxwell
module load mamba
. mamba-init

mamba activate cohomology

BASE=/data/dust/user/$USER/storage/ask-before-answer

export HF_HOME=$BASE/hf_cache
export HF_DATASETS_CACHE=$BASE/hf_cache
export TMPDIR=$BASE/tmp

mkdir -p $BASE/{datasets,logs,tmp}

cd ~/projects/ask-before-answer

echo "Starting preprocessing..."

python scripts/preprocess_data.py

echo "Preprocessing complete."
```
