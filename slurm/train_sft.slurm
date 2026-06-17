#!/bin/bash
#SBATCH --job-name=ask_sft
#SBATCH --partition=maxgpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --output=/data/dust/user/%u/storage/ask-before-answer/logs/sft_%j.out
#SBATCH --error=/data/dust/user/%u/storage/ask-before-answer/logs/sft_%j.err

set -e

unset LD_PRELOAD

module load maxwell
module load mamba
. mamba-init

mamba activate cohomology

BASE=/data/dust/user/$USER/storage/ask-before-answer

export HF_HOME=$BASE/hf_cache
export WANDB_DIR=$BASE/wandb
export TMPDIR=$BASE/tmp

mkdir -p $BASE/{logs,checkpoints,tmp}

cd ~/projects/ask-before-answer

echo "Starting SFT..."
make train-sft
