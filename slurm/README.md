# AskBeforeAnswer Slurm Configuration

This directory contains the modernized Slurm scripts designed to seamlessly wrap the project's DVC and Makefile architecture.

## Principles
1. **Never Bypass DVC:** These scripts do not call python scripts directly. They execute `make` targets (which wrap `dvc repro`), guaranteeing that environments and pipelines execute exactly the same on the cluster as they do locally.
2. **Universal Preamble:** Each script dynamically injects your Conda environment, the `LD_LIBRARY_PATH` CXXABI fix, and `.env` secrets into the compute nodes.

## Scripts & Usage

### 1. Preprocessing (`01_preprocess.slurm`)
Generates the Apache Arrow dataset caches without wasting GPU credits.
```bash
sbatch slurm/01_preprocess.slurm
```

### 2. Ablation Suite (`02_ablation_suite.slurm`)
The heavy lifter. Requests an 80GB GPU node and runs the full A-Z pipeline (SFT, DPO, GRPO, ORPO) along with the LLM-as-a-judge and Multi-Turn evaluations.
```bash
sbatch slurm/02_ablation_suite.slurm
```

### 3. Sequential Sweeps (`03_sweep_agent.slurm`)
Leverages **Slurm Arrays** with sequential throttling (`%1`) to create an infinite continuous pipeline. This perfectly optimizes Bayesian search (by ensuring each trial learns from the last) and seamlessly bypasses strict 24h queue limits.

**Architecture Note:** We explicitly separate sweep *creation* from sweep *execution*. If the script created the sweep internally, the Slurm Array would spawn 8 completely independent sweeps that wouldn't share Bayesian history. By initializing it once on the login node and passing the ID to the array, all 8 nodes seamlessly contribute to the exact same global Bayesian model.

1. Initialize the sweep on your login node (do NOT use `make sweep`, as that will start training on the login node!):
```bash
wandb sweep sweeps/sft.yaml
```
*(This will print out a Sweep ID, e.g., `rl4aa/ask-before-answer/xyz123`)*
2. Launch the sequential array pipeline on the cluster:
```bash
sbatch --export=ALL,SWEEP_ID="rl4aa/ask-before-answer/xyz123",COUNT=5 slurm/03_sweep_agent.slurm
```
3. Generate the final sweep report (requires passing the Slurm Job ID returned from the previous step):
```bash
sbatch --dependency=afterok:<ARRAY_JOB_ID> --wrap="python scripts/generate_sweep_report.py --fine-tune-method sft --sweep-id rl4aa/ask-before-answer/xyz123"
```

### 4. Interactive Demo Tunnel (`04_demo_tunnel.slurm`)
Runs the Streamlit app on a GPU compute node and provides instructions for SSH tunneling it to your local Macbook.
```bash
sbatch slurm/04_demo_tunnel.slurm
```
