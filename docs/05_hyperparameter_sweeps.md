# Hyperparameter Optimization & Sweeps

In LLM post-training (SFT, DPO, GRPO), plotting Validation Curves (how performance changes as a specific hyperparameter changes) is critical for discovering a stable model configuration.

Because algorithms like DPO and GRPO are notoriously sensitive to hyperparameters, plotting these curves is often the only way to find the narrow optimal boundary where the model actually learns human preferences without destroying its baseline capabilities (reward hacking).

This repository fully automates this process using **Weights & Biases (W&B) Sweeps** paired with **Data Version Control (DVC)**.

## 1. The Two-Step Architecture

Our codebase completely isolates the hyperparameter sweep trials from your final baseline model evaluations. 

### Step 1: The Sweep Report
1. When you run a sweep command (e.g., `make sweep FINE_TUNE_METHOD=dpo COUNT=10`), it launches a W&B Agent that executes `scripts/run_sweep_trial.py`.
2. That script injects specific hyperparameters into your YAML configs and runs a full DVC training trial, streaming metrics live to the W&B servers.
3. Once the trials complete, the Makefile automatically triggers `scripts/generate_sweep_report.py`. This script pulls the raw metrics from the cloud, groups them by Sweep ID, plots the Validation Curves, and statelessly regenerates the `docs/sweep_report.md` leaderboard.

### Step 2: Human-in-the-Loop Promotion
Crucially, **no automated script picks the top model from the sweep trials and promotes it.** The filtering mechanism relies entirely on a human-in-the-loop workflow using DVC and the W&B API:
1. **Review:** You review the leaderboard and identify the absolute best trial (e.g., Run ID `5cxs95q7`).
2. **Apply:** You tell DVC to restore that winning model to your active workspace by running `dvc exp apply sweep_5cxs95q7`. This permanently locks the winning hyperparameters into your local YAML config.
3. **Train the Final Baseline:** You run `make ablation-suite` (which executes standard `dvc repro`). This runs the training script *manually*, outside of the W&B sweep agent.
4. **Isolate the Ablation Report:** Because the manual run was not executed by the agent, W&B does *not* tag it with the `.sweep` metadata property. When `scripts/generate_ablation_report.py` generates the ablation report, it loops through the cloud and skips any run that possesses a `.sweep` tag, ensuring your report contains only your clean, final baseline rows.

## 2. Advanced Interactive W&B Charts

While our scripts pull down raw historical metrics to embed standard Validation Curves locally, the true power of W&B lies in its **Web Dashboard**. W&B automatically builds interactive charts:
- **Parallel Coordinates Chart:** Visually trace how combinations of variables (e.g., high Learning Rate + low Batch Size) flow toward the final Eval Loss.
- **Hyperparameter Importance Matrix:** W&B trains a Random Forest model on your sweep results in real-time to calculate Feature Importance.
- **Interactive Validation Curves:** Interactive scatter plots for Performance vs. Hyperparameters.

## 3. Supported Sweep Strategies

The repository supports Bayesian hyperparameter sweeps for all three major training stages. 

### SFT Sweeps (`make sweep-sft`)
Optimize the model's ability to imitate the 4-field schema without catastrophic forgetting.
- **Key Hyperparameters:** Learning Rate, Batch Size, Epochs, and LoRA Rank ($r$).
- **Strategy:** Balance the learning rate so the model learns the strict formatting schema without forgetting foundational knowledge.

### DPO Sweeps (`make sweep-dpo`)
Maximize the margin between chosen and rejected samples.
- **Key Hyperparameters:** Beta ($\beta$) and Learning Rate.
- **Strategy:** Beta controls the KL Divergence penalty. Sweeping $\beta$ exposes the optimal F1 score peak without reward hacking.

### GRPO Sweeps (`make sweep-grpo`)
Optimize programmatic reward functions rather than static contrastive pairs.
- **Key Hyperparameters:** Beta ($\beta$), Learning Rate, and Reward Coefficients.
- **Strategy:** Sweeps help identify the exact boundary where the model maximizes deterministic formatting and factual accuracy without experiencing policy collapse.
