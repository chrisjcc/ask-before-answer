# Hyperparameter Optimization (Validation Curves)

In LLM post-training (SFT, DPO, GRPO), plotting Validation Curves (how performance changes as a specific hyperparameter changes) is critical for discovering a stable model configuration.

Because algorithms like DPO and GRPO are notoriously sensitive to hyperparameters, plotting these curves is often the only way to find the narrow optimal boundary where the model actually learns human preferences without destroying its baseline capabilities (reward hacking).

This repository fully automates this process using **Weights & Biases (W&B) Sweeps** paired with **Data Version Control (DVC)**.

## How it works mechanically: The Two-Step Architecture

Our codebase completely isolates the hyperparameter sweep trials from your final baseline model evaluations. 

### Step 1: The Sweep Report
1. When you run a sweep command (e.g., `make sweep-dpo COUNT=10`), it launches a W&B Agent that executes `scripts/run_sweep_trial.py`.
2. That script injects specific hyperparameters into your YAML configs and runs a full DVC training trial, streaming metrics live to the W&B servers.
3. Once the trials complete, the Makefile automatically triggers `scripts/generate_sweep_report.py`. This script pulls the raw metrics from the cloud, groups them by Sweep ID, plots the Validation Curves, and statelessly regenerates the `docs/sweep_report.md` leaderboard.

### Step 2: Human-in-the-Loop Promotion
Crucially, **no automated script picks the top model from the sweep trials and promotes it.** The filtering mechanism relies entirely on a human-in-the-loop workflow using DVC and the W&B API:
1. **Review:** You review the `docs/sweep_report.md` leaderboard and identify the absolute best trial (e.g., Run ID `5cxs95q7`).
2. **Apply:** You tell DVC to restore that winning model to your active workspace by running `dvc exp apply sweep_5cxs95q7`. This permanently locks the winning hyperparameters into your local YAML config.
3. **Train the Final Baseline:** You run `make ablation-suite` (which executes standard `dvc repro`). This runs the training script *manually*, outside of the W&B sweep agent.
4. **Isolate the Ablation Report:** Because the manual run was not executed by the agent, W&B does *not* tag it with the `.sweep` metadata property. When `scripts/generate_report.py` generates the final `docs/ablation_report.md`, it loops through the cloud and instantly skips any run that possesses a `.sweep` tag. It only allows manual runs to pass through, ensuring your ablation report contains only your clean, final baseline rows (Base vs SFT vs DPO vs GRPO) rather than being cluttered by dozens of sweep trials!

---

## Advanced Interactive W&B Charts
While our `generate_report.py` script pulls down raw historical metrics to embed a standard Validation Curve locally, the true power of W&B lies in its **Web Dashboard**. W&B automatically builds and updates advanced interactive charts live on the web dashboard the moment your sweep trials start completing!

You can find the following interactive visualizations exclusively on the W&B Web UI for your sweep:
- **Parallel Coordinates Chart:** It instantly draws this to help you visually trace how combinations of variables (e.g., high Learning Rate + low Batch Size) flow toward the final Eval Loss.
- **Hyperparameter Importance Matrix:** Under the hood, W&B actually trains a lightweight Random Forest model on your sweep results in real-time! It uses this to calculate "Feature Importance," outputting a matrix that explicitly tells you (for example) "Beta was 85% responsible for the changes in your F1 score, while Batch Size was only 15% responsible."
- **Standard Validation Curves:** It plots the interactive scatter plots for Performance vs. Hyperparameters, allowing you to hover over individual dots to inspect the specific run.

---

## Supported Sweep Strategies

The repository supports Bayesian hyperparameter sweeps for all three major training stages. 

### 1. SFT Sweeps (`make sweep-sft`)
In Supervised Fine-Tuning, the goal is to optimize the model's ability to imitate the 4-field schema (Action, Reasoning, Facets, Response) without catastrophic forgetting.
- **Key Hyperparameters:** Learning Rate, Batch Size, Epochs, and LoRA Rank ($r$).
- **Strategy:** If the learning rate is too high, the model catastrophically forgets its foundational knowledge (spitting out gibberish). If it is too low, it fails to learn the strict formatting schema.

### 2. DPO Sweeps (`make sweep-dpo`)
In Direct Preference Optimization, we want to maximize the margin between the chosen and rejected samples.
- **Key Hyperparameters:** Beta ($\beta$) and Learning Rate.
- **Strategy:** Beta controls the KL Divergence penalty—meaning how much the model is allowed to deviate from the original SFT model. 
  - If $\beta$ is too low, the model over-optimizes and destroys its formatting (a phenomenon known as reward hacking).
  - If $\beta$ is too high, the model is penalized too heavily for changing its weights and outright refuses to learn the new preference data. 
  By sweeping $\beta$ across `[0.01, 0.05, 0.1, 0.2, 0.5]`, the generated validation curve clearly exposes the optimal F1 score peak.

### 3. GRPO Sweeps (`make sweep-grpo`)
In Group Relative Policy Optimization, the model relies on programmatic reward functions rather than static contrastive pairs.
- **Key Hyperparameters:** Beta ($\beta$), Learning Rate, and Reward Coefficients.
- **Strategy:** GRPO is extremely unstable if the learning rate is too high. Sweeps help identify the exact boundary where the model maximizes its deterministic formatting and factual accuracy rewards relative to the group average, without experiencing policy collapse.
