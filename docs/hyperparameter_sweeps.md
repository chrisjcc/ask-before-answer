# Hyperparameter Optimization (Validation Curves)

In LLM post-training (SFT, DPO, GRPO), plotting Validation Curves (how performance changes as a specific hyperparameter changes) is critical for discovering a stable model configuration.

Because algorithms like DPO and GRPO are notoriously sensitive to hyperparameters, plotting these curves is often the only way to find the narrow optimal boundary where the model actually learns human preferences without destroying its baseline capabilities (reward hacking).

This repository fully automates this process using **Weights & Biases (W&B) Sweeps** paired with **Data Version Control (DVC)**.

## How it works mechanically
Our codebase does *not* wait until the very end to manually plot things. Instead:
1. When you run a sweep command (e.g., `make sweep-dpo`), it launches a W&B Agent that executes `scripts/run_sweep_trial.py`.
2. That script injects specific hyperparameters (like $\beta=0.1$) into your YAML configs and runs a full DVC training trial, streaming the evaluation loss and metrics live to the W&B servers.
3. Later, when you run `make ablation-suite`, our script `scripts/generate_report.py` queries the W&B API, pulls down the exact raw metrics for every single trial, plots the **Validation Curves** locally using `seaborn`, and injects them directly into `docs/ablation_report.md`.

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
