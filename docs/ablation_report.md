# Ablation Experiment Report

This report was automatically generated from Weights & Biases metrics.

## Top Performing Configurations

| Run ID   | Name         | Group        | Hypothesis                                                                                                                                                                                                                                                                                                    | Sweep ID   |   Learning Rate |   Batch Size |   Eval Loss | URL                                                    |
|:---------|:-------------|:-------------|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-----------|----------------:|-------------:|------------:|:-------------------------------------------------------|
| 7omeutrj | dpo_training | dpo_baseline | Supervised Fine-Tuning (SFT) alone is insufficient to align the model to consistently ask clarification questions without hallucinating. A two-stage pipeline (SFT followed by DPO) will achieve a lower evaluation loss and better disambiguation formatting than either the SFT-only or DPO-only baselines. | N/A        |          5e-05  |            1 | 3.99771e-05 | https://wandb.ai/rl4aa/ask-before-answer/runs/7omeutrj |
| buzbquuo | dpo_training | dpo_baseline | Supervised Fine-Tuning (SFT) alone is insufficient to align the model to consistently ask clarification questions without hallucinating. A two-stage pipeline (SFT followed by DPO) will achieve a lower evaluation loss and better disambiguation formatting than either the SFT-only or DPO-only baselines. | N/A        |          5e-05  |            1 | 0.000113545 | https://wandb.ai/rl4aa/ask-before-answer/runs/buzbquuo |
| lln0thjk | sft_training | sft_baseline | Supervised Fine-Tuning (SFT) alone is insufficient to align the model to consistently ask clarification questions without hallucinating. A two-stage pipeline (SFT followed by DPO) will achieve a lower evaluation loss and better disambiguation formatting than either the SFT-only or DPO-only baselines. | N/A        |          0.0002 |            1 | 0.43139     | https://wandb.ai/rl4aa/ask-before-answer/runs/lln0thjk |
| nxep4c1i | sft_training | sft_baseline | Supervised Fine-Tuning (SFT) alone is insufficient to align the model to consistently ask clarification questions without hallucinating. A two-stage pipeline (SFT followed by DPO) will achieve a lower evaluation loss and better disambiguation formatting than either the SFT-only or DPO-only baselines. | N/A        |          0.0002 |            1 | 0.432052    | https://wandb.ai/rl4aa/ask-before-answer/runs/nxep4c1i |

## Learning Curves

![Training Loss](plots/train_loss_comparison.png)

## LLM-as-a-Judge Evaluation Leaderboard

The following scores were computed using W&B Weave with a Gemini-based judge scorer.

| Model    |   ambiguity_detection |   clarification_quality |   usefulness |
|:---------|----------------------:|------------------------:|-------------:|
| base     |                 0.904 |                   0.788 |        0.894 |
| sft_only |                 0.526 |                   0.61  |        0.718 |
| dpo_only |                 0.858 |                   0.746 |        0.842 |
| sft      |                 0.464 |                   0.586 |        0.712 |
| sft_dpo  |                 0.856 |                   0.74  |        0.838 |
