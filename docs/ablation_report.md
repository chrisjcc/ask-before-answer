# Ablation Experiment Report

This report was automatically generated from Weights & Biases metrics.

## Top Performing Configurations

| Run ID   | Name         | Group        | Hypothesis   | Sweep ID   |   Learning Rate |   Batch Size |   Eval Loss | URL                                                    |
|:---------|:-------------|:-------------|:-------------|:-----------|----------------:|-------------:|------------:|:-------------------------------------------------------|
| vpvdqbla | dpo_training | dpo_baseline | N/A          | N/A        |           5e-07 |            1 |    0.158069 | https://wandb.ai/rl4aa/ask-before-answer/runs/vpvdqbla |
| 4n4vxwes | sft_training | sft_baseline | N/A          | N/A        |           2e-05 |            1 |    0.226521 | https://wandb.ai/rl4aa/ask-before-answer/runs/4n4vxwes |
| wgoc96ik | dpo_training | dpo_baseline | N/A          | N/A        |           5e-07 |            1 |    0.66954  | https://wandb.ai/rl4aa/ask-before-answer/runs/wgoc96ik |

## Learning Curves

![Training Loss](plots/train_loss_comparison.png)


## LLM-as-a-Judge Evaluation Leaderboard

The following scores were computed using W&B Weave 
with a Gemini-based judge scorer.

| Metric                |      base |   dpo_only |      sft |   sft_dpo |   clarifier_lora |     grpo |
|:----------------------|----------:|-----------:|---------:|----------:|-----------------:|---------:|
| ambiguity_detection   | 0.966     |  0.944     | 0.972    |  0.966    |          0.994   | 0.958    |
| clarification_quality | 0.784     |  0.784     | 0.794    |  0.796    |          0.796   | 0.788    |
| usefulness            | 0.88      |  0.88      | 0.896    |  0.896    |          0.896   | 0.884    |
| model_accuracy        | 0.62      |  0.6       | 0.64     |  0.64     |          0.6     | 0.64     |
| clarify_precision     | 0.617021  |  0.604167  | 0.65     |  0.657895 |          0.6     | 0.636364 |
| clarify_recall        | 0.966667  |  0.966667  | 0.866667 |  0.833333 |          1       | 0.933333 |
| clarify_f1            | 0.753247  |  0.74359   | 0.742857 |  0.735294 |          0.75    | 0.756757 |
| answer_f1             | 0.173913  |  0.0909091 | 0.4      |  0.4375   |          0       | 0.307692 |
| macro_f1              | 0.46358   |  0.417249  | 0.571429 |  0.586397 |          0.375   | 0.532225 |
| answer_accuracy       | 0.05      |  0.05      | 0        |  0.05     |          0       | 0.1      |
| facet_generation_rate | 0.0851064 |  0.104167  | 1        |  1        |          0       | 1        |
| clarify_ratio         | 1.56667   |  1.6       | 1.33333  |  1.26667  |          1.66667 | 1.46667  |
