# Ablation Experiment Report

This report was automatically generated from Weights & Biases metrics.

## 1. Top Performing Configurations

| Run ID   | Name          | Group         | Hypothesis   | Sweep ID   |   Learning Rate |   Batch Size |   Eval Loss | URL                                                    |
|:---------|:--------------|:--------------|:-------------|:-----------|----------------:|-------------:|------------:|:-------------------------------------------------------|
| rs04nh5a | dpo_training  | dpo_baseline  | N/A          | N/A        |           5e-07 |            1 |   0.0101895 | https://wandb.ai/rl4aa/ask-before-answer/runs/rs04nh5a |
| bp50774y | dpo_training  | dpo_baseline  | N/A          | N/A        |           5e-07 |            1 |   0.0148286 | https://wandb.ai/rl4aa/ask-before-answer/runs/bp50774y |
| g6awb9l8 | sft_training  | sft_baseline  | N/A          | N/A        |           2e-05 |            1 |   0.223156  | https://wandb.ai/rl4aa/ask-before-answer/runs/g6awb9l8 |
| lduzpogd | sft_training  | sft_baseline  | N/A          | N/A        |           2e-05 |            1 |   0.223236  | https://wandb.ai/rl4aa/ask-before-answer/runs/lduzpogd |
| qgosp3j0 | dpo_training  | dpo_baseline  | N/A          | N/A        |           5e-07 |            1 |   0.549293  | https://wandb.ai/rl4aa/ask-before-answer/runs/qgosp3j0 |
| 4gqg7kuq | dpo_training  | dpo_baseline  | N/A          | N/A        |           5e-07 |            1 |   0.556035  | https://wandb.ai/rl4aa/ask-before-answer/runs/4gqg7kuq |
| tusrhca7 | orpo_training | orpo_baseline | N/A          | N/A        |           5e-06 |            1 |   1.36969   | https://wandb.ai/rl4aa/ask-before-answer/runs/tusrhca7 |
| fl9cyvti | orpo_training | orpo_baseline | N/A          | N/A        |           5e-06 |            1 |   1.37168   | https://wandb.ai/rl4aa/ask-before-answer/runs/fl9cyvti |

## 2. Learning Curves

![Training Loss](plots/train_loss_comparison.png)

![Eval Loss](plots/eval_loss_comparison.png)

## 3. Validation Curves

![Validation Curve](plots/val_curve_lr.png)

## 4. LLM-as-a-Judge Evaluation Leaderboard

The following scores were computed using W&B Weave with a Gemini-based judge scorer.

| Metric                |      base |   dpo_only |      sft |   sft_dpo |   clarifier_lora |     orpo |     grpo | grpo_u8xjwcjv |
|:----------------------|----------:|-----------:|---------:|----------:|-----------------:|---------:|---------:|--------------:|
| ambiguity_detection   | 0.966     |   0.968    | 0.97     |  0.946    |          0.994   | 0.968    | 0.97     |      0.97     |
| clarification_quality | 0.784     |   0.786    | 0.796    |  0.778    |          0.796   | 0.796    | 0.8      |      0.8      |
| usefulness            | 0.88      |   0.882    | 0.896    |  0.878    |          0.896   | 0.896    | 0.9      |      0.892    |
| model_accuracy        | 0.62      |   0.64     | 0.62     |  0.62     |          0.6     | 0.62     | 0.62     |      0.62     |
| clarify_precision     | 0.617021  |   0.630435 | 0.641026 |  0.657143 |          0.6     | 0.622222 | 0.634146 |      0.634146 |
| clarify_recall        | 0.966667  |   0.966667 | 0.833333 |  0.766667 |          1       | 0.933333 | 0.866667 |      0.866667 |
| clarify_f1            | 0.753247  |   0.763158 | 0.724638 |  0.707692 |          0.75    | 0.746667 | 0.732394 |      0.732394 |
| action_f1_answer      | 0.173913  |   0.25     | 0.387097 |  0.457143 |          0       | 0.24     | 0.344828 |      0.344828 |
| macro_f1              | 0.46358   |   0.506579 | 0.555867 |  0.582418 |          0.375   | 0.493333 | 0.538611 |      0.538611 |
| answer_accuracy       | 0.05      |   0.05     | 0.05     |  0        |          0       | 0.05     | 0.1      |      0.15     |
| facet_generation_rate | 0.0851064 |   0.108696 | 1        |  1        |          0       | 0.955556 | 1        |      1.0      |
| clarify_ratio         | 1.56667   |   1.53333  | 1.3      |  1.16667  |          1.66667 | 1.5      | 1.36667  |      1.37     |
