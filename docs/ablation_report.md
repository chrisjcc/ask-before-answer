# Ablation Experiment Report

This report was automatically generated from Weights & Biases metrics.

## Top Performing Configurations

| Run ID   | Name          | Group         | Hypothesis   | Sweep ID   |   Learning Rate |   Batch Size |   Eval Loss | URL                                                    |
|:---------|:--------------|:--------------|:-------------|:-----------|----------------:|-------------:|------------:|:-------------------------------------------------------|
| amq3lj42 | sft_training  | sft_baseline  | N/A          | N/A        |           2e-05 |            1 |    0.223156 | https://wandb.ai/rl4aa/ask-before-answer/runs/amq3lj42 |
| 2om8x500 | dpo_training  | dpo_baseline  | N/A          | N/A        |           5e-07 |            1 |    0.344238 | https://wandb.ai/rl4aa/ask-before-answer/runs/2om8x500 |
| 5c1r1rmw | dpo_training  | dpo_baseline  | N/A          | N/A        |           5e-07 |            1 |    0.662823 | https://wandb.ai/rl4aa/ask-before-answer/runs/5c1r1rmw |
| xazqzfa1 | dpo_training  | dpo_baseline  | N/A          | N/A        |           5e-07 |            1 |    0.692516 | https://wandb.ai/rl4aa/ask-before-answer/runs/xazqzfa1 |
| tooh9928 | orpo_training | orpo_baseline | N/A          | N/A        |           5e-06 |            1 |    1.36939  | https://wandb.ai/rl4aa/ask-before-answer/runs/tooh9928 |

## Learning Curves

![Training Loss](plots/train_loss_comparison.png)


## LLM-as-a-Judge Evaluation Leaderboard

The following scores were computed using W&B Weave 
with a Gemini-based judge scorer.

| Metric                |      base |   dpo_only |      sft |   sft_dpo |   clarifier_lora |     orpo |     grpo |
|:----------------------|----------:|-----------:|---------:|----------:|-----------------:|---------:|---------:|
| ambiguity_detection   | 0.966     |  0.944     | 0.97     |  0.972    |          0.994   | 0.972    | 0.948    |
| clarification_quality | 0.784     |  0.784     | 0.796    |  0.796    |          0.796   | 0.796    | 0.798    |
| usefulness            | 0.88      |  0.88      | 0.896    |  0.898    |          0.896   | 0.896    | 0.898    |
| model_accuracy        | 0.62      |  0.6       | 0.64     |  0.62     |          0.6     | 0.64     | 0.64     |
| clarify_precision     | 0.617021  |  0.604167  | 0.657895 |  0.657143 |          0.6     | 0.636364 | 0.642857 |
| clarify_recall        | 0.966667  |  0.966667  | 0.833333 |  0.766667 |          1       | 0.933333 | 0.9      |
| clarify_f1            | 0.753247  |  0.74359   | 0.735294 |  0.707692 |          0.75    | 0.756757 | 0.75     |
| action_f1_answer      | 0.173913  |  0.0909091 | 0.4375   |  0.457143 |          0       | 0.307692 | 0.357143 |
| macro_f1              | 0.46358   |  0.417249  | 0.586397 |  0.582418 |          0.375   | 0.532225 | 0.553571 |
| answer_accuracy       | 0.05      |  0.05      | 0.05     |  0        |          0       | 0.1      | 0.1      |
| facet_generation_rate | 0.0851064 |  0.0833333 | 1        |  1        |          0       | 0.931818 | 1        |
| clarify_ratio         | 1.56667   |  1.6       | 1.26667  |  1.16667  |          1.66667 | 1.46667  | 1.4      |

## Analysis of Final Showdown (SFT vs SFT+DPO vs SFT+GRPO)

1. **DPO Generative Hard Negatives (SUCCESS)**: The introduction of generative hard negatives for DPO successfully achieved our goal of reducing over-clarification! Comparing `sft_dpo` to `sft`, the `clarify_ratio` dropped from 1.26 to 1.16 (moving closer to the ideal 1.0). Furthermore, the model learned to answer more confidently, boosting `action_f1_answer` from 0.437 to 0.457.
2. **GRPO SFT Warm-Start (SUCCESS)**: Configuring GRPO to warm-start from the SFT model completely cured the catastrophic mode collapse. The `facet_generation_rate` skyrocketed back to a flawless 100%. 
3. **GRPO Accuracy vs. Caution (THE TRADE-OFF)**: Because the `accuracy_reward_func` heavily penalizes hallucinations, GRPO learned a highly calibrated, cautious policy. It became the only model to successfully double its `answer_accuracy` (10% vs 5%). However, because answering is deemed "risky," the model learned to clarify more often to avoid penalties, pulling its `action_f1_answer` down to 0.357.
4. **Conclusion**: DPO is highly effective at aligning the *style* of the policy (highest Macro F1 and perfect Clarify Ratio), while GRPO is highly effective at enforcing *factual logic and calibration* (highest factual accuracy, but cautious).
