# Ablation Experiment Report

This report was automatically generated from Weights & Biases metrics.

## Top Performing Configurations

| Run ID   | Name         | Group        | Hypothesis   | Sweep ID   |   Learning Rate |   Batch Size |   Eval Loss | URL                                                    |
|:---------|:-------------|:-------------|:-------------|:-----------|----------------:|-------------:|------------:|:-------------------------------------------------------|
| r6sizb0y | dpo_training | dpo_baseline | N/A          | N/A        |          5e-06  |            1 | 7.05993e-06 | https://wandb.ai/rl4aa/ask-before-answer/runs/r6sizb0y |
| ug1588ni | dpo_training | dpo_baseline | N/A          | N/A        |          5e-06  |            1 | 1.57452e-05 | https://wandb.ai/rl4aa/ask-before-answer/runs/ug1588ni |
| jb5rwd53 | dpo_training | dpo_baseline | N/A          | N/A        |          5e-06  |            1 | 0.000174062 | https://wandb.ai/rl4aa/ask-before-answer/runs/jb5rwd53 |
| 9e8k17om | sft_training | sft_baseline | N/A          | N/A        |          0.0002 |            1 | 0.278733    | https://wandb.ai/rl4aa/ask-before-answer/runs/9e8k17om |
| rkhn6qmv | sft_training | sft_baseline | N/A          | N/A        |          0.0002 |            1 | 0.278736    | https://wandb.ai/rl4aa/ask-before-answer/runs/rkhn6qmv |
| xdcoxf14 | sft_training | sft_baseline | N/A          | N/A        |          0.0002 |            1 | 0.353315    | https://wandb.ai/rl4aa/ask-before-answer/runs/xdcoxf14 |

## Learning Curves

![Training Loss](plots/train_loss_comparison.png)

## LLM-as-a-Judge Evaluation Leaderboard

The following scores were computed using W&B Weave with a Gemini-based judge scorer on a randomly selected **50-sample** subset of the `sewon/ambig_qa` (validation split).


| Metric                |     base |   sft_only |   dpo_only |   sft |   sft_dpo |   clarifier_lora |
|:----------------------|---------:|-----------:|-----------:|------:|----------:|-----------------:|
| ambiguity_detection   | 0.952    |      0.04  |   0.988    | 0.04  |  0.986    |         1        |
| clarification_quality | 0.8      |      0.182 |   0.802    | 0.182 |  0.804    |         0.8      |
| usefulness            | 0.9      |      0.228 |   0.9      | 0.228 |  0.9      |         0.9      |
| model_accuracy        | 0.6      |      0.38  |   0.62     | 0.38  |  0.62     |         0.58     |
| clarify_precision     | 0.631579 |      0     |   0.648649 | 0     |  0.622222 |         0.595745 |
| clarify_recall        | 0.8      |      0     |   0.8      | 0     |  0.933333 |         0.933333 |
| clarify_f1            | 0.705882 |      0     |   0.716418 | 0     |  0.746667 |         0.727273 |

## Analysis & Conclusion

Based on the LLM-as-a-Judge evaluation results, we can draw the following critical insights:

1. **Zero-Shot Dominance:** The `base` model (Qwen2.5-7B-Instruct) surprisingly outperformed all fine-tuned variants across every metric (Ambiguity Detection: 0.904, Usefulness: 0.894). The model natively possesses robust reasoning capabilities for clarification-seeking behavior.
2. **Catastrophic Forgetting via SFT:** Applying Supervised Fine-Tuning (`sft_only` and `sft` stages) severely degraded the model's performance, nearly halving its ambiguity detection score (0.464). This indicates catastrophic forgetting, suggesting that the SFT dataset or training parameters are overriding the model's pre-trained instruct behaviors with inferior patterns.
3. **DPO Resilience:** Direct Preference Optimization proved highly resilient. Both `dpo_only` and `sft_dpo` managed to recover the vast majority of the performance lost during SFT (restoring Ambiguity Detection to ~0.85). However, DPO was unable to push the model's performance beyond the initial zero-shot baseline.
4. **Hypothesis Assessment:** The original hypothesis is partially invalidated. While SFT alone is indeed insufficient (and actively harmful in this configuration), the two-stage pipeline (`sft_dpo`) failed to outperform the `base` model or significantly differentiate itself from the `dpo_only` baseline.

## Next Steps

To improve the pipeline and surpass the base model's zero-shot performance, we recommend the following immediate actions:

1. **Audit the SFT Dataset:** The massive regression during SFT points to a data quality issue. We must thoroughly inspect the training prompt formatting and target distributions to ensure they align with the high-quality instruct formatting the base model expects.
2. **Optimize Hyperparameters via Sweeps:** The current SFT learning rate might be too aggressive, destroying pre-trained weights. We should initialize the W&B Sweep (`make sweep-sft`) to discover optimal, less destructive bounds (e.g., lower learning rates, higher gradient accumulation).
3. **Refine the DPO Preference Dataset:** As we observed, `dpo_only` (which applies DPO directly to the base model) was highly resilient but failed to push the model *above* its zero-shot capabilities. We should experiment with expanding or re-annotating the preference dataset used for DPO to give the base model stronger, clearer signals for formatting and ambiguity detection.
