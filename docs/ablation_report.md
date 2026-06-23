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

The following scores were computed using W&B Weave with a local Gemma-based judge scorer (google/gemma-2-9b-it).

| Model    |   ambiguity_detection |   clarification_quality |   usefulness |
|:---------|----------------------:|------------------------:|-------------:|
| base     |                 0.904 |                   0.788 |        0.894 |
| sft_only |                 0.526 |                   0.61  |        0.718 |
| dpo_only |                 0.858 |                   0.746 |        0.842 |
| sft      |                 0.464 |                   0.586 |        0.712 |
| sft_dpo  |                 0.856 |                   0.74  |        0.838 |

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
