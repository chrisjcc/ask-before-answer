# Model Evaluation & Performance Analysis

This document summarizes the post-training evaluation results for the AskBeforeAnswer models.

Six model variants were evaluated on the `sewon_ambig_qa_eval` benchmark using both the `LocalGemmaJudge` evaluator (subjective quality metrics) and the `ActionScorer` evaluator (decision and task performance).

## 1. Observability & Systematic Evaluation (W&B Weave)

This project integrates tightly with **Weights & Biases Weave** to provide comprehensive LLM observability and systematic evaluation pipelines. 

The automated evaluation pipeline (`scripts/evaluate.py`) uses a dual-scoring approach to systematically evaluate all model configurations against the test dataset:

**1. LLM-as-a-Judge (Gemini 2.5 Flash / Gemma 4):**
Evaluates the subjective nuance and quality of the response:
- Ambiguity Detection F1
- Clarification Quality F1
- Clarification Usefulness

**2. Rule-Based Programmatic Scoring (`ActionScorer`):**
Evaluates the deterministic structural accuracy of the agent's chosen action:
- Model Accuracy (Raw percentage of correct `Action` choices—Clarify vs. Answer—compared to the ground-truth labels).

To run the full suite and generate a dynamic leaderboard on Weave:
```bash
make evaluate
```

To run a specific evaluation configuration (e.g. `configs/evaluation/custom.yaml`) without modifying the default:
```bash
make evaluate EVAL_CONFIG=custom
```

## 1. The Core Trade-off: Clarification vs Answering

The post-training evaluation demonstrates that there is **no single model that dominates every metric**. The evaluated models naturally separate into two distinct policies:

1. **Clarification-first models**: Prioritize identifying ambiguity and asking clarifying questions, even at the expense of providing final answers.
2. **Balanced models**: Preserve strong clarification behavior while also producing accurate and useful final responses once sufficient information has been obtained.

For a general-purpose interactive assistant, the **balanced policy** is strictly preferred, as a model that refuses to answer unambiguous questions (e.g., the Clarifier LoRA) suffers from mode collapse.

## 2. Definitive Model Ranking (Balanced Policy)

Based on the holistic balance of both structural performance (Macro F1, Answer F1) and subjective quality (Usefulness, Clarification Quality), here is the definitive ranking of the top 3 models:

| Rank | Model | Macro F1 | Answer F1 | Clarify F1 | Assessment |
|------|-------|----------|-----------|------------|-------------------------|
| 🥇 **1st** | **sft** | **0.615** | **0.485** | 0.746 | The undisputed overall winner. It achieved the highest Macro F1 and Answer F1, proving it has the best calibration for knowing exactly when it is safe to answer directly versus when it must ask for clarification. Its `clarify_ratio` (1.23) is the closest to the ideal 1.0. |
| 🥈 **2nd** | **dpo_only** | 0.553 | 0.357 | 0.750 | A surprisingly strong performer. By applying DPO with Hard Negatives directly to the base model, it successfully learned semantic boundaries, doubling the answer capability of the base model while maintaining strong ambiguity detection (0.964). |
| 🥉 **3rd** | **grpo** | 0.532 | 0.308 | **0.757** | The Reinforcement Learning variant. While it struggles slightly with answer confidence compared to `sft` and `dpo_only`, it achieved the **highest `clarify_f1` (0.757) of any functional model**, proving the trial-and-error rollout phase made it highly precise at extracting relevant facets. |

### Note on other models:
- **sft_dpo** (4th) suffered from excessive caution, dropping its `answer_f1` to 0.296.
- **base** (5th) heavily biased toward over-clarification and failed to confidently answer unambiguous questions.
- **clarifier_lora** (6th) suffered complete mode collapse, refusing to answer any questions (`answer_f1` = 0).

## 3. Key Metric Takeaways

*   **Ambiguity Detection:** Variance here is marginal; all models perform exceptionally well (>94%). The base model already possesses strong foundational phrasing.
*   **Clarification Quality:** Post-training slightly bumps quality, but again, the baseline is already strong.
*   **Model Accuracy:** SFT-backed models consistently hit the ceiling of 0.64. The gap is small but consistent.
