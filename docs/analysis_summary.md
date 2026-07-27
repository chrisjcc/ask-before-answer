# Model Evaluation Summary (Updated July 2026)

## Overview

Six model variants were evaluated on the `sewon_ambig_qa_eval` benchmark using both the `LocalGemmaJudge` evaluator (subjective quality metrics) and the `ActionScorer` evaluator (decision and task performance).

The evaluated models correspond to the following training configurations:

| Model | Training Configuration |
|--------|------------------------|
| **base** | Qwen2.5 baseline |
| **sft** | Supervised Fine-Tuning |
| **dpo_only** | Direct Preference Optimization only |
| **sft_dpo** | SFT followed by DPO |
| **clarifier_lora** | Clarification-specialized LoRA |
| **grpo** | Group Relative Policy Optimization |

---

# Overall Ranking

Based on the holistic balance of both structural performance (Macro F1, Answer F1) and subjective quality (Usefulness, Clarification Quality), here is the definitive ranking of the top 3 models:

| Rank | Model | Macro F1 | Answer F1 | Clarify F1 | Why it earned this rank |
|------|-------|----------|-----------|------------|-------------------------|
| 🥇 **1st** | **sft** | **0.615** | **0.485** | 0.746 | The undisputed overall winner. It achieved the highest Macro F1 and Answer F1, proving it has the best calibration for knowing exactly when it is safe to answer directly versus when it must ask for clarification. Its `clarify_ratio` (1.23) is the closest to the ideal 1.0. |
| 🥈 **2nd** | **dpo_only** | 0.553 | 0.357 | 0.750 | A surprisingly strong performer. By applying DPO with Hard Negatives directly to the base model, it successfully learned semantic boundaries, doubling the answer capability of the base model while maintaining strong ambiguity detection (0.964). |
| 🥉 **3rd** | **grpo** | 0.532 | 0.308 | **0.757** | The new Reinforcement Learning variant edges out `sft_dpo` for third place. While it struggles slightly with answer confidence compared to `sft` and `dpo_only`, it achieved the **highest `clarify_f1` (0.757) of any functional model**, proving the trial-and-error rollout phase made it highly precise at extracting relevant facets when asking questions. |

*Note on other models:* 
- `sft_dpo` (4th) suffered from excessive caution, dropping its `answer_f1` to 0.296.
- `base` (5th) heavily biased toward over-clarification and failed to confidently answer unambiguous questions.
- `clarifier_lora` (6th) suffered complete mode collapse, refusing to answer any questions (`answer_f1` = 0).

---

# Detailed Analysis

## 1. SFT (🥇 First Place)

The SFT model represents the strongest overall performer and achieves the best balance between asking clarification questions and answering directly when appropriate.

Although its ambiguity detection score (97.2%) is marginally below several other variants, it substantially improves downstream decision quality:

- Highest Action Accuracy (66%)
- Highest Clarification Precision (67.6%)
- Highest Answer F1 (48.5%)
- Highest Macro F1 (61.6%)

These improvements indicate that SFT not only detects ambiguity effectively, but also learns when clarification is unnecessary, reducing over-clarification while preserving strong answer quality. Overall, this represents the best production-ready trade-off.

## 2. DPO Only (🥈 Second Place)

Applying DPO without an initial SFT stage produces a balanced and capable model.

Performance remains relatively consistent across both judge-based and action-based metrics, with:

- Strong clarification recall (90%)
- Good ambiguity detection (96.4%)
- Solid Answer F1 (35.7%)

Compared with SFT, the model is noticeably weaker at determining when it is safe to answer directly, resulting in lower overall Macro F1. However, it still significantly outperforms the base model, suggesting the Hard Negatives dataset was highly effective.

## 3. GRPO (🥉 Third Place)

Group Relative Policy Optimization (GRPO) introduces a trial-and-error rollout phase, scoring generated answers against our reward functions. 

The results show that GRPO is highly effective at structuring its thoughts:
- It achieved the highest `clarify_f1` (0.757) of any functional model.
- It maintained a perfect `facet_generation_rate` (1.0).
- It achieved a solid ambiguity detection score (94.6%).

However, similar to `sft_dpo`, the reinforcement learning phase made the model slightly too conservative. Its `answer_f1` (0.308) trails behind the purely supervised approaches, indicating the reward function might be penalizing incorrect answers too harshly, causing the model to default to clarification.

## 4. SFT + DPO (Honorable Mention)

Adding DPO after SFT slightly improves the subjective quality metrics:

- Highest Clarification Quality (80%)
- Highest Usefulness (90%)
- Slightly improved ambiguity detection (97.6%)

However, these gains do not translate into improved downstream task performance. Compared to SFT, the model shows significantly reduced Answer F1 (29.6%) and lower Macro F1 (51.8%). This suggests that DPO shifts the policy toward more conservative clarification behaviour, producing higher-quality clarification questions at the expense of answering capability.
