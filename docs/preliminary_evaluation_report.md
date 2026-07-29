```markdown
# Analysis of Post-Training Evaluation Results

## Overview

The post-training evaluation demonstrates that there is **no single model that dominates every metric**. Instead, the evaluated models naturally separate into two distinct categories:

1. **Clarification-first models**, which prioritise identifying ambiguity and asking clarifying questions, even at the expense of providing final answers.

2. **Balanced models**, which preserve strong clarification behaviour while also producing accurate and useful final responses once sufficient information has been obtained.

This distinction is important because the intended deployment objective determines which class of model is preferable. A clarification-first model may be desirable for applications where avoiding incorrect assumptions is paramount, whereas a balanced model is generally better suited for interactive assistants that must both clarify when necessary and eventually answer the user's question.

---

# Clarification-First Analysis

The clarification-first ranking emphasises metrics associated with recognising ambiguity and requesting clarification:

- Ambiguity Detection
- Clarification Quality
- Clarify Recall
- Clarify Precision
- Clarify F1

Answer generation metrics are intentionally given lower weight in this ranking.

| Rank | Model | Assessment |
|------|-------|------------|
| **1** | **Clarifier LoRA** | Outstanding ambiguity detection and clarification behaviour, but heavily biased towards requesting clarification. |
| **2** | **SFT + DPO** | Excellent clarification quality while maintaining a more balanced policy. |
| **3** | **SFT** | Nearly identical to SFT+DPO with slightly higher clarification recall. |
| **4** | **GRPO** | Very strong clarification performance with the highest Clarify F1. |
| **5** | **Qwen-2.5 Base** | Surprisingly capable ambiguity detector despite no task-specific training. |
| **6** | **DPO Only** | Similar behaviour to the base model but generally weaker across clarification metrics. |

## Discussion

The **Clarifier LoRA** model achieves the strongest clarification-oriented performance across nearly every clarification-related metric.

Its strengths include:

- Highest ambiguity detection.
- Highest clarification quality.
- Perfect clarification recall.
- Highest usefulness score.

However, these strengths come at a significant cost. The model exhibits essentially no answer generation capability, with an Answer F1 of zero and no successful answer accuracy. This suggests that the model has learned a policy similar to:

> *When uncertain, always ask another question.*

rather than

> *Clarify when necessary, then answer once enough information has been obtained.*

Consequently, while Clarifier LoRA represents the strongest clarification policy, it is not suitable as a complete Clarify-or-Act assistant.

The remaining models demonstrate progressively more balanced behaviour, with SFT+DPO and SFT preserving strong clarification performance while avoiding excessive clarification.

---

# Balanced Model Analysis

The balanced ranking evaluates each model as a complete Clarify-or-Act system.

The following metrics are weighted most heavily:

- Macro F1
- Answer F1
- Clarify F1
- Clarification Quality
- Usefulness
- Accuracy

This ranking rewards models that successfully determine **when to clarify** and **when to provide a final answer**.

| Rank | Model | Assessment |
|------|-------|------------|
| **1** | **SFT + DPO** | Best overall balance between clarification and answering. |
| **2** | **SFT** | Extremely close to SFT+DPO and an excellent overall baseline. |
| **3** | **GRPO** | Strong compromise between clarification and answer generation. |
| **4** | **Qwen-2.5 Base** | Surprisingly competitive despite no fine-tuning. |
| **5** | **DPO Only** | Improves little over the base model and underperforms SFT-based approaches. |
| **6** | **Clarifier LoRA** | Outstanding clarification behaviour but fails as a balanced assistant due to lack of answer generation. |

## Discussion

### SFT + DPO

SFT+DPO emerges as the strongest overall model.

It does not necessarily achieve the highest score in every individual metric, but it consistently performs near the top across all major evaluation criteria.

Notable characteristics include:

- Highest Macro F1.
- Highest Answer F1.
- Highest clarification precision.
- Excellent clarification quality.
- Strong ambiguity detection.

Importantly, there are no obvious weaknesses. The model neither over-clarifies nor under-clarifies and demonstrates the strongest overall balance between recognising ambiguity and producing correct final answers.

---

### SFT

The supervised fine-tuned model performs remarkably close to SFT+DPO.

Its strengths include:

- Strong clarification quality.
- Excellent clarification recall.
- High answer quality.
- High overall accuracy.

The differences between SFT and SFT+DPO are relatively small. SFT tends to favour slightly higher clarification recall, whereas SFT+DPO achieves better answer quality and overall Macro F1.

This suggests that DPO primarily serves as a refinement stage rather than introducing fundamentally different behaviour.

---

### GRPO

The GRPO model occupies a clear middle ground.

Its primary strengths are:

- Highest Clarify F1.
- Excellent clarification recall.
- Improved answer quality relative to the base model.

However, answer generation remains weaker than both SFT-based models, resulting in a lower Macro F1.

Overall, GRPO demonstrates promising behaviour but has not yet surpassed the supervised training pipeline.

---

### Qwen-2.5 Base

Perhaps the most surprising result is the strength of the unmodified base model.

Despite receiving no task-specific training, it demonstrates:

- Strong ambiguity detection.
- Excellent clarification recall.
- Competitive Clarify F1.

Its primary weakness lies in answer generation, where performance drops substantially compared to the fine-tuned models.

Nevertheless, the results indicate that the underlying base model already possesses a strong implicit understanding of conversational ambiguity.

---

### DPO Only

Applying Direct Preference Optimisation without a preceding supervised fine-tuning stage produces the weakest fine-tuned model.

Although clarification behaviour remains reasonable, answer quality deteriorates and overall Macro F1 decreases relative to the base model.

These findings are consistent with the broader literature, where DPO is typically used to refine an existing policy rather than to learn task behaviour from scratch.

---

### Clarifier LoRA

Although Clarifier LoRA ranks first under the clarification-first evaluation, it ranks last as a balanced Clarify-or-Act model.

This apparent contradiction highlights the distinction between optimisation objectives.

Clarifier LoRA successfully learns to identify ambiguous situations and request clarification, but it fails to transition into answer generation once sufficient information has been obtained.

As a result, it behaves more like a dedicated ambiguity detector than a complete conversational agent.

---

# Training Method Insights

Several important observations emerge from the comparison of training strategies.

## Supervised Fine-Tuning provides the largest improvement

Moving from the Qwen-2.5 base model to SFT yields the largest overall improvement.

SFT substantially improves:

- Answer generation.
- Overall Macro F1.
- Clarification precision.
- Overall balance.

This indicates that supervised fine-tuning successfully teaches the desired Clarify-or-Act behaviour.

---

## DPO is most effective as a refinement stage

Comparing SFT and SFT+DPO shows consistent, albeit modest, improvements.

These improvements include:

- Higher Answer F1.
- Higher Macro F1.
- Better precision-recall balance.

Rather than dramatically changing behaviour, DPO appears to refine the policy learned during supervised fine-tuning.

---

## DPO alone is insufficient

The DPO-only model consistently underperforms SFT and even trails the base model on several balanced evaluation metrics.

This suggests that preference optimisation alone is insufficient for learning the desired clarification policy and is considerably more effective when applied after supervised fine-tuning.

---

## GRPO is promising but not yet superior

GRPO demonstrates a strong balance between clarification and answer generation while preserving excellent clarification behaviour.

Although it does not outperform SFT+DPO, it substantially exceeds both the base model and DPO-only, indicating that reinforcement learning represents a promising alternative direction for future work.

---

## Clarification objectives must remain balanced

The Clarifier LoRA model demonstrates that optimising too aggressively for clarification behaviour can degrade the overall conversational experience.

A successful Clarify-or-Act system must not only determine **when clarification is necessary**, but must also confidently provide an answer once sufficient information has been gathered.

Over-emphasising clarification alone risks producing assistants that repeatedly ask questions without successfully completing the interaction.

---

# Overall Conclusions

The evaluation clearly demonstrates that different training strategies optimise different aspects of conversational behaviour.

When viewed purely as ambiguity detectors, Clarifier LoRA produces the strongest clarification policy.

However, when evaluated as complete Clarify-or-Act systems, the supervised training approaches clearly outperform the remaining models.

Overall, **SFT+DPO** provides the strongest balance between clarification and answer generation, with **SFT** performing nearly as well. **GRPO** represents a promising reinforcement learning approach but has not yet surpassed the supervised pipeline. Meanwhile, **DPO Only** confirms that preference optimisation is substantially more effective when used to refine an existing supervised policy rather than replacing supervised learning entirely.

These findings reinforce the broader conclusion that there is no universally best model. Instead, the optimal choice depends on whether the deployment objective prioritises **maximising clarification behaviour** or **maintaining a balanced Clarify-or-Act interaction**.
```
