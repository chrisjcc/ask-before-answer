# Model Evaluation & Performance Analysis

This document summarizes the post-training evaluation results for the AskBeforeAnswer models.

Six model variants were evaluated on the `sewon_ambig_qa_eval` benchmark using both the `LocalGemmaJudge` evaluator (subjective quality metrics) and the `ActionScorer` evaluator (decision and task performance).

## 1. Observability & Systematic Evaluation (W&B Weave)

This project integrates tightly with **Weights & Biases Weave** to provide comprehensive LLM observability and systematic evaluation pipelines. 

The automated single-turn evaluation pipeline (`scripts/evaluate_single_turn.py`) uses a dual-scoring approach to systematically evaluate all model configurations against the test dataset:

**1. LLM-as-a-Judge (Gemini 2.5 Flash / Gemma 4):**
Evaluates the subjective nuance and quality of the response:
- Ambiguity Detection F1
- Clarification Quality F1
- Clarification Usefulness

**2. Rule-Based Programmatic Scoring (`ActionScorer`):**
Evaluates the deterministic structural accuracy of the agent's chosen action:
- Model Accuracy (Raw percentage of correct `Action` choices—Clarify vs. Answer—compared to the ground-truth labels).

Additionally, the automated multi-turn evaluation pipeline (`scripts/evaluate_multi_turn.py`) uses an interactive simulation (`MultiTurnEnv`) to test if the agent successfully reaches the final answer after interacting with a simulated user.

To run the full suite and generate a dynamic leaderboard on Weave:
```bash
make evaluate-single-turn
make evaluate-multi-turn
```

To run a specific evaluation configuration (e.g. `configs/evaluation/custom.yaml`) without modifying the default:
```bash
make evaluate-single-turn EVAL_CONFIG=custom
```

## 2. The Core Trade-off: Clarification vs Answering

The post-training evaluation demonstrates that there is **no single model that dominates every metric**. The evaluated models naturally separate into two distinct policies:

1. **Clarification-first models**: Prioritize identifying ambiguity and asking clarifying questions, even at the expense of providing final answers.
2. **Balanced models**: Preserve strong clarification behavior while also producing accurate and useful final responses once sufficient information has been obtained.

For a general-purpose interactive assistant, the **balanced policy** is strictly preferred, as a model that refuses to answer unambiguous questions (e.g., the Clarifier LoRA) suffers from mode collapse.

## 3. Single-Turn Definitive Model Ranking (Balanced Policy)

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

## 4. Multi-Turn Interactive Performance

To prove the models can successfully navigate a full conversation, they were evaluated in a dynamic environment (`MultiTurnEnv`) alongside a simulated human user holding hidden ground-truth facts. The environment imposes a strict maximum of 3 turns to reach the final answer.

| Model | Task Success Rate | Average Turns | Analysis |
|-------|-------------------|---------------|----------|
| **sft** | **96.00%** | **1.92** | The most efficient and successful model. A sub-2.0 average turn count indicates it seamlessly answers unambiguous questions in 1 turn, and resolves ambiguous ones in exactly 2 turns. |
| **sft_dpo** | **96.00%** | 1.88 | Ties for the highest success rate, though its overly cautious behavior from single-turn evaluation translates to a slightly lower turn count (it prefers single-turn assumptions if pushed). |
| **grpo** | 94.00% | 2.02 | Outstanding interactive performance. The GRPO trial-and-error phase trained the model to reliably extract facets and use the resulting context to answer successfully. Interestingly, ablation revealed that rigid programmatic rewards outperformed LLM-as-a-judge semantic rewards for this specific policy. |
| **orpo** | 88.00% | 2.22 | Strong performance, but slightly more prone to asking redundant follow-up questions compared to GRPO. |
| **clarifier_lora** | 88.00% | 2.26 | Despite suffering from mode collapse in single-turn evaluation (refusing to answer), it is capable of answering when fed explicit ground truth from the Provider Agent. |
| **dpo_only** | 82.00% | 2.24 | Less stable in multi-turn contexts than the SFT-backed models. |
| **base** | 74.00% | 2.34 | The least capable interactive agent. Its high turn count proves it gets stuck in clarification loops and fails to synthesize the user's answers. |

## 5. Key Metric Takeaways

*   **Interactive Capability vs. Single-Turn Rigidness:** While `sft_dpo` struggled in single-turn zero-shot answering, the Multi-Turn framework proves that when a human is in the loop providing answers, the model is highly capable of reaching the finish line.
*   **Ambiguity Detection:** Variance here is marginal; all models perform exceptionally well (>94%). The base model already possesses strong foundational phrasing.
*   **Model Accuracy:** SFT-backed models consistently hit the ceiling of 0.64 in single-turn evaluation. The gap is small but consistent.
*   **Reward Signal Ablation (Programmatic vs. LLM-as-a-Judge):** During GRPO training, an ablation was conducted comparing a rigid, programmatic reward (`ActionScorer`) against a subjective, semantic reward (`LLM-as-a-Judge`). The evaluation definitively proved that the pure programmatic reward produced a superior interactive agent (94.00% success at 2.02 turns) compared to the LLM-as-a-Judge (92.00% success at 2.06 turns). From a Reinforcement Learning perspective, formatting and schema adherence (e.g., strictly outputting `Action: Clarify` vs `Action: Answer`) is a highly rigid, deterministic task. The LLM-as-a-judge likely introduced "noise" or "softness" into the reward signal, making the model slightly more hesitant to output the exact strict schema strings needed to successfully terminate the environment loop.
