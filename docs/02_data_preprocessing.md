# Data Preprocessing & Generation Pipeline

This document conceptually outlines the end-to-end data generation pipeline used in this repository. It provides a clear blueprint for how raw AmbigNQ data is transformed into structured, contrastive training examples using the `qwen2.5-7b-instruct` model. 

The pipeline is designed to enforce Chain-of-Thought (CoT) reasoning and structural schema alignment, ensuring that the final preference dataset is perfectly symmetric for Direct Preference Optimization (DPO).

## 1. Execution

Data preprocessing is the first ML pipeline stage and is tracked by DVC.

```bash
make preprocess
```

The preprocessing code lives under `src/ask_before_answer/data/` and produces the datasets consumed by the subsequent training stages (`sft_train.jsonl`, `dpo_train.jsonl`, etc.).

## 2. Pipeline Flow

```text
[Raw AmbigNQ Data] 
       │
       ▼
[Stage 1: Schema Enforcement & Open-Ended Extraction (7B)]
       │
       ▼
[Stage 2: SFT "Chosen" Synthesis (Positive Targets)]
       │
       ▼
[Stage 3: DPO "Rejected" Synthesis (Prompt-Guided Synthetic Generation)]
```

## 3. Stage 1: Base Data and Schema Enforcement

The foundation of our dataset is the raw AmbigNQ corpus. To align our model to a "clarify-first" behavior, we process this raw data using a lightweight instructor model (`qwen2.5-7b-instruct`). 

To prevent the model from generating unstructured conversational hallucinations, the instructor model is prompted to adhere to a strict, 4-field structural schema for every response:

*   **Action:** A binary classification (`Clarify` or `Answer`).
*   **Reasoning:** A Chain-of-Thought (CoT) trace explaining *why* the question is ambiguous or clear.
*   **Facets:** A parsed JSON array of the specific semantic attributes missing from the query.
*   **Response:** The final conversational text.

> [!NOTE]
> **Open-Ended Facet Extraction:** The facets extracted by the model are **open-ended**. The prompt does *not* restrict the model to a finite, hardcoded list of available facets. Instead, the 7B model dynamically generates the semantic categories.

## 4. Stage 2: SFT "Chosen" Synthesis (Positive CoT Distillation)

The raw `sewon/ambig_qa` dataset only contains questions and ground-truth QA annotations—it does **not** come with Chain-of-Thought (CoT) reasoning traces.

During `make preprocess`, `SyntheticGenerator` (in `src/ask_before_answer/data/preprocess.py`, configured via `configs/data/default.yaml`) uses `unsloth/qwen2.5-7b-instruct-unsloth-bnb-4bit` to synthesize the "Gold" (chosen) response using `generate_batch()` and `SYSTEM_PROMPT`. This produces **Positive Reasoning (`reasoning`)**, explaining *why* the query has multiple plausible interpretations (for `Clarify`) or why it is unambiguous (for `Answer`).

**How SFT Uses Reasoning Traces:** By including the `Reasoning` field in the target, SFT teaches the model to internalize the Chain-of-Thought process—decomposing the ambiguity state so it can extract accurate `Facets` and synthesize a grounded `Response`.

## 5. Stage 3: DPO "Rejected" Synthesis (Adversarial Hard-Negative CoT)

To perform Direct Preference Optimization (DPO) and Odds Ratio Preference Optimization (ORPO), the algorithms require a contrastive pair for every question: a "chosen" target ($y^+$) and a "rejected" target ($y^-$). 

Instead of expensive Model-Grounded Mining, we use **Prompt-Guided Synthetic Generation** via `generate_negative_batch()` and `NEGATIVE_SYSTEM_PROMPT`. We explicitly instruct the 7B instructor model to intentionally generate an *incorrect but plausible-sounding* reasoning chain (`neg_reasoning`) justifying the wrong action. The 7B model generates a structurally valid 4-field block containing logically flawed reasoning.

**How DPO & ORPO Use Reasoning Traces:** This Prompt-Guided Synthetic Generation process ensures that the dataset structure is **perfectly symmetric** between the chosen and rejected samples. Because DPO and ORPO optimize log-probability margins over the *entire* generated sequence, they directly penalize the *flawed logic* inside the rejected `neg_reasoning` trace while shifting probability mass toward the valid `reasoning` trace. This structurally symmetric design prevents the model from "reward hacking" based on surface formatting.

## 6. Autoregressive Ordering Nuance (`Action` $\rightarrow$ `Reasoning` $\rightarrow$ `Facets` $\rightarrow$ `Response`)

> [!NOTE]
> Because the 4-field schema places `Action:` on Line 1 and `Reasoning:` on Line 2:
> ```text
> Action: Clarify | Answer
> Reasoning: <Chain-of-Thought rationale explaining the ambiguity state>
> Facets: [<List of missing informational attributes if Clarify, else empty>]
> Response: <Targeted clarifying question OR direct terminal answer>
> ```
> In causal (left-to-right) autoregressive decoding, the `Reasoning:` tokens are generated *after* the binary `Action:` token is already chosen, and *before* `Facets:` and `Response:`. Thus, at generation time, the `Reasoning:` trace primarily acts as an explicit **conditioning bridge** that decomposes the ambiguity so the model can extract accurate `Facets` and formulate a grounded clarifying question in `Response`.
>
> *For a complete breakdown of how each training algorithm (SFT, DPO, ORPO, and GRPO) optimizes the `Reasoning:` tag during training and evaluation, see [Model Training](03_model_training.md).*

