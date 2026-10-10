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

## 4. Stage 2: SFT "Chosen" Synthesis (Positive Targets)

To generate the SFT dataset, the 7B instructor model is prompted to act as an expert agent. It analyzes the ground-truth question and synthesizes the "Gold" (chosen) response using the 4-field schema.

**How SFT Uses Reasoning Traces:** By including the `Reasoning` field in the target, SFT teaches the model to internalize the Chain-of-Thought process—forcing it to logically deduce the ambiguity state *before* it predicts the final action or response.

## 5. Stage 3: DPO "Rejected" Synthesis 

To perform Direct Preference Optimization (DPO), the algorithm requires a contrastive pair for every question: a "chosen" target ($y^+$) and a "rejected" target ($y^-$). 

Instead of expensive Model-Grounded Mining, we use a simpler, cheaper, and native approach called **Prompt-Guided Synthetic Generation**.

We explicitly instruct the 7B instructor model (using a negative system prompt) to intentionally generate an *incorrect but plausible-sounding* reasoning chain. The 7B model generates a structurally perfect but logically flawed response.

**How DPO Uses Reasoning Traces:** This Prompt-Guided Synthetic Generation process ensures that the dataset structure is **perfectly symmetric** between the chosen and rejected samples. Because DPO optimizes the log-probability margins over the *entire* generated sequence, the algorithm directly penalizes the *flawed logic* inside the rejected reasoning trace. This structurally symmetric design prevents the model from "reward hacking" based on formatting.
