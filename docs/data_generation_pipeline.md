# Conceptual Description of Our Data Generation Pipeline

This document conceptually outlines the end-to-end data generation pipeline used in this repository. It provides a clear blueprint for how raw AmbigNQ data is transformed into structured, contrastive training examples using the `qwen2.5-7b-instruct` model. 

The pipeline is designed to enforce Chain-of-Thought (CoT) reasoning and structural schema alignment, ensuring that the final preference dataset is perfectly symmetric for Direct Preference Optimization (DPO).

## Stage 1: Base Data and Schema Enforcement

The foundation of our dataset is the raw AmbigNQ corpus, which contains questions that may have single or multiple interpretations. To align our model to a "clarify-first" behavior, we process this raw data using a lightweight instructor model (`qwen2.5-7b-instruct`).

To prevent the model from generating unstructured conversational hallucinations, the instructor model is prompted to adhere to a strict, 4-field structural schema for every response:
1. **Action:** A binary classification (`Clarify` or `Answer`).
2. **Reasoning:** A Chain-of-Thought (CoT) trace explaining *why* the question is ambiguous or clear.
3. **Facets:** A JSON array of the specific semantic attributes missing from the query.
4. **Response:** The final conversational text (a targeted clarifying question or a direct answer).

> [!NOTE]
> **Open-Ended Facet Extraction:** The facets extracted by the model are **open-ended**. The prompt does *not* restrict the model to a finite, hardcoded list of available facets (e.g., temporal, spatial). Instead, the 7B model dynamically generates the semantic categories it determines are missing based on the inherent context of the user's question.

## Stage 2: SFT "Chosen" Synthesis (Positive Targets)

To generate the Supervised Fine-Tuning (SFT) dataset, the 7B instructor model is prompted to act as an expert agent. It analyzes the ground-truth AmbigNQ question and synthesizes the "Gold" (chosen) response using the 4-field schema. 

For example, if a question is ambiguous, the model correctly outputs `Action: Clarify`, generates a valid, logical reasoning trace explaining the ambiguity, extracts the open-ended missing facets, and synthesizes a high-quality clarifying question as the `Response`.

**How SFT Uses Reasoning Traces:** During SFT, the model is trained via standard cross-entropy loss over the entire generated string. By including the `Reasoning` field in the target, SFT teaches the model to internalize the Chain-of-Thought process—forcing it to logically deduce the ambiguity state *before* it predicts the final action or response.

## Stage 3: DPO "Rejected" Synthesis (Synthetic Hard Negatives)

To perform Direct Preference Optimization (DPO), the algorithm requires a contrastive pair for every question: a "chosen" target ($y^+$) and a "rejected" target ($y^-$).

Instead of using naive heuristics (such as randomly swapping the label or corrupting the final answer), we explicitly prompt the 7B instructor model to generate an adversarial, **synthetic hard negative**. The model is instructed with a `NEGATIVE_SYSTEM_PROMPT` to intentionally generate an *incorrect but plausible-sounding* reasoning chain.

* If the correct action is `Clarify`, the model is forced to hallucinate a fake reasoning trace justifying why the question is "clear," and then outputs a hallucinated direct `Answer`.
* If the correct action is `Answer`, the model hallucinates a reason for why the question is "ambiguous" and generates an unnecessary clarifying question.

**How DPO Uses Reasoning Traces:** This synthetic generation process ensures that the dataset structure is **perfectly symmetric** between the chosen ($y^+$) and rejected ($y^-$) samples. Both targets contain the exact same 4-field JSON schema, and crucially, both contain a full `Reasoning` trace. Because DPO optimizes the log-probability margins over the *entire* generated sequence, the algorithm directly penalizes the *flawed logic* inside the rejected reasoning trace, rather than just penalizing the final incorrect response. This structurally symmetric design is critical: it prevents the model from "reward hacking" (e.g., learning to distinguish chosen/rejected pairs simply based on length disparities or missing formatting fields).

## Summary of Our Pipeline

Our data generation pipeline produces a highly structured, CoT-driven synthetic preference dataset entirely powered by the 7B model. By strictly enforcing a 4-field schema (`Action`, `Reasoning`, `Facets`, `Response`), allowing open-ended facet extraction, and generating structurally symmetric hard negatives, the pipeline ensures that both SFT and DPO algorithms optimize for deep semantic reasoning rather than superficial pattern matching. Because the 7B model's reasoning is naturally noisier than massive frontier models, this pipeline sets a challenging but robust foundation for Stage 2 **Group Relative Policy Optimization (GRPO)** to mathematically correct and refine the model's logic during post-training.
