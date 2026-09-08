# Technical Design Document (TDD)
**Project Name:** AskBeforeAnswer
**Date:** September 2026
**Audience:** Machine Learning Engineers, MLOps, and Backend Developers

## 1. System Architecture & Components
The AskBeforeAnswer project implements a robust two-stage training pipeline (SFT followed by DPO/GRPO) and a lightweight inference architecture.

### 1.1 Inference Architecture (ClarifyOrActPipeline)
At runtime, the user's query is passed into the `ClarifyOrActPipeline`. This module wraps the fine-tuned LLM and enforces the 4-field generation schema (`Action`, `Reasoning`, `Facets`, `Response`).
- If `Action == Clarify`, the pipeline renders a UI component prompting the user to disambiguate using the extracted facets.
- If `Action == Answer`, the pipeline renders the direct factual answer.

## 2. Technology & Tooling Stack
The project relies on a strictly integrated MLOps stack to guarantee reproducibility and scalability.

1. **Python (`python-dotenv`):** Core application language. Secret management (API keys, Hugging Face tokens) is explicitly handled via local `.env` files loaded by `python-dotenv`, preventing credential leakage in public repositories.
2. **Makefiles:** Serve as the unified entry point for developers (e.g., `make train-sft`, `make evaluate`), wrapping complex orchestration commands into simple, reproducible aliases.
3. **Hydra:** Manages dynamic, hierarchical YAML configurations (`configs/`). It allows developers to compose and override training hyperparameters on the fly without modifying Python source code.
4. **Data Version Control (DVC):** Tracks large datasets and model weights (`dvc.yaml`). It constructs a Directed Acyclic Graph (DAG) for the entire training pipeline, intelligently caching intermediate stages (Data -> SFT -> DPO) to avoid redundant compute.
5. **Weights & Biases (W&B) & Weave:** 
   - **W&B Sweeps:** Orchestrates Bayesian hyperparameter optimization across distributed agent nodes.
   - **W&B Registry:** Acts as the internal single source of truth for model lifecycle management.
   - **Weave:** Provides real-time LLM trace logging (observability for the Streamlit UI) and systematic LLM-as-a-Judge evaluations.
6. **Hugging Face Hub:** The public distribution vector. Models are pushed to the Hub dynamically from the W&B Registry (`make deploy-hf`).
7. **GitHub Actions:** Manages Continuous Integration (CI) for linting and formatting, as well as Continuous Deployment (CD). The Streamlit Demo UI is automatically deployed to a Hugging Face Space upon triggering a GitHub Release (`.github/workflows/deploy-hf-demo.yml`).

## 3. Data Pipeline & Synthetic Generation
Rather than relying on expensive human-annotated contrastive pairs, the preprocessing pipeline (`src/ask_before_answer/data/preprocess.py`) implements **Prompt-Guided Synthetic Generation**.
- **SFT Targets:** An instructor model (`qwen2.5-7b-instruct`) generates high-quality reasoning traces and extracts facets based on the AmbigNQ dataset.
- **Hard Negatives for DPO:** The same instructor model is given a negative system prompt instructing it to explicitly hallucinate logically flawed reasoning chains (e.g., aggressively justifying why an ambiguous question is actually clear). This produces perfectly symmetric, structurally sound data that forces DPO to learn semantic logic rather than formatting exploits.

## 4. Hardware Acceleration Stack
To prevent Out-Of-Memory (OOM) faults during memory-intensive algorithms like GRPO, the training engine (`src/ask_before_answer/training/trainer.py`) integrates three layers of optimization:
1. **FlashAttention 2:** Computes exact attention with $O(N)$ memory complexity instead of $O(N^2)$.
2. **Unsloth:** Implements custom Triton kernels for Cross-Entropy loss and RoPE embeddings, bypassing standard PyTorch overhead. *(Note: Requires `module load cuda/12.6` on HPC clusters).*
3. **bitsandbytes:** Executes standard 4-bit and 8-bit QLoRA weight quantization.

## 5. Evaluation Framework (Dual-Scoring)
The systematic evaluation pipeline (`scripts/evaluate.py`) evaluates model artifacts using two concurrent methodologies:
1. **LLM-as-a-Judge:** Uses `Gemini 2.5 Flash` to assess subjective NLP nuances such as Clarification Quality, Usefulness, and Ambiguity Detection.
2. **Rule-Based `ActionScorer`:** Deterministically evaluates the raw policy accuracy of the chosen action against the ground truth, computing exact precision, recall, and Macro F1 scores. 
All evaluations generate dynamic leaderboards directly within the W&B Weave dashboard.
