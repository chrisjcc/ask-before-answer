# Model Training Pipeline

This document describes the model training phase of the AskBeforeAnswer lifecycle. The training pipeline is built around three complementary systems:

* **DVC** provides reproducible pipeline execution, data/model dependencies, and experiment tracking.
* **Hydra** provides structured training configuration and runtime parameter overrides.
* **Weights & Biases (W&B)** provides experiment tracking, hyperparameter sweeps, and model artifact management.

## 1. Training Architecture

The design intentionally distinguishes between a **model**, a **fine-tuning method**, and a **training variant**:

* A **model** is the trained parameterized artifact produced by applying a learning algorithm to data.
* A **fine-tuning method** describes the learning procedure, such as SFT, DPO, ORPO, or GRPO.
* A **training variant** identifies a particular DVC training configuration or experimental variant, such as `sft`, `sft-only`, `dpo-only`, `orpo`, or `grpo`.

```text
         DVC
          │
          ▼
Training configuration
          │
          ▼
  Training script
          │
  ┌───────┼────────┬────────┐
  ▼       ▼        ▼        ▼
 SFT     DPO      ORPO     GRPO
  │       │        │        │
  └───────┴────────┴────────┘
                  │
                  ▼
            Model artifact
                  │
                  ▼
            DVC tracking
```

## 2. Running Training

Training is orchestrated through DVC and exposed through the Makefile.

A specific training variant can be run with:

```bash
make train TRAIN_VARIANT=sft
```

For the common variants, readable aliases are provided:

```bash
make train-sft
make train-dpo
make train-orpo
make train-grpo
```

The training implementation lives under `src/ask_before_answer/training/` with the individual training entry points exposed through the corresponding scripts.

## 3. Hardware Acceleration Stack

To maximize GPU utilization and prevent Out-Of-Memory (OOM) errors during memory-hungry post-training algorithms (especially GRPO), AskBeforeAnswer integrates a highly optimized, layered acceleration stack.

### Flash Attention (`flash-attn`)
Flash Attention computes attention without materializing the massive $N \times N$ matrix in memory, reducing attention memory usage from $O(N^2)$ to $O(N)$. It is highly optimized for Ampere GPUs and newer (e.g., RTX A6000, RTX 3090, RTX 4090).

### xFormers
`xformers` is Meta’s library for optimized transformer building blocks, providing fallback support for older GPUs (like Turing or Volta) or edge-case architectures.

### Unsloth
Unsloth relies on Flash Attention for the core attention calculations, but writes custom Triton kernels to optimize everything else:
- LoRA weight updates
- Cross-Entropy Loss function (drastically reducing memory at the very end of the network)
- Rotary Position Embeddings (RoPE)
- MLP (Feed-Forward) blocks

**Integration:**
Unsloth is seamlessly integrated into the training pipeline via `src/training/trainer.py`. To enable it, set `use_unsloth: true` in your model's YAML configuration (e.g., `configs/model/qwen2_5_7b.yaml`).
If enabled, the trainer will intercept the standard Hugging Face loading process and load the model using `unsloth.FastLanguageModel`. 

*(Note: If `use_unsloth: true` is enabled but the library is not installed—e.g., on a CPU-only MacBook—the pipeline gracefully falls back to native Hugging Face loading to ensure code portability.)*

## 4. GRPO Reward Shaping

GRPO relies entirely on its reward functions to shape the model's behavior. We utilize four distinct reward functions to holistically enforce both formatting and factual accuracy:
1. **`format_reward_func`**: Enforces strict structural adherence. The model receives a positive reward only if it outputs all required syntax headers (`Action:`, `Reasoning:`, `Facets:`, `Response:`).
2. **`action_reward_func`**: Penalizes the model for choosing the wrong path (e.g. trying to answer an ambiguous question, or clarifying a clear question) by checking the predicted action against the dataset ground-truth.
3. **`facet_logic_reward_func`**: Enforces logical consistency. If the action is `Clarify`, the facets list *must* be non-empty. If the action is `Answer`, the facets list *must* be empty.
4. **`accuracy_reward_func`**: The most critical reward for mitigating hallucination. For direct answers, it calculates a Token F1 overlap between the model's generated response and the factual ground-truth answer. It heavily rewards exact factual retrieval while severely penalizing hallucinations, solving the "reward hacking" problem where a model learns perfect formatting but generates factually incorrect text.

> [!TIP]
> **Configurable Reward Shaping:** All proportionality weights and penalties for these four reward functions are fully decoupled from the source code. You can easily adjust them or run automated W&B Hyperparameter Sweeps by modifying the `reward_weights:` block inside `configs/training/grpo.yaml`.
