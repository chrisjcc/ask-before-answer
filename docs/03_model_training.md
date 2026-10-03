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
- LoRA weight updates (using `bitsandbytes` for standard 4-bit/8-bit QLoRA training)
- Cross-Entropy Loss function (drastically reducing memory at the very end of the network)
- Rotary Position Embeddings (RoPE)
- MLP (Feed-Forward) blocks

> [!NOTE]
> **HPC Cluster Training:** Because Unsloth compiles Triton kernels on the fly, it strictly requires the CUDA toolkit to be present in your environment path. If you are training on an institutional HPC or Slurm cluster, you must load the CUDA module (e.g., `module load cuda/12.6` or `module load cuda/12.1`) before running the training pipeline, otherwise Unsloth will fail to initialize.

**Integration:**
Unsloth is seamlessly integrated into the training pipeline via `src/training/trainer.py`. To enable it, set `use_unsloth: true` in your model's YAML configuration (e.g., `configs/model/qwen2_5_7b.yaml`).
If enabled, the trainer will intercept the standard Hugging Face loading process and load the model using `unsloth.FastLanguageModel`. 

*(Note: If `use_unsloth: true` is enabled but the library is not installed—e.g., on a CPU-only MacBook—the pipeline gracefully falls back to native Hugging Face loading to ensure code portability.)*

## 4. GRPO Reward Shaping

GRPO relies entirely on reward functions to shape the model's behavior. To keep the training loop completely decoupled from evaluation heuristics, AskBeforeAnswer utilizes a native, modular evaluation architecture comprised of a `Rubric`, `Criterion` objects, and a `SingleTurnEnv`.

### `SingleTurnEnv` (The TRL Bridge)
The `SingleTurnEnv` abstraction acts as the bridge between the raw datasets and Hugging Face TRL (`GRPOTrainer`). It dynamically formats the data and injects the system prompt, ensuring the training script remains clean and declarative.

### The `Rubric` & `Criterion` Data Model
Rather than injecting inline reward functions, the environment utilizes a declarative `Rubric` which aggregates multiple `Criterion` objects. This allows the model to be graded holistically across multiple metrics:

1. **`FormatCriterion`**: Enforces strict structural adherence. The model receives a positive reward only if it outputs all required syntax headers (`Action:`, `Reasoning:`, `Facets:`, `Response:`).
2. **`ActionCriterion`**: Penalizes the model for choosing the wrong path (e.g. trying to answer an ambiguous question) by checking the predicted action against the dataset ground-truth.
3. **`FacetLogicCriterion`**: Enforces logical consistency. If the action is `Clarify`, the facets list *must* be non-empty.
4. **`AccuracyCriterion`**: Computes Token F1 overlap for direct answers, mitigating hallucination by enforcing factual retrieval.
5. **`JudgeRubric` (LLM-as-a-Judge)**: For advanced evaluation, this criterion runs a completely local, batched PyTorch forward-pass using a smaller LLM (e.g., `google/gemma-2-2b-it`) to grade semantic usefulness without external network API calls.

> [!TIP]
> **Configurable Reward Shaping:** The `Rubric` dynamically wraps each `Criterion` into isolated TRL-compliant adapter functions. This automatically forces Weights & Biases (W&B) to log decomposed reward histograms (e.g., `reward/format`, `reward/action`), allowing for highly granular hyperparameter sweeps of the `reward_weights:` block inside `configs/training/grpo.yaml`.
