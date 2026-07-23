# Hardware Acceleration Stack

To maximize GPU utilization and prevent Out-Of-Memory (OOM) errors during memory-hungry post-training algorithms (especially GRPO), AskBeforeAnswer integrates a highly optimized, layered acceleration stack.

It is highly recommended to use **all three** of the following tools together for optimal performance on Ampere+ architecture GPUs (e.g., RTX A6000, RTX 3090, RTX 4090).

---

### 1. Flash Attention (`flash-attn`)
Flash Attention is a low-level algorithm written directly in CUDA/Triton that fundamentally rewrites how the **Attention Matrix** is computed. 
* **The Problem:** Standard attention requires holding a massive $N \times N$ matrix in VRAM (where $N$ is sequence length). As your context length grows, VRAM usage explodes quadratically.
* **What it adds:** It computes attention without materializing that massive matrix in memory. It reduces attention memory usage from $O(N^2)$ to $O(N)$.
* **Hardware:** It is highly optimized for Ampere GPUs and newer (making it a perfect fit for an RTX A6000).

### 2. xFormers
`xformers` is Meta’s library for optimized transformer building blocks. 
* **The overlap:** It includes its own implementation of "memory-efficient attention" which is very similar to Flash Attention. 
* **The difference:** While Flash Attention is hyper-optimized for modern GPUs (Ampere/Hopper), `xformers` often provides better fallback support for older GPUs (like Turing or Volta) or edge-case architectures. Unsloth uses it heavily under the hood as a base dependency.

### 3. Unsloth
Unsloth is a higher-level wrapper library. **Unsloth does not replace Flash Attention; it relies on it!**
* Flash Attention and xFormers *only* optimize the **Attention** mechanism. But attention is only one part of an LLM.
* Unsloth writes custom Triton kernels to optimize **everything else**:
  * **LoRA weight updates:** Making adapters train faster.
  * **The Cross-Entropy Loss function:** Drastically reducing memory at the very end of the network.
  * **Rotary Position Embeddings (RoPE).**
  * **The MLP (Feed-Forward) blocks.**

### Integration in AskBeforeAnswer
Unsloth is seamlessly integrated into the training pipeline via `src/training/trainer.py`. To enable it, simply set `use_unsloth: true` in your model's YAML configuration (e.g., `configs/model/qwen2_5_7b.yaml`).

If enabled, the trainer will intercept the standard Hugging Face loading process and load the model using `unsloth.FastLanguageModel`, automatically applying the Triton-optimized LoRA adapters and delegating the core attention calculations to `flash-attn`.

**Note for Local Development:**
If `use_unsloth: true` is enabled but the library is not installed (e.g., when running on a CPU-only MacBook), the pipeline gracefully falls back to native Hugging Face loading to ensure code portability.
