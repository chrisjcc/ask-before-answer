# High-Throughput Evaluation Pipeline (vLLM + Weave)

This document details the architectural decisions behind `scripts/evaluate.py` and `src/inference/pipeline.py`. 

Because evaluating multiple Language Models across hundreds of dataset rows is computationally expensive, our evaluation pipeline employs a **Two-Step Architecture** designed to maximize GPU utilization and minimize evaluation time.

## 1. The Inference Step (vLLM)
Instead of relying on the standard Hugging Face `transformers` pipeline, this project strictly uses **vLLM** for evaluation inference. 

### Why vLLM?
Standard Hugging Face inference is memory-inefficient because the KV Cache (which stores the mathematical context of the prompt) becomes heavily fragmented in VRAM. 
vLLM utilizes **PagedAttention**, an algorithm inspired by OS virtual memory management, to chunk the KV Cache into small, non-contiguous blocks. This reduces VRAM waste from ~60% down to <5%, allowing the GPU to process enormous batches of inputs simultaneously.

### The Singleton Engine & Dynamic LoRA Swapping
When you run `make ablation-suite`, the script needs to evaluate four different model variants (e.g., SFT only, SFT+DPO, SFT+GRPO). Loading the 7B base model into VRAM four separate times wastes significant time (upwards of 40 seconds per model load).

To solve this, `src/inference/pipeline.py` implements a **Singleton Engine**:
1. The unquantized base model (`unsloth/Qwen2.5-7B-Instruct`) is loaded into VRAM exactly *once* at the beginning of the script.
2. When the script evaluates the SFT model, vLLM dynamically injects the SFT LoRA adapter weights directly into the active base model in milliseconds.
3. When it evaluates the DPO model, it instantly unloads the SFT LoRA and swaps in the DPO LoRA.

> [!IMPORTANT]
> **Hardware Requirements:** Because dynamic LoRA swapping with vLLM requires precise unquantized matrix multiplication, the base model must be loaded in `bfloat16` or `float16`. You must have enough disk space (15.2 GB) and VRAM (~14 GB) to support the full 7B model, unlike the training stages which utilize 4-bit quantization.

## 2. The Judging Step (W&B Weave & Offline Batching)
Once the models generate text, they are evaluated systematically by a Gemini-based LLM-as-a-judge using Weights & Biases **Weave**. 

### The Offline Batching Strategy
A major challenge when pairing Weave with local GPU generation is concurrency. If Weave sends 50 parallel requests to the local GPU, it will instantly trigger an Out-Of-Memory (OOM) crash as the GPU tries to allocate 50 simultaneous execution graphs.

To prevent this without sacrificing speed, the pipeline uses an **Offline Batching Strategy**:
1. **Intercept:** Before Weave even starts, `scripts/evaluate.py` extracts all 50 questions from the dataset.
2. **Batch Execute:** It passes all 50 questions to vLLM in a single batch. Because vLLM uses PagedAttention, it can safely and efficiently generate all 50 answers simultaneously in seconds.
3. **Cache:** The answers are saved in memory to a Python dictionary (`_VLLM_OFFLINE_CACHE`).
4. **Concurrent Judging:** When Weave finally begins the evaluation phase, the `ClarifyOrActModel` simply performs a dictionary lookup. Because this requires zero GPU compute, we can safely set `WEAVE_PARALLELISM="50"`. Weave blasts all 50 generated text strings to the external Gemini API concurrently, finishing the LLM-as-a-judge scoring phase nearly instantaneously.
