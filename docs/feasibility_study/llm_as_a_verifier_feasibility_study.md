# Feasibility Study: Integrating `LLM-as-a-Verifier` into AskBeforeAnswer

* **Subject:** Technical & Scientific Feasibility Assessment of *LLM-as-a-Verifier: A General-Purpose Verification Framework*
* **Reference Paper:** [arXiv:2607.05391](https://arxiv.org/pdf/2607.05391) (Kwok et al., Stanford / UC Berkeley / NVIDIA Research)
* **Reference Repository:** [github.com/llm-as-a-verifier/llm-as-a-verifier](https://github.com/llm-as-a-verifier/llm-as-a-verifier) (`pip install llm-verifier`, v0.3.0, MIT License)
* **Target Codebase:** `ask-before-answer` (Single-Turn & Multi-Turn Clarification-Seeking LLM Pipeline)

---

## 1. Executive Summary & Verdict

| Evaluation Dimension | Assessment | Rating |
| :--- | :--- | :---: |
| **Problem-Space Alignment** | Directly addresses open bottlenecks in our evaluation judges, GRPO semantic reward shaping, and multi-turn trajectory scoring. | **High (9.5 / 10)** |
| **Scientific Value Add** | Replaces brittle, discrete JSON-text judge scores with calibrated, continuous logit-expectation scores ($G=20$ granularity), eliminating GRPO advantage ties. | **High (9.0 / 10)** |
| **Engineering Feasibility** | Lightweight Python package (`llm-verifier`) + core mathematical formulation can be implemented natively in our local PyTorch/Unsloth stack in $<100$ LOC. | **High (9.0 / 10)** |
| **Compute & Latency Impact** | **Reduces** local judge inference cost by $\sim 10\times$ when using single-token prefill logit extraction (`max_new_tokens=1`) compared to generating 300-token JSON blocks. | **Positive** |

### Final Verdict: **STRONGLY RECOMMENDED (High Value, Low Friction)**

`LLM-as-a-Verifier` is highly applicable to **AskBeforeAnswer**. Specifically, it directly solves **Open Question #1** (*"Semantic Facet Reward Models"*) and **Open Question #2** (*"Multi-Turn Clarification Dynamics"*) identified in Section 10 of [`docs/specifications/ARTICLE.md`](../specifications/ARTICLE.md#L519-L520), while upgrading our existing LLM-as-a-Judge infrastructure in [`src/ask_before_answer/evaluation/judge.py`](../../src/ask_before_answer/evaluation/judge.py) and [`src/ask_before_answer/rewards/judge.py`](../../src/ask_before_answer/rewards/judge.py).

---

## 2. Overview of `LLM-as-a-Verifier`

Standard **LLM-as-a-Judge** implementations prompt a language model to emit a discrete text score (e.g., an integer `1–5` or a JSON float like `0.8`). This paradigm suffers from three well-documented failure modes:
1. **Score Quantization & Mode Collapse:** Autoregressive text generation overwhelmingly collapses onto a few favorite numbers (e.g., `0.0`, `0.5`, `0.8`, `1.0`), creating massive ties between candidate rollouts.
2. **Stochastic Sampling & Parsing Variance:** Sampling free-form text introduces formatting failures (broken JSON) and high variance across evaluation runs.
3. **Monolithic Prompt Complexity:** Asking a judge to evaluate multiple rubric dimensions simultaneously in a single generation degrades calibration.

### How `LLM-as-a-Verifier` Solves This
Instead of reading a sampled text number, `LLM-as-a-Verifier` prompts the verifier model with an ordered scale of $G=20$ letter tokens (`A` through `T`, where `A = 20` is best and `T = 1` is worst) and **computes the mathematical expectation over the token probability distribution (logprobs)**:

$$\mathcal{R}(t, \tau) = \frac{1}{C \cdot K} \sum_{c=1}^{C} \sum_{k=1}^{K} \sum_{g=1}^{G} p_\theta(v_g \mid t, c, \tau) \cdot \phi(v_g)$$

Where:
* **$G = 20$ (Score Granularity):** Ordered letter tokens $v_g \in \{\text{A}, \text{B}, \dots, \text{T}\}$ mapped via $\phi(v_g)$ to $[0, 1]$. Taking the expectation $\sum_g p_\theta(v_g)\phi(v_g)$ produces a **continuous, high-resolution scalar** (e.g., `0.73104` vs. `0.72891`) that virtually eliminates ties.
* **$C$ (Criteria Decomposition):** Evaluates distinct rubric dimensions (e.g., *Ambiguity Detection*, *Facet Completeness*, *Question Usefulness*) independently rather than in one entangled prompt.
* **$K$ (Repeated Evaluation):** Aggregates across $K$ directional or permutation passes to cancel out positional bias and variance.
* **Probabilistic Pivot Tournament (`llm_verifier.select`):** Ranks $N$ candidates in $O(N \cdot P)$ comparisons (typically $P=2$ pivots) rather than $O(N^2)$ all-pairs comparisons.
* **Step-Level Progress Tracking (`llm_verifier.track`):** Evaluates partial multi-turn agent trajectories at intermediate checkpoints $[1, 2, \dots, T]$ to measure whether each turn moves the agent closer to resolving the task.

---

## 3. Specific Pain Points in `AskBeforeAnswer` Solved by This Framework

### 3.1 Pain Point 1: Discrete Ties & JSON Fragility in `LocalGemmaJudge` and `GeminiJudge`
* **Current State ([`src/ask_before_answer/evaluation/judge.py`](../../src/ask_before_answer/evaluation/judge.py)):**
  * Both `GeminiJudge` and `LocalGemmaJudge` prompt the judge to generate up to `max_new_tokens=300` of JSON text containing `"ambiguity_detection"`, `"clarification_quality"`, and `"usefulness"`.
  * `LocalGemmaJudge` requires regex fallback logic (lines 206–226) to recover from broken JSON formatting, and the emitted scores cluster heavily on discrete tenths (`0.0`, `0.5`, `0.8`, `1.0`).
* **Value of `LLM-as-a-Verifier`:**
  * By decomposing the 3 criteria ($C=3$) and reading the logprob expectation over the `A–T` token distribution, every model variant receives a **continuous, parse-error-free score in $[0, 1]$**.
  * Furthermore, paper benchmarks demonstrate **Weak-to-Strong Verification** and **Self-Verification**: a small local model (`Qwen2.5-7B-Instruct` or `gemma-2-2b-it`) using logprob expectation accurately verifies and ranks outputs from stronger or equal-sized models.

### 3.2 Pain Point 2: Sparse / Tie-Prone Rewards in Online GRPO (`JudgeRubric` & `FacetLogicCriterion`)
* **Current State ([`src/ask_before_answer/rewards/criteria.py`](../../src/ask_before_answer/rewards/criteria.py) & [`rewards/judge.py`](../../src/ask_before_answer/rewards/judge.py)):**
  * During GRPO training (`G = 8` rollouts per prompt), `FormatCriterion`, `ActionCriterion`, and `FacetLogicCriterion` are **binary step functions** (e.g., `+0.5` if `len(facets) > 0`, `-0.5` otherwise).
  * Once the SFT-warmed policy learns the basic schema, all 8 rollouts in a group frequently output valid headers and non-empty facets. As a result, all 8 rollouts receive the **exact same programmatic reward**, causing the group advantage standard deviation $\text{std}(\mathcal{R}_1, \dots, \mathcal{R}_8)$ to collapse to zero!
  * Even when `JudgeRubric` (`src/ask_before_answer/rewards/judge.py`) is enabled, it uses `model.generate(max_new_tokens=20, do_sample=False)` and regex-parses a single float—which still produces ties among similar rollouts.
* **Value of `LLM-as-a-Verifier`:**
  * `LLM-as-a-Verifier` was explicitly validated on **GRPO reinforcement learning** in the paper, showing substantial sample-efficiency gains.
  * Because logit-expectation produces continuous scores with 5+ significant digits (or pairwise preferences via `llm_verifier.compare`), **no two distinct rollouts in a $G=8$ GRPO group ever tie**.
  * This provides a dense, well-calibrated gradient that teaches the policy to generate *semantically superior facets and more natural clarifying questions*, directly solving **Open Question #1** in [`ARTICLE.md`](../specifications/ARTICLE.md#L519).

### 3.3 Pain Point 3: Terminal-Only Scoring in `MultiTurnEnv` (`src/ask_before_answer/evaluation/multi_turn.py`)
* **Current State ([`src/ask_before_answer/evaluation/multi_turn.py`](../../src/ask_before_answer/evaluation/multi_turn.py)):**
  * Our `MultiTurnEnv` runs up to `max_turns = 3` between the `SeekerAgent` and `ProviderAgent`.
  * Currently, evaluation only measures terminal `task_success` (whether the final `Action: Answer` matches the ground-truth answer via Exact Match / F1) and `turns_taken`.
  * If an agent asks a brilliant clarifying question on Turn 1 that narrows down 90% of the ambiguity, vs. an agent that asks a useless repetitive question on Turn 1, both trajectories look identical to the terminal scorer if Turn 2 succeeds or fails.
* **Value of `LLM-as-a-Verifier` (`llm_verifier.track`):**
  * Using `llm_verifier.track(problem=question, steps=dialogue_history, checkpoint_steps=[1, 2, 3])`, we can compute **Information Gain / Clarification Progress Scores** after every conversational turn (e.g., Turn 1: `0.64` $\rightarrow$ Turn 2: `0.98`).
  * This enables **turn-level diagnostic curves** in W&B Weave and provides step-wise reward shaping ($\Delta \text{Progress}_t = s_t - s_{t-1}$) for future multi-turn RL rollouts!

### 3.4 Pain Point 4: Synthetic Data Curation & Hard-Negative Filtering (`src/ask_before_answer/data/preprocess.py`)
* **Current State ([`src/ask_before_answer/data/preprocess.py`](../../src/ask_before_answer/data/preprocess.py)):**
  * During `make preprocess`, `SyntheticGenerator` generates a single greedy completion (`do_sample=False`) for each AmbigQA question and uses basic rule checks (`filter_bad_rows`) to drop malformed rows.
* **Value of `LLM-as-a-Verifier` (`llm_verifier.select` & `compare`):**
  * We can sample $N=3$ candidate positive completions per question and run `llm_verifier.select()` (self-verification using the same 7B model) to keep the highest-quality gold target ($y^+$) and verify that the adversarial hard negative ($y^-$) is genuinely inferior on logic while remaining formatting-compliant.

---

## 4. Architectural Fit & Integration Blueprint

### 4.1 Mapping to Existing Modules

```text
ask-before-answer/
├── src/ask_before_answer/
│   ├── rewards/
│   │   ├── criteria.py          # Existing deterministic Format, Action, FacetLogic, Accuracy criteria
│   │   ├── judge.py             # UPGRADE: Replace 20-token text generation in JudgeRubric with
│   │   │                        #          1-step A-T Logit Expectation (10x faster, continuous)
│   │   └── verifier_criterion.py# NEW: Pairwise / Pivot-Tournament GRPO group ranker criterion
│   │
│   └── evaluation/
│       ├── judge.py             # UPGRADE: Add VerifierJudge (Weave Scorer using C=3 criteria
│       │                        #          decomposition + A-T logprob expectation)
│       └── multi_turn.py        # UPGRADE: Integrate per-turn progress tracking (llm_verifier.track)
```

### 4.2 Important Technical Nuance: API vs. Local GPU Logprob Extraction

A critical engineering finding from inspecting `llm_verifier/fine_grained_reward.py` in the official repository:
1. **External API Requirement in Stock `llm-verifier`:**
   * Stock `llm-verifier` supports three backends: (a) any OpenAI-compatible server exposing token logprobs (`vllm serve` or SGLang via `OPENAI_BASE_URL`), (b) DeepSeek (`DEEPSEEK_API_KEY`), or (c) Google Vertex AI (`VERTEX_API_KEY`).
   * **Note on Gemini:** As documented in `llm_verifier/fine_grained_reward.py` (line 119), the standard consumer Gemini API (`GEMINI_API_KEY` via AI Studio) *does not* expose token-level logprobs—only Vertex AI (`vertexai=True`) does.
2. **Why Our Local PyTorch Stack Has a Major Advantage:**
   * In `ask-before-answer`, both `LocalGemmaJudge` (`evaluation/judge.py`) and `JudgeRubric` (`rewards/judge.py`) load models directly via Hugging Face `AutoModelForCausalLM` on the local GPU!
   * With a local `AutoModelForCausalLM`, we do **not** even need `model.generate()` or an external API to compute the `LLM-as-a-Verifier` score: a **single forward pass (`model(**inputs).logits[:, -1, :]`)** gives us the exact log-probability distribution over the 20 score tokens (`A` through `T`) in **$\sim 25\text{ms}$**!

### 4.3 Concrete Prototype: Local PyTorch Logit-Expectation Verifier (`< 50` Lines of Code)

Below is how the exact `LLM-as-a-Verifier` mathematical expectation ($G=20$, tokens `A`–`T`) plugs directly into our existing [`src/ask_before_answer/rewards/judge.py`](../../src/ask_before_answer/rewards/judge.py):

```python
import torch
import torch.nn.functional as F
from typing import List, Dict, Any
from ask_before_answer.rewards.base import Criterion

GRANULARITY = 20
LETTERS = [chr(65 + i) for i in range(GRANULARITY)]  # ['A', 'B', ..., 'T']
# Map 'A' -> 1.0 (best), 'T' -> 0.0 (worst)
WEIGHTS = [float(GRANULARITY - 1 - i) / (GRANULARITY - 1) for i in range(GRANULARITY)]


class LogitVerifierCriterion(Criterion):
    """Implements LLM-as-a-Verifier (arXiv:2607.05391) via 1-step logit expectation."""

    def __init__(self, name: str, weight: float, model, tokenizer, criterion_desc: str):
        super().__init__(name=name, weight=weight, description=criterion_desc)
        self.model = model
        self.tokenizer = tokenizer
        self.criterion_desc = criterion_desc
        # Pre-resolve token IDs for letters A through T
        self.letter_token_ids = [
            self.tokenizer.encode(letter, add_special_tokens=False)[-1]
            for letter in LETTERS
        ]

    @torch.inference_mode()
    def evaluate_batch(
        self, prompts: List[str], completions: List[str], kwargs: Dict[str, Any]
    ) -> List[float]:
        verifier_prompts = [
            f"Task Question: {p}\n\nCandidate Agent Response:\n{c}\n\n"
            f"Criterion: {self.criterion_desc}\n"
            f"Rate the response on a 20-point scale using a single letter A (best) through T (worst).\n"
            f"Score Letter:"
            for p, c in zip(prompts, completions)
        ]

        inputs = self.tokenizer(
            verifier_prompts, return_tensors="pt", padding=True, truncation=True
        ).to(self.model.device)

        # Single forward pass (no autoregressive decoding needed!)
        logits = self.model(**inputs).logits[:, -1, :]  # [batch_size, vocab_size]
        letter_logits = logits[:, self.letter_token_ids]  # [batch_size, 20]
        probs = F.softmax(letter_logits.float(), dim=-1)  # [batch_size, 20]

        weight_tensor = torch.tensor(WEIGHTS, device=probs.device, dtype=probs.dtype)
        expected_scores = torch.matmul(probs, weight_tensor)  # [batch_size] in [0.0, 1.0]

        return expected_scores.cpu().tolist()
```

---

## 5. Cost, Risk, and Trade-Off Analysis

| Risk / Consideration | Severity | Mitigation Strategy |
| :--- | :---: | :--- |
| **Consumer `GEMINI_API_KEY` lacks logprobs** | Medium | Use `llm-verifier` with `OPENAI_BASE_URL` (vLLM on HPC), `DEEPSEEK_API_KEY`, `VERTEX_API_KEY`, or use our native PyTorch `LogitVerifierCriterion` on local GPU (`gemma-2-2b-it` / `Qwen2.5-7B`). |
| **VRAM Contention during Online GRPO** | Low | Because `LLM-as-a-Verifier` supports **Self-Verification**, we do not even need to load a second judge model into VRAM during GRPO! We can temporarily disable the LoRA adapter (`with model.disable_adapter():`) on the already-loaded `Qwen2.5-7B` base model to compute the clean reference verifier logprobs with **0 MB of extra VRAM**. |
| **Positional Bias in Pairwise Mode** | Low | `llm-verifier` natively runs directed pairs in both `(A, B)` and `(B, A)` orders and averages the expectations. |

> [!TIP]
> **Zero-VRAM Self-Verification in GRPO:** One of the most impactful findings of the `LLM-as-a-Verifier` paper is that a base model can verify its own rollouts (lifting Pass@1 from $78.7\%$ to $88.0\%$ on Terminal-Bench 2.1). In `ask-before-answer`, because we train QLoRA adapters on top of `Qwen2.5-7B-Instruct`, we can call `with model.disable_adapter():` inside the GRPO reward step to use the frozen 7B base model as a zero-overhead `LLM-as-a-Verifier`!

---

## 6. Recommended Phased Implementation Plan

If you decide to adopt `LLM-as-a-Verifier` in `ask-before-answer` (and carry the pattern forward to `ask-the-database`), we recommend a 3-stage rollout:

### Stage 1: Single-Turn Evaluation Upgrade (`src/ask_before_answer/evaluation/judge.py`)
* Add `VerifierGemmaJudge` (or integrate `llm_verifier` via vLLM/DeepSeek) alongside `LocalGemmaJudge`.
* Decompose evaluation into the $C=3$ criteria (*Ambiguity Detection*, *Facet Completeness*, *Clarification Usefulness*) using the $G=20$ (`A–T`) logprob expectation scale.
* **Benefit:** Eliminates JSON parse errors, speeds up local judge evaluation by $\sim 10\times$ (1 forward pass per criterion vs. 300 generated tokens), and produces calibrated continuous leaderboard metrics.

### Stage 2: Online GRPO Semantic Facet & Question Quality Reward (`src/ask_before_answer/rewards/judge.py`)
* Upgrade `JudgeRubric` to compute the $G=20$ logit expectation (or use `llm_verifier.select` Pivot Tournament across the $G=8$ rollouts in each GRPO group).
* Support zero-VRAM self-verification by toggling `model.disable_adapter()` on the Qwen 2.5 7B base model.
* **Benefit:** Solves **Open Question #1** in `ARTICLE.md` by giving GRPO a dense, tie-free semantic reward signal for facet and question quality.

### Stage 3: Multi-Turn Progress Tracking (`src/ask_before_answer/evaluation/multi_turn.py`)
* Integrate `llm_verifier.track()` into `MultiTurnEnv.rollout()` to log per-turn clarification progress scores ($[s_1, s_2, s_3]$) in `results/multi_turn_eval_summary.json` and `docs/ablation_report.md`.
* **Benefit:** Measures *how much ambiguity each clarifying question resolves per turn*, rather than only scoring whether the final turn succeeded.
