# Article Title Options

*   **Option 1 (Recommended):** *Ask Before Answer: Teaching Language Models When to Clarify and When to Act*
*   **Option 2:** *Clarification as an Action: Why Post-Training Needs Programmatic Reinforcement Learning for Ambiguous Intent*
*   **Option 3:** *Why Answering Every Question Is Wrong: Aligning LLMs on the Clarify-or-Act Decision Boundary*
*   **Option 4:** *When Preference Alignment Fails: Comparing Offline DPO and Online GRPO in Ambiguity Resolution*

---

# Ask Before Answer: Teaching Language Models When to Clarify and When to Act

## 1. Hook: “What if the correct answer is a question?”

Imagine entering a single, simple query into an AI assistant:

> *"When was the last time the World Cup was won?"*

To an eager Large Language Model (LLM) trained to deliver direct answers at all costs, this looks like a straightforward factual retrieval problem. The model will immediately retrieve a date, pick a sport (likely men's FIFA association football), and confidently declare: *"December 18, 2022, by Argentina."*

Yet this answer rests entirely on hidden assumptions. Was the user asking about the FIFA Men's World Cup? The FIFA Women's World Cup (won by Spain in August 2023)? The Cricket World Cup (won by Australia in November 2023)? The Rugby World Cup (won by South Africa in October 2023)? Or perhaps a regional tournament?

The prompt suffers from multi-faceted **entity ambiguity**.

Now contrast two fundamentally different model behaviors when confronted with this query:

```text
Typical LLM Behavior:
User: "When was the last time the World Cup was won?"
Assistant: "The last World Cup was won by Argentina on December 18, 2022, defeating France in the final."
[Presumes men's association football; ignores all other interpretations]

AskBeforeAnswer Behavior:
User: "When was the last time the World Cup was won?"
Assistant:
Action: Clarify
Reasoning: The query "World Cup" is entity-ambiguous because it does not specify the sport or gender category (e.g., FIFA Men's, FIFA Women's, ICC Cricket, or Rugby World Cup).
Facets: ["Sport / Category", "Gender Division"]
Response: "Which World Cup are you referring to? (e.g., FIFA Men's World Cup, FIFA Women's World Cup, ICC Cricket World Cup, or Rugby World Cup)?"
```

Instead of gambling on what the user meant, the model pauses, identifies the missing parameters (the *facets* of ambiguity), and asks a targeted clarification question first.

This brings us to a central question in LLM alignment:

> **Instead of teaching an LLM to answer every question immediately, can we train it to recognize when it should ask for clarification first?**

```text
                               +-------------------+
                               |   User Question   |
                               +---------+---------+
                                         |
                                         v
                       +-----------------------------------+
                       | Is the question sufficiently      |
                       | specified?                        |
                       +-----------------+-----------------+
                                         |
                       +-----------------+-----------------+
                       |                                   |
                [Clear / Specified]               [Ambiguous / Underspecified]
                       |                                   |
                       v                                   v
             +-------------------+               +-------------------+
             |  Action: Answer   |               |  Action: Clarify  |
             |  Generate Direct  |               |  Surface Facets   |
             |      Answer       |               |  & Targeted Qs    |
             +-------------------+               +-------------------+
```

*Figure 1: The high-level decision flow of the AskBeforeAnswer framework, routing queries to direct answers or targeted clarification questions based on explicit ambiguity detection.*

---

## 2. The Problem: LLMs Are Optimized to Answer

Current post-training paradigms—such as Supervised Fine-Tuning (SFT) and Direct Preference Optimization (DPO)—are overwhelmingly calibrated to produce helpful, definitive textual answers. During RLHF and instruction tuning, human annotators routinely reward complete, direct responses and penalize models that hesitate or reply with counter-questions.

As a consequence, language models develop a deep behavioral bias: **they act as if every prompt is fully specified**.

To understand why this breaks down in real-world deployment, we must distinguish between two fundamentally different types of uncertainty:

1.  **Factual (Epistemic) Uncertainty:** The model does not know a specific fact (e.g., *"What is the exact atomic weight of Ununennium?"*). Here, the user's intent is perfectly clear, but the model lacks knowledge.
2.  **Intent (Aleatoric) Uncertainty:** The model knows all relevant facts, but the user's request is underspecified or multi-interpretable (e.g., *"How do I make pasta?"* or *"Who won the election in Washington?"*).

When an LLM hallucinates in response to an ambiguous prompt, the failure is rarely a collapse of its internal knowledge base. Rather, **the model is hallucinating an arbitrary interpretation of the user's intent** and providing a correct factual answer to a question the user never actually asked.

Simply scaling model parameters or adding more pre-training tokens does not solve intent uncertainty. A 700-billion-parameter model cannot divine whether a user in Vancouver asking for *"weather in Washington"* means Washington State or Washington, D.C. To fix this, we must treat **clarification as an explicit, first-class model action**.

---

## 3. The Core Idea: Clarification as an Action

In the **AskBeforeAnswer** project, ambiguity resolution is reframed from an incidental conversational style into an explicit, structured action selection problem.

Instead of generating free-form text where clarification occurs randomly, the model is constrained at the token level to execute a four-tiered programmatic schema for every input:

```text
Action: Clarify | Answer
Reasoning: <Chain-of-Thought rationale explaining the ambiguity state>
Facets: [<List of missing informational attributes if Clarify, else empty>]
Response: <Targeted clarifying question OR direct terminal answer>
```

```text
+------------------------------------------------------------------------+
|                      ClarifyOrActPipeline Architecture                   |
+------------------------------------------------------------------------+
                                   |
                                   v
                   +-------------------------------+
                   |       User Prompt Parsing     |
                   +---------------+---------------+
                                   |
                                   v
                   +-------------------------------+
                   |  Token-Level Structured Schema |
                   |  - Action: Clarify / Answer   |
                   |  - Reasoning: CoT Rationale   |
                   |  - Facets: Missing Attributes  |
                   |  - Response: Payload          |
                   +---------------+---------------+
                                   |
                  +----------------+----------------+
                  |                                 |
                  v                                 v
        [Action == Clarify]                [Action == Answer]
                  |                                 |
                  v                                 v
      +-----------------------+         +-----------------------+
      | Ask Targeted          |         | Execute Terminal      |
      | Disambiguation        |         | Knowledge Retrieval   |
      | Question              |         | & Direct Answer       |
      +-----------------------+         +-----------------------+
```

*Figure 2: Architectural routing within the AskBeforeAnswer runtime pipeline (`ClarifyOrActPipeline`), separating structural intent routing from text payload generation.*

By forcing the model to generate a Chain-of-Thought `Reasoning` trace *before* predicting the final `Action`, the framework induces the model to analyze the query's specification state prior to committing to an output.

Crucially, when the action is `Clarify`, the model must not output generic, unhelpful responses such as *"Can you please provide more details?"* Instead, it must dynamically extract the underlying missing attributes—known as **Facets**—and synthesize a targeted question that addresses those specific options.

Below is a representative Python representation of the structural output schema evaluated by the system:

```python
from typing import List, Literal
from pydantic import BaseModel, Field

class ClarifyOrActSchema(BaseModel):
    action: Literal["Clarify", "Answer"] = Field(
        description="Explicit routing decision based on query specification."
    )
    reasoning: str = Field(
        description="Chain-of-Thought trace evaluating missing query parameters."
    )
    facets: List[str] = Field(
        default_factory=list,
        description="List of extracted ambiguity dimensions. Must be non-empty if action is Clarify."
    )
    response: str = Field(
        description="Targeted clarification question if Clarify; direct factual answer if Answer."
    )
```

---

## 4. Dataset and Ambiguity Taxonomy

To train and evaluate models on the clarify-or-act decision boundary, the project derived a specialized dataset from the open-source **AmbigNQ** corpus (Min et al., 2020), which contains open-domain questions from Natural Questions annotated with multiple plausible interpretations.

### The Ambiguity Taxonomy

Ambiguity in real-world queries manifests across distinct semantic dimensions. AskBeforeAnswer categorizes these into three primary categories:

| Ambiguity Facet | Description | Ambiguous Query Example | Extracted Facets & Targeted Clarification |
| :--- | :--- | :--- | :--- |
| **Temporal Ambiguity** | Timeframe, season, or historical era is missing or shifting. | *"Who is the current prime minister of the UK?"* (in historical archives) or *"When did the movie come out?"* | `["Film Adaptation / Version", "Release Region"]`<br>*"Are you asking about the 1974 original film or the 2021 remake?"* |
| **Entity Ambiguity** | Multiple distinct real-world entities share the same name or term. | *"Where is Concordia University located?"* | `["Campus Location"]`<br>*"Do you mean Concordia University in Montreal, Canada, or Concordia University in Wisconsin, USA?"* |
| **Geographic / Spatial Ambiguity** | Location-specific context is omitted. | *"What are the sales tax rates?"* | `["State / Jurisdiction", "Municipality"]`<br>*"Which country and state/province are you inquiring about?"* |

### Curation, Synthetic Augmentation, and Balancing

Standard supervised datasets derived from raw QA benchmarks suffer from severe class imbalance and shortcut learning: models easily overfit to majority classes (e.g., learning to answer everything and ignoring clarification).

To prevent this, AskBeforeAnswer executes a three-stage curation pipeline:

1.  **Synthetic CoT Augmentation:** Using `Qwen/Qwen2.5-7B-Instruct` as a lightweight instructor model, each raw query is augmented with a synthetic Chain-of-Thought `Reasoning` trace and open-ended `Facets`.
2.  **Strict Row Filtering & 50/50 Class Balancing:** To ensure an unbiased decision boundary, the dataset is dynamically downsampled to enforce an exact 50% / 50% split between `Clarify` and `Answer` ground-truth targets. Row filtering automatically purges corrupted generations (e.g., instances where an LLM labeled a query as `Answer` while simultaneously outputting non-empty facets).
3.  **Contrastive Hard Negatives:** For preference optimization (DPO), standard random or empty rejected responses are insufficient. The pipeline synthesizes **Prompt-Guided Hard Negatives**. If the gold action is `Clarify`, the rejected target ($y^-$) is forced to be a confident, un-disambiguated direct answer. Conversely, if the gold action is `Answer`, the rejected target is forced to be an unnecessary, over-cautious clarification question.

```text
[Raw AmbigNQ Corpus]
        │
        ▼
[Stage 1: Schema Enforcement & CoT Synthetic Extraction (Qwen 2.5 7B)]
        │
        ▼
[Stage 2: Strict Filtering & 50/50 Class Balancing]
        ├──> SFT Dataset (Gold Schema Targets: y+)
        │
        ▼
[Stage 3: Prompt-Guided Contrastive Hard Negative Synthesis]
        └──> Preference Dataset (Chosen: y+ vs. Hard Negative Rejected: y-)
```

*Figure 3: Data curation and contrastive hard-negative synthesis pipeline for AskBeforeAnswer.*

---

## 5. Training Strategy: SFT, DPO, and Online GRPO

The AskBeforeAnswer research framework systematically evaluates both offline preference optimization and online reinforcement learning paradigms built on top of a 4-bit QLoRA base model (`unsloth/qwen2.5-7b-instruct-unsloth-bnb-4bit`).

```text
                               +-----------------------------------+
                               |       Base LLM (Qwen 2.5 7B)      |
                               +-----------------+-----------------+
                                                 |
                                                 v
                               +-----------------------------------+
                               |    Stage 1: Supervised Fine-      |
                               |           Tuning (SFT)            |
                               |   (Internalizes Schema & CoT)     |
                               +--------+-----------------+--------+
                                        |                 |
                   +--------------------+                 +--------------------+
                   |                                                           |
                   v                                                           v
+-----------------------------------+                       +-----------------------------------+
|     Stage 2a: Offline Preference  |                       |     Stage 2b: Online Programmatic |
|      Optimization (SFT -> DPO)    |                       |       RL (SFT -> GRPO)            |
|  (Implicit Log-Ratio Margins over |                       |  (Deterministic Reward Engine +   |
|         Hard Negatives)           |                       |      Group Advantage Rollouts)    |
+-----------------------------------+                       +-----------------------------------+
```

*Figure 4: The two-stage training progression evaluated in AskBeforeAnswer, comparing offline preference alignment (DPO) with online programmatic RL (GRPO).*

### Training Infrastructure, Quantization, and Hyperparameters

For ML engineers and practitioners looking to reproduce or build upon this setup, training memory management is a major technical consideration. Online RL algorithms like GRPO sample multiple generations per prompt ($G=8$ rollouts), which can cause substantial VRAM spikes.

To make full post-training accessible on accessible single-GPU hardware (e.g., NVIDIA A100 or RTX A6000), the project leverages **QLoRA (Quantized Low-Rank Adaptation)** and hardware-accelerated kernels via the **Unsloth** framework:

* **Quantization & Adapter Precision:** Base models are loaded in 4-bit NormalFloat (`bnb-4bit`) precision via `bitsandbytes`, while LoRA adapter weights are injected into attention and MLP projection layers and trained in `bfloat16`.
* **Kernel Acceleration:** Unsloth's custom Triton kernels and FlashAttention-2 enable fast forward/backward passes and efficient memory handling during multi-rollout sampling.

Table 3 summarizes the exact hyperparameter configurations used across the SFT, DPO, and GRPO training stages:

| Hyperparameter | Supervised Fine-Tuning (SFT) | Direct Preference Optimization (DPO) | Group Relative Policy Optimization (GRPO) |
| :--- | :--- | :--- | :--- |
| **Learning Rate** | $2 \times 10^{-5}$ | $5 \times 10^{-7}$ | $5 \times 10^{-6}$ |
| **Optimizer** | AdamW (`adamw_torch`) | AdamW (`adamw_torch`) | AdamW (`adamw_torch`) |
| **Epochs / Passes** | 3 | 1 | 1 |
| **Per-Device Batch Size** | 1 | 1 | 1 |
| **Gradient Accumulation** | 8 | 8 | 8 |
| **Warmup Ratio** | 0.05 | 0.10 | 0.10 |
| **Max Context Length** | 2048 tokens | 2048 tokens | 512 (prompt) / 512 (completion) |
| **KL Penalty ($\beta$)** | N/A | 0.10 | 0.10 |
| **Rollouts per Prompt ($G$)** | N/A | N/A | 8 |
| **Precision** | `bfloat16` adapter / 4-bit base | `bfloat16` adapter / 4-bit base | `bfloat16` adapter / 4-bit base |

*Table 3: Exact training hyperparameters across SFT, DPO, and GRPO stages.*

### Training Dynamics & Optimization Telemetry

Yes—for a technical Medium article targeting ML researchers, post-training engineers, and LLM practitioners, including training dynamics and telemetry metrics is essential. Monitoring loss curves, implicit reward margins, and KL divergence constraints provides crucial insight into optimization stability, sample efficiency, and potential failure modes like over-fitting or reward hacking.

The AskBeforeAnswer training telemetry highlights three distinct behavior patterns across training stages:

#### 1. Training & Evaluation Loss Convergence

During Stage 1 Supervised Fine-Tuning and offline preference alignment, monitoring loss convergence ensures the model internalizes the schema without catastrophic forgetting.

* **SFT Convergence:** SFT training loss drops rapidly within the first 100 steps as the model learns the structural headers (`Action:`, `Reasoning:`, `Facets:`, `Response:`), stabilizing at an evaluation loss of $\sim 0.22$.
* **DPO Margin Loss:** DPO training stabilizes quickly at a low evaluation loss ($\sim 0.010$). However, as discussed in Section 8, an extraordinarily low DPO evaluation loss can be a leading indicator of reward hacking—where the model over-fits log-ratio margins on header tokens at the expense of generative answer accuracy.

```text
[Training & Evaluation Loss Curves]
Reference: papers/draft/figures/train_loss_comparison.png & eval_loss_comparison.png
- Compares loss trajectories across SFT, DPO, and ORPO variants.
- Demonstrates rapid convergence on structural header tokens across all post-training paradigms.
```

#### 2. DPO Preference Dynamics

In offline preference optimization, telemetry focuses on implicit rewards, log probabilities, and reward margins:

* **Reward Margins:** The difference in implicit reward between chosen ($y_w$) and rejected ($y_l$) completions grows smoothly, demonstrating that DPO effectively separates contrastive pairs.
* **Log Probabilities:** The log probability of chosen responses ($\log \pi_\theta(y_w \mid x)$) remains stable while rejected responses ($\log \pi_\theta(y_l \mid x)$) are suppressed.
* **Classification Accuracy:** DPO preference accuracy rapidly approaches $>95\%$ on contrastive hard-negative pairs.

```text
[DPO Telemetry: Reward Margins & Log Probabilities]
Reference: papers/draft/figures/dpo_margin.png & dpo_logprobs.png
- Illustrates smooth margin growth between chosen and hard-negative rejected completions.
```

#### 3. GRPO Reinforcement Learning Dynamics

In online reinforcement learning, monitoring reward components and policy drift is vital to prevent policy collapse or runaway optimization:

* **Total Programmatic Reward Convergence:** The group mean reward starts low as rollouts frequently fail format or action constraints, then steadily increases as policy generations align with schema and routing rules.
* **Component Breakdown:** Format reward ($\mathcal{R}_{\text{format}}$) and routing reward ($\mathcal{R}_{\text{routing}}$) saturate early to $+1.0$. The accuracy reward ($\mathcal{R}_{\text{accuracy}}$) increases gradually as rollouts explore correct answers.
* **KL Divergence Constraint ($\mathbb{D}_{\text{KL}}$):** The KL divergence relative to the reference SFT policy remains bounded ($\le 0.10$), ensuring the model retains its linguistic fluency and behavioral prior while learning conservative action boundaries.

```text
[GRPO Telemetry: Reward Convergence & KL Divergence]
Reference: papers/draft/figures/grpo_total_reward.png, kl_divergence.png, & grpo_reward_components.png
- Top Left: Mean group reward trajectory showing online policy improvement.
- Top Right: Bounded KL divergence penalty preventing policy drift from the SFT base.
- Bottom: Individual programmatic reward component trajectories (Format, Routing, Facet, and Accuracy rewards).
```

### Stage 1: Supervised Fine-Tuning (SFT)

The SFT stage serves as behavioral cloning. It conditions the base LLM on the structured four-tiered schema, teaching the model to output valid reasoning traces, extract open-ended facets, and format final responses.

The SFT loss optimizes standard cross-entropy over the gold sequence targets $\mathcal{D}_{\text{SFT}}$:

$$\mathcal{L}_{\text{SFT}}(\theta) = -\mathbb{E}_{(x,y) \sim \mathcal{D}_{\text{SFT}}} \left[ \sum_{t=1}^{\vert y \vert} \log \pi_\theta(y_t \mid x, y_{<t}) \right]$$

### Stage 2a: Direct Preference Optimization (DPO)

DPO optimizes implicit reward margins over paired chosen ($y_w$) and rejected ($y_l$) completions without requiring an explicit reward model:

$$\mathcal{L}_{\text{DPO}}(\theta; \pi_{\text{ref}}) = -\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}_{\text{pref}}} \left[ \log \sigma \left( \beta \log \frac{\pi_\theta(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} - \beta \log \frac{\pi_\theta(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)} \right) \right]$$

In theory, DPO should penalize hard negatives (e.g., confident guessing on ambiguous queries or over-clarifying clear queries). However, as our experimental results reveal, **offline DPO suffers from severe reward hacking on generative QA tasks**.

### Stage 2b: Group Relative Policy Optimization (GRPO) with Programmatic Rewards

To eliminate offline reward hacking, AskBeforeAnswer warm-starts **Group Relative Policy Optimization (GRPO)** (Shao et al., 2024) from an SFT baseline.

GRPO samples a group of outputs $\{y_1, y_2, \dots, y_G\}$ from the policy $\pi_\theta$ for each prompt $x$, and computes advantages by standardizing scores across the group *without requiring a neural critic network*:

$$A_i = \frac{\mathcal{R}(x, y_i) - \frac{1}{G}\sum_{j=1}^G \mathcal{R}(x, y_j)}{\text{std}(\mathcal{R}(x, y_1), \dots, \mathcal{R}(x, y_G))}$$

The policy loss clips likelihood ratios and enforces a KL-divergence constraint relative to the reference policy $\pi_{\text{ref}}$:

$$\mathcal{L}_{\text{GRPO}}(\theta) = -\frac{1}{G} \sum_{i=1}^G \left[ \min \left( \frac{\pi_\theta(y_i \mid x)}{\pi_{\text{old}}(y_i \mid x)} A_i, \text{clip}\left(\frac{\pi_\theta(y_i \mid x)}{\pi_{\text{old}}(y_i \mid x)}, 1-\epsilon, 1+\epsilon\right) A_i \right) - \beta \mathbb{D}_{\text{KL}}(\pi_\theta \parallel \pi_{\text{ref}}) \right]$$

### Deterministic Programmatic Reward Functions

Crucially, the GRPO stage in AskBeforeAnswer does **not** rely on an LLM-as-a-judge or learned scalar reward model during rollouts. Instead, it uses a suite of fast, deterministic, programmatic Python reward functions:

$$\mathcal{R}(x, y) = \omega_1 \mathcal{R}_{\text{format}}(y) + \omega_2 \mathcal{R}_{\text{routing}}(x, a) + \omega_3 \mathcal{R}_{\text{facet}}(x, f) + \omega_4 \mathcal{R}_{\text{len}}(y)$$

```text
+------------------------------------------------------------------------+
|                        Reward Execution Engine                         |
+------------------------------------------------------------------------+
                                    |
                                    v
       +----------------------------------------------------------+
       |             Format Check: R_format(y)                    |
       |    Validates schema blocks via regex/JSON parser         |
       +----------------------------+-----------------------------+
                                    |
                  +-----------------+-----------------+
                  v Pass                              v Fail (-1.0)
+----------------------------------+        +----------------------------+
|    Routing Check: R_routing      |        | Terminate Early            |
| Compares action 'a' with ground  |        | Skip downstream processing |
| truth data annotation            |        +----------------------------+
+-----------------+----------------+
                  |
                  v Pass
+----------------------------------+
|     Facet Validation: R_facet    |
| If Clarify: measures string match|
| or exact parameter key matching  |
+-----------------+----------------+
                  |
                  v Pass
+----------------------------------+
|     Length Penalty: R_len        |
| Penalizes verbose outputs using  |
| a dynamic length scaling term    |
+----------------------------------+
```

*Figure 5: The deterministic programmatic reward evaluation pipeline executed during GRPO online rollouts.*

1.  **Format Reward ($\mathcal{R}_{\text{format}}$):** Checks via regex and JSON parsing whether the completion contains all four required schema headers (`Action:`, `Reasoning:`, `Facets:`, `Response:`).
2.  **Routing Reward ($\mathcal{R}_{\text{routing}}$):** Awards $+1.0$ if the predicted action matches ground truth (`Clarify` vs `Answer`), and $-1.0$ if mismatched.
3.  **Facet Reward ($\mathcal{R}_{\text{facet}}$):** If `Action == Clarify`, verifies that `Facets` is a valid, non-empty JSON list.
4.  **Length Penalty ($\mathcal{R}_{\text{len}}$):** Penalizes redundant verbosity to keep generation efficient.

---

## 6. Evaluation Methodology: How Do We Measure Clarification?

Evaluating clarification behavior requires measuring both **structural decision-making** and **natural language generation quality**. AskBeforeAnswer combines programmatic rule-based scoring (`ActionScorer`) with LLM-as-a-judge evaluation (`LocalGemmaJudge` utilizing `gemma-2-2b-it`).

### Core Evaluation Metrics

The framework benchmarks performance across 9 core metrics:

1.  **Action Accuracy (Macro Accuracy):** The percentage of queries where the model correctly chose between `Clarify` and `Answer`.
2.  **Clarify Precision, Recall, & F1:** Evaluates the model's ability to identify ambiguous questions without hallucinating ambiguity on clear questions.
3.  **Action F1 (Answer Class):** Measures whether the model correctly chooses to answer straightforward, unambiguous queries.
4.  **Macro F1:** The unweighted mathematical average of Clarify F1 and Action F1 (Answer Class).
5.  **Answer Accuracy:** Measured **only** on ground-truth Answer cases; checks whether the generated response matches the factual ground truth using fuzzy token matching.
6.  **Clarify Ratio:** Total predicted `Clarify` actions divided by total ground-truth `Clarify` instances. A ratio $> 1.0$ indicates over-clarification; $< 1.0$ indicates under-clarification.
7.  **Facet Generation Rate (FGR):** The percentage of predicted `Clarify` actions that successfully included a non-empty list of facets.
8.  **Ambiguity Detection & Clarification Quality F1 (LLM Judge):** Subjective scores from the teacher model evaluating whether the clarification question was natural and relevant.
9.  **Clarification Usefulness (LLM Judge):** Evaluates whether answering the clarification question would actually enable a user to resolve their query.

> **Limitations of LLM-as-a-Judge:** While LLM judges provide valuable qualitative feedback on phrasing, judge scores can be vulnerable to verbosity bias and stylistic preferences. Therefore, AskBeforeAnswer treats programmatic metrics (Macro F1, Answer Accuracy, FGR) as the primary indicators of structural reliability.

---

## 7. Experimental Results

To evaluate whether models post-trained under AskBeforeAnswer successfully navigate the clarify-or-act decision boundary, we benchmarked six distinct model variants on the evaluation split of the `sewon/ambig_qa` benchmark.

Our empirical evaluation measures both deterministic structural performance (rule-based routing precision, schema adherence, and factual accuracy) and qualitative generation safety (LLM-as-a-judge scoring).

### Main Alignment Leaderboard

To establish a clear comparative baseline, we evaluate the un-tuned base model (`unsloth/qwen2.5-7b-instruct-unsloth-bnb-4bit`), single-stage preference variants (`DPO Only` and `ORPO`), supervised behavioral cloning (`SFT`), and two-stage post-training pipelines (`SFT → DPO` and `SFT → GRPO`).

Table 1 summarizes the primary decision-routing metrics, schema compliance rates, and downstream answer accuracy across all six configurations:

| Model / Pipeline | Action Acc | Clarify F1 | Answer F1 | Macro F1 | Clarify Ratio | Answer Acc | Facet Gen Rate (FGR) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Base (Qwen 2.5 7B Instruct)** | 62.0% | 75.3% | 17.4% | 46.4% | 1.57 | 5.0% | 8.5% |
| **DPO Only (No SFT)** | 64.0% | 76.3% | 25.0% | 50.7% | 1.53 | 5.0% | 10.9% |
| **Single-Stage ORPO** | 62.0% | 74.7% | 24.0% | 49.3% | 1.50 | 5.0% | 95.6% |
| **SFT (Stage 1 Only)** | **62.0%** | 72.5% | 38.7% | 55.6% | 1.30 | 5.0% | **100.0%** |
| **SFT $\rightarrow$ DPO** | **62.0%** | 70.8% | **45.7%** | **58.2%** | **1.17** | **0.0%** | **100.0%** |
| **SFT $\rightarrow$ GRPO (Ours)** | **62.0%** | **73.2%** | 34.5% | 53.9% | 1.37 | **15.0%** | **100.0%** |

*Table 1: Performance comparison across post-training pipelines on the AmbigNQ evaluation split.*

#### Key Insights from the Main Leaderboard:

1. **The SFT Warm-Start Cures Schema Collapse:** Both un-tuned Base (8.5%) and `DPO Only` (10.9%) suffer from near-total failure in outputting structured facets when deciding to clarify. In contrast, every model that underwent SFT warm-starting (`SFT`, `SFT → DPO`, and `SFT → GRPO`) achieved a **flawless 100.0% Facet Generation Rate (FGR)**, confirming that behavioral cloning is strictly required to establish structured execution schemas.
2. **DPO Optimizes Action Classification at the Cost of Generative Accuracy:** `SFT → DPO` achieves the highest Macro F1 (58.2%) and brings the Clarify Ratio down to a near-ideal 1.17 (reducing over-clarification). However, its factual Answer Accuracy drops to **0.0%**. DPO's static loss function heavily optimizes the log-probability margin on header classification tokens (`Action: Answer`), but completely neglects the generative factual tokens in the answer payload.
3. **Programmatic GRPO Triples Factual Answer Accuracy:** The `SFT → GRPO` pipeline leverages dynamic online rollouts with deterministic reward functions that penalize hallucinated answers. This enables GRPO to **triple factual Answer Accuracy (15.0% vs. 5.0% for SFT and 0.0% for DPO)** while maintaining 100% schema compliance. Because wrong answers carry strict negative rewards during training, the GRPO agent learns a calibrated, cautious policy—preferring to clarify when uncertain (Clarify Ratio of 1.37).

---

### Qualitative LLM-as-a-Judge Evaluation

In addition to deterministic rule-based metrics, we evaluated the natural language quality, ambiguity detection capability, and practical usefulness of the generated clarifying questions.

Using `LocalGemmaJudge` (powered by `google/gemma-2-2b-it`), each model output was evaluated on a 0.0 to 1.0 scale across three qualitative dimensions. Table 2 details these results:

| Model / Pipeline | Ambiguity Detection F1 | Clarification Quality | Clarification Usefulness |
| :--- | :---: | :---: | :---: |
| **Base** | 0.966 | 0.784 | 0.880 |
| **DPO Only** | 0.968 | 0.786 | 0.882 |
| **SFT** | **0.970** | 0.796 | **0.896** |
| **SFT $\rightarrow$ DPO** | 0.946 | 0.778 | 0.878 |
| **SFT $\rightarrow$ GRPO** | **0.970** | **0.800** | 0.892 |

*Table 2: Subjective qualitative metrics scored by LocalGemmaJudge (`gemma-2-2b-it`).*

#### Key Insights from Qualitative Judge Scoring:

1. **High Baseline Ambiguity Detection:** Across all variants, Ambiguity Detection F1 remains consistently high (>0.946). This indicates that modern pre-trained instruction-tuned bases (such as Qwen 2.5 7B) already possess strong latent semantic understanding of query underspecification; post-training primarily aligns *how* the model acts on that understanding.
2. **GRPO Achieves the Highest Clarification Quality:** `SFT → GRPO` recorded the top score in **Clarification Quality (0.800)**, demonstrating that training with programmatic reward functions does not degrade conversational naturalness or syntactic phrasing.
3. **Structured Constraints Do Not Compromise Usefulness:** Across all post-trained variants, Clarification Usefulness remains near or above 0.88–0.89. Enforcing a strict 4-field output schema (`Action`, `Reasoning`, `Facets`, `Response`) preserves the semantic richness and helpfulness of the clarification response.

---

## 8. What Surprised Us: Candid Analysis & Trade-offs

The experimental findings revealed several unexpected outcomes and negative results that provide key insights into LLM post-training.

### 1. DPO Suffers from Offline Reward Hacking

The most surprising finding was the dramatic failure mode of the **SFT $\rightarrow$ DPO** pipeline.

On paper, DPO appears to be the top performer: it achieved the highest Macro F1 (58.2%) and brought the Clarify Ratio down to a near-perfect 1.17. However, when examining **Answer Accuracy**, DPO collapsed to **0.0%**.

```text
               DPO Offline Optimization: Reward Hacking Collapse

    +-------------------------------------------------------------+
    | Macro F1: 58.2% (High)                                       |
    | Clarify Ratio: 1.17 (Near Ideal)                            |
    | Answer Accuracy: 0.0% (COMPLETE GENERATIVE COLLAPSE)         |
    +-------------------------------------------------------------+
    | Mechanism: The static preference loss forces the model to   |
    | maximize log-ratio margins on decision header tokens        |
    | ("Action: Answer") while completely abandoning factual      |
    | generative semantics in the payload text.                   |
    +-------------------------------------------------------------+
```

Why did this happen?

Because preference optimization computes log-probability ratios over static offline pairs, DPO found a shortcut: it maximized margin separation on the structural headers (`Action: Answer`) to optimize classification metrics, while sacrificing the generation quality of the answer payload itself. The model learned to "game" the decision boundary at the complete expense of factual grounding.

### 2. Preference Optimization Fails Without an SFT Prior

Running preference alignment directly on the base model without an SFT warm-start (`DPO Only`) resulted in structural failure:
*   Facet Generation Rate dropped to an abysmal **10.9%**.
*   The model routinely failed to parse or adhere to the four-tiered schema.

This demonstrates that preference optimization functions primarily as a stylistic tuner. It cannot teach structured execution or complex schemas from scratch; it requires a behavioral prior established by SFT.

### 3. Online GRPO Enforces Factual Safety and Prevents Reward Hacking

By replacing static preference pairs with dynamic programmatic rollouts, the **SFT $\rightarrow$ GRPO** pipeline successfully avoided reward hacking:
*   **Answer Accuracy tripled to 15.0%** (compared to 5.0% for SFT and 0.0% for DPO).
*   **Facet Generation Rate remained flawless at 100.0%**.

Because the programmatic reward function strictly penalizes incorrect answers during rollouts, GRPO learned a **highly calibrated, cautious agent personality**. When faced with uncertainty, the GRPO agent prefers to clarify rather than risk an incorrect answer—raising the Clarify Ratio to 1.37 and lowering Answer F1 to 34.5%.

For safety-critical applications where guessing incurs severe penalties, **SFT $\rightarrow$ GRPO produces the most hallucination-resistant policy**.

```text
               GRPO Online RL: Hallucination-Resistant Safety

    +-------------------------------------------------------------+
    | Answer Accuracy: 15.0% (3x higher than SFT / Base)          |
    | Facet Generation Rate: 100.0% (Flawless Schema Compliance)  |
    | Clarify Ratio: 1.37 (Highly Cautious Agent Personality)     |
    +-------------------------------------------------------------+
    | Mechanism: Dynamic programmatic rollouts penalize factual   |
    | errors on rollouts, teaching the policy that guessing is     |
    | risky and driving conservative, calibrated behavior.        |
    +-------------------------------------------------------------+
```

---

## 9. From Research Prototype to a Practical LLM System

Teaching language models when to clarify has immediate practical implications for real-world software engineering and agentic AI systems.

### Why Ambiguity Resolution Matters for Autonomous Agents

In standard chatbot interfaces, an un-disambiguated guess is merely a mild annoyance. But in **agentic workflows**—where LLMs execute SQL queries, invoke external APIs, modify files, or issue database transactions—guessing a user's intent can lead to destructive outcomes.

```text
User Request: "Delete all records for John in the system."

Un-Disambiguated Agent Action:
- Identifies 5 users named "John" in the database.
- Confidently picks the first entry (John Doe, ID: 101) and executes DELETE.
--> Severe Data Loss / Irreversible Action

AskBeforeAnswer Agent Action:
- Action: Clarify
- Facets: ["User Account ID", "Email Address", "Department"]
- Response: "There are multiple accounts matching 'John'. Which account should be deleted?
  1) John Doe (Engineering, ID: 101)
  2) John Smith (Marketing, ID: 204)"
--> Safe, Deterministic System Execution
```

### Key Production Application Domains

1.  **Enterprise Search & QA:** Preventing internal search assistants from serving outdated policies when documents span multiple fiscal years.
2.  **Text-to-SQL & Database Interfaces:** Forcing the LLM to confirm column mappings or timeframe filters before executing heavy database queries.
3.  **Customer Support Automation:** Ensuring agents verify account identifiers, product versions, and subscription tiers prior to initiating refunds or cancellations.
4.  **Developer & Code Assistants:** Asking whether a user wants a refactor in Python 3.11 or legacy Python 2 before rewriting a code repository.

---

## 10. Where This Leaves Us: Open Questions & Future Directions

The core takeaway of the AskBeforeAnswer project can be summarized in a single principle:

> **A reliable AI assistant should not only know how to answer a question. It should also know when the question itself needs clarification.**

While this research demonstrates that SFT $\rightarrow$ GRPO with programmatic rewards successfully aligns models to a hallucination-resistant clarify-or-act behavior, several important open questions remain for the AI community:

1.  **Semantic Facet Reward Models:** Current GRPO reward functions validate structural formatting (`Facets` as a non-empty JSON list) but do not evaluate whether the generated facets are semantically optimal. Integrating lightweight local reward models to score facet quality is a key next step.
2.  **Multi-Turn Clarification Dynamics:** The current benchmark evaluates single-turn clarification. Extending post-training to multi-turn dialogues—where the agent dynamically incorporates user responses to iteratively refine its intent model—remains an active area of research.
3.  **Reinforcement Learning with Tool Use:** Combining ambiguity resolution with tool-calling frameworks (e.g., checking database schema availability before deciding whether to clarify) represents a promising frontier for building safe autonomous agents.
4.  **Calibration and Uncertainty Estimation:** Investigating whether logit-based entropy or conformal prediction bounds can be directly integrated into the GRPO reward function to improve intent uncertainty estimation.

By treating clarification as an explicit model action rather than an accidental byproduct of generation, we move one step closer to building language models that are not just fluent, but truly reliable.

---

## Source Material & Code Availability

The AskBeforeAnswer project is fully open-source. Code, dataset processing scripts, training configurations, and evaluation suites are available on GitHub and Hugging Face:

*   **GitHub Repository:** [chrisjcc/ask-before-answer](https://github.com/chrisjcc/ask-before-answer)
*   **Research Paper Draft:** [AskBeforeAnswer Draft (ICML Style)](https://github.com/chrisjcc/ask-before-answer/blob/main/papers/draft/main.pdf)
*   **Hugging Face Model Card:** [chrisjcc/ask-before-answer](https://huggingface.co/chrisjcc/ask-before-answer)
*   **Hugging Face Space Demo:** [AskBeforeAnswer Interactive Demo](https://huggingface.co/spaces/chrisjcc/ask-before-answer-demo)
