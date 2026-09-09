import sys
sys.path.insert(0, "/home/claude/build")
from doc_style import *

RUNNING_TITLE = "TECHNICAL REQUIREMENTS DOCUMENT  |  AskBeforeAnswer Clarify-or-Answer Model"
FOOTER_LEFT = "INTERNAL / ENGINEERING  |  AskBeforeAnswer ML"

story = []

# ---------------------------------------------------------------- title --
story.append(P("TECHNICAL REQUIREMENTS", "DocTitle"))
story.append(P("DOCUMENT", "DocTitle"))
story.append(P("AskBeforeAnswer &mdash; Clarify-or-Answer Decision Model", "DocSubtitle"))
story.append(meta_table([
    ("Document ID", "TRD-ASKBEFOREANSWER-001", False),
    ("Version", "1.0.0", False),
    ("Status", "Draft &mdash; For Review", True),
    ("Date", "September 2026", False),
    ("Owner", "ML Research &amp; Applied NLP", False),
    ("Audience", "ML Engineering, MLOps, Applied Research, Product", False),
]))
story.append(P("Classification: INTERNAL / ENGINEERING", "Classification"))

story.append(h1(1, "Purpose &amp; Scope"))
story.append(P(
    "This Technical Requirements Document (TRD) specifies the functional and non-functional "
    "requirements for AskBeforeAnswer, a fine-tuned Qwen 2.5 7B language model that decides, for "
    "every incoming user question, whether to <b>ask a targeted clarifying question</b> or "
    "<b>answer directly</b>. The document defines the dataset, training, evaluation, reproducibility, "
    "experiment-tracking, versioning, and deployment requirements the system must satisfy prior to "
    "production promotion."
))
story.append(P("The scope of this document covers:"))
story.extend(bullets([
    "The clarify-or-answer decision objective and its structured 4-field output schema",
    "Dataset construction, splits, and quality requirements for SFT and DPO/GRPO training",
    "Minimum model-performance bars and the evaluation methodology used to certify them",
    "Training reproducibility, configuration discipline, and experiment-tracking requirements",
    "Model registry promotion, versioning, and Hugging Face Hub release gating",
    "Deployment and inference-serving requirements for the production Streamlit application",
]))

story.append(h1(2, "Reference Documents &amp; Standards"))
story.append(data_table(
    ["Reference", "Source", "Applicability"],
    [
        ["AmbigNQ Corpus", "Sewon Min et al. / AI2", "Base ambiguous-question source data"],
        ["DVC Documentation", "dvc.org", "Data &amp; pipeline versioning"],
        ["Hydra Documentation", "hydra.cc", "Configuration composition and override"],
        ["Weights &amp; Biases Docs", "docs.wandb.ai", "Experiment tracking, sweeps, model registry"],
        ["W&amp;B Weave Docs", "weave-docs.wandb.ai", "LLM observability, evaluation, tracing"],
        ["Hugging Face Hub Docs", "huggingface.co/docs", "Model &amp; dataset publication"],
        ["DPO Paper (Rafailov et al., 2023)", "arXiv:2305.18290", "Preference-optimization objective"],
        ["Unsloth Documentation", "unsloth.ai/docs", "Accelerated LoRA fine-tuning"],
    ],
    col_widths=[2.05 * inch, 1.75 * inch, 2.55 * inch],
))

story.append(h1(3, "Objective &amp; Problem Statement"))
story.append(P(
    "Open-domain question-answering models frequently guess a user's intent when a query is "
    "ambiguous (e.g. <i>&ldquo;How do I make pasta?&rdquo;</i> admits many valid interpretations), "
    "which leads to hallucinated or mismatched answers. The <b>primary training objective</b> of "
    "this project is to align a Qwen 2.5 7B base model to a <b>clarification-first policy</b>: "
    "for every input question the model must emit a decision, <tt>Action &isin; {Clarify, Answer}</tt>, "
    "together with the reasoning and content required to act on that decision."
))
story.append(h2("3.1 Structured Output Contract"))
story.append(P("Every model response, in both training targets and production inference, must conform to a strict 4-field schema:"))
story.append(data_table(
    ["Field", "Type", "Description"],
    [
        ["Action", "Enum {Clarify, Answer}", "Binary routing decision for the query"],
        ["Reasoning", "Free text (CoT)", "Chain-of-Thought trace explaining why the question is ambiguous or clear"],
        ["Facets", "JSON array of strings", "Open-ended semantic attributes missing from the query; empty when Action=Answer"],
        ["Response", "Free text", "The targeted clarifying question, or the final direct answer"],
    ],
    col_widths=[1.1 * inch, 1.6 * inch, 3.65 * inch],
))
story.append(note_box("NOTE", (
    "Facets are <b>open-ended</b>. The generation prompts used to build the training data do not "
    "restrict the model to a fixed, hardcoded facet taxonomy &mdash; the instructor model dynamically "
    "produces the semantic categories relevant to each question."
)))

story.append(h1(4, "Functional Requirements"))
story.append(data_table(
    ["ID", "Requirement", "Priority"],
    [
        ["FR-1", "Given a user question, the model MUST output a single Action of Clarify or Answer.", "P0"],
        ["FR-2", "When Action = Clarify, the Facets array MUST be non-empty and the Response MUST contain a targeted clarifying question.", "P0"],
        ["FR-3", "When Action = Answer, the Facets array MUST be empty and the Response MUST contain a direct, factual answer.", "P0"],
        ["FR-4", "The model output MUST always include a Reasoning field produced before the Action/Response (Chain-of-Thought first).", "P0"],
        ["FR-5", "The training pipeline MUST support at least four fine-tuning methods: SFT, DPO, ORPO, and GRPO, selectable as independent training variants.", "P0"],
        ["FR-6", "The inference pipeline (ClarifyOrActPipeline) MUST parse the raw generation into the 4-field schema and route the UI response accordingly.", "P0"],
        ["FR-7", "Model promotion to production MUST require explicit human review of generated telemetry reports (human-in-the-loop gate).", "P0"],
        ["FR-8", "GRPO reward shaping MUST evaluate format adherence, action correctness, facet/action logical consistency, and factual accuracy as independent, re-weightable reward terms.", "P1"],
    ],
    col_widths=[0.6 * inch, 4.55 * inch, 0.75 * inch],
))

story.append(h1(5, "Dataset Requirements"))
story.append(h2("5.1 Source &amp; Construction"))
story.append(data_table(
    ["Property", "Detail"],
    [
        ["Source corpus", "AmbigNQ (ambiguous open-domain QA)"],
        ["Generator / instructor model", "qwen2.5-7b-instruct"],
        ["Schema", "Strict 4-field JSON: Action, Reasoning, Facets, Response"],
        ["Chosen (y+) synthesis", "Instructor model acting as expert agent, producing the SFT-quality gold target"],
        ["Rejected (y-) synthesis", "Prompt-Guided Synthetic Generation &mdash; negative system prompt yields structurally perfect but logically flawed reasoning"],
        ["Symmetry requirement", "Chosen/rejected pairs MUST be structurally symmetric so DPO margins isolate reasoning quality, not formatting"],
        ["Pipeline stages", "(1) Schema enforcement &amp; open-ended facet extraction &rarr; (2) SFT chosen synthesis &rarr; (3) DPO rejected synthesis"],
    ],
    col_widths=[2.1 * inch, 4.25 * inch],
))
story.append(h2("5.2 Dataset Quality Requirements"))
story.append(data_table(
    ["ID", "Requirement"],
    [
        ["DR-1", "Every record MUST validate against the 4-field JSON schema before being written to sft_train.jsonl / dpo_train.jsonl."],
        ["DR-2", "Facets MUST be a parseable JSON array for every Clarify-labeled record and MUST be empty for every Answer-labeled record."],
        ["DR-3", "Rejected samples MUST NOT be produced via expensive model-grounded mining; Prompt-Guided Synthetic Generation is the required (cheaper, native) method."],
        ["DR-4", "Dataset generation MUST be executed and tracked via DVC (make preprocess) so raw &rarr; processed lineage is reproducible."],
        ["DR-5", "The final dataset MUST be published to the Hugging Face Hub as a versioned dataset repository with an accompanying Data Card."],
    ],
    col_widths=[0.6 * inch, 5.75 * inch],
))
story.append(note_box("NOTE", (
    "Chosen and rejected DPO targets must remain perfectly symmetric in structure. Because DPO "
    "optimizes log-probability margins over the entire sequence, asymmetric formatting between "
    "chosen and rejected samples would let the model reward-hack on formatting instead of learning "
    "correct reasoning."
)))

story.append(h1(6, "Model Performance Requirements"))
story.append(P(
    "Six model variants (base, sft, dpo_only, sft_dpo, clarifier_lora, orpo, grpo) are evaluated on "
    "the sewon_ambig_qa_eval benchmark. The following minimum bars apply to any variant considered "
    "for production promotion under the <b>balanced policy</b> (strong clarification behavior "
    "without refusing to answer unambiguous questions):"
))
story.append(data_table(
    ["Gate", "Threshold", "Action on Failure"],
    [
        ["Macro F1 (Action classification)", "&ge; 0.55", "Block promotion"],
        ["Answer F1", "&ge; 0.35", "Block promotion; flag mode collapse risk"],
        ["Clarify F1", "&ge; 0.70", "Block promotion"],
        ["Ambiguity Detection F1", "&ge; 0.94", "Warning; manual review"],
        ["Facet Generation Rate", "&ge; 0.95", "Block; indicates schema non-adherence"],
        ["Clarify Ratio (pred/gold clarify rate)", "0.9 &ndash; 1.3", "Warning; indicates over- or under-clarification"],
    ],
    col_widths=[2.7 * inch, 1.6 * inch, 2.05 * inch],
))
story.append(note_box("IMPORTANT", (
    "No single model dominates every metric. A model that refuses to answer unambiguous questions "
    "(e.g. the clarifier_lora variant, answer_f1 = 0) suffers from mode collapse and MUST be "
    "rejected for general-purpose deployment even if its Clarify F1 is highest."
), bg=colors.HexColor("#FDE2D0"), border=colors.HexColor("#E8956E")))

story.append(h1(7, "Evaluation Metrics &amp; Methodology"))
story.append(h2("7.1 Dual-Scoring Evaluation"))
story.append(data_table(
    ["Evaluator", "Method", "Captures"],
    [
        ["LLM-as-a-Judge (Gemini 2.5 Flash / Gemma 4)", "W&amp;B Weave scorer", "Ambiguity Detection F1, Clarification Quality F1, Usefulness"],
        ["ActionScorer (rule-based)", "Deterministic string/JSON comparison", "Model Accuracy &mdash; raw Action correctness vs. ground truth"],
    ],
    col_widths=[2.6 * inch, 1.9 * inch, 1.85 * inch],
))
story.append(h2("7.2 Definitive Model Ranking (Balanced Policy)"))
story.append(data_table(
    ["Rank", "Model", "Macro F1", "Answer F1", "Clarify F1", "Assessment"],
    [
        ["1st", "sft", "0.615", "0.485", "0.746", "Best calibration; clarify_ratio (1.23) closest to ideal 1.0"],
        ["2nd", "dpo_only", "0.553", "0.357", "0.750", "Doubles base answer capability while holding ambiguity detection at 0.964"],
        ["3rd", "grpo", "0.532", "0.308", "0.757", "Highest clarify_f1 of any functional model; precise facet extraction"],
    ],
    col_widths=[0.55 * inch, 0.85 * inch, 0.7 * inch, 0.75 * inch, 0.75 * inch, 2.2 * inch],
))
story.append(P(
    "sft_dpo (4th) over-corrects toward caution (answer_f1 = 0.296); base (5th) over-clarifies "
    "unambiguous questions; clarifier_lora (6th) exhibits complete mode collapse (answer_f1 = 0) "
    "and MUST NOT be promoted."
))
story.append(P("Run the full evaluation suite and publish the Weave leaderboard with:"))
story.append(code_block(["make evaluate"]))

story.append(h1(8, "Training Requirements"))
story.append(data_table(
    ["ID", "Requirement"],
    [
        ["TR-1", "Training MUST be orchestrated exclusively through DVC pipeline stages defined in dvc.yaml; ad-hoc script invocation outside DVC is prohibited for tracked runs."],
        ["TR-2", "All hyperparameters and dependencies MUST be resolved through Hydra configuration composition; no hardcoded training constants are permitted in source code."],
        ["TR-3", "The training stack MUST support Unsloth-accelerated LoRA fine-tuning with Flash Attention (Ampere+) and xFormers fallback (Turing/Volta) for attention computation."],
        ["TR-4", "When use_unsloth: true is set but the Unsloth package is unavailable (e.g. CPU-only macOS), the trainer MUST gracefully fall back to native Hugging Face model loading."],
        ["TR-5", "GRPO training MUST expose four independently weighted reward functions (format, action, facet-logic, accuracy) via configs/training/grpo.yaml reward_weights."],
        ["TR-6", "HPC / Slurm training environments MUST load a CUDA toolkit module compatible with Unsloth's on-the-fly Triton kernel compilation prior to pipeline execution."],
        ["TR-7", "Each training variant (sft, sft-only, dpo-only, orpo, grpo) MUST be runnable both via a generic make train TRAIN_VARIANT=&lt;name&gt; and via a readable alias (e.g. make train-dpo)."],
    ],
    col_widths=[0.6 * inch, 5.75 * inch],
))

story.append(h1(9, "Reproducibility Requirements"))
story.append(data_table(
    ["ID", "Requirement"],
    [
        ["RP-1", "A full pipeline re-run (dvc repro) from a given dvc.lock + conf/ snapshot MUST reproduce identical metrics within floating-point tolerance."],
        ["RP-2", "Every model artifact promoted toward production MUST have traceable lineage: DVC experiment ID + Git SHA + Hydra config snapshot + W&amp;B run ID."],
        ["RP-3", "Large or generated ML artifacts (datasets, checkpoints) MUST NOT be committed to Git; DVC is the system of record for all binary artifacts."],
        ["RP-4", "Sweep trials MUST be distinguishable from clean baseline runs via a W&amp;B .sweep tag so ablation reports can programmatically exclude sweep noise."],
        ["RP-5", "Applying a sweep's winning configuration MUST be a single deterministic operation: dvc exp apply sweep_&lt;Run ID&gt;, followed by a Git commit of the updated YAML defaults."],
    ],
    col_widths=[0.6 * inch, 5.75 * inch],
))

story.append(h1(10, "Experiment Tracking Requirements"))
story.append(data_table(
    ["ID", "Requirement"],
    [
        ["ET-1", "All training runs MUST stream live metrics (loss curves, reward components, eval scores) to Weights &amp; Biases."],
        ["ET-2", "Hyperparameter sweeps MUST use W&amp;B Bayesian Sweeps for SFT, DPO, and GRPO, each isolating its own key hyperparameter set (LR/batch/epochs/LoRA rank for SFT; beta/LR for DPO; beta/LR/reward coefficients for GRPO)."],
        ["ET-3", "Post-sweep, scripts/generate_sweep_report.py MUST statelessly regenerate docs/sweep_report.md, including Validation Curves grouped by Sweep ID."],
        ["ET-4", "All qualitative and quantitative evaluation runs MUST be logged to W&amp;B Weave, producing a dynamic leaderboard and full input/output traces."],
        ["ET-5", "The production Streamlit application MUST log every user interaction, prompt, and generation to Weave for real-time observability."],
    ],
    col_widths=[0.6 * inch, 5.75 * inch],
))

story.append(h1(11, "Model / Versioning Requirements"))
story.append(P(
    "Hugging Face deployment is permitted <b>only</b> from a W&amp;B production artifact whose exact "
    "identity and immutable digest have been recorded in release provenance and independently "
    "verified at deployment time."
))
story.append(data_table(
    ["Stage", "Command", "Gate Performed"],
    [
        ["1. Promote DVC experiment", "make promote-dvc MODEL=&lt;model&gt; EXPERIMENT=&lt;id&gt;", "Marks the DVC experiment as the selected source of truth (does not deploy)"],
        ["2. Publish W&amp;B artifact", "make publish-model-artifact MODEL=&lt;model&gt; EXPERIMENT=&lt;id&gt; STAGE=production", "Validates model, creates/verifies immutable artifact, checks registry digest + production alias, records provenance"],
        ["3. Deploy to Hugging Face", "make deploy-hf", "Reads verified provenance, confirms digest, pushes exact artifact + updated Model/Data Card"],
    ],
    col_widths=[1.5 * inch, 2.55 * inch, 2.3 * inch],
))

story.append(h1(12, "Deployment / Inference Requirements"))
story.append(data_table(
    ["ID", "Requirement"],
    [
        ["DI-1", "The deployed inference application MUST require only src/ask_before_answer/inference/pipeline.py and app/ &mdash; the data, training, and evaluation modules MUST NOT be part of the deployment payload."],
        ["DI-2", "Deployment MUST be triggered exclusively by a published GitHub Release, executed by .github/workflows/deploy-hf-demo.yml."],
        ["DI-3", "The Streamlit application MUST load the production model from the Hugging Face model repository at runtime, not from a bundled checkpoint."],
        ["DI-4", "Minimum inference hardware: 1x NVIDIA T4/L4/RTX 3060 (8&ndash;12GB VRAM) at 4-bit/8-bit precision; CPU-only inference is supported but not recommended for production."],
        ["DI-5", "All user interactions on the deployed application MUST be traced to W&amp;B Weave for observability."],
    ],
    col_widths=[0.6 * inch, 5.75 * inch],
))

story.append(h1(13, "Constraints &amp; Acceptance Criteria"))
story.append(data_table(
    ["Category", "Requirement", "Specification"],
    [
        ["Reproducibility", "Full pipeline re-run", "dvc repro from dvc.lock + conf/ snapshot MUST produce identical metrics within floating-point tolerance"],
        ["Governance", "Promotion gate", "No model reaches production without explicit human review of generated telemetry (human-in-the-loop, non-negotiable)"],
        ["Performance", "Minimum viable policy", "Promoted model MUST satisfy Section 6 thresholds AND avoid mode collapse (answer_f1 &gt; 0)"],
        ["CI Scope", "CI environment", "GitHub Actions CI runs CPU-only lint + tests; GPU training dependencies (CUDA PyTorch, Unsloth, xFormers, Flash Attention) are intentionally excluded from CI"],
        ["Artifact hygiene", "Git repository", "No large or generated ML artifacts committed to Git; DVC is authoritative for data/model binaries"],
        ["Security", "Secrets management", "No credentials in configuration files; all secrets (HF_TOKEN, GEMINI_API_KEY, WANDB keys) via .env / GitHub Secrets"],
    ],
    col_widths=[1.2 * inch, 1.6 * inch, 3.55 * inch],
))

story.append(h1(14, "Dependencies &amp; Version Requirements"))
story.append(data_table(
    ["Package", "Min Version", "Purpose"],
    [
        ["Python", "3.10 (&ge;3.9 supported)", "Runtime environment"],
        ["PyTorch", "2.6+", "Core tensor / training framework (CUDA 12.1+ or 13.0)"],
        ["transformers", "4.4x", "Base model loading and generation"],
        ["unsloth", "latest", "Accelerated LoRA fine-tuning, custom Triton kernels"],
        ["xformers", "latest", "Fallback attention kernels for older GPU architectures"],
        ["flash-attn", "2.x", "Memory-efficient attention (Ampere+ GPUs)"],
        ["bitsandbytes", "latest", "4-bit / 8-bit QLoRA quantization"],
        ["dvc", "3.x", "Data &amp; pipeline versioning, reproducible experiments"],
        ["hydra-core", "1.3", "Configuration composition and override"],
        ["wandb", "latest", "Experiment tracking, sweeps, model registry"],
        ["weave", "latest", "LLM observability and evaluation leaderboards"],
        ["huggingface_hub", "latest", "Model / dataset publication"],
        ["streamlit", "latest", "Production inference UI"],
        ["pytest", "7.4", "Unit and integration tests"],
    ],
    col_widths=[1.7 * inch, 1.45 * inch, 3.2 * inch],
))

story.append(h1(15, "Open Items &amp; Decisions Required"))
story.append(data_table(
    ["#", "Open Item", "Owner", "Target Date"],
    [
        ["1", "Finalize minimum-viable thresholds for facet_generation_rate under GRPO reward shaping", "ML Research", "TBD"],
        ["2", "Decide whether sft_dpo should remain a supported production candidate given its answer_f1 regression", "Applied NLP", "TBD"],
        ["3", "Confirm long-term storage / retention policy for DVC-tracked model artifacts", "MLOps", "TBD"],
        ["4", "Evaluate whether a dedicated accelerated serving backend is warranted for the HF Space demo", "ML Engineering", "TBD"],
        ["5", "Define SLA for Weave-observed production drift and the associated re-training trigger", "ML Research", "TBD"],
    ],
    col_widths=[0.35 * inch, 3.55 * inch, 1.35 * inch, 1.1 * inch],
))

story.append(h1(16, "Revision History"))
story.append(data_table(
    ["Version", "Date", "Author", "Change Summary"],
    [["1.0.0", "Sep 2026", "ML Research &amp; Applied NLP", "Initial draft for review"]],
    col_widths=[0.9 * inch, 1.1 * inch, 2.1 * inch, 2.25 * inch],
))
story.append(spacer(14))
story.append(P("END OF DOCUMENT | TRD-ASKBEFOREANSWER-001 v1.0.0 | INTERNAL / ENGINEERING", "Footer"))

build_doc("/mnt/user-data/outputs/TRD_AskBeforeAnswer_ClarifyOrAnswer.pdf", RUNNING_TITLE, FOOTER_LEFT, story)
print("done")
