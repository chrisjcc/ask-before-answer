# ruff: noqa: E501, I001
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from doc_style import (
    P,
    TA_LEFT,
    build_doc,
    bullets,
    code_block,
    colors,
    data_table,
    diagram_block,
    h1,
    h2,
    inch,
    meta_table,
    note_box,
    spacer,
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "..", "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

RUNNING_TITLE = "TECHNICAL DESIGN DOCUMENT  |  AskBeforeAnswer Clarify-or-Answer Model"
FOOTER_LEFT = "INTERNAL / ENGINEERING  |  AskBeforeAnswer ML"

story = []

# ---------------------------------------------------------------- title --
story.append(P("TECHNICAL DESIGN", "DocTitle"))
story.append(P("DOCUMENT", "DocTitle"))
story.append(
    P("AskBeforeAnswer &mdash; System &amp; Pipeline Architecture", "DocSubtitle")
)
story.append(
    meta_table(
        [
            ("Document ID", "TDD-ASKBEFOREANSWER-001", False),
            ("Version", "1.0.0", False),
            ("Status", "Draft &mdash; For Review", True),
            ("Date", "September 2026", False),
            ("Owner", "ML Engineering &amp; MLOps", False),
            ("Audience", "ML Engineering, MLOps, Applied Research, Platform", False),
        ]
    )
)
story.append(P("Classification: INTERNAL / ENGINEERING", "Classification"))

story.append(h1(1, "Purpose &amp; Scope"))
story.append(
    P(
        "This Technical Design Document (TDD) describes <b>how</b> AskBeforeAnswer is built: the "
        "repository and module architecture, the Qwen 2.5 7B / Unsloth model design, the SFT &rarr; "
        "DPO/GRPO training pipeline, configuration and versioning tooling, experiment tracking and "
        "publication integrations, CI/CD, and the production deployment architecture. It complements "
        "TRD-ASKBEFOREANSWER-001, which defines <i>what</i> the system must satisfy; this document "
        "defines the concrete engineering design that satisfies it."
    )
)
story.append(P("The scope of this document covers:"))
story.extend(
    bullets(
        [
            "Repository layout, Python package structure, and separation-of-responsibility design",
            "Model architecture and hardware acceleration stack (Unsloth, Flash Attention, xFormers)",
            "The SFT &rarr; DPO/GRPO training pipeline and GRPO reward-shaping design",
            "Hydra configuration hierarchy and DVC pipeline / remote storage design",
            "W&amp;B / Weave and Hugging Face Hub integration points",
            "CI/CD workflow design, artifact/data flow, model promotion, testing, and deployment architecture",
        ]
    )
)

story.append(h1(2, "Reference Documents"))
story.append(
    data_table(
        ["Reference", "Source", "Applicability"],
        [
            [
                "TRD-ASKBEFOREANSWER-001",
                "Internal",
                "Functional &amp; performance requirements this design satisfies",
            ],
            [
                "DVC Documentation",
                "dvc.org",
                "Pipeline stage &amp; remote storage design",
            ],
            ["Hydra Documentation", "hydra.cc", "Configuration composition patterns"],
            [
                "Weights &amp; Biases Docs",
                "docs.wandb.ai",
                "Sweeps, registry, artifact lineage API",
            ],
            [
                "Unsloth Documentation",
                "unsloth.ai/docs",
                "FastLanguageModel loading, Triton kernels",
            ],
            [
                "Hugging Face Hub Docs",
                "huggingface.co/docs",
                "Model/dataset repo &amp; Spaces deployment API",
            ],
        ],
        col_widths=[2.1 * inch, 1.75 * inch, 2.5 * inch],
    )
)

# ---------------------------------------------------------------- Sec 3 --
story.append(h1(3, "Repository / Package Architecture"))
story.append(
    P(
        "The central design principle is that <b>large or generated ML artifacts do not become part of "
        "the Git repository</b>. Git tracks code and configuration; DVC tracks the data and model "
        "artifacts those code paths produce."
    )
)
story.append(
    data_table(
        ["Path", "Purpose"],
        [
            ["app/", "Streamlit Hugging Face Space UI (app.py, requirements.txt)"],
            [
                "configs/",
                "Hydra YAML configuration hierarchy (model, data, training, infra)",
            ],
            ["data/", "Processed dataset files &mdash; DVC-tracked, ignored by Git"],
            ["docs/", "Comprehensive lifecycle documentation (9-phase architecture)"],
            ["models/", "Model checkpoints &mdash; DVC-tracked, ignored by Git"],
            [
                "scripts/",
                "Executable CLI entry points (train_sft.py, evaluate.py, run_sweep_trial.py, ...)",
            ],
            [
                "src/ask_before_answer/data/",
                "Preprocessing &amp; synthetic data-generation logic",
            ],
            [
                "src/ask_before_answer/training/",
                "Training orchestrators for SFT / DPO / ORPO / GRPO",
            ],
            [
                "src/ask_before_answer/evaluation/",
                "Dual-scoring evaluation logic (LLM-judge + ActionScorer)",
            ],
            [
                "src/ask_before_answer/inference/",
                "ClarifyOrActPipeline generation &amp; parsing logic",
            ],
            ["sweeps/", "W&amp;B sweep orchestration configurations"],
            ["tests/", "Pytest unit tests"],
            [
                ".github/workflows/",
                "GitHub Actions CI and HF Space deployment workflows",
            ],
            [
                "dvc.yaml / .dvc/",
                "DVC pipeline stage definitions and internal cache metadata",
            ],
            ["Makefile", "Reproducible command aliases across the full lifecycle"],
            [
                ".env.example",
                "Environment secrets template (HF_TOKEN, GEMINI_API_KEY, WANDB_*)",
            ],
        ],
        col_widths=[2.35 * inch, 4.0 * inch],
    )
)

# ---------------------------------------------------------------- Sec 4 --
story.append(h1(4, "Python Module Structure"))
story.append(
    P(
        "The package under src/ask_before_answer/ is organized by lifecycle stage rather than by "
        "model type, so a single fine-tuning method can be added or removed without touching unrelated "
        "modules."
    )
)
story.append(
    diagram_block(
        [
            "src/ask_before_answer/",
            "  |-- data/         preprocessing.py, schema.py, facet_extraction.py",
            "  |-- training/     trainer.py, sft.py, dpo.py, orpo.py, grpo.py, rewards.py",
            "  |-- evaluation/   evaluate.py, action_scorer.py, judge_scorer.py",
            "  |-- inference/    pipeline.py  (ClarifyOrActPipeline)",
            "  `-- common/       config_schema.py, logging.py, io.py",
        ],
        align=TA_LEFT,
    )
)
story.append(h2("4.1 Design Rules"))
story.append(
    data_table(
        ["ID", "Design Rule"],
        [
            [
                "MR-1",
                "training/ modules depend only on data/ output contracts and common/, never on evaluation/ or inference/.",
            ],
            [
                "MR-2",
                "inference/pipeline.py is the only module imported by app/app.py, keeping the deployment payload minimal (Section 15).",
            ],
            [
                "MR-3",
                "Every fine-tuning method (sft.py, dpo.py, orpo.py, grpo.py) implements a shared TrainerBase interface so the Makefile dispatch layer (Section 11) is uniform across variants.",
            ],
            [
                "MR-4",
                "Reward functions for GRPO live in a dedicated rewards.py module, decoupled from grpo.py, so reward_weights overrides in Hydra do not require code changes.",
            ],
        ],
        col_widths=[0.6 * inch, 5.75 * inch],
    )
)

# ---------------------------------------------------------------- Sec 5 --
story.append(h1(5, "Qwen / Unsloth Model Architecture"))
story.append(
    P(
        "The base model is <b>Qwen 2.5 7B Instruct</b>, fine-tuned with LoRA adapters. Model loading is "
        "abstracted so the same training entry point transparently uses either accelerated Unsloth "
        "loading or native Hugging Face loading, depending on environment capability."
    )
)
story.append(h2("5.1 Hardware Acceleration Stack"))
story.append(
    data_table(
        ["Layer", "Role"],
        [
            [
                "Flash Attention (flash-attn)",
                "Computes attention without materializing the full N&times;N matrix, reducing memory from O(N^2) to O(N). Targets Ampere+ GPUs (RTX A6000, 3090, 4090).",
            ],
            [
                "xFormers",
                "Meta's optimized transformer building blocks; fallback attention path for older architectures (Turing, Volta).",
            ],
            [
                "Unsloth",
                "Sits atop Flash Attention for core attention; supplies custom Triton kernels for LoRA weight updates (via bitsandbytes 4-bit/8-bit QLoRA), the Cross-Entropy loss, RoPE, and MLP blocks.",
            ],
        ],
        col_widths=[1.9 * inch, 4.45 * inch],
    )
)
story.append(
    diagram_block(
        [
            "configs/model/qwen2_5_7b.yaml   (use_unsloth: true)",
            "            |",
            "            v",
            "     src/training/trainer.py",
            "            |",
            "   +--------+---------+",
            "   |                  |",
            "   v                  v",
            "unsloth available   unsloth NOT available",
            "   |                  |",
            "   v                  v",
            "FastLanguageModel   native HF AutoModel",
            "(Triton kernels)    (portable fallback)",
        ]
    )
)
story.append(
    note_box(
        "NOTE",
        (
            "Unsloth compiles Triton kernels on the fly and strictly requires the CUDA toolkit on the "
            "environment path. On institutional HPC/Slurm clusters, the CUDA module (e.g. module load "
            "cuda/12.6) MUST be loaded before the training pipeline runs, or Unsloth initialization fails."
        ),
    )
)

# ---------------------------------------------------------------- Sec 6 --
story.append(h1(6, "SFT &rarr; DPO / GRPO Training Pipeline"))
story.append(
    P(
        "Training deliberately separates the notion of a <b>model</b> (the trained artifact), a "
        "<b>fine-tuning method</b> (SFT, DPO, ORPO, GRPO), and a <b>training variant</b> (a specific DVC "
        "configuration, e.g. sft-only or dpo-only)."
    )
)
story.append(
    diagram_block(
        [
            "                 DVC",
            "                  |",
            "                  v",
            "        Training configuration (Hydra)",
            "                  |",
            "                  v",
            "            Training script",
            "     +--------+--------+--------+",
            "     |        |        |        |",
            "     v        v        v        v",
            "    SFT      DPO      ORPO     GRPO",
            "     |        |        |        |",
            "     +--------+--------+--------+",
            "                  |",
            "                  v",
            "            Model artifact",
            "                  |",
            "                  v",
            "             DVC tracking",
        ]
    )
)
story.append(h2("6.1 Stage Dispatch"))
story.append(
    code_block(
        [
            "make train TRAIN_VARIANT=sft      # generic dispatch",
            "make train-sft                    # readable alias",
            "make train-dpo",
            "make train-orpo",
            "make train-grpo",
        ]
    )
)
story.append(h2("6.2 GRPO Reward Shaping Design"))
story.append(
    P(
        "GRPO relies entirely on programmatic reward functions rather than static contrastive pairs. "
        "Four decoupled reward functions, each independently weighted via configs/training/grpo.yaml "
        "reward_weights, jointly shape the policy:"
    )
)
story.append(
    data_table(
        ["Function", "Design Intent"],
        [
            [
                "format_reward_func",
                "Positive reward only if all four schema headers (Action:, Reasoning:, Facets:, Response:) are present",
            ],
            [
                "action_reward_func",
                "Penalizes an incorrect Action choice against dataset ground truth",
            ],
            [
                "facet_logic_reward_func",
                "Enforces Clarify &rarr; non-empty Facets, and Answer &rarr; empty Facets",
            ],
            [
                "accuracy_reward_func",
                "Token F1 overlap between generated Response and the factual ground-truth answer; the primary anti-hallucination signal, and the term that resolves format-only reward hacking",
            ],
        ],
        col_widths=[1.9 * inch, 4.45 * inch],
    )
)

# ---------------------------------------------------------------- Sec 7 --
story.append(h1(7, "Hydra Configuration Structure"))
story.append(
    P(
        "All pipeline hyperparameters, data paths, preprocessing switches, evaluation thresholds, and "
        "infrastructure settings are managed through Hydra (v1.3+). No hardcoded values are permitted "
        "in pipeline code."
    )
)
story.append(
    data_table(
        ["Config Path", "Governs"],
        [
            [
                "conf/config.yaml",
                "Root config &mdash; imports all sub-configs via the defaults list",
            ],
            [
                "conf/model/qwen2_5_7b.yaml",
                "Base model identity, LoRA rank, use_unsloth toggle",
            ],
            [
                "conf/data/ambignq.yaml",
                "AmbigNQ source paths, schema settings, split ratios",
            ],
            [
                "conf/training/sft.yaml",
                "SFT learning rate, batch size, epochs, LoRA rank",
            ],
            [
                "conf/training/dpo.yaml",
                "DPO beta, learning rate, chosen/rejected dataset paths",
            ],
            [
                "conf/training/grpo.yaml",
                "GRPO beta, learning rate, reward_weights block (Section 6.2)",
            ],
            [
                "conf/infra/local.yaml / hpc.yaml",
                "Runtime environment: local GPU vs. HPC/Slurm CUDA module settings",
            ],
        ],
        col_widths=[2.6 * inch, 3.75 * inch],
    )
)
story.append(
    P(
        "Runtime parameter overrides follow the standard Hydra CLI pattern, e.g. "
        "python scripts/train_dpo.py training.beta=0.1 training.learning_rate=5e-7, allowing sweep "
        "agents (Section 9) to override configuration without editing YAML."
    )
)

# ---------------------------------------------------------------- Sec 8 --
story.append(h1(8, "DVC Pipeline &amp; Remote Storage Design"))
story.append(
    P(
        "DVC provides reproducible pipeline execution, data/model dependency tracking, and experiment "
        "management. Every pipeline stage is declared in dvc.yaml with explicit deps and outs, so "
        "dvc repro can recompute only the stages whose inputs changed."
    )
)
story.append(
    diagram_block(
        [
            "[data/raw] --preprocess--> [data/processed]",
            "                 |",
            "                 v",
            "        [train: sft|dpo|orpo|grpo]",
            "                 |",
            "                 v",
            "           [models/<variant>]",
            "                 |",
            "                 v",
            "             [evaluate]",
            "                 |",
            "                 v",
            "     [reports/metrics, reports/figures]",
        ]
    )
)
story.append(h2("8.1 Experiments &amp; Sweep Isolation"))
story.append(
    P(
        "Because DVC tracks the exact YAML config state for every sweep trial, winning hyperparameters "
        "never need to be copy-pasted: dvc exp apply sweep_&lt;Run ID&gt; reverts local configuration "
        "files to that exact optimal state, ready for a Git commit as the new defaults."
    )
)
story.append(
    note_box(
        "NOTE",
        (
            "Manual ablation runs (make ablation-suite, plain dvc repro) are not tagged with the W&amp;B "
            ".sweep property. The ablation report generator skips any run carrying that tag, keeping the "
            "final baseline leaderboard free of sweep noise."
        ),
    )
)

# ---------------------------------------------------------------- Sec 9 --
story.append(h1(9, "W&amp;B / Weave Integration"))
story.append(
    data_table(
        ["Integration Point", "Design"],
        [
            [
                "Training telemetry",
                "Every DVC training stage streams loss, reward components, and eval metrics live to W&amp;B via the Trainer callback interface",
            ],
            [
                "Bayesian sweeps",
                "scripts/run_sweep_trial.py is the W&amp;B sweep agent entry point; it fetches hyperparameters, patches the local Hydra config, and runs dvc exp run",
            ],
            [
                "Sweep reporting",
                "scripts/generate_sweep_report.py pulls cloud metrics grouped by Sweep ID, plots Validation Curves, and statelessly regenerates docs/sweep_report.md",
            ],
            [
                "Model registry",
                "publish-model-artifact creates/verifies an immutable W&amp;B artifact and checks the production alias + registry digest before any Hugging Face push is permitted",
            ],
            [
                "Weave evaluation",
                "scripts/evaluate.py runs the dual-scoring pipeline (LLM-as-a-judge + ActionScorer) through Weave, producing a dynamic leaderboard",
            ],
            [
                "Weave tracing",
                "app/app.py wraps every user interaction, prompt, and generation in a Weave trace for real-time production observability",
            ],
        ],
        col_widths=[1.9 * inch, 4.45 * inch],
    )
)
story.append(h2("9.1 Advanced Interactive Charts (W&amp;B Dashboard)"))
story.extend(
    bullets(
        [
            "<b>Parallel Coordinates Chart</b> &mdash; traces how hyperparameter combinations flow toward final Eval Loss",
            "<b>Hyperparameter Importance Matrix</b> &mdash; a Random Forest trained on sweep results in real time to rank feature importance",
            "<b>Interactive Validation Curves</b> &mdash; scatter plots of performance vs. hyperparameter, explored live in the browser",
        ]
    )
)

# ---------------------------------------------------------------- Sec 10--
story.append(h1(10, "Hugging Face Integration"))
story.append(
    P(
        "Hugging Face Hub is the publication surface for the final dataset, the production model, and "
        "the public inference Space. Deployment to Hugging Face is only permitted from a verified W&amp;B "
        "production artifact (Section 13)."
    )
)
story.append(
    data_table(
        ["Artifact", "Hugging Face Repository Type", "Publication Trigger"],
        [
            [
                "AskBeforeAnswer dataset",
                "Dataset repo",
                "make deploy-hf, after Data Card regeneration",
            ],
            [
                "AskBeforeAnswer model",
                "Model repo",
                "make deploy-hf, gated by W&amp;B provenance verification",
            ],
            [
                "Inference demo",
                "Space repo (Docker/Streamlit)",
                "GitHub Release publish event (Section 14)",
            ],
        ],
        col_widths=[1.9 * inch, 2.15 * inch, 2.3 * inch],
    )
)

# ---------------------------------------------------------------- Sec 11--
story.append(h1(11, "Makefile Command Reference"))
story.append(
    data_table(
        ["Command", "Purpose"],
        [
            [
                "make install / make install-dvc",
                "Install Python dependencies and DVC (via uv or pipx)",
            ],
            [
                "make preprocess",
                "Run the 3-stage synthetic data generation pipeline (DVC-tracked)",
            ],
            [
                "make run-pipeline",
                "Full DVC pipeline: data &rarr; SFT &rarr; SFT eval &rarr; DPO &rarr; SFT+DPO eval",
            ],
            [
                "make train TRAIN_VARIANT=&lt;v&gt; / make train-sft / -dpo / -orpo / -grpo",
                "Run a specific training variant",
            ],
            [
                "make sweep FINE_TUNE_METHOD=&lt;m&gt; COUNT=&lt;n&gt; / make sweep-sft / -dpo / -grpo",
                "Initialize and run a W&amp;B Bayesian sweep",
            ],
            [
                "make ablation-suite",
                "Run the clean baseline dvc repro and regenerate the ablation report",
            ],
            [
                "make evaluate",
                "Run the dual-scoring evaluation suite and publish the Weave leaderboard",
            ],
            ["make infer", "Interactive CLI inference"],
            ["make run-app", "Launch the local Streamlit UI"],
            [
                "make promote-dvc MODEL=&lt;m&gt; EXPERIMENT=&lt;id&gt;",
                "Select the winning DVC experiment as release source of truth",
            ],
            [
                "make publish-model-artifact MODEL=&lt;m&gt; EXPERIMENT=&lt;id&gt; STAGE=production",
                "W&amp;B release gate: validate, create/verify artifact, record provenance",
            ],
            ["make deploy-hf", "Push the verified model/data artifact to Hugging Face"],
            ["make lint / make test", "CI-equivalent static checks and pytest suite"],
        ],
        col_widths=[2.9 * inch, 3.45 * inch],
    )
)

# ---------------------------------------------------------------- Sec 12--
story.append(h1(12, "Inference Architecture"))
story.append(
    P(
        "At the core of the inference framework is the <b>ClarifyOrActPipeline</b> "
        "(src/ask_before_answer/inference/pipeline.py). It parses a raw model generation into the "
        "4-field schema and routes the response through a dual-action mechanism: an ambiguous query "
        "returns Action: Clarify with facets and a targeted question; a clear query returns "
        "Action: Answer and a direct response."
    )
)
story.append(
    diagram_block(
        [
            "  user question",
            "        |",
            "        v",
            "  model.generate()  (HF Transformers / Unsloth inference)",
            "        |",
            "        v",
            "  schema parser  -->  Action == Clarify?",
            "        |                     |",
            "        | no                  | yes",
            "        v                     v",
            "  return Response       return Facets + Response",
            "  (direct answer)       (clarifying question)",
        ]
    )
)
story.append(h2("12.1 Serving Configuration"))
story.append(
    data_table(
        ["Tier", "Hardware", "Precision"],
        [
            [
                "Production GPU serving",
                "1x NVIDIA T4 / L4 / RTX 3060 (8&ndash;12GB VRAM)",
                "4-bit / 8-bit quantized (bitsandbytes)",
            ],
            [
                "CPU-only inference",
                "Any x86/ARM host, no GPU required",
                "bitsandbytes or llama.cpp quantization",
            ],
        ],
        col_widths=[2.1 * inch, 2.85 * inch, 2.4 * inch],
    )
)
story.append(
    note_box(
        "NOTE",
        (
            "The current production serving path is Hugging Face Transformers (optionally Unsloth-accelerated) "
            "behind the Streamlit application, not a dedicated high-throughput server. A vLLM (or "
            "TGI-class) serving backend is a reasonable future upgrade if concurrent request volume grows "
            "beyond what the single-process Streamlit deployment can sustain; it is tracked as an open item "
            "in the companion TRD rather than a current requirement."
        ),
    )
)

# ---------------------------------------------------------------- Sec 13--
story.append(h1(13, "CI/CD &mdash; GitHub Actions"))
story.append(
    P(
        "Continuous Integration runs on every push and pull request to main, in a CPU-only environment. "
        "CI deliberately does not reproduce the GPU training environment, since training has "
        "substantially different dependencies (CUDA-enabled PyTorch, Unsloth, xFormers, Flash "
        "Attention)."
    )
)
story.append(
    diagram_block(
        [
            "GitHub push / pull request",
            "          |",
            "          v",
            "     GitHub Actions",
            "          |",
            "          v",
            "   CPU-only CI environment",
            "     +----+-----+",
            "     |          |",
            "     v          v",
            "   Lint       Tests",
        ]
    )
)
story.append(h2("13.1 Deployment Workflow"))
story.append(
    P(
        "The HF Space demo is deployed through a separate workflow, triggered by a published GitHub "
        "Release rather than by the model publication command, so the model and the application can "
        "evolve independently."
    )
)
story.append(
    code_block(
        [
            "# .github/workflows/deploy-hf-demo.yml",
            "on:",
            "  release:",
            "    types: [published]",
        ]
    )
)
story.append(
    diagram_block(
        [
            "GitHub Release",
            "      |",
            "      v",
            "deploy-hf-demo.yml (GitHub Action)",
            "      |",
            "      v",
            "Check out released source revision",
            "      |",
            "      v",
            "Construct minimal app payload",
            "      |",
            "      v",
            "Push to Hugging Face Space repository",
            "      |",
            "      v",
            "HF builds Docker image",
            "      |",
            "      v",
            "Streamlit application goes live",
        ]
    )
)

# ---------------------------------------------------------------- Sec 14--
story.append(h1(14, "Data Flow &amp; Artifact Flow"))
story.append(
    P(
        "End-to-end, the system routes control and artifacts through five distinct systems, each with a "
        "deliberately limited responsibility, enforcing a strict chain of custody from raw data to a "
        "publicly served model."
    )
)
story.append(
    data_table(
        ["System", "Primary Responsibility"],
        [
            ["Git / GitHub", "Source code, configuration, CI, releases"],
            ["DVC", "Versioned datasets, model artifacts, reproducible experiments"],
            [
                "W&amp;B",
                "Training telemetry, hyperparameter sweeps, evaluation results, model registry",
            ],
            ["Hugging Face Hub", "Public model and dataset publication"],
            ["GitHub Actions", "Continuous integration and application deployment"],
            ["Hugging Face Space", "Public Streamlit inference application"],
        ],
        col_widths=[1.9 * inch, 4.45 * inch],
    )
)
story.append(
    diagram_block(
        [
            "[AmbigNQ raw data]",
            "        |  DVC: make preprocess",
            "        v",
            "[sft_train.jsonl / dpo_train.jsonl]  --push-->  [HF Dataset repo]",
            "        |  DVC: make train-*",
            "        v",
            "[DVC experiment / model artifact]",
            "        |  make promote-dvc",
            "        v",
            "[W&B production artifact]  --verify digest-->  [release provenance]",
            "        |  make deploy-hf",
            "        v",
            "[HF Model repo]  <---loaded at runtime---  [HF Space / Streamlit app]",
        ]
    )
)

# ---------------------------------------------------------------- Sec 15--
story.append(h1(15, "Model Promotion &amp; Provenance"))
story.append(
    P(
        "The deployment pipeline is deliberately split into two stages to preserve chain of custody and "
        "enforce a human-in-the-loop review: automation executes sweeps, generates telemetry, and "
        "securely deploys finalized assets, but never automatically selects the promotion &ldquo;winner.&rdquo;"
    )
)
story.append(
    data_table(
        ["Stage", "Command", "Output"],
        [
            [
                "DVC promotion",
                "make promote-dvc MODEL=&lt;model&gt; EXPERIMENT=&lt;id&gt;",
                "Selected DVC experiment marked as release source of truth",
            ],
            [
                "W&amp;B release gate",
                "make publish-model-artifact ... STAGE=production",
                "Immutable W&amp;B artifact + verified registry digest + production alias + local provenance record",
            ],
            [
                "Hugging Face deployment",
                "make deploy-hf",
                "Verified artifact pushed to HF, Model/Data Card updated with release notes and metrics",
            ],
        ],
        col_widths=[1.55 * inch, 2.55 * inch, 2.15 * inch],
    )
)
story.append(
    note_box(
        "IMPORTANT",
        (
            "Hugging Face deployment is permitted only from a W&amp;B production artifact whose exact "
            "identity and immutable digest have been recorded in release provenance and independently "
            "verified at deployment time. promote-dvc alone never deploys a model."
        ),
        bg=colors.HexColor("#FDE2D0"),
        border=colors.HexColor("#E8956E"),
    )
)

# ---------------------------------------------------------------- Sec 16--
story.append(h1(16, "Testing Architecture"))
story.append(
    data_table(
        ["Layer", "Scope", "Tooling"],
        [
            [
                "Unit tests",
                "Schema validation, facet parsing, reward function logic, config composition",
                "pytest, tests/",
            ],
            [
                "Static checks",
                "Lint, formatting, type checks",
                "GitHub Actions CI (CPU-only)",
            ],
            [
                "Pipeline smoke tests",
                "make preprocess / make evaluate against a small fixture split",
                "DVC + pytest fixtures",
            ],
            [
                "Evaluation regression",
                "Six-variant leaderboard compared against Section 6 thresholds of the TRD",
                "W&amp;B Weave",
            ],
        ],
        col_widths=[1.6 * inch, 3.1 * inch, 1.6 * inch],
    )
)
story.append(
    P(
        "CI must pass make lint and make test before merge; GPU-dependent training and inference code "
        "paths are exercised outside CI, in the DVC-orchestrated training stage, since they require "
        "CUDA hardware not present in the CI runner."
    )
)

# ---------------------------------------------------------------- Sec 17--
story.append(h1(17, "Deployment Architecture"))
story.append(
    P(
        "The demo application's payload is intentionally minimal: app/ plus the single inference module "
        "src/ask_before_answer/inference/pipeline.py. The data, training, and evaluation modules are "
        "excluded, keeping the deployed image lightweight."
    )
)
story.append(
    diagram_block(
        [
            "app/",
            " |- app.py",
            " \\- requirements.txt",
            "+ src/ask_before_answer/inference/pipeline.py",
        ]
    )
)
story.append(
    data_table(
        ["Property", "Detail"],
        [
            [
                "Application",
                "Streamlit inference UI, Docker-packaged on Hugging Face Spaces",
            ],
            [
                "Deployment trigger",
                "Published GitHub Release (Section 13.1), not model publication",
            ],
            [
                "Model source at runtime",
                "Hugging Face model repository (never a bundled checkpoint)",
            ],
            [
                "Observability",
                "All prompts and generations traced to W&amp;B Weave in real time",
            ],
            [
                "Independent evolution",
                "Model and application version independently; a new model release does not require redeploying the app, and vice versa",
            ],
        ],
        col_widths=[1.9 * inch, 4.45 * inch],
    )
)

story.append(h1(18, "Revision History"))
story.append(
    data_table(
        ["Version", "Date", "Author", "Change Summary"],
        [
            [
                "1.0.0",
                "Sep 2026",
                "ML Engineering &amp; MLOps",
                "Initial draft for review",
            ]
        ],
        col_widths=[0.9 * inch, 1.1 * inch, 2.1 * inch, 2.25 * inch],
    )
)
story.append(spacer(14))
story.append(
    P(
        "END OF DOCUMENT | TDD-ASKBEFOREANSWER-001 v1.0.0 | INTERNAL / ENGINEERING",
        "Footer",
    )
)

build_doc(
    os.path.join(OUTPUT_DIR, "TDD_AskBeforeAnswer_SystemArchitecture.pdf"),
    RUNNING_TITLE,
    FOOTER_LEFT,
    story,
)
print("done")
