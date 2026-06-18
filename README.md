# AskBeforeAnswer 🤖

[![AskBeforeAnswer CI](https://github.com/chrisjcc/ask-before-answer/actions/workflows/ci.yml/badge.svg)](https://github.com/chrisjcc/ask-before-answer/actions/workflows/ci.yml)

> A production-grade, clarification-seeking language model based on Qwen 2.5 7B.

When a question is ambiguous ("How do I make pasta?"), multiple valid interpretations exist. This project trains an LLM to surface *which* interpretation a user intends by generating **facets** — structured disambiguation options — and asking targeted clarification questions before answering.

## 🚀 Overview & Motivation

Open-domain question answering models often hallucinate or guess the user's intent when faced with ambiguous queries. **AskBeforeAnswer** addresses this by aligning the model to a clarification-first behavior using a two-stage training pipeline (Supervised Fine-Tuning followed by Direct Preference Optimization).

This repository has been restructured into a modular, highly maintainable Python codebase suitable for:
- Research reproducibility and ablation studies
- Lightweight production deployments (Streamlit / Docker)
- Future experimentation with RLHF, ORPO, or Reward Modeling

---

## 💻 System Requirements

**Software:**
- Python: `>= 3.9` (3.10 recommended)
- OS: Linux (Ubuntu 20.04/22.04 recommended) or macOS (for limited CPU-only inference)
- CUDA: `11.8` or `12.1` for training/GPU inference

**Hardware:**
- **Training (SFT/DPO):** Minimum 1x NVIDIA A100 (40GB/80GB) or RTX 3090/4090 (24GB VRAM) using 8-bit quantization and LoRA.
- **Inference (GPU):** 1x NVIDIA T4/L4/RTX 3060 (minimum 8-12GB VRAM using 4-bit/8-bit precision).
- **Inference (CPU):** Possible via `bitsandbytes` or `llama.cpp` quantization, but significantly slower. Not recommended for production.

---

## 🛠️ Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/chrisjcc/ask-before-answer.git
   cd ask-before-answer
   ```

2. **Setup virtual environment & install dependencies:**
   ```bash
   make install
   make install-dvc  # Recommended: Installs DVC globally via uv or pipx
   ```

3. **Configure Environment Variables:**
   Copy the example config and add your keys:
   ```bash
   cp .env.example .env
   # Edit .env with your HF_TOKEN, GEMINI_API_KEY, WANDB_ENTITY, and WANDB_PROJECT
   ```

---

## 🔬 Training Workflow & Ablation Studies

The training pipeline is modular and configured via [Hydra](https://hydra.cc/). It supports full ablation testing to measure the impact of different training stages.

### 1. Data Preprocessing
Prepares the AmbigNQ dataset for SFT and DPO stages.
```bash
python scripts/preprocess_data.py
```

### 2. Full Pipeline (via DVC)
Run the complete automated pipeline (Data -> SFT -> SFT Eval -> DPO -> SFT+DPO Eval) using Data Version Control (DVC):
```bash
make run-pipeline
# Alternatively, use: dvc repro
```

### Running Individual Stages
- **Supervised Fine-Tuning (SFT):** `make train-sft` (Prefix with `CUDA_VISIBLE_DEVICES=1` if your primary GPU is full)
- **Direct Preference Optimization (DPO):** `make train-dpo`

By default, models and checkpoints are saved to `models/sft/` and `models/dpo/`.

### Orchestrating Sweeps (W&B + DVC)
This project leverages **Weights & Biases Sweeps** to orchestrate Bayesian hyperparameter optimization, and **DVC** to track the reproducibility and caching of those trials.

To launch a sweep optimizing the learning rate against the SFT evaluation loss:
```bash
# 1. Initialize the sweep (this will return a SWEEP_ID)
make sweep

# 2. Start the sweep agent
wandb agent <USERNAME>/<PROJECT>/<SWEEP_ID>
```
The agent script (`scripts/run_sweep_trial.py`) will automatically fetch the hyperparameters from W&B, update the local Hydra configuration, and invoke `dvc exp run` to securely version and execute the pipeline trial.

---

## 📊 Evaluation

The evaluation pipeline uses Gemini 2.5 Flash as an LLM-as-a-judge to grade the outputs on:
1. Ambiguity Detection F1
2. Clarification Quality F1
3. Clarification Usefulness

**Evaluate a specific model checkpoint:**
```bash
python scripts/evaluate.py model_name=base
python scripts/evaluate.py model_name=sft
python scripts/evaluate.py model_name=sft_dpo
```

Metrics are saved to `results/<model_name>_metrics.json`.

---

## 💬 Inference & UI

### CLI Inference
Run interactive inference in the terminal:
```bash
make infer
# Or specify a model: python scripts/infer.py model_name=sft_dpo
```

### Streamlit Web App
Launch a local Hugging Face Spaces-compatible Streamlit UI:
```bash
make run-app
```

### Docker Deployment
Build and run the Streamlit application via Docker:
```bash
make docker-build
docker run -p 8501:8501 askbeforeanswer-app
```

---

## 🗂️ Data Version Control (DVC) Integration

This project uses **DVC (Data Version Control)** to manage large datasets, model checkpoints, and orchestrated training pipelines alongside Git. 

### Key Features Used:
1. **Tracking Large Datasets & Artifacts:**
   Large files in `data/` and `models/` are tracked via DVC. This prevents your Git repository from becoming bloated while still allowing you to version the `.dvc` tracking files.
2. **Managing Model Checkpoints:**
   LoRA weights generated in `models/sft` and `models/dpo` are managed by DVC. You can use `dvc checkout` to revert to previous model weights perfectly synced with your Git commits.
3. **Pipeline Reproducibility (`dvc.yaml`):**
   The entire training pipeline (data preprocessing, SFT, DPO, and evaluation) is defined declaratively in `dvc.yaml` as a Directed Acyclic Graph (DAG). DVC automatically detects dependency changes (e.g., if you edit `configs/training/sft.yaml`) and intelligently re-runs only the required stages, skipping unchanged ones.
4. **Metrics Tracking:**
   Evaluation results generated in `results/` are configured as DVC metrics. You can compare how changes affect your model's F1 scores across branches using `dvc metrics diff`.

---

## 🏗️ Repository Structure

```
ask-before-answer/
├── app/                  # Streamlit Hugging Face Space UI
├── configs/              # Hydra YAML configurations (model, data, training)
├── data/                 # Processed dataset files (ignored in git)
├── models/               # Model checkpoints (ignored in git)
├── scripts/              # Executable CLI entry points
├── src/                  # Core Python modules
│   ├── data/             # Preprocessing logic
│   ├── evaluation/       # Gemini judge and metrics
│   ├── inference/        # Generation pipeline
│   └── training/         # SFT and DPO LoRA trainers
├── tests/                # Pytest unit tests
├── .env.example          # Environment secrets template
├── Dockerfile            # Container deployment definition
├── dvc.yaml              # DVC pipeline definition
├── Makefile              # Reproducible command aliases
├── pyproject.toml        # Project metadata and linting config
└── requirements.txt      # Python dependencies
```

---

## 🤝 Contribution & Future Work

Future iterations of AskBeforeAnswer can easily extend the current pipeline to support:
- Multi-turn clarification datasets
- ORPO (Odds Ratio Preference Optimization)
- Constitutional AI and Reward Modeling

To contribute, ensure code passes the CI pipeline (`make lint` and `make test`).
