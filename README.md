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

> **Architecture Note:** At the core of the inference framework lies the `ClarifyOrActPipeline`. This module is designed to autonomously parse user queries and route them using a dual-action mechanism: if the query is ambiguous, it returns an `Action: Clarify` schema with facets and a targeted question; if the query is clear, it routes to `Action: Answer` and answers directly.

## 📚 Comprehensive Documentation

The complete end-to-end lifecycle of this project—from synthetic data generation and hyperparameter sweeps to W&B model promotion and HF Space deployment—is thoroughly documented in the `docs/` directory.

👉 **[Start here: Documentation Overview (9-Phase Architecture)](docs/README.md)**

---

## 🌐 Hugging Face Artifacts
You can explore the deployed final model, the dataset, and interact with the UI demo on Hugging Face:
- **🤗 Space Demo:** [AskBeforeAnswer Demo](https://huggingface.co/spaces/chrisjcc/ask-before-answer-demo)
- **🤗 Model Card:** [AskBeforeAnswer Qwen Model](https://huggingface.co/chrisjcc/ask-before-answer)
- **🤗 Dataset:** [AskBeforeAnswer Data](https://huggingface.co/datasets/chrisjcc/ask-before-answer-dataset)

---

## 💻 System Requirements
**Software:**
- Python: `3.10` recommended (>= 3.9 supported)
- OS: Linux (Ubuntu 20.04/22.04 recommended) or macOS (for limited CPU-only inference)
- CUDA: `12.1+` or `13.0` for training/GPU inference (Required for PyTorch 2.6+)

**Hardware:**
- **Training (SFT/DPO):** Minimum 1x NVIDIA RTX A6000 / A100 (40GB/48GB/80GB) or RTX 3090/4090 (24GB VRAM) using 8-bit quantization and LoRA.
- **Inference (GPU):** 1x NVIDIA T4/L4/RTX 3060 (minimum 8-12GB VRAM using 4-bit/8-bit precision).
- **Inference (CPU):** Possible via `bitsandbytes` or `llama.cpp` quantization, but significantly slower. Not recommended for production.

*For details on our Unsloth and FlashAttention hardware acceleration stack, see [Model Training](docs/03_model_training.md).*

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
   ```bash
   cp .env.example .env
   # Edit .env with your HF_TOKEN, GEMINI_API_KEY, WANDB_ENTITY, and WANDB_PROJECT
   ```

---

## 🔬 Quick Start: Training & Evaluation

The training pipeline is modular, configured via [Hydra](https://hydra.cc/), and versioned via **DVC**. 

### 1. Data Preprocessing
Prepares the AmbigNQ dataset for SFT and DPO stages. 
```bash
make preprocess
```
*For a detailed breakdown of our strict curation and contrastive negative synthesis, see [Data Preprocessing](docs/02_data_preprocessing.md).*

### 2. Full Pipeline (via DVC)
Run the complete automated pipeline (Data -> SFT -> SFT Eval -> DPO -> SFT+DPO Eval) using Data Version Control (DVC):
```bash
make run-pipeline
```

### 3. Individual Training Stages
- **Supervised Fine-Tuning (SFT):** `make train-sft`
- **Direct Preference Optimization (DPO):** `make train-dpo`
- **Odds Ratio Preference Optimization (ORPO):** `make train-orpo`
- **Group Relative Policy Optimization (GRPO):** `make train-grpo`

*For advanced information on GRPO reward shaping or emergency checkpoint recovery, see [Model Training](docs/03_model_training.md).*

### 4. Hyperparameter Sweeps
This project leverages **Weights & Biases Sweeps** to orchestrate Bayesian hyperparameter optimization.
```bash
make sweep-sft
wandb agent <USERNAME>/<PROJECT>/<SWEEP_ID> --count 10
make ablation-suite
```
*For comprehensive instructions on how sweeps are orchestrated, validated, and applied using our human-in-the-loop architecture, see [Hyperparameter Sweeps](docs/05_hyperparameter_sweeps.md) and the [Ablation Study](docs/06_ablation_study.md).*

### 5. Systematic Evaluation
The automated evaluation pipeline uses a dual-scoring approach (LLM-as-a-judge & rule-based scoring).
```bash
make evaluate
```
*For a deep dive into the 6 distinct model variants and our final leaderboard, see [Evaluation & Analysis](docs/04_evaluation_analysis.md).*

---

## 💬 Inference & UI

### CLI Inference
Run interactive inference in the terminal:
```bash
make infer
```

### Streamlit Web App
Launch a local Hugging Face Spaces-compatible Streamlit UI:
```bash
make run-app
```

### Docker Deployment
Pull the pre-compiled Docker image directly from the GitHub Container Registry (GHCR):
```bash
docker pull ghcr.io/chrisjcc/ask-before-answer:latest
docker run --env-file .env -p 8501:8501 ghcr.io/chrisjcc/ask-before-answer:latest
```
*For instructions on how the Streamlit demo is automatically deployed to Hugging Face, see [Demo Deployment](docs/09_demo_deployment.md).*

---

## 🏗️ Repository Structure

```
ask-before-answer/
├── app/                  # Streamlit Hugging Face Space UI
├── configs/              # Hydra YAML configurations (model, data, training)
├── data/                 # Processed dataset files (ignored in git)
├── docs/                 # Comprehensive 9-phase lifecycle documentation
├── models/               # Model checkpoints (ignored in git)
├── scripts/              # Executable CLI entry points (e.g., train_sft.py, evaluate.py)
├── src/                  # Core Python modules
│   ├── data/             # Preprocessing logic
│   ├── evaluation/       # Evaluation logic
│   ├── inference/        # Generation pipeline
│   └── training/         # Training orchestrators
├── sweeps/               # W&B sweep orchestration configurations
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
- Constitutional AI and Reward Modeling

To contribute, ensure code passes the CI pipeline (`make lint` and `make test`).
