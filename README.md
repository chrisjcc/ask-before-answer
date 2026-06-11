# AskBeforeAnswer 🤖

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
   git clone https://github.com/chrisjcc/AskBeforeAnswer.git
   cd AskBeforeAnswer
   ```

2. **Setup virtual environment & install dependencies:**
   ```bash
   make install
   ```

3. **Configure Environment Variables:**
   Copy the example config and add your keys:
   ```bash
   cp .env.example .env
   # Edit .env with your HF_TOKEN and GEMINI_API_KEY
   ```

---

## 🔬 Training Workflow & Ablation Studies

The training pipeline is modular and configured via [Hydra](https://hydra.cc/). It supports full ablation testing to measure the impact of different training stages.

### 1. Data Preprocessing
Prepares the AmbigNQ dataset for SFT and DPO stages.
```bash
python scripts/preprocess_data.py
```

### 2. Full Ablation Pipeline
Run the complete automated pipeline (Data -> SFT -> DPO -> Evaluation):
```bash
make run-ablation
```

### Running Individual Stages
- **Supervised Fine-Tuning (SFT):** `make train-sft`
- **Direct Preference Optimization (DPO):** `make train-dpo`

By default, models and checkpoints are saved to `models/sft_output/` and `models/dpo_output/`.

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

## 🏗️ Repository Structure

```
AskBeforeAnswer/
├── app/                  # Streamlit Hugging Face Space UI
├── configs/              # Hydra YAML configurations (model, data, training)
├── data/                 # Processed dataset files (ignored in git)
├── legacy/               # Original CS546 notebooks and report
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
