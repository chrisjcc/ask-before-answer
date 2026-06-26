.PHONY: help install install-dvc run-pipeline train train-sft train-dpo train-sft-only train-dpo-only ablation-suite evaluate infer sweep-sft sweep-dpo format lint test docker-build run-app clean

# -------------------------
# Help
# -------------------------
help:
	@echo ""
	@echo "DVC-driven ML pipeline"
	@echo ""
	@echo "Core commands:"
	@echo "  make install            Install dependencies"
	@echo "  make run-pipeline       Run full DVC pipeline"
	@echo "  make train              Alias for run-pipeline"
	@echo ""
	@echo "Training variants (DVC stages):"
	@echo "  make train-sft          Run SFT stage"
	@echo "  make train-dpo          Run DPO stage (requires SFT)"
	@echo "  make train-sft-only     Run SFT-only baseline"
	@echo "  make train-dpo-only     Run DPO-only baseline"
	@echo "  make ablation-suite     Run all experimental variants"
	@echo ""
	@echo "Utils:"
	@echo "  make evaluate           Run evaluation scripts"
	@echo "  make infer              Run inference"
	@echo ""

# -------------------------
# Setup
# -------------------------
install:
	pip install -r requirements.txt
	pip install -e .

install-dvc:
	@echo "Installing DVC..."
	@if command -v uv >/dev/null 2>&1; then \
		uv tool install dvc; \
	elif command -v pipx >/dev/null 2>&1; then \
		pipx install dvc; \
	else \
		echo "Install uv or pipx first."; \
		exit 1; \
	fi

# -------------------------
# Core pipeline (DVC is source of truth)
# -------------------------
run-pipeline:
	dvc repro

train: run-pipeline

# -------------------------
# DVC training targets
# -------------------------
train-sft:
	dvc repro train_sft

train-dpo:
	dvc repro train_dpo

train-sft-only:
	dvc repro train_sft_only

train-dpo-only:
	dvc repro train_dpo_only

ablation-suite:
	@echo "Running all experimental baselines..."
	dvc repro train_sft_only train_dpo_only train_sft train_dpo
	@echo "Evaluating all models with LLM-as-a-Judge..."
	python scripts/evaluate.py
	@echo "Synthesizing experiment results into docs/ablation_report.md..."
	python scripts/generate_report.py

# -------------------------
# Evaluation / inference (optional shortcuts)
# -------------------------
evaluate:
	python scripts/evaluate.py

infer:
	python scripts/infer.py

sweep-sft:
	@echo "Initializing SFT W&B Sweep..."
	wandb sweep sweeps/sft.yaml
	@echo "Copy the sweep ID above and run:"
	@echo "  wandb agent <USERNAME>/<PROJECT>/<SWEEP_ID> --count 10"

sweep-dpo:
	@echo "Initializing DPO W&B Sweep..."
	wandb sweep sweeps/dpo.yaml
	@echo "Copy the sweep ID above and run:"
	@echo "  wandb agent <USERNAME>/<PROJECT>/<SWEEP_ID> --count 10"

# -------------------------
# Dev tools
# -------------------------
format:
	black src scripts tests app
	ruff check --fix src scripts tests app

lint:
	black --check src scripts tests app
	ruff check src scripts tests app

test:
	pytest tests/

# -------------------------
# Deployment
# -------------------------
deploy-hf:
	@echo "Deploying datasets and final model to Hugging Face Hub..."
	python scripts/push_to_hub.py

docker-build:
	docker build -t askbeforeanswer-app .

run-app:
	streamlit run app/app.py

clean:
	rm -rf models/* outputs/* results/* wandb/
