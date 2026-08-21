-include .env

export

# Default sweep agent trial count
COUNT ?= 10

.PHONY: help install install-dvc preprocess run-pipeline train \
	train-sft train-dpo train-sft-only train-dpo-only train-orpo \
	train-grpo train-grpo-dpo ablation-suite evaluate infer \
	sweep-sft sweep-dpo sweep-grpo promote format lint test \
	docker-build run-app deploy-hf clean clean-cache clean-locks

# -------------------------
# Help
# -------------------------

help:
	@echo ""
	@echo "DVC-driven ML pipeline"
	@echo ""
	@echo "Core commands:"
	@echo "  make install             Install dependencies"
	@echo "  make preprocess          Run data preprocessing"
	@echo "  make run-pipeline        Run full DVC pipeline"
	@echo "  make train               Alias for run-pipeline"
	@echo ""
	@echo "Training variants (DVC stages):"
	@echo "  make train-sft           Run SFT stage"
	@echo "  make train-dpo           Run DPO stage (requires SFT)"
	@echo "  make train-sft-only      Run SFT-only baseline"
	@echo "  make train-dpo-only      Run DPO-only baseline"
	@echo "  make train-orpo          Run ORPO baseline"
	@echo "  make train-grpo          Run GRPO baseline"
	@echo "  make train-grpo-dpo      Run GRPO->DPO pipeline"
	@echo "  make ablation-suite      Run all experimental variants"
	@echo ""
	@echo "Evaluation & Inference:"
	@echo "  make evaluate            Run evaluation scripts"
	@echo "  make infer               Run inference"
	@echo ""
	@echo "Model Promotion:"
	@echo "  make promote MODEL=<sft|dpo|grpo|orpo> EXPERIMENT=<id>"
	@echo "                          Apply DVC experiment and commit promotion"
	@echo ""
	@echo "Deployment:"
	@echo "  make deploy-hf           Push final datasets and best model to Hugging Face"
	@echo ""
	@echo "Hyperparameter Optimization (Sweeps):"
	@echo "  make sweep-sft           Run hyperparameter sweep for SFT stage"
	@echo "  make sweep-dpo           Run hyperparameter sweep for DPO stage"
	@echo "  make sweep-grpo          Run hyperparameter sweep for GRPO stage"
	@echo ""
	@echo "Dev tools:"
	@echo "  make format              Format code with isort, black, and ruff"
	@echo "  make lint                Check code style and linting"
	@echo "  make test                Run pytest test suite"
	@echo ""
	@echo "App & Docker:"
	@echo "  make run-app             Launch the Streamlit demo application"
	@echo "  make docker-build        Build the Docker container image"
	@echo ""
	@echo "Cleanup:"
	@echo "  make clean               Remove all outputs, models, and W&B cache"
	@echo "  make clean-cache         Prune old DVC cache"
	@echo "  make clean-locks         Forcefully remove DVC lock files"
	@echo ""

# -------------------------
# Setup
# -------------------------

install:
	pip install -r requirements.txt
	pip uninstall -y torchao
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
# Core pipeline / data
# DVC is the source of truth
# -------------------------

run-pipeline:
	dvc repro

train: run-pipeline

preprocess:
	dvc repro preprocess

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

train-orpo:
	dvc repro train_orpo

train-grpo:
	dvc repro train_grpo

train-grpo-dpo:
	dvc repro train_grpo train_dpo

ablation-suite:
	@echo "Running all experimental baselines..."
	dvc repro train_sft train_dpo train_dpo_only train_orpo train_grpo
	@echo "Evaluating all models with LLM-as-a-Judge..."
	python scripts/evaluate.py
	@echo "Synthesizing experiment results into docs/ablation_report.md..."
	python scripts/generate_report.py

# -------------------------
# Evaluation / inference
# -------------------------

evaluate:
	python scripts/evaluate.py

infer:
	python scripts/infer.py

# -------------------------
# DVC experiment promotion
# -------------------------

promote:
	@if [ -z "$(MODEL)" ]; then \
		echo "ERROR: MODEL is required."; \
		echo "Usage: make promote MODEL=<sft|dpo|grpo|orpo> EXPERIMENT=<id>"; \
		exit 1; \
	fi
	@if [ -z "$(EXPERIMENT)" ]; then \
		echo "ERROR: EXPERIMENT is required."; \
		echo "Usage: make promote MODEL=<sft|dpo|grpo|orpo> EXPERIMENT=<id>"; \
		exit 1; \
	fi
	@echo "=========================================================="
	@echo "Promoting DVC experiment"
	@echo "=========================================================="
	@echo "Model:       $(MODEL)"
	@echo "Experiment:  $(EXPERIMENT)"
	@echo "=========================================================="
	@python scripts/promote_experiment.py \
		--model "$(MODEL)" \
		--experiment "$(EXPERIMENT)"

# -------------------------
# Hyperparameter Optimization
# -------------------------

sweep-sft:
	@echo "Initializing SFT W&B Sweep and launching agent..."
	@OUTPUT=$$(wandb sweep sweeps/sft.yaml 2>&1); \
	echo "$$OUTPUT"; \
	SWEEP_ID=$$(echo "$$OUTPUT" | grep -oE "ID: [a-zA-Z0-9]+" | awk '{print $$2}' | tail -1); \
	if [ -z "$$SWEEP_ID" ]; then \
		echo "Failed to extract Sweep ID from wandb output."; \
		exit 1; \
	fi; \
	echo "Parsed Sweep ID: $$SWEEP_ID. Starting agent..."; \
	wandb agent $(WANDB_ENTITY)/$(WANDB_PROJECT)/$$SWEEP_ID --count $(COUNT)
	@echo "Sweep complete! Generating sweep report..."
	python scripts/generate_sweep_report.py

sweep-dpo:
	@echo "Initializing DPO W&B Sweep and launching agent..."
	@OUTPUT=$$(wandb sweep sweeps/dpo.yaml 2>&1); \
	echo "$$OUTPUT"; \
	SWEEP_ID=$$(echo "$$OUTPUT" | grep -oE "ID: [a-zA-Z0-9]+" | awk '{print $$2}' | tail -1); \
	if [ -z "$$SWEEP_ID" ]; then \
		echo "Failed to extract Sweep ID from wandb output."; \
		exit 1; \
	fi; \
	echo "Parsed Sweep ID: $$SWEEP_ID. Starting agent..."; \
	wandb agent $(WANDB_ENTITY)/$(WANDB_PROJECT)/$$SWEEP_ID --count $(COUNT)
	@echo "Sweep complete! Generating sweep report..."
	python scripts/generate_sweep_report.py

sweep-grpo:
	@echo "Initializing GRPO W&B Sweep and launching agent..."
	@OUTPUT=$$(wandb sweep sweeps/grpo.yaml 2>&1); \
	echo "$$OUTPUT"; \
	SWEEP_ID=$$(echo "$$OUTPUT" | grep -oE "ID: [a-zA-Z0-9]+" | awk '{print $$2}' | tail -1); \
	if [ -z "$$SWEEP_ID" ]; then \
		echo "Failed to extract Sweep ID from wandb output."; \
		exit 1; \
	fi; \
	echo "Parsed Sweep ID: $$SWEEP_ID. Starting agent..."; \
	wandb agent $(WANDB_ENTITY)/$(WANDB_PROJECT)/$$SWEEP_ID --count $(COUNT)
	@echo "Sweep complete! Generating sweep report..."
	python scripts/generate_sweep_report.py

# -------------------------
# Dev tools
# -------------------------

format:
	isort src scripts tests app
	black src scripts tests app
	ruff check --fix src scripts tests app

lint:
	isort --check-only src scripts tests app
	black --check src scripts tests app
	ruff check src scripts tests app

test:
	pytest tests/

# -------------------------
# Deployment
# -------------------------

deploy-hf:
	@echo "=========================================================="
	@echo "🚀 Deploying to Hugging Face Hub"
	@echo "➔ Dataset: https://huggingface.co/datasets/chrisjcc/ask-before-answer-data"
	@echo "➔ Model:   https://huggingface.co/chrisjcc/ask-before-answer"
	@echo "=========================================================="
	@RELEASE_TAG=$$(git describe --tags --abbrev=0) && \
	echo "Detected latest release tag: $$RELEASE_TAG" && \
	python scripts/push_to_hub.py deployment.release_tag="$$RELEASE_TAG"

# -------------------------
# App & Docker
# -------------------------

docker-build:
	docker build -t askbeforeanswer-app .

run-app:
	streamlit run app/app.py

# -------------------------
# Cleanup
# -------------------------

clean:
	rm -rf models/* outputs/* results/* wandb/

clean-cache:
	@echo "Pruning DVC Cache to free up disk space..."
	@echo "WARNING: This will permanently delete the weights of old sweep trials!"
	dvc gc -w -f

clean-locks:
	@echo "Forcefully removing leftover DVC locks..."
	rm -f .dvc/lock .dvc/tmp/lock .dvc/tmp/rwlock
