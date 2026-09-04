-include .env

export

# -------------------------
# Dynamic Environment Fixes (HPC / Conda Compatibility)
# -------------------------
# 1. Prevent Python from loading or installing into the global ~/.local user-site directory,
#    ensuring strict isolation for Conda environments.
export PYTHONNOUSERSITE := 1

# Default sweep agent trial count
COUNT ?= 10

# W&B Model Registry promotion defaults
REGISTRY_NAME ?= Model
REGISTRY_COLLECTION ?= AskBeforeAnswer-Models
PRODUCTION_ALIAS ?= production
PROVENANCE_FILE ?= provenance/model_promotion.json

# -------------------------
# Phony targets
# -------------------------

.PHONY: \
	help \
	install install-dvc \
	preprocess run-pipeline train \
	train-sft train-dpo train-sft-only train-dpo-only train-orpo \
	train-grpo ablation-suite \
	evaluate infer \
	promote-dvc publish-model-artifact promote-model \
	sweep \
	format lint test \
	publish-hf-release \
	docker-build run-app \
	push pull \
	clean clean-cache clean-locks

# -------------------------
# Help
# -------------------------

help:
	@echo ""
	@echo "DVC-driven ML pipeline"
	@echo ""
	@echo "Core commands:"
	@echo "  make install                 Install dependencies"
	@echo "  make install-dvc             Install DVC"
	@echo "  make preprocess              Run data preprocessing"
	@echo "  make run-pipeline            Run full DVC pipeline"
	@echo ""

	@echo "Training variants (DVC stages):"
	@echo "  make train TRAIN_VARIANT=<variant>"
	@echo "                               Run selected DVC training variant"
	@echo "                               Supported: sft, dpo, sft-only, dpo-only, orpo, grpo"
	@echo ""
	@echo "  make train-sft               Run SFT training variant"
	@echo "  make train-dpo               Run DPO training variant (requires SFT)"
	@echo "  make train-sft-only          Run SFT-only baseline"
	@echo "  make train-dpo-only          Run DPO-only baseline"
	@echo "  make train-orpo              Run ORPO baseline"
	@echo "  make train-grpo              Run GRPO baseline"
	@echo "  make ablation-suite          Run all experimental variants"
	@echo ""

	@echo "Evaluation & Inference:"
	@echo "  make evaluate                Run evaluation scripts"
	@echo "  make infer                   Run inference"
	@echo ""

	@echo "Model Promotion & Publication:"
	@echo ""
	@echo "Release workflow:"
	@echo "  1. make promote-dvc MODEL=<model> EXPERIMENT=<id>"
	@echo "     Promote a DVC experiment to the promoted model"
	@echo "  2. make publish-model-artifact MODEL=<model> EXPERIMENT=<id> STAGE=<stage>"
	@echo "     Publish and promote the verified DVC model to W&B production"
	@echo "  3. make publish-hf-release"
	@echo "     Verify production provenance and publish Model and Data cards to Hugging Face"
	@echo ""
	@echo "Direct/alternative W&B promotion:"
	@echo "  make promote-model ARTIFACT_REF=<exact-wandb-artifact-ref>"
	@echo "     Directly promote an existing W&B artifact"
	@echo "     (alternative to publish-model-artifact)"
	@echo ""

	@echo "W&B promotion requires an immutable: vN artifact reference."
	@echo "Example: rl4aa/ask-before-answer/Clarifier-grpo:v17"
	@echo ""

	@echo "Deployment:"
	@echo "  make publish-hf-release"
	@echo "     Publish the verified production artifact to Hugging Face"
	@echo "     Requires $(PROVENANCE_FILE)"
	@echo ""

	@echo "Hyperparameter Optimization (Sweeps):"
	@echo "  make sweep FINE_TUNE_METHOD=<sft|dpo|orpo|grpo> COUNT=<n>"
	@echo "                               Run W&B hyperparameter sweep"
	@echo "                               COUNT defaults to $(COUNT)"
	@echo ""

	@echo "Dev tools:"
	@echo "  make format                  Format code with isort, black, and ruff"
	@echo "  make lint                    Check code style and linting"
	@echo "  make test                    Run pytest test suite"
	@echo ""

	@echo "App & Docker:"
	@echo "  make run-app                 Launch the Streamlit demo application"
	@echo "  make docker-build            Build the Docker container image"
	@echo ""

	@echo "Cleanup:"
	@echo "  make clean                   Remove all outputs, models, and W&B cache"
	@echo "  make clean-cache             Prune old DVC cache"
	@echo "  make clean-locks             Forcefully remove DVC lock files"
	@echo ""

# -------------------------
# Setup
# -------------------------

install:
	pip install -r requirements.txt
	pip uninstall -y torchao
	pip install -e .

install-dvc:
	@echo "Checking DVC installation..."
	@if command -v dvc >/dev/null 2>&1; then \
		echo "DVC already installed: $$(dvc --version)"; \
	else \
		echo "DVC not found. Installing..."; \
		if command -v uv >/dev/null 2>&1; then \
			uv tool install dvc --force; \
		elif command -v pipx >/dev/null 2>&1; then \
			pipx install dvc --force; \
		else \
			echo "ERROR: Neither uv nor pipx is installed."; \
			echo "Install uv or pipx first."; \
			exit 1; \
		fi; \
	fi

# -------------------------
# Core pipeline / data (DVC is source of truth)

# DVC is the source of truth
# -------------------------

GPU := $(shell nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | \
       sort -t',' -k2 -nr | head -1 | cut -d',' -f1 | tr -d ' ')

run-pipeline:
	dvc repro

preprocess:
	@echo "Selecting GPU $(GPU)"
	CUDA_VISIBLE_DEVICES=$(GPU) dvc repro preprocess

push:
	@echo "=========================================================="
	@echo "Pushing DVC artifacts to remote storage:"
	@dvc remote list
	@echo "=========================================================="
	dvc push

pull:
	@echo "=========================================================="
	@echo "Pulling DVC artifacts from remote storage:"
	@dvc remote list
	@echo "=========================================================="
	dvc pull

# -------------------------
# DVC training targets
# -------------------------

# Supported DVC training variants.

TRAIN_VARIANTS := sft dpo sft-only dpo-only orpo grpo

# Generic training interface.
#
# TRAIN_VARIANT uses hyphens for CLI readability while the corresponding
# DVC stage uses underscores, e.g.:
#
#   sft-only -> train_sft_only
#   dpo-only -> train_dpo_only

train:
	@if [ -z "$(TRAIN_VARIANT)" ]; then \
		echo "ERROR: TRAIN_VARIANT is required."; \
		echo "Usage: make train TRAIN_VARIANT=<sft|dpo|sft-only|dpo-only|orpo|grpo>"; \
		exit 1; \
	fi
	@if ! echo "$(TRAIN_VARIANTS)" | grep -qw "$(TRAIN_VARIANT)"; then \
		echo "ERROR: Unsupported training variant='$(TRAIN_VARIANT)'."; \
		echo "Supported training variants: $(TRAIN_VARIANTS)"; \
		exit 1; \
	fi
	dvc repro train_$(subst -,_,$(TRAIN_VARIANT))

# Convenience aliases for the generic training interface.

train-sft:
	$(MAKE) train TRAIN_VARIANT=sft

train-dpo:
	$(MAKE) train TRAIN_VARIANT=dpo

train-sft-only:
	$(MAKE) train TRAIN_VARIANT=sft-only

train-dpo-only:
	$(MAKE) train TRAIN_VARIANT=dpo-only

train-orpo:
	$(MAKE) train TRAIN_VARIANT=orpo

train-grpo:
	$(MAKE) train TRAIN_VARIANT=grpo

ablation-suite:
	@echo "Running all experimental baselines..."
	dvc repro train_sft train_dpo train_dpo_only train_orpo train_grpo
	@echo "Evaluating all models with LLM-as-a-Judge..."
	python scripts/evaluate.py
	@echo "Synthesizing experiment results into docs/ablation_report.md..."
	python scripts/generate_report.py
	@echo "Saving DVC experiment..."
	dvc exp save -n ablation_suite

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

promote-dvc:
	@if [ -z "$(MODEL)" ]; then \
		echo "ERROR: MODEL is required."; \
		echo "Usage: make promote-dvc MODEL=<model> EXPERIMENT=<id>"; \
		exit 1; \
	fi
	@if [ -z "$(EXPERIMENT)" ]; then \
		echo "ERROR: EXPERIMENT is required."; \
		echo "Usage: make promote-dvc MODEL=<model> EXPERIMENT=<id>"; \
		exit 1; \
	fi
	@python scripts/promote_experiment.py \
		--model "$(MODEL)" \
		--experiment "$(EXPERIMENT)"

# -------------------------
# Publish model artifact to W&B
# -------------------------

publish-model-artifact:
	@if [ -z "$(MODEL)" ]; then \
		echo "ERROR: MODEL is required."; \
		echo "Usage: make publish-model-artifact MODEL=<model> EXPERIMENT=<id> STAGE=<stage>"; \
		exit 1; \
	fi
	@if [ -z "$(EXPERIMENT)" ]; then \
		echo "ERROR: EXPERIMENT is required."; \
		echo "Usage: make publish-model-artifact MODEL=<model> EXPERIMENT=<id> STAGE=<stage>"; \
		exit 1; \
	fi
	@if [ -z "$(STAGE)" ]; then \
		echo "ERROR: STAGE is required."; \
		echo "Usage: make publish-model-artifact MODEL=<model> EXPERIMENT=<id> STAGE=<stage>"; \
		exit 1; \
	fi
	@echo "=========================================================="
	@echo "Publishing promoted model to W&B"
	@echo "=========================================================="
	@echo "Model:       $(MODEL)"
	@echo "Experiment:  $(EXPERIMENT)"
	@echo "Stage:       $(STAGE)"
	@echo "=========================================================="
	@python scripts/publish_model_artifact.py \
		publication_model="$(MODEL)" \
		experiment="$(EXPERIMENT)" \
		stage="$(STAGE)"
	@echo "=========================================================="
	@echo "W&B model artifact publication complete."
	@echo "=========================================================="

# -------------------------
# W&B-level model promotion
# -------------------------

promote-model:
	@if [ -z "$(ARTIFACT_REF)" ]; then \
		echo "ERROR: ARTIFACT_REF is required."; \
		echo "Usage:"; \
		echo "  make promote-model ARTIFACT_REF=<exact-wandb-artifact-ref>"; \
		echo "Example:"; \
		echo "  make promote-model ARTIFACT_REF=rl4aa/ask-before-answer/Clarifier-grpo:v17"; \
		exit 1; \
	fi
	@echo "=========================================================="
	@echo "W&B Model Registry Promotion"
	@echo "=========================================================="
	@echo "Source artifact: $(ARTIFACT_REF)"
	@echo "Registry:        $(REGISTRY_NAME)"
	@echo "Collection:      $(REGISTRY_COLLECTION)"
	@echo "Alias:           $(PRODUCTION_ALIAS)"
	@echo "Provenance:      $(PROVENANCE_FILE)"
	@echo "=========================================================="
	@python scripts/promote_model.py \
		--artifact-ref "$(ARTIFACT_REF)" \
		--registry-name "$(REGISTRY_NAME)" \
		--registry-collection "$(REGISTRY_COLLECTION)" \
		--production-alias "$(PRODUCTION_ALIAS)" \
		--provenance-file "$(PROVENANCE_FILE)"
	@echo "=========================================================="
	@echo "W&B Model Registry promotion complete."
	@echo "=========================================================="

# -------------------------
# Hyperparameter Optimization
# -------------------------

# Supported fine-tuning methods.

FINE_TUNE_METHODS := sft dpo orpo grpo

sweep:
	@if [ -z "$(FINE_TUNE_METHOD)" ]; then \
		echo "ERROR: FINE_TUNE_METHOD is required."; \
		echo "Usage: make sweep FINE_TUNE_METHOD=<sft|dpo|orpo|grpo> [COUNT=<n>]"; \
		exit 1; \
	fi
	@if ! echo "$(FINE_TUNE_METHODS)" | grep -qw "$(FINE_TUNE_METHOD)"; then \
		echo "ERROR: Unsupported fine-tuning method='$(FINE_TUNE_METHOD)'."; \
		echo "Supported fine-tuning methods: $(FINE_TUNE_METHODS)"; \
		exit 1; \
	fi
	@if [ -z "$(WANDB_ENTITY)" ]; then \
		echo "ERROR: WANDB_ENTITY is not set."; \
		exit 1; \
	fi
	@if [ -z "$(WANDB_PROJECT)" ]; then \
		echo "ERROR: WANDB_PROJECT is not set."; \
		exit 1; \
	fi
	@if [ ! -f "sweeps/$(FINE_TUNE_METHOD).yaml" ]; then \
		echo "ERROR: Sweep configuration not found: sweeps/$(FINE_TUNE_METHOD).yaml"; \
		exit 1; \
	fi
	@echo "=========================================================="
	@echo "Initializing W&B Sweep"
	@echo "=========================================================="
	@echo "Fine-tune method: $(FINE_TUNE_METHOD)"
	@echo "Sweep config:     sweeps/$(FINE_TUNE_METHOD).yaml"
	@echo "Entity:           $(WANDB_ENTITY)"
	@echo "Project:          $(WANDB_PROJECT)"
	@echo "Trial count:      $(COUNT)"
	@echo "=========================================================="
	@OUTPUT=$$(wandb sweep sweeps/$(FINE_TUNE_METHOD).yaml 2>&1) || { \
		echo "$$OUTPUT"; \
		echo "ERROR: Failed to create W&B sweep."; \
		exit 1; \
	}; \
	echo "$$OUTPUT"; \
	SWEEP_ID=$$(echo "$$OUTPUT" | grep -oE "ID: [a-zA-Z0-9]+" | awk '{print $$2}' | tail -1); \
	if [ -z "$$SWEEP_ID" ]; then \
		echo "ERROR: Failed to extract Sweep ID from wandb output."; \
		exit 1; \
	fi; \
	echo "Parsed Sweep ID: $$SWEEP_ID"; \
	echo "Starting W&B sweep agent..."; \
	wandb agent $(WANDB_ENTITY)/$(WANDB_PROJECT)/$$SWEEP_ID --count $(COUNT) || { \
		echo "ERROR: W&B sweep agent failed."; \
		exit 1; \
	}; \
	echo "Generating sweep report..."; \
	python scripts/generate_sweep_report.py \
		--fine-tune-method "$(FINE_TUNE_METHOD)" \
		--sweep-id "$$SWEEP_ID"

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

publish-hf-release:
	@echo "=========================================================="
	@echo "🚀 Deploying verified production model to Hugging Face"
	@echo "➔ Dataset: https://huggingface.co/datasets/chrisjcc/ask-before-answer-dataset"
	@echo "➔ Model:   https://huggingface.co/chrisjcc/ask-before-answer"
	@echo "=========================================================="
	@test -f "$(PROVENANCE_FILE)" || ( \
		echo "ERROR: $(PROVENANCE_FILE) not found."; \
		echo "Run 'make promote-model ARTIFACT_REF=<exact-artifact-ref>' first."; \
		exit 1; \
	)
	@RELEASE_TAG=$$(git describe --tags --abbrev=0) && \
	echo "Detected latest release tag: $$RELEASE_TAG" && \
	echo "Using W&B promotion provenance: $(PROVENANCE_FILE)" && \
	python scripts/push_to_hub.py \
		deployment.release_tag="$$RELEASE_TAG"
	@echo "=========================================================="
	@echo "Hugging Face deployment complete."
	@echo "=========================================================="

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
