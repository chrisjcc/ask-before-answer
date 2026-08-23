export

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
	train-grpo train-grpo-dpo ablation-suite \
	evaluate infer \
	promote-dvc publish-model-artifact promote-model \
	sweep-sft sweep-dpo sweep-grpo \
	format lint test \
	deploy-hf \
	docker-build run-app \
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
	@echo "  make train                   Alias for run-pipeline"
	@echo ""
	@echo "Training variants (DVC stages):"
	@echo "  make train-sft               Run SFT stage"
	@echo "  make train-dpo               Run DPO stage (requires SFT)"
	@echo "  make train-sft-only          Run SFT-only baseline"
	@echo "  make train-dpo-only          Run DPO-only baseline"
	@echo "  make train-orpo              Run ORPO baseline"
	@echo "  make train-grpo              Run GRPO baseline"
	@echo "  make train-grpo-dpo          Run GRPO->DPO pipeline"
	@echo "  make ablation-suite          Run all experimental variants"
	@echo ""
	@echo "Evaluation & Inference:"
	@echo "  make evaluate                Run evaluation scripts"
	@echo "  make infer                   Run inference"
	@echo ""
	@echo "Model Promotion & Publication:"
	@echo "  make promote-dvc MODEL=<sft|dpo|grpo|orpo> EXPERIMENT=<id>"
	@echo "                              Promote a DVC experiment"
	@echo "  make publish-model-artifact MODEL=<model> EXPERIMENT=<id> STAGE=<stage>"
	@echo "                              Publish promoted DVC model to W&B"
	@echo "  make promote-model ARTIFACT_REF=<exact-wandb-artifact-ref>"
	@echo "                              Verify and promote exact W&B artifact"
	@echo "                              to Model Registry + production alias"
        @echo ""
	@echo "Deployment:"
	@echo "  make deploy-hf              Deploy Registry-verified model to Hugging Face"
	@echo "                              using provenance/model_promotion.json"
	@echo ""
	@echo "W&B promotion requires an immutable :vN artifact reference."
	@echo "Example: rl4aa/ask-before-answer/Clarifier-grpo:v17"
	@echo ""
	@echo "Hyperparameter Optimization (Sweeps):"
	@echo "  make sweep-sft              Run hyperparameter sweep for SFT stage"
	@echo "  make sweep-dpo              Run hyperparameter sweep for DPO stage"
	@echo "  make sweep-grpo             Run hyperparameter sweep for GRPO stage"
	@echo "                              Override trial count with COUNT=<n>"
	@echo ""
	@echo "Dev tools:"
	@echo "  make format                 Format code with isort, black, and ruff"
	@echo "  make lint                   Check code style and linting"
	@echo "  make test                   Run pytest test suite"
	@echo ""
	@echo "App & Docker:"
	@echo "  make run-app                Launch the Streamlit demo application"
	@echo "  make docker-build           Build the Docker container image"
	@echo ""
	@echo "Cleanup:"
	@echo "  make clean                  Remove all outputs, models, and W&B cache"
	@echo "  make clean-cache            Prune old DVC cache"
	@echo "  make clean-locks            Forcefully remove DVC lock files"
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

promote-dvc:
	@if [ -z "$(MODEL)" ]; then \
		echo "ERROR: MODEL is required."; \
		echo "Usage: make promote-dvc MODEL=<sft|dpo|grpo|orpo> EXPERIMENT=<id>"; \
		exit 1; \
	fi
	@if [ -z "$(EXPERIMENT)" ]; then \
		echo "ERROR: EXPERIMENT is required."; \
		echo "Usage: make promote-dvc MODEL=<sft|dpo|grpo|orpo> EXPERIMENT=<id>"; \
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
		publication-model="$(MODEL)" \
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
	@echo "Registry:         $(REGISTRY_NAME)"
	@echo "Collection:       $(REGISTRY_COLLECTION)"
	@echo "Alias:            $(PRODUCTION_ALIAS)"
	@echo "Provenance:       $(PROVENANCE_FILE)"
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
	@echo "🚀 Deploying verified production model to Hugging Face"
	@echo "➔ Dataset: https://huggingface.co/datasets/chrisjcc/ask-before-answer-dataset"
	@echo "➔ Model:   https://huggingface.co/chrisjcc/ask-before-answer"
	@echo "=========================================================="
	@test -f "provenance/model_promotion.json" || ( \
		echo "ERROR: provenance/model_promotion.json not found."; \
		echo "Run 'make promote-model ARTIFACT_REF=<exact-artifact-ref>' first."; \
		exit 1; \
	)
	@RELEASE_TAG=$$(git describe --tags --abbrev=0) && \
	echo "Detected latest release tag: $$RELEASE_TAG" && \
	echo "Using W&B promotion provenance: provenance/model_promotion.json" && \
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
