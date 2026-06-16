.PHONY: help install install-dvc run-pipeline train evaluate infer format lint test docker-build run-app clean

# -------------------------
# Help
# -------------------------
help:
	@echo ""
	@echo "DVC-driven ML pipeline"
	@echo ""
	@echo "Core commands:"
	@echo "  make install        Install dependencies"
	@echo "  make run-pipeline   Run full DVC pipeline"
	@echo "  make train          Alias for run-pipeline"
	@echo ""
	@echo "Utils:"
	@echo "  make evaluate       Run evaluation scripts"
	@echo "  make infer          Run inference"
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
# Evaluation / inference (optional shortcuts)
# -------------------------
evaluate:
	python scripts/evaluate.py

infer:
	python scripts/infer.py

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
docker-build:
	docker build -t askbeforeanswer-app .

run-app:
	streamlit run app/app.py

clean:
	rm -rf models/* outputs/* results/*
