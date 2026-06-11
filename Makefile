.PHONY: install install-dvc train-sft train-dpo run-pipeline evaluate infer format lint test docker-build run-app

install:
	pip install -r requirements.txt
	pip install -e .

install-dvc:
	@echo "Installing DVC globally..."
	@if command -v uv >/dev/null 2>&1; then \
		uv tool install dvc; \
	elif command -v pipx >/dev/null 2>&1; then \
		pipx install dvc; \
	else \
		echo "Neither uv nor pipx found. Please install uv (https://github.com/astral-sh/uv) or pipx to install DVC globally."; \
		exit 1; \
	fi

train-sft:
	python scripts/train_sft.py

train-dpo:
	python scripts/train_dpo.py

run-pipeline:
	dvc repro

evaluate:
	python scripts/evaluate.py

infer:
	python scripts/infer.py

format:
	black src scripts tests app
	ruff check --fix src scripts tests app

lint:
	black --check src scripts tests app
	ruff check src scripts tests app

test:
	pytest tests/

docker-build:
	docker build -t askbeforeanswer-app .

run-app:
	streamlit run app/app.py
