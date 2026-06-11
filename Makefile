.PHONY: install train-sft train-dpo run-ablation evaluate infer format lint test docker-build run-app

install:
	pip install -r requirements.txt
	pip install -e .

train-sft:
	python scripts/train_sft.py

train-dpo:
	python scripts/train_dpo.py

run-ablation:
	python scripts/run_ablation.py

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
