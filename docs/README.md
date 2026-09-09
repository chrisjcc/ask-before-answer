# Documentation Overview

This directory contains the comprehensive documentation architecture covering the full project lifecycle of AskBeforeAnswer, from data generation to Hugging Face deployment.

The documents are designed to be read sequentially:

| Phase | Document | Description |
|---|---|---|
| 1 | [01_project_architecture.md](01_project_architecture.md) | High-level architecture, CI, and the human-in-the-loop philosophy. |
| 2 | [02_data_preprocessing.md](02_data_preprocessing.md) | The 3-stage synthetic data generation pipeline (SFT targets & prompt-guided hard negatives). |
| 3 | [03_model_training.md](03_model_training.md) | DVC training workflow and the Unsloth/FlashAttention acceleration stack. |
| 4 | [04_evaluation_analysis.md](04_evaluation_analysis.md) | The definitive ranking and analysis of the 6 model variants (SFT vs DPO vs GRPO). |
| 5 | [05_hyperparameter_sweeps.md](05_hyperparameter_sweeps.md) | The two-step sweep architecture and human-in-the-loop application. |
| 6 | [06_ablation_study.md](06_ablation_study.md) | Leaderboard, top runs, and performance analysis isolated from sweep noise. |
| 7 | [07_data_card.md](07_data_card.md) | The structural schema and splits of the dataset (for HF Hub). |
| 8 | [08_model_card_release.md](08_model_card_release.md) | W&B model registry promotion and HF deployment commands. |
| 9 | [09_demo_deployment.md](09_demo_deployment.md) | Github Actions workflow for deploying the Streamlit UI to an HF Space. |
