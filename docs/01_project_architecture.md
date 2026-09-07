# Project Architecture & Overview

This document describes the end-to-end architecture used by AskBeforeAnswer to move from data preparation and model experimentation to a reproducible production release.

The system separates **model development**, **model/data publication**, and **application deployment** into distinct stages. DVC provides experiment and artifact tracking during development, Weights & Biases (W&B) provides experiment telemetry and the production model registry, the Hugging Face Hub provides the public model and dataset repositories, and GitHub Actions deploys the inference application to the Hugging Face Space.

## 1. Architectural Motivation

Our deployment pipeline is deliberately split into two distinct stages to maintain a strict "chain of custody." 

This architectural choice enforces a **human-in-the-loop** design pattern. The pipeline is heavily automated for executing sweeps, generating telemetry reports, and securely deploying the finalized assets, but it intentionally stops short of automatically selecting the "winner." 

Because the best model is often not just the one with the lowest loss (requiring qualitative evaluation of generation quality, perplexity, etc.), the system relies on the developer to review the generated telemetry reports and explicitly pull the trigger on which experiment gets promoted.

## 2. Separation of Responsibilities

AskBeforeAnswer uses several systems, each with a deliberately limited responsibility.

| System             | Primary responsibility                                                            |
| ------------------ | --------------------------------------------------------------------------------- |
| Git / GitHub       | Source code, configuration, CI, releases                                          |
| DVC                | Versioned datasets, model artifacts, and reproducible experiments                 |
| W&B                | Training telemetry, hyperparameter sweeps, evaluation results, and model registry |
| Hugging Face Hub   | Public model and dataset publication                                              |
| GitHub Actions     | Continuous integration and application deployment                                 |
| Hugging Face Space | Public Streamlit inference application                                            |

The central design principle is that **large or generated ML artifacts do not become part of the Git repository**. Git tracks the code and configuration that produced them, while DVC tracks the corresponding data and model artifacts.

## 3. Continuous Integration

Continuous Integration runs through GitHub Actions on pushes and pull requests to `main`.
The CI workflow performs static checks and tests in a CPU-only environment.

```text
GitHub push / pull request
          │
          ▼
     GitHub Actions
          │
          ▼
   CPU-only CI environment
          │
     ┌────┴─────┐
     ▼          ▼
   Lint       Tests
```

CI deliberately does not reproduce the GPU training environment. Training has substantially different dependencies, including CUDA-enabled PyTorch and GPU-specific packages such as Unsloth, xFormers, and Flash Attention.

This separation keeps CI relatively lightweight while still validating the Python package, application code, tests, and static quality.
