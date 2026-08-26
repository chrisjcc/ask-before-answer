# Deployment and Release Workflow

This document describes the end-to-end release architecture for
AskBeforeAnswer, from local DVC experiments through model promotion,
Hugging Face model/data publication, and deployment of the Streamlit
demonstration application.

The architecture deliberately separates three concerns:

1. **Model development and experimentation**
   - DVC
   - local/GPU training environments
   - evaluation
   - W&B experiment tracking

2. **Model and dataset publication**
   - W&B Model Registry
   - Hugging Face Model Repository
   - Hugging Face Dataset Repository
   - Model Card and Data Card containing evaluation results

3. **Application deployment**
   - GitHub Releases
   - GitHub Actions
   - Hugging Face Spaces
   - CPU-only Streamlit inference

These are related but independent deployment paths.

---

## 1. Architecture Overview

The production workflow is:

```text
                    Git Repository
                          │
             ┌────────────┴────────────┐
             │                         │
             ▼                         ▼
        GitHub CI                Local/GPU Environment
             │                         │
      lint / test / QA                  │
                                       ▼
                                DVC Training Pipeline
                                       │
                             ┌─────────┼─────────┐
                             │         │         │
                             ▼         ▼         ▼
                         preprocess  train    evaluate
                                       │
                                       ▼
                                DVC Experiment
                                       │
                                       ▼
                              W&B Model Registry
                                       │
                               production promotion
                                       │
                    ┌──────────────────┴─────────────────┐
                    │                                    │
                    ▼                                    ▼
             Hugging Face Hub                     GitHub Release
             Model/Data Release                         │
                    │                                    │
          ┌─────────┴─────────┐                          ▼
          ▼                   ▼                  GitHub Actions
      Model Card          Data Card                     │
          │                   │                         ▼
          └─────────┬─────────┘                 Hugging Face Space
                    │                                  │
                    │                                  ▼
                    │                           Streamlit Demo
                    │                                  │
                    └──────────────┐                   ▼
                                   │             Production Model
                                   └─────────────► (loaded dynamically)
