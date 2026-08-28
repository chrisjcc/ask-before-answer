# End-to-End Training, Evaluation, and Deployment Workflow

This document describes the end-to-end architecture used by AskBeforeAnswer to move from data preparation and model experimentation to a reproducible production release.

The system separates **model development**, **model/data publication**, and **application deployment** into distinct stages. DVC provides experiment and artifact tracking during development, Weights & Biases (W&B) provides experiment telemetry and the production model registry, the Hugging Face Hub provides the public model and dataset repositories, and GitHub Actions deploys the inference application to the Hugging Face Space.

The resulting architecture has two coordinated release paths:

```text
                         PRODUCTION RELEASE
                                │
                 ┌──────────────┴──────────────┐
                 │                             │
                 ▼                             ▼
          Model/Data Release              App Release
                 │                             │
             W&B → HF Hub              GitHub Release
                 │                             │
                 ▼                             ▼
       Model Card + Data Card       GitHub Action → HF Space
       + evaluation results                  │
                                             ▼
                                      Streamlit inference
                                             │
                                             ▼
                                      HF model repository
```

The model/data path determines **what model is released**. The application path determines **what inference code is deployed**. Keeping these concerns separate allows either side to evolve independently while maintaining an explicit connection between the released application and the production model repository it consumes.

---

## 1. Architecture Overview

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

Likewise, the Hugging Face Space is not a copy of the training repository. It contains only the runtime components required to run the inference application.

The deployed Space has the following payload:

```text
hf-space/
├── Dockerfile
├── pyproject.toml
├── README.md
├── app/
│   ├── app.py
│   └── requirements.txt
└── src/
    └── ask_before_answer/
        ├── __init__.py
        └── inference/
            ├── __init__.py
            └── pipeline.py
```

The Space therefore represents an **inference application**, not the complete training system.

---

## 2. Continuous Integration

Continuous Integration runs through GitHub Actions on:

* pushes to `main`
* pull requests targeting `main`

The CI workflow performs static checks and tests in a CPU-only environment.

The workflow:

1. Checks out the repository.
2. Installs Python.
3. Installs the dedicated CI dependency set from `requirements-ci.txt`.
4. Installs the `askbeforeanswer` package.
5. Runs Ruff.
6. Runs Black.
7. Runs isort.
8. Runs the test suite.

Conceptually:

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
     │          │
     └────┬─────┘
          ▼
       CI result
```

CI deliberately does not reproduce the GPU training environment. Training has substantially different dependencies, including CUDA-enabled PyTorch and GPU-specific packages such as Unsloth, xFormers, and Flash Attention.

This separation keeps CI relatively lightweight while still validating the Python package, application code, tests, and static quality.

The CI environment uses `requirements-ci.txt`, while the training environment continues to use the GPU-oriented `requirements.txt`.

Python 3.10 is currently retained for CI and runtime compatibility. Migration to Python 3.11 is a separate dependency-validation task and should not be performed until the GPU training environment has been verified against Python 3.11.

---

## 3. Data Preprocessing

Data preprocessing is the first ML pipeline stage.

Run:

```bash
make preprocess
```

The preprocessing code lives under:

```text
src/ask_before_answer/data/
```

and produces the datasets consumed by the subsequent training stages.

DVC tracks the resulting artifacts so that generated data does not need to be committed directly to Git.

The conceptual flow is:

```text
Raw / source dataset
        │
        ▼
make preprocess
        │
        ▼
Data preprocessing
        │
        ├── SFT training data
        ├── SFT validation data
        ├── DPO/ORPO/GRPO training data
        └── DPO/ORPO/GRPO validation data
        │
        ▼
      DVC
```

The preprocessing stage is therefore shared by the different fine-tuning strategies.

When DVC determines that preprocessing outputs are already current and available in its cache, the stage does not need to be recomputed. If inputs or dependencies have changed, DVC can reproduce the stage.

---

## 4. Model Training

Training is orchestrated through DVC and exposed through the Makefile.

A specific training variant can be run with:

```bash
make train TRAIN_VARIANT=sft
```

For the common variants, readable aliases are provided:

```bash
make train-sft
make train-dpo
make train-orpo
make train-grpo
```

The training implementation lives under:

```text
src/ask_before_answer/training/
```

with the individual training entry points exposed through the corresponding scripts.

The training architecture is:

```text
                 DVC
                  │
                  ▼
        Training configuration
                  │
                  ▼
          Training script
                  │
          ┌───────┼────────┬────────┐
          ▼       ▼        ▼        ▼
         SFT     DPO      ORPO     GRPO
          │       │        │        │
          └───────┴────────┴────────┘
                          │
                          ▼
                    Model artifact
                          │
                          ▼
                    DVC tracking
```

The training environment is intentionally separate from the CPU-only CI and HF Space environments. The training dependency set contains GPU-specific packages required for efficient fine-tuning.

Training outputs are treated as experiment artifacts rather than ordinary source files.

---

## 5. Evaluation

Evaluation measures the resulting model against the project's evaluation criteria and records the resulting metrics.

The evaluation implementation is under:

```text
src/ask_before_answer/evaluation/
```

and includes both evaluation/judging and metric calculation.

Evaluation can be invoked through the project's Makefile workflow, including:

```bash
make evaluate
```

Evaluation is conceptually downstream of training:

```text
Training
   │
   ▼
Model artifact
   │
   ▼
make evaluate
   │
   ├── Evaluation / judging
   ├── Metric calculation
   ├── W&B / Weave telemetry
   └── Evaluation results
```

Evaluation results are subsequently used when determining which trained model should be promoted for release.

The evaluation artifacts also become part of the publication workflow, where the Model Card and Data Card include the relevant evaluation information.

---

## 6. Hyperparameter Optimization

Hyperparameter optimization is implemented using W&B Sweeps.

A sweep is launched with:

```bash
make sweep FINE_TUNE_METHOD=sft COUNT=10
```

The supported fine-tuning methods are:

```text
sft
dpo
orpo
grpo
```

For example:

```bash
make sweep FINE_TUNE_METHOD=dpo COUNT=10
```

The sweep architecture is:

```text
                 W&B Sweep
                     │
          ┌──────────┼──────────┐
          ▼          ▼          ▼
        Trial 1    Trial 2    Trial N
          │          │          │
          ▼          ▼          ▼
       DVC experiment runs
          │          │          │
          └──────────┼──────────┘
                     ▼
             W&B run telemetry
                     │
                     ▼
              Sweep evaluation
                     │
                     ▼
             Sweep report
```

Each trial can override training parameters without modifying the tracked training configuration permanently.

The sweep report is generated from the W&B sweep itself:

```bash
python scripts/generate_sweep_report.py \
    --fine-tune-method sft \
    --sweep-id <sweep-id>
```

The resulting report records the optimized parameters, objective metric, completed configurations, and, where applicable, a parameter/objective plot.

This makes W&B the appropriate system for comparing sweep trials, while DVC remains responsible for reproducible local experiment execution and artifact tracking.

---

## 7. DVC Experiment Promotion

Not every training experiment should become a production model.

DVC provides the experiment layer where different model variants can be created, compared, and retained without immediately modifying the production state.

Once an experiment has been identified as the desired candidate, it can be promoted through the project's promotion workflow.

The promotion process establishes a durable relationship between the selected experiment and the model artifact that should proceed toward production.

The important distinction is:

```text
DVC experiment
     │
     │ select successful experiment
     ▼
Production candidate
     │
     ▼
W&B model artifact
     │
     ▼
Production registry
```

Promotion therefore represents an explicit decision point rather than an automatic consequence of completing training.

---

## 8. W&B Model Registry

The W&B Model Registry is the production model-selection layer.

Training and evaluation may produce many candidate models. The registry provides a stable mechanism for identifying which artifact is considered the production model.

The production flow is:

```text
DVC experiment
      │
      ▼
W&B model artifact
      │
      ▼
W&B Model Registry
      │
      ▼
production alias
```

The production alias is the important deployment pointer. It allows the deployment process to identify the selected model artifact without hardcoding a particular experiment ID into the deployment application.

The promotion command uses the exact W&B artifact reference selected for release, for example:

```bash
make promote-model ARTIFACT_REF=<exact-artifact-ref>
```

The resulting provenance information is retained locally so that the subsequent Hugging Face publication can verify that a production model has explicitly been selected.

This gives the architecture a clear separation:

* **DVC:** Which experiment and artifacts were produced?
* **W&B:** Which model artifact is selected for production?
* **Hugging Face:** Which production model and dataset are publicly published?

---

## 9. Hugging Face Model and Dataset Publication

The Hugging Face Hub is the public distribution layer for the model and dataset.

The project publishes:

```text
Hugging Face
├── ask-before-answer
│   └── Model Card + model release
│
└── ask-before-answer-dataset
    └── Data Card + dataset release
```

The publication workflow is initiated with:

```bash
make deploy-hf
```

Before publication, the deployment process verifies the production provenance established during model promotion.

The model/data publication flow is:

```text
W&B production model
          │
          ▼
   Verified provenance
          │
          ▼
     make deploy-hf
          │
          ▼
     Hugging Face Hub
       ┌────┴────┐
       ▼         ▼
     Model     Dataset
      Card       Card
       │         │
       └────┬────┘
            ▼
    Evaluation results
```

The Model Card and Data Card provide the public-facing record of the release, including the relevant evaluation results.

This path publishes the **model/data release**. It does not deploy the Streamlit application.

---

## 10. Hugging Face Demo Application

The Hugging Face Demo is a separate inference application.

Its source code consists of:

```text
app/
├── app.py
└── requirements.txt
```

and the minimal portion of the Python package required for inference:

```text
src/
└── ask_before_answer/
    ├── __init__.py
    └── inference/
        ├── __init__.py
        └── pipeline.py
```

The application imports the inference pipeline through the installed package namespace:

```python
from ask_before_answer import ClarifyOrActPipeline
```

The application does **not** require:

```text
ask_before_answer.data
ask_before_answer.training
ask_before_answer.evaluation
```

Consequently, these components are intentionally excluded from the HF Space deployment payload.

The inference application loads the released AskBeforeAnswer model from the Hugging Face model repository:

```text
HF Space
   │
   ▼
Streamlit app
   │
   ▼
ClarifyOrActPipeline
   │
   ▼
Hugging Face model repository
   │
   ▼
AskBeforeAnswer model
```

The Space currently performs inference on CPU. It does not require the GPU-oriented training environment.

This is why the Space has its own `app/requirements.txt` rather than installing the complete training dependency set.

---

## 11. Demo Application Deployment

The HF Demo application is deployed through a GitHub Release rather than through the model publication command.

The workflow is defined in:

```text
.github/workflows/deploy-hf-demo.yml
```

It is triggered when a GitHub Release is published:

```yaml
on:
  release:
    types: [published]
```

The deployment sequence is:

```text
GitHub Release
      │
      ▼
GitHub Actions
      │
      ▼
Check out released source revision
      │
      ▼
Clone HF Space repository
      │
      ▼
Replace application payload
      │
      ▼
Commit deployment
      │
      ▼
Push to HF Space
      │
      ▼
Hugging Face builds Docker image
      │
      ▼
Streamlit application
```

The GitHub Action checks out the source corresponding to the release tag. It then constructs the minimal Space payload rather than copying the entire repository.

The deployed payload contains:

```text
Dockerfile
pyproject.toml
README.md
app/
src/ask_before_answer/
```

with only the inference package required by the application.

The HF Space subsequently builds and runs the application using its Dockerfile.

This makes the GitHub Release the explicit version boundary for the application itself.

---

## 12. Complete Release Workflow

A complete production release therefore consists of two related but independent operations.

### Phase A: Train and evaluate

```bash
make preprocess

make train-sft
# or:
make train-dpo
make train-orpo
make train-grpo

make evaluate
```

For hyperparameter optimization:

```bash
make sweep FINE_TUNE_METHOD=sft COUNT=10
```

and then generate the corresponding sweep report:

```bash
python scripts/generate_sweep_report.py \
    --fine-tune-method sft \
    --sweep-id <sweep-id>
```

### Phase B: Select and publish the production model

After identifying the desired model:

```bash
make promote-model ARTIFACT_REF=<exact-artifact-ref>
```

Then publish the verified production model and dataset:

```bash
make deploy-hf
```

This produces the public Model Card and Data Card, including the associated evaluation results.

### Phase C: Release the inference application

Create and publish a GitHub Release corresponding to the application version.

Publishing the release triggers:

```text
GitHub Release
      │
      ▼
deploy-hf-demo.yml
      │
      ▼
HF Space repository
      │
      ▼
Docker build
      │
      ▼
Streamlit application
```

The deployed application then loads the model from:

```text
https://huggingface.co/chrisjcc/ask-before-answer
```

The complete system therefore becomes:

```text
                 DEVELOPMENT
                     │
        ┌────────────┼────────────┐
        ▼            ▼            ▼
   preprocess      train       sweeps
        │            │            │
        └────────────┼────────────┘
                     ▼
                 evaluate
                     │
                     ▼
              DVC experiment
                     │
                     ▼
              W&B promotion
                     │
                     ▼
              ┌──────┴──────┐
              │             │
              ▼             ▼
        Model/Data       App Release
          Release            │
              │              │
              ▼              ▼
          W&B → HF      GitHub Action
              │              │
              ▼              ▼
        Model + Data     HF Space
          Cards             │
              │              ▼
              │        Streamlit inference
              │              │
              └──────────────┤
                             ▼
                    HF model repository
```

---

## 13. Separation of Responsibilities

The architecture intentionally assigns different responsibilities to different systems.

### GitHub

GitHub is the source-of-truth for:

* Python source code
* configuration
* Makefile commands
* CI workflows
* deployment workflows
* Git history
* GitHub Releases

Git does not store large generated model or dataset artifacts.

### DVC

DVC is responsible for:

* dataset artifacts
* model artifacts
* reproducible pipeline stages
* experiments
* relationships between code/configuration and generated artifacts

DVC is primarily the **development and experiment layer**.

### W&B

W&B is responsible for:

* training telemetry
* sweep orchestration
* run comparison
* evaluation telemetry
* model artifacts
* production model registry
* production model selection

W&B is primarily the **experiment observability and model-selection layer**.

### Hugging Face Hub

The Hugging Face Hub is responsible for public distribution of:

* the production model
* the production dataset
* Model Card
* Data Card
* evaluation information associated with the release

### Hugging Face Space

The Hugging Face Space is responsible for:

* hosting the Streamlit demo
* running inference
* providing the public demonstration interface
* loading the production model from the Hugging Face model repository

It is not a training environment.

### GitHub Actions

GitHub Actions is responsible for automation between GitHub and external deployment targets.

There are two distinct workflows:

```text
CI
GitHub → test/lint/format validation

Deployment
GitHub Release → HF Space deployment
```

This prevents application deployment from being coupled to ordinary pushes to `main`.

---

## 14. Operational Summary

The following commands summarize the normal lifecycle.

### Prepare data

```bash
make preprocess
```

### Train a specific variant

```bash
make train TRAIN_VARIANT=sft
```

or:

```bash
make train-sft
make train-dpo
make train-orpo
make train-grpo
```

### Evaluate

```bash
make evaluate
```

### Run a W&B sweep

```bash
make sweep FINE_TUNE_METHOD=sft COUNT=10
```

Supported methods:

```text
sft
dpo
orpo
grpo
```

### Generate a sweep report

```bash
python scripts/generate_sweep_report.py \
    --fine-tune-method sft \
    --sweep-id <sweep-id>
```

### Promote a model artifact

```bash
make promote-model ARTIFACT_REF=<exact-artifact-ref>
```

### Publish the production model and dataset

```bash
make deploy-hf
```

### Deploy the demo application

Create and publish a GitHub Release.

The release automatically triggers:

```text
.github/workflows/deploy-hf-demo.yml
```

which deploys the released application revision to the Hugging Face Space.

---

## 15. End-to-End Architecture at a Glance

The complete architecture can be summarized as:

```text
                           GITHUB
                             │
                  ┌──────────┴──────────┐
                  │                     │
                  ▼                     ▼
               CI/CD              GitHub Release
                  │                     │
                  ▼                     ▼
          Lint / Test             Deploy HF Demo
                                        │
                                        ▼
                                   HF Space
                                        │
                                        ▼
                                  Streamlit App
                                        │
                                        ▼
                                  HF Model Repo
                                        ▲
                                        │
                             ┌──────────┴──────────┐
                             │                     │
                           W&B                   HF Hub
                             ▲                     ▲
                             │                     │
                        Model Registry      Model + Dataset
                             ▲                publication
                             │                     ▲
                             │                     │
                           DVC ────────────────────┘
                             ▲
                             │
                    Training / Evaluation
                             ▲
                             │
                        Preprocessing
```

The important architectural boundary is between **experimentation and release**.

During experimentation, DVC and W&B provide the machinery needed to generate, compare, evaluate, and select models. Once a model is explicitly promoted, the W&B registry identifies the production artifact that is published to the Hugging Face Hub.

Application deployment follows a separate release boundary. A GitHub Release freezes the application source revision, and GitHub Actions deploys that revision to the HF Space. The Space contains only the inference application and its required runtime code and dependencies.

The result is a reproducible relationship between:

```text
source revision
      │
      ├── training/evaluation artifacts
      │          │
      │          ▼
      │      W&B production model
      │          │
      │          ▼
      │      HF model repository
      │
      └── GitHub application release
                 │
                 ▼
             HF Space
                 │
                 ▼
        inference against the
        released HF model
```

This separation keeps the training environment, production model, public model/data artifacts, and inference application independently manageable while preserving a clear path from experiment to production release.
