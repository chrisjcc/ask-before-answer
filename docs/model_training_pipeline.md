# Model Training and Release Pipeline

This document describes the end-to-end model development lifecycle for AskBeforeAnswer, from prepared training data through model training, evaluation, hyperparameter optimization, model promotion, and deployment.

The pipeline is built around three complementary systems:

* **DVC** provides reproducible pipeline execution, data/model dependencies, and experiment tracking.
* **Hydra** provides structured training configuration and runtime parameter overrides.
* **Weights & Biases (W&B)** provides experiment tracking, hyperparameter sweeps, run comparison, and model artifact management.

The Makefile provides the human-facing command interface to these systems.

The design intentionally distinguishes between a **model**, a **fine-tuning method**, and a **training variant**:

* A **model** is the trained parameterized artifact produced by applying a learning algorithm to data.
* A **fine-tuning method** describes the learning procedure, such as SFT, DPO, ORPO, or GRPO.
* A **training variant** identifies a particular DVC training configuration or experimental variant, such as `sft`, `sft-only`, `dpo-only`, `orpo`, or `grpo`.

This distinction prevents the term `MODEL` from being overloaded to mean both a learning method and the resulting trained model.

---

## 1. End-to-End Architecture

The overall lifecycle is:

```text
                    Training Data
                         │
                         ▼
                ┌─────────────────┐
                │  Preprocessing  │
                │  make preprocess │
                └────────┬────────┘
                         │
                         ▼
                ┌─────────────────┐
                │ DVC Training    │
                │                 │
                │ TRAIN_VARIANT   │
                │ sft / dpo /     │
                │ orpo / grpo ... │
                └────────┬────────┘
                         │
                         ▼
                 Trained Model
                         │
             ┌───────────┴───────────┐
             │                       │
             ▼                       ▼
       Evaluation              W&B Tracking
       make evaluate            / Sweeps
             │                       │
             │                FINE_TUNE_METHOD
             │                       │
             │                       ▼
             │                 Best Experiment
             │                       │
             └───────────┬───────────┘
                         │
                         ▼
                 Model Promotion
                         │
              ┌──────────┴──────────┐
              │                     │
              ▼                     ▼
        DVC release path       Direct W&B path
        promote-dvc            promote-model
              │                     │
              ▼                     │
    publish-model-artifact          │
              │                     │
              └──────────┬──────────┘
                         ▼
                  W&B Production
                         │
                         ▼
                    Deployment
                     publish-hf-release
```

The workflow separates **model development** from **model release**.

Training and evaluation determine which experiment produces a model worth promoting. Promotion then establishes the model as a release artifact, after which deployment consumes the verified production artifact.

---

# 2. Data Preprocessing

Data preprocessing is the first executable stage of the model-development pipeline.

```bash
make preprocess
```

This invokes:

```bash
dvc repro preprocess
```

DVC is the source of truth for the preprocessing stage and its dependencies/outputs.

The preprocessing stage produces the datasets consumed by downstream training stages. The training pipeline therefore does not assume that raw or intermediate data have already been prepared manually.

This provides two important properties:

1. **Reproducibility:** DVC can determine whether preprocessing needs to be rerun.
2. **Dependency tracking:** downstream training stages can depend explicitly on the generated datasets.

The detailed data-generation process is documented separately in:

`docs/data_generation_pipeline.md`

The purpose of this document is therefore to treat preprocessing as the boundary between data generation and model training rather than duplicate the detailed dataset-generation documentation.

---

# 3. DVC Training Variants

## 3.1 Generic Training Interface

Training is exposed through a single parameterized Make target:

```bash
make train TRAIN_VARIANT=sft
```

The supported variants are:

```text
sft
dpo
sft-only
dpo-only
orpo
grpo
```

The Makefile maps the human-readable training variant to the corresponding DVC stage.

For example:

```bash
make train TRAIN_VARIANT=sft
```

invokes:

```bash
dvc repro train-sft
```

Similarly:

```bash
make train TRAIN_VARIANT=sft-only
```

maps to:

```bash
dvc repro train-sft-only
```

The Makefile performs this mapping by converting the hyphenated CLI representation to the underscore-based DVC stage name.

This allows the public command interface to use readable names while preserving the naming conventions of the DVC pipeline.

---

## 3.2 Convenience Aliases

For frequently used variants, the Makefile also provides explicit aliases:

```bash
make train-sft
make train-dpo
make train-sft-only
make train-dpo-only
make train-orpo
make train-grpo
```

These aliases delegate to the generic training interface rather than duplicating the underlying DVC command.

For example:

```makefile
train-sft:
	$(MAKE) train TRAIN_VARIANT=sft
```

This gives the project two useful interfaces:

```bash
make train TRAIN_VARIANT=sft
```

for a generic, parameterized interface, and:

```bash
make train-sft
```

for a concise, self-documenting command.

The actual training implementation remains centralized in one Make target.

---

# 4. Training Configuration and Reproducibility

The Makefile does not contain the training hyperparameters themselves.

Training configuration is maintained by the project's configuration system and consumed by the DVC stages.

A typical training stage therefore has the following conceptual structure:

```text
Makefile
    │
    │ TRAIN_VARIANT=sft
    ▼
DVC stage
    │
    ▼
Hydra configuration
    │
    ├── model configuration
    ├── training configuration
    ├── dataset configuration
    └── logging configuration
    │
    ▼
Training script
    │
    ▼
Trained model artifact
```

This separation is deliberate.

The Makefile defines **what operation should be performed**, while DVC and Hydra determine **how the operation is reproduced and configured**.

This avoids embedding experiment-specific hyperparameters in the command interface.

---

# 5. Model Evaluation

Evaluation is a separate operation from training.

```bash
make evaluate
```

invokes:

```bash
python scripts/evaluate.py
```

Therefore:

```bash
make train TRAIN_VARIANT=sft
```

does **not** inherently mean:

```text
train → evaluate
```

It means:

```text
run the selected DVC training stage
```

Evaluation must be invoked separately when evaluating an individual trained model.

The distinction is important because it allows training and evaluation to be performed independently.

For example:

```bash
make train TRAIN_VARIANT=sft
make evaluate
```

This separation also allows the same evaluation procedure to be applied to multiple trained models without retraining them.

---

## 5.1 Ablation Suite

The `ablation-suite` target is an explicit exception to the normal separation.

It runs the relevant DVC training stages and then evaluates the resulting models:

```text
Training variants
      │
      ▼
Evaluation
      │
      ▼
Ablation report
```

The current implementation runs:

```bash
dvc repro train-sft train-dpo train-dpo-only train-orpo train-grpo
```

followed by:

```bash
python scripts/evaluate.py
```

and then generates the experiment report.

Thus, evaluation is part of the **ablation workflow**, but not part of an individual `make train TRAIN_VARIANT=...` invocation.

---

# 6. Hyperparameter Optimization

Hyperparameter optimization is deliberately separated from ordinary training.

A normal training command selects a training variant:

```bash
make train TRAIN_VARIANT=sft
```

A W&B sweep selects a **fine-tuning method**:

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

This terminology reflects the distinction between the learning method and the resulting model.

---

## 6.1 Sweep Architecture

The sweep workflow is:

```text
make sweep
     │
     │ FINE_TUNE_METHOD=sft
     ▼
sweeps/sft.yaml
     │
     ▼
W&B Sweep
     │
     ▼
W&B Sweep Agent
     │
     ├── trial 1
     ├── trial 2
     ├── ...
     └── trial N
          │
          ▼
     DVC Experiment
          │
          ▼
     Training Stage
          │
          ▼
     Trained Model
```

For example:

```bash
make sweep FINE_TUNE_METHOD=sft COUNT=10
```

selects:

```text
sweeps/sft.yaml
```

and creates the corresponding W&B sweep.

Each W&B trial can then provide a hyperparameter value to the training pipeline. The sweep integration passes those values into the DVC experiment as configuration overrides.

This preserves the separation:

```text
W&B
  │
  │ hyperparameter selection
  ▼
DVC Experiment
  │
  │ reproducible execution
  ▼
Hydra
  │
  │ resolved configuration
  ▼
Training
```

The resulting experiments remain reproducible and independently identifiable.

The detailed sweep implementation, including W&B configuration, sweep agents, DVC experiment integration, and sweep reporting, is documented in:

`docs/hyperparameter_sweeps.md`

---

# 7. Selecting a Model for Promotion

Hyperparameter optimization and ordinary training can produce multiple candidate models.

The release process therefore does not automatically promote every trained model.

Instead, a particular DVC experiment is selected for promotion.

The normal release entry point is:

```bash
make promote-dvc MODEL=<model> EXPERIMENT=<id>
```

For example:

```bash
make promote-dvc MODEL=sft EXPERIMENT=<id>
```

The important distinction is:

```text
TRAIN_VARIANT
    ↓
defines how the model is trained

EXPERIMENT
    ↓
identifies a particular reproducible training run

MODEL
    ↓
identifies the resulting trained model being promoted
```

The promotion operation therefore occurs **after** training and experiment selection.

---

# 8. Release Workflow

The normal release path is:

```text
DVC Experiment
      │
      ▼
promote-dvc
      │
      ▼
publish-model-artifact
      │
      ├── resolve DVC experiment
      ├── validate model
      ├── create W&B model artifact
      └── record provenance
      │
      ▼
W&B Model Registry
      │
      ▼
Production alias
      │
      ▼
publish-hf-release
      │
      ▼
Hugging Face
```

The release workflow intentionally separates:

1. **training**
2. **experiment selection**
3. **artifact publication**
4. **production promotion**
5. **deployment**

This prevents a training run from implicitly becoming a production model.

---

## 8.1 DVC Promotion Path

The normal release process begins with:

```bash
make promote-dvc MODEL=<model> EXPERIMENT=<id>
```

This identifies the specific DVC experiment and model that should enter the release workflow.

The next step is publication of the verified model artifact:

```bash
make publish-model-artifact \
    MODEL=<model> \
    EXPERIMENT=<id> \
    STAGE=<stage>
```

This creates the W&B model artifact associated with the promoted DVC experiment.

The W&B artifact then becomes the immutable model artifact used by the downstream release process.

---

# 9. Alternative Direct W&B Promotion

The project also supports a direct W&B promotion path:

```bash
make promote-model ARTIFACT_REF=<exact-wandb-artifact-ref>
```

This is an alternative to the DVC-based publication path when an existing W&B artifact is already available and has been independently verified.

For example:

```bash
make promote-model \
    ARTIFACT_REF=rl4aa/ask-before-answer/Clarifier-grpo:v17
```

The use of an immutable artifact reference is intentional.

A mutable model name does not uniquely identify a particular set of weights. An immutable artifact version does.

Therefore, the production promotion mechanism operates on:

```text
artifact:vN
```

rather than an ambiguous or mutable model reference.

---

# 10. Deployment

Deployment is deliberately downstream of model promotion.

```bash
make publish-hf-release
```

does not select an arbitrary training output.

Instead, it verifies the W&B promotion provenance and deploys the verified production model.

Conceptually:

```text
Training
   │
   ▼
DVC Experiment
   │
   ▼
Promotion
   │
   ▼
W&B Production Artifact
   │
   ▼
Provenance Verification
   │
   ▼
make publish-hf-release
   │
   ▼
Hugging Face
```

This prevents deployment from becoming an accidental side effect of training.

The detailed deployment behavior is documented separately in:

`docs/deployment_workflow.md`

---

# 11. Complete Operational Workflows

## 11.1 Train a Specific Variant

Generic interface:

```bash
make train TRAIN_VARIANT=sft
```

Convenience alias:

```bash
make train-sft
```

The same pattern applies to the other variants:

```bash
make train TRAIN_VARIANT=dpo
make train TRAIN_VARIANT=sft-only
make train TRAIN_VARIANT=dpo-only
make train TRAIN_VARIANT=orpo
make train TRAIN_VARIANT=grpo
```

---

## 11.2 Evaluate a Trained Model

```bash
make evaluate
```

Training and evaluation are intentionally separate:

```bash
make train TRAIN_VARIANT=sft
make evaluate
```

---

## 11.3 Run Hyperparameter Optimization

For example:

```bash
make sweep FINE_TUNE_METHOD=sft COUNT=10
```

Other supported methods:

```bash
make sweep FINE_TUNE_METHOD=dpo COUNT=10
make sweep FINE_TUNE_METHOD=orpo COUNT=10
make sweep FINE_TUNE_METHOD=grpo COUNT=10
```

---

## 11.4 Promote a DVC Experiment

After identifying the experiment to release:

```bash
make promote-dvc MODEL=<model> EXPERIMENT=<id>
```

The model can then be published as a W&B artifact through the normal release path.

---

## 11.5 Directly Promote an Existing W&B Artifact

Alternatively:

```bash
make promote-model \
    ARTIFACT_REF=<exact-wandb-artifact-ref>
```

This bypasses the DVC experiment publication step when an existing immutable W&B artifact is already suitable for promotion.

---

## 11.6 Deploy the Production Model

After production promotion:

```bash
make publish-hf-release
```

Deployment consumes the verified production artifact rather than an arbitrary local training directory.

---

# 12. Design Rationale

## 12.1 DVC Is the Pipeline Source of Truth

DVC defines the actual computational pipeline and its dependencies.

The Makefile is a user-facing interface over that pipeline.

For example:

```text
make train TRAIN_VARIANT=sft
        │
        ▼
dvc repro train-sft
```

This means that the Makefile does not independently implement the training pipeline. It delegates execution to DVC.

This keeps reproducibility and dependency tracking centralized.

---

## 12.2 The Makefile Provides a Stable Human Interface

The Makefile provides short, discoverable commands without requiring users to know the underlying DVC, Hydra, or W&B implementation details.

For example:

```bash
make preprocess
make train-sft
make evaluate
make sweep FINE_TUNE_METHOD=sft COUNT=10
make promote-dvc MODEL=<model> EXPERIMENT=<id>
make publish-hf-release
```

The underlying implementation can evolve while these high-level commands remain stable.

---

## 12.3 Training Is Parameterized Rather Than Duplicated

The training interface uses:

```bash
make train TRAIN_VARIANT=<variant>
```

rather than implementing independent DVC commands for every variant.

The convenience aliases delegate to the generic target.

This avoids having multiple independent implementations of the same orchestration logic.

---

## 12.4 Fine-Tuning Method and Training Variant Are Different Concepts

The project deliberately uses different names for these concepts.

### Fine-tuning method

```text
FINE_TUNE_METHOD
```

Examples:

```text
sft
dpo
orpo
grpo
```

This terminology is used by the W&B sweep interface.

### Training variant

```text
TRAIN_VARIANT
```

Examples:

```text
sft
dpo
sft-only
dpo-only
orpo
grpo
```

This terminology is used by the DVC training interface.

A training variant may therefore encode an experimental configuration or baseline distinction beyond the underlying fine-tuning method.

---

## 12.5 Models Are Distinct from Training Methods

The project also avoids using `MODEL` as a synonym for a training method.

Conceptually:

```text
Base model
    +
Training data
    +
Fine-tuning method
    +
Hyperparameters
    ↓
Training run / DVC experiment
    ↓
Trained model
    ↓
Model artifact
```

For example, SFT is not itself the model. SFT is the learning method used to produce a model.

This distinction becomes particularly important once multiple experiments, model variants, and immutable W&B artifacts exist.

---

## 12.6 Training Does Not Imply Evaluation

The normal interface keeps:

```bash
make train ...
```

and:

```bash
make evaluate
```

separate.

This is intentional because training and evaluation are different computational operations with different purposes.

A training run can be evaluated later, potentially using a different evaluation configuration or evaluation set.

The exception is the ablation workflow, which explicitly combines training, evaluation, and report generation.

---

## 12.7 Promotion Is Separate from Training

A successful training run does not automatically become a production model.

Instead:

```text
Training
   ↓
Evaluation / experiment analysis
   ↓
Experiment selection
   ↓
Promotion
   ↓
Artifact publication
   ↓
Production
   ↓
Deployment
```

This separation establishes a clear boundary between **research/experimentation** and **production release**.

It also makes it possible to retain many experiments without promoting all of them.

---

# 13. Relationship to Other Documentation

This document provides the lifecycle-level view. The following documents provide deeper detail for individual parts of the system:

| Document                           | Responsibility                                                              |
| ---------------------------------- | --------------------------------------------------------------------------- |
| `docs/data_generation_pipeline.md` | Data generation and preparation                                             |
| `docs/model_training_pipeline.md`  | End-to-end model training, evaluation, sweeps, and release lifecycle        |
| `docs/hyperparameter_sweeps.md`    | W&B sweep configuration and execution                                       |
| `docs/release_architecture.md`     | Model promotion, artifact publication, provenance, and release architecture |
| `docs/deployment_workflow.md`      | Production deployment workflow                                              |

The goal is to avoid duplicating implementation details across documents while still giving a new contributor a single document from which the complete model-development lifecycle can be understood.

---

# 14. Quick Reference

### Data preparation

```bash
make preprocess
```

### Train a DVC variant

```bash
make train TRAIN_VARIANT=sft
```

or:

```bash
make train-sft
```

### Evaluate

```bash
make evaluate
```

### Hyperparameter optimization

```bash
make sweep FINE_TUNE_METHOD=sft COUNT=10
```

### Promote a DVC experiment

```bash
make promote-dvc MODEL=<model> EXPERIMENT=<id>
```

### Alternative direct W&B promotion

```bash
make promote-model \
    ARTIFACT_REF=<exact-wandb-artifact-ref>
```

### Deploy the verified production model

```bash
make publish-hf-release
```

---

# 15. Conceptual Summary

The complete architecture can be summarized as:

```text
                     DATA
                      │
                      ▼
               make preprocess
                      │
                      ▼
                  DVC data
                      │
                      ▼
          ┌────────────────────────┐
          │     TRAINING           │
          │                        │
          │ TRAIN_VARIANT          │
          │                        │
          │ sft                    │
          │ dpo                    │
          │ sft-only               │
          │ dpo-only               │
          │ orpo                   │
          │ grpo                   │
          └───────────┬────────────┘
                      │
                      ▼
                DVC Experiment
                      │
             ┌────────┴────────┐
             │                 │
             ▼                 ▼
         Evaluation       W&B Sweeps
         make evaluate    FINE_TUNE_METHOD
                              │
                              ▼
                       Best Experiment
                              │
             ┌────────────────┴────────────────┐
             │                                 │
             ▼                                 ▼
       DVC release path                 Direct W&B path
       promote-dvc                      promote-model
             │                                 │
             ▼                                 │
       publish-model-artifact                │
             │                                 │
             └────────────────┬────────────────┘
                              ▼
                       W&B Production
                              │
                              ▼
                         publish-hf-release
                              │
                              ▼
                       Production Model
```

The central design principle is that **training, evaluation, experiment selection, promotion, and deployment are distinct lifecycle stages**. DVC provides reproducibility for the computational pipeline, W&B provides experiment and artifact management, and the Makefile provides a concise interface that exposes those capabilities without conflating their responsibilities.
