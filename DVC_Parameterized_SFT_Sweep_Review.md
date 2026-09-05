# DVC Parameterized SFT Sweep: Summary Review

## 1. Objective

The goal is to establish a reproducible hyperparameter sweep workflow in which **Weights & Biases (W&B)** manages the sweep and trial-level orchestration, while **DVC** remains responsible for experiment reproducibility, parameter tracking, dependency management, and experiment versioning.

The specific feature we are trying to achieve is a **parameterized SFT sweep**, where individual sweep trials dynamically override training parameters in the DVC pipeline without requiring manual modification of `params.yaml`, `dvc.yaml`, or the training configuration between trials.

The intended end state is:

1. Define the sweep space in W&B.
2. Let W&B select a hyperparameter configuration for each trial.
3. Pass those parameters into the trial execution layer.
4. Translate the selected parameters into DVC parameter overrides.
5. Execute the SFT pipeline through DVC.
6. Track the resulting model, metrics, parameters, and experiment state in both W&B and DVC.
7. Preserve enough metadata to associate a W&B run with its corresponding DVC experiment.

---

## 2. What We Have Accomplished

### 2.1 Established the DVC-based training pipeline

The SFT training workflow has been integrated into a DVC pipeline rather than being treated as an isolated training script.

The current pipeline structure allows stages such as preprocessing and SFT training to be reproduced through commands such as:

```bash
dvc repro train-sft
```

DVC correctly identifies whether upstream dependencies, parameters, or outputs have changed and determines which stages need to be rerun.

This establishes the reproducibility foundation required for parameterized experiments.

### 2.2 Established parameterized training configuration

The training configuration has been structured so that important SFT hyperparameters can be represented as DVC parameters rather than hard-coded values.

This is important because the sweep system needs to be able to override individual parameters for each trial.

The intended mechanism is conceptually:

```bash
dvc exp run \
  -S train.learning_rate=... \
  -S train.batch_size=... \
  -S train.num_train_epochs=...
```

rather than modifying configuration files directly for every experiment.

### 2.3 Established W&B sweep configuration

A W&B sweep has been configured for SFT training.

The sweep currently uses a Bayesian search strategy and optimizes the evaluation loss:

```yaml
method: bayes

metric:
  name: eval/loss
  goal: minimize
```

The sweep also includes Hyperband-based early termination, allowing poorly performing trials to be stopped before consuming the full training budget.

For example:

```yaml
early_terminate:
  type: hyperband
  min_iter: 50
  eta: 3
```

The learning-rate search space is also parameterized rather than fixed, using a logarithmic distribution.

This establishes W&B as the component responsible for selecting candidate hyperparameter configurations.

### 2.4 Established a dedicated sweep-trial execution layer

A dedicated script, `scripts/run_sweep_trial.py`, has been introduced to act as the bridge between W&B and DVC.

This is an important architectural decision.

Rather than having W&B directly execute the training script, the sweep agent invokes the trial wrapper, which can:

1. Initialize the W&B run.
2. Read the selected sweep configuration.
3. Translate the W&B configuration into DVC parameter overrides.
4. Execute the appropriate DVC experiment.
5. Forward relevant metadata and metrics.
6. Associate the W&B run with the corresponding DVC experiment.

This creates a clean separation between **sweep orchestration** and **experiment execution**.

### 2.5 Confirmed that DVC supports parameter overrides

DVC's `exp run` mechanism supports parameter overrides through the `-S` option.

For example:

```bash
dvc exp run -S train.learning_rate=0.0001
```

This provides the fundamental mechanism needed for parameterized sweep trials.

The key advantage is that the parameter override can be supplied at runtime without permanently modifying the repository's tracked configuration.

### 2.6 Confirmed DVC experiment isolation

DVC experiments provide a natural mechanism for representing individual parameter configurations as separate experiments.

This is particularly useful for sweeps because each W&B trial can correspond conceptually to one DVC experiment:

```text
W&B Trial 1  <->  DVC Experiment 1
W&B Trial 2  <->  DVC Experiment 2
W&B Trial 3  <->  DVC Experiment 3
...
```

This gives the sweep a reproducible experiment layer underneath the W&B orchestration layer.

---

## 3. What We Are Trying to Accomplish

The overall architecture is intended to look approximately like this:

```text
                         ┌──────────────────────┐
                         │      W&B Sweep       │
                         │                      │
                         │ Search algorithm     │
                         │ Bayesian / Hyperband │
                         └──────────┬───────────┘
                                    │
                                    │ sampled parameters
                                    ▼
                         ┌──────────────────────┐
                         │   W&B Sweep Agent    │
                         │                      │
                         │ launches each trial  │
                         └──────────┬───────────┘
                                    │
                                    ▼
                    ┌──────────────────────────────┐
                    │ scripts/run_sweep_trial.py   │
                    │                              │
                    │ • initialize W&B run         │
                    │ • read wandb.config          │
                    │ • map parameters             │
                    │ • construct DVC command     │
                    │ • forward metadata           │
                    └──────────────┬───────────────┘
                                   │
                                   │ dvc exp run -S ...
                                   ▼
                    ┌──────────────────────────────┐
                    │          DVC                 │
                    │                              │
                    │ Parameter overrides          │
                    │ Experiment management        │
                    │ Pipeline reproduction        │
                    │ Cache / artifact tracking    │
                    └──────────────┬───────────────┘
                                   │
                                   ▼
                    ┌──────────────────────────────┐
                    │       SFT Pipeline           │
                    │                              │
                    │ preprocess → train-sft       │
                    └──────────────┬───────────────┘
                                   │
                                   ▼
                    ┌──────────────────────────────┐
                    │       Training Outputs       │
                    │                              │
                    │ model / metrics / artifacts  │
                    └──────────────┬───────────────┘
                                   │
                    ┌──────────────┴───────────────┐
                    │                              │
                    ▼                              ▼
             ┌──────────────┐              ┌──────────────┐
             │     W&B      │              │     DVC      │
             │              │              │              │
             │ metrics      │              │ experiment   │
             │ parameters   │              │ params       │
             │ run ID       │◄────────────►│ artifacts    │
             │ sweep ID     │  metadata    │ provenance   │
             └──────────────┘              └──────────────┘
```

The central architectural principle is that **W&B chooses the experiment configuration, while DVC executes and records the reproducible experiment**.

---

## 4. What We Have Confirmed

The following aspects of the architecture have been established with reasonable confidence:

### Confirmed

* DVC can execute the SFT pipeline through `dvc repro` / `dvc exp run`.
* DVC supports runtime parameter overrides through `dvc exp run -S`.
* DVC correctly detects changes to parameters, dependencies, and outputs.
* The SFT training pipeline is represented as a DVC stage.
* W&B can define and manage the hyperparameter search space.
* W&B sweep agents can invoke `scripts/run_sweep_trial.py`.
* The trial wrapper can access the W&B-selected configuration.
* Hyperparameters can therefore be programmatically translated into DVC parameter overrides.
* DVC experiments provide an appropriate abstraction for representing individual sweep trials.
* W&B and DVC can therefore be connected without requiring W&B to replace DVC's experiment-management functionality.

These pieces establish that the underlying architecture is technically viable.

---

## 5. What We Are Still Verifying

The remaining work is primarily around **integration semantics, metadata propagation, and reproducibility guarantees**, rather than whether DVC can perform parameter overrides at all.

### 5.1 Parameter propagation

We need to verify that the complete path works reliably:

```text
W&B sweep configuration
        ↓
wandb.config
        ↓
run_sweep_trial.py
        ↓
DVC -S parameter overrides
        ↓
dvc.yaml / params.yaml
        ↓
SFT training script
```

In particular, we need to confirm that every parameter exposed by the sweep is mapped to the correct DVC parameter path and ultimately reaches the training code with the expected value and type.

### 5.2 DVC experiment identity

We need to establish the preferred naming convention for DVC experiments generated by sweep trials.

For example:

```text
sweep_trial_001
sweep_trial_002
...
```

or a W&B-derived identifier:

```text
wandb_<run_id>
```

The important requirement is that the identity is deterministic and allows a reviewer to move from a W&B run to the corresponding DVC experiment.

### 5.3 Bi-directional metadata linking

One of the more important remaining objectives is establishing a reliable relationship between:

```text
W&B Run ID
        ↕
DVC Experiment
        ↕
DVC parameters
        ↕
Training outputs
```

The proposed approach is to include the W&B run ID and relevant sweep metadata in the DVC experiment metadata or message.

Conversely, the W&B run should record the DVC experiment identifier.

This would allow either system to be used as the starting point for tracing an experiment.

### 5.4 Metric propagation

Another area requiring verification is the relationship between the metrics produced by training, DVC, and W&B.

The desired behavior is:

```text
SFT training
      ↓
evaluation metrics
      ├──→ W&B
      └──→ DVC
```

The primary sweep metric, such as:

```text
eval/loss
```

must reliably reach W&B so that the sweep controller can compare trials.

At the same time, the relevant metrics should remain associated with the corresponding DVC experiment for reproducibility.

### 5.5 DVC cache and experiment execution behavior

We also need to verify how parameterized sweep trials interact with DVC's cache.

A successful implementation should avoid unnecessarily recomputing unchanged upstream stages while still executing stages whose parameter dependencies have changed.

For example, if preprocessing does not depend on the SFT learning rate, changing:

```text
train.learning_rate
```

should not cause preprocessing to be recomputed unnecessarily.

This is one of the main advantages of keeping DVC underneath the sweep framework.

### 5.6 Failure and interruption handling

The behavior of the system still needs to be verified when:

* a training trial fails;
* a W&B run is interrupted;
* Hyperband terminates a trial early;
* DVC returns a non-zero exit code;
* the GPU process fails;
* a parameter combination produces an invalid configuration.

The desired behavior is for W&B and DVC to remain consistent enough that failed or interrupted trials can still be diagnosed.

---

## 6. Target Workflow

The target workflow for one sweep trial is:

```text
1. W&B selects hyperparameters
             │
             ▼
2. Sweep agent starts run_sweep_trial.py
             │
             ▼
3. W&B initializes run
             │
             ▼
4. run_sweep_trial.py reads wandb.config
             │
             ▼
5. Parameters are mapped to DVC parameter paths
             │
             ▼
6. dvc exp run -S ... is constructed
             │
             ▼
7. DVC creates/runs the parameterized experiment
             │
             ▼
8. DVC executes only affected pipeline stages
             │
             ▼
9. SFT training runs
             │
             ├───────────────┐
             ▼               ▼
        metrics          artifacts
             │               │
             └───────┬───────┘
                     ▼
              DVC experiment
                     │
                     ▼
             metadata linkage
                     │
                     ▼
                 W&B run
                     │
                     ▼
            sweep controller
```

---

## 7. Desired End State

The final system should allow a command such as:

```bash
wandb agent <ENTITY>/<PROJECT>/<SWEEP_ID>
```

to launch multiple trials without manually modifying the DVC configuration between runs.

Each trial should independently define its parameter configuration while remaining reproducible through DVC.

Conceptually:

```text
                     W&B Sweep
                         │
          ┌──────────────┼──────────────┐
          ▼              ▼              ▼
       Trial A        Trial B        Trial C
          │              │              │
          ▼              ▼              ▼
       DVC Exp A      DVC Exp B      DVC Exp C
          │              │              │
          ▼              ▼              ▼
       SFT Model A    SFT Model B    SFT Model C
          │              │              │
          └──────────────┼──────────────┘
                         ▼
                  W&B comparison
                         │
                         ▼
                 Best configuration
```

The important distinction is that **W&B is responsible for optimization and trial selection, while DVC is responsible for reproducible experiment execution and provenance**.

---

## 8. Current Assessment

The core feasibility of the DVC parameterized SFT sweep approach has effectively been demonstrated. The individual components already exist and the interfaces between them are understood.

The remaining work is primarily integration hardening:

1. Verify complete parameter propagation.
2. Standardize DVC experiment naming.
3. Implement reliable W&B ↔ DVC metadata linking.
4. Verify metric propagation and sweep optimization.
5. Verify DVC cache behavior across trials.
6. Test failure, interruption, and early-termination cases.
7. Confirm that a completed sweep can be reproduced from the recorded DVC experiment state.

Once these points are verified, the resulting system should provide a clean experimental architecture in which **W&B handles hyperparameter search and optimization, while DVC provides reproducible execution, parameter versioning, pipeline provenance, and artifact management**.
