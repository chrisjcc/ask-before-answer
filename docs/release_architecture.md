# Model Release, W&B Registry Promotion, Provenance, and Hugging Face Deployment

## Overview

The production release pipeline uses **DVC, Weights & Biases (W&B), and Hugging Face** as distinct stages in a controlled promotion and deployment chain.

The central principle is:

> **Hugging Face deployment is permitted only from a W&B production artifact whose exact identity and immutable digest have been recorded in release provenance and independently verified at deployment time.**

The normal release workflow consists of three commands:

```text
DVC experiment
      │
      ▼
promote-dvc
      │
      ▼
Promoted DVC model
      │
      ▼
publish-model-artifact
      │
      ├── resolve DVC experiment
      ├── validate model
      ├── create/verify immutable W&B artifact
      ├── verify Registry digest
      ├── verify production alias
      └── record provenance
      │
      ▼
deploy-hf
      │
      ├── validate provenance
      ├── verify immutable artifact digest
      ├── verify production digest
      └── deploy exact artifact
      │
      ▼
Hugging Face
```

This architecture separates **model selection**, **W&B production promotion**, and **deployment**, while using the artifact digest as the integrity anchor across the release boundary.

---

# Release Architecture

The release process has three stages.

## 1. Promote the DVC experiment

```bash
make promote-dvc MODEL=<model> EXPERIMENT=<id>
```

This establishes the relationship:

```text
DVC experiment
      │
      ▼
promoted model
```

The DVC experiment remains the source of truth for the model being selected for release.

`promote-dvc` does not deploy the model and does not by itself make the model available on Hugging Face.

---

## 2. Publish and promote the model through W&B

```bash
make publish-model-artifact \
    MODEL=<model> \
    EXPERIMENT=<id> \
    STAGE=<stage>
```

Despite its name, `publish-model-artifact` is more than a simple artifact upload. It is the primary W&B release gate in the normal workflow.

It performs the operations necessary to establish and verify the W&B production release:

```text
promoted DVC model
       │
       ▼
publish-model-artifact
       │
       ├── resolve DVC experiment
       ├── validate model
       ├── create/verify immutable W&B artifact
       ├── verify Registry digest
       ├── verify production alias
       └── record provenance
```

The result is an immutable W&B artifact associated with the production Registry state and a persistent release record:

```text
provenance/model_promotion.json
```

An artifact reference has the form:

```text
<entity>/<project>/<artifact>:vN
```

For example:

```text
rl4aa/ask-before-answer/Clarifier-grpo:v17
```

The `:vN` version is important because release promotion operates on an exact artifact version rather than an unversioned or mutable artifact name.

---

## 3. Deploy the verified production artifact to Hugging Face

```bash
make deploy-hf
```

The deployment process does not simply deploy whatever artifact currently has the W&B `production` alias.

Instead, it reads the promotion provenance and independently verifies the W&B state before allowing deployment.

Conceptually:

```text
provenance/model_promotion.json
              │
              ▼
       deployment gate
              │
       ┌──────┴──────┐
       ▼             ▼
 verify artifact   verify alias
    digest           digest
       │             │
       └──────┬──────┘
              ▼
       Deploy to HF
```

The deployment gate therefore connects the approved W&B release to the actual artifact being deployed.

---

# The W&B Publication and Promotion Gate

The normal W&B release operation is intentionally consolidated into `publish-model-artifact`.

The command performs the following logical sequence:

```text
DVC experiment
      │
      ▼
resolve experiment
      │
      ▼
validate promoted model
      │
      ▼
create/verify W&B source artifact
      │
      ▼
immutable W&B Registry artifact :vN
      │
      ├── obtain Registry digest
      │
      ▼
verify Registry digest
      │
      ▼
verify production alias
      │
      ▼
write provenance/model_promotion.json
```

This means that publication and W&B production promotion are not two independent user-facing commands in the normal workflow.

Instead:

> **`publish-model-artifact` is the normal W&B publication and promotion gate.**

This is an important architectural distinction.

---

# Provenance

The file:

```text
provenance/model_promotion.json
```

is the persistent release record connecting W&B production promotion to Hugging Face deployment.

The provenance records information corresponding to:

```text
artifact_ref
artifact_digest
wandb.registry_alias
promoted_at
```

Conceptually:

```text
provenance/model_promotion.json
              │
              ├── artifact_ref
              │       │
              │       ▼
              │   immutable
              │   W&B artifact :vN
              │
              ├── artifact_digest
              │
              ├── wandb.registry_alias
              │
              └── promoted_at
```

The provenance record answers:

* Which exact W&B artifact was promoted?
* What was its immutable content digest?
* Which W&B registry alias represents production?
* When was the artifact promoted?

This makes the release auditable after the promotion operation has completed.

---

# Artifact Identity and Digest

An artifact reference such as:

```text
rl4aa/ask-before-answer/Clarifier-grpo:v17
```

identifies a specific W&B artifact version.

The provenance additionally records the artifact's immutable digest:

```text
artifact_ref    = .../Clarifier-grpo:v17
artifact_digest = 8dd12fd8...
```

The two provide complementary forms of identity:

```text
artifact_ref
      │
      ▼
exact W&B artifact/version
      │
      ▼
artifact_digest
      │
      ▼
exact artifact contents
```

The deployment gate verifies that the artifact currently resolved from W&B still has the digest recorded during promotion.

The required condition is:

```text
current artifact digest
        ==
provenance artifact_digest
```

If the digests differ:

```text
FAIL
```

and deployment must not proceed.

---

# Production Alias Verification

The W&B `production` alias is treated as a **mutable pointer**, not as the immutable release identity.

The deployment process therefore does not trust the alias by name alone.

Instead, it resolves the production alias and verifies that its target corresponds to the artifact recorded in the provenance.

Conceptually:

```text
provenance
    │
    ├── expected artifact_ref
    │
    ├── expected artifact_digest
    │
    └── expected registry alias
                 │
                 ▼
          resolve production
                 │
                 ▼
          obtain production
             artifact
                 │
                 ▼
          obtain its digest
                 │
                 ▼
       compare with provenance
          artifact_digest
```

The required invariant is:

```text
production alias digest
        ==
provenance artifact_digest
```

If the production alias points to a different artifact or a different digest, deployment fails.

This prevents a later alias change from silently changing what gets deployed to Hugging Face.

---

# Publication Gate

The publication gate can be summarized as:

```text
publish-model-artifact
          │
          ▼
   DVC experiment
          │
          ▼
   validate model
          │
          ▼
   W&B source artifact
          │
          ▼
immutable Registry artifact :vN
          │
          ├── digest
          │
          ▼
   verify Registry digest
          │
          ▼
   verify production alias
          │
          ▼
       provenance
```

The important property is that the production state is established **before** Hugging Face deployment is attempted.

---

# Deployment Gate

`make deploy-hf` is the final release gate.

The deployment process uses `push_to_hub.py` and performs the following logical sequence:

```text
make deploy-hf
       │
       ▼
push_to_hub.py
       │
       ├── read provenance
       │
       ├── validate provenance
       │
       ├── verify registry_alias == production
       │
       ├── resolve artifact_ref
       │
       ├── obtain current W&B artifact digest
       │
       ├── compare current digest
       │      with provenance artifact_digest
       │
       ├── resolve production alias
       │
       ├── obtain production artifact digest
       │
       ├── compare production digest
       │      with provenance artifact_digest
       │
       ▼
   deploy exact verified artifact
```

The Hugging Face deployment is therefore conditional on the W&B state satisfying the recorded provenance.

---

# Full Release Gate Chain

The complete release integrity chain is:

```text
DVC experiment
      │
      ▼
promote-dvc
      │
      ▼
Promoted DVC model
      │
      ▼
publish-model-artifact
      │
      ├── resolve DVC experiment
      ├── validate model
      ├── create/verify immutable W&B artifact
      ├── verify Registry digest
      ├── verify production alias
      └── write provenance
      │
      ▼
provenance/model_promotion.json
      │
      ▼
deploy-hf validation
      │
      ├── provenance valid
      ├── artifact_ref resolves
      ├── artifact digest matches
      ├── production alias resolves
      └── production digest matches
      │
      ▼
Hugging Face deployment
```

The central integrity invariant is:

```text
provenance artifact_digest
        ==
immutable artifact digest
        ==
production alias target digest
```

Only when this invariant holds should deployment proceed.

---

# Normal Release Workflow

The normal release workflow consists of three commands:

```text
1. make promote-dvc MODEL=<model> EXPERIMENT=<id>
                         │
                         ▼
                Promoted DVC model
                         │
                         ▼
2. make publish-model-artifact \
       MODEL=<model> \
       EXPERIMENT=<id> \
       STAGE=<stage>
                         │
                         ├── resolve experiment
                         ├── validate model
                         ├── create/verify artifact
                         ├── verify Registry digest
                         ├── verify production alias
                         └── write provenance
                         │
                         ▼
3. make deploy-hf
                         │
                         ├── validate provenance
                         ├── verify artifact digest
                         ├── verify production digest
                         └── deploy verified artifact
                         │
                         ▼
                    Hugging Face
```

In compact form:

```text
DVC
 │
 ▼
promote-dvc
 │
 ▼
publish-model-artifact
 │
 ▼
immutable W&B artifact
 │
 ▼
production + provenance
 │
 ▼
deploy-hf
 │
 ▼
Hugging Face
```

---

# Direct / Alternative W&B Promotion Path

`promote-model` remains available as a lower-level W&B promotion command for cases where an immutable W&B artifact already exists and should be promoted directly.

```bash
make promote-model \
    ARTIFACT_REF=<exact-wandb-artifact-ref>
```

This provides an alternative to the W&B publication/promotion portion of the normal workflow.

Conceptually:

```text
Existing immutable W&B artifact :vN
              │
              ▼
        promote-model
              │
              ├── verify/promote artifact
              └── write provenance
              │
              ▼
          deploy-hf
```

The direct path does not bypass the deployment gate.

`deploy-hf` still reads the resulting provenance and performs the same artifact and production digest verification before deployment.

Therefore:

```text
Normal path:

DVC
 │
 ▼
promote-dvc
 │
 ▼
publish-model-artifact
 │
 ▼
deploy-hf
```

while:

```text
Alternative W&B path:

existing immutable W&B artifact
 │
 ▼
promote-model
 │
 ▼
deploy-hf
```

---

# Why the Artifact Must Be Immutable

Release promotion requires an exact W&B artifact reference:

```text
<entity>/<project>/<artifact>:vN
```

For example:

```text
rl4aa/ask-before-answer/Clarifier-grpo:v17
```

The version identifies the specific artifact that was approved.

The digest then provides a stronger content-level identity:

```text
artifact_ref
      │
      ▼
Clarifier-grpo:v17
      │
      ▼
artifact_digest
      │
      ▼
exact artifact contents
```

This gives the release process two complementary forms of identity:

* **Artifact reference** identifies the W&B artifact/version.
* **Artifact digest** identifies the artifact contents.

The deployment gate verifies both.

---

# Separation of Responsibilities

Each component has a clearly defined responsibility.

| Component                | Responsibility                                                                                                                                                                 |
| ------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| DVC                      | Tracks experiments, stages, parameters, and model lineage                                                                                                                      |
| `promote-dvc`            | Selects and promotes the DVC experiment/model                                                                                                                                  |
| `publish-model-artifact` | Resolves and validates the promoted model, creates/verifies the immutable W&B artifact, verifies Registry state, verifies the production alias, and records release provenance |
| `promote-model`          | Provides a direct/alternative W&B artifact promotion path for an already-existing immutable artifact                                                                           |
| `model_promotion.json`   | Persistent release record containing artifact identity, digest, registry information, and promotion time                                                                       |
| `deploy-hf`              | Executes the final deployment gate and deploys only after W&B provenance verification                                                                                          |
| Hugging Face             | Final deployed model artifact                                                                                                                                                  |

This separation gives each system a clear role.

DVC answers:

> **Which experiment/model was selected?**

W&B answers:

> **Which immutable artifact was published and promoted to production?**

The provenance record answers:

> **Which exact artifact and digest were approved for deployment?**

The deployment gate answers:

> **Does the current W&B production state still correspond to the approved artifact?**

Hugging Face receives the model only after those checks pass.

---

# Failure Conditions

The deployment gate should fail rather than deploy if any critical verification fails.

For example:

```text
provenance/model_promotion.json does not exist
        │
        └── FAIL
```

```text
artifact_ref cannot be resolved
        │
        └── FAIL
```

```text
current artifact digest
    !=
provenance artifact_digest
        │
        └── FAIL
```

```text
production alias cannot be resolved
        │
        └── FAIL
```

```text
production alias digest
    !=
provenance artifact_digest
        │
        └── FAIL
```

The underlying principle is **fail closed**:

```text
verification passes
       │
       ▼
   deployment

verification fails
       │
       ▼
   no deployment
```

---

# Makefile Interface

The Makefile exposes the release workflow directly through:

```bash
make help
```

The relevant section is:

```text
Model Promotion & Publication:

Release workflow:

  1. make promote-dvc MODEL=<sft|dpo|grpo|orpo> EXPERIMENT=<id>
     Promote a DVC experiment to the promoted model

  2. make publish-model-artifact MODEL=<model> EXPERIMENT=<id> STAGE=<stage>
     Publish the promoted DVC model as an immutable W&B artifact

  3. make deploy-hf
     Verify production provenance and deploy to Hugging Face

Direct/alternative W&B promotion:

  make promote-model ARTIFACT_REF=<exact-wandb-artifact-ref>
     Directly promote an existing W&B artifact
     (alternative to publish-model-artifact)
```

W&B promotion requires an immutable artifact version:

```text
W&B promotion requires an immutable :vN artifact reference.
```

For example:

```text
rl4aa/ask-before-answer/Clarifier-grpo:v17
```

---

# Release Model

The resulting architecture can be summarized as:

```text
                    MODEL DEVELOPMENT
                           │
                           ▼
                    DVC experiment
                           │
                           │ promote-dvc
                           ▼
                    Promoted model
                           │
                           │ publish-model-artifact
                           ▼
                 W&B publication/promotion
                           │
                    ┌──────┴──────┐
                    │             │
                    ▼             ▼
              immutable :vN   production
                    │             │
                    └──────┬──────┘
                           │
                           ▼
              model_promotion.json
                           │
                           │ deploy-hf
                           ▼
                    Deployment gate
                           │
                 ┌─────────┴─────────┐
                 │                   │
                 ▼                   ▼
          verify artifact       verify production
              digest                digest
                 │                   │
                 └─────────┬─────────┘
                           │
                      PASS / FAIL
                           │
                    PASS ──┴── FAIL
                     │          │
                     ▼          ▼
               Hugging Face    STOP
                deployment
```

## Core Invariant

The release system ultimately enforces one central invariant:

```text
Approved artifact
      ==
Current artifact
      ==
Production alias target
      ==
Recorded provenance digest
```

Hugging Face deployment is allowed only when this invariant is satisfied.

The result is a release process in which:

1. **DVC identifies the model being released.**
2. **W&B establishes and verifies the immutable production artifact.**
3. **Provenance records the exact artifact and digest approved for deployment.**
4. **`deploy-hf` independently verifies that W&B still resolves to that exact artifact.**
5. **Hugging Face is updated only after all release integrity checks pass.**

This makes the immutable W&B artifact and its digest, rather than the mutable `production` alias alone, the basis of deployment trust.
