# Model Release & W&B Registry Promotion

The production release pipeline uses **DVC, Weights & Biases (W&B), and Hugging Face** as distinct stages in a controlled promotion and deployment chain.

The central principle is:
> **Hugging Face deployment is permitted only from a W&B production artifact whose exact identity and immutable digest have been recorded in release provenance and independently verified at deployment time.**

## 1. Promote the DVC experiment
First, select the winning DVC experiment to promote:

```bash
make promote-dvc MODEL=<model> EXPERIMENT=<id>
```
The DVC experiment remains the source of truth for the model being selected for release. *Note: `promote-dvc` does not deploy the model and does not by itself make the model available on Hugging Face.*

## 2. Publish and promote through W&B

```bash
make publish-model-artifact MODEL=<model> EXPERIMENT=<id> STAGE=production
```

This command acts as the primary W&B release gate:
1. Resolves the DVC experiment.
2. Validates the model.
3. Creates/verifies an immutable W&B artifact.
4. Verifies the Registry digest and production alias.
5. Records the deployment provenance locally.

## 3. Hugging Face Deployment

```bash
make deploy-hf
```

This final step reads the verified provenance, confirms the digest, and pushes the exact model artifact to the Hugging Face Hub, updating the Model Card and Data Card with the release notes and evaluation metrics.
