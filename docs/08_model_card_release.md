# Model Release & W&B Registry Promotion

The production release pipeline uses **DVC, Weights & Biases (W&B), and Hugging Face** as distinct stages in a controlled promotion and deployment chain.

The central principle is:
> **Hugging Face deployment is permitted only from a W&B production artifact whose exact identity and immutable digest have been recorded in release provenance and independently verified at deployment time.**

## Step 1. Promote the DVC experiment
First, lock the optimal hyperparameters from the sweep into your local `params.yaml`:

```bash
make promote-dvc MODEL=<model> EXPERIMENT=<id>
```
The DVC experiment remains the source of truth for the model being selected for release. *Note: `promote-dvc` does not deploy the model and does not by itself make the model available on Hugging Face.*

## Step 2. Publish and promote through W&B

```bash
make publish-model-artifact MODEL=<model> EXPERIMENT=<id> STAGE=production
```

This command acts as the primary W&B release gate. It uploads the model weights to the cloud, tags it as the official `production` artifact in the W&B Model Registry, and creates a local `provenance/model_promotion.json` file to cryptographically prove the deployment pipeline.

## Step 3. Git Commit & Tag

Hugging Face deployments are tied to Git release tags. Commit the updated `params.yaml`, `dvc.lock`, and `provenance/` files, then tag the release (for example, `v1.0.0`):

```bash
git add params.yaml dvc.lock provenance/
git commit -m "chore: promote <model> <id> to production"
git push origin main
git tag v1.0.0
git push origin v1.0.0
```

## Step 4. Hugging Face Deployment

```bash
make publish-hf-release
```

This final command reads the local provenance file you just committed, verifies the W&B registry signatures, and automatically pushes the exact model weights to the Hugging Face Hub while dynamically generating and updating the Model Card and Data Card.
