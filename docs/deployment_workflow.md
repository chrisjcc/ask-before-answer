# End-to-End Deployment & Version Control Workflow

This document outlines the architecture and step-by-step instructions for managing datasets, model weights, and deployments in the AskBeforeAnswer project. It explains how we seamlessly bridge local experimentation (DVC) with production deployment using the **Weights & Biases Model Registry** and the **Hugging Face Hub**.

## Architectural Overview: The "Messy Kitchen" vs. The "Dining Room"

When training machine learning models, you do not want to pollute your public repository or public Hugging Face Hub with broken or intermediate files. We solve this using a two-tiered architecture:

### 1. The "Messy Kitchen" (DVC + Internal Cloud Storage)
During development, we use **Data Version Control (DVC)**. DVC acts like Git for large binary files. It silently tracks every dataset generation and model training run you execute locally, mapping them to your Git commits. 

### 2. The "Dining Room" (W&B Model Registry + Hugging Face Hub)
Once an experiment is successful and finalized, we want to serve it to the public. 
Rather than hardcoding deployment logic, we use the **Weights & Biases (W&B) Model Registry** as our single source of truth for deployment decisions. By tagging an experimental model (e.g., `sft_only` or `dpo`) with the `production` alias in the W&B web UI, our local deployment script dynamically resolves the winner, injects the Weave evaluation leaderboard into the Model Card, and pushes the model straight to Hugging Face!

---

## Step-by-Step Workflow

Follow these instructions to execute the end-to-end pipeline from local development to public Hugging Face deployment.

### 1. The Development Phase (Locally)
When you want to process data or train the models, let DVC orchestrate the pipeline so it can automatically track the outputs.

```bash
# This automatically orchestrates the entire ablation suite (SFT, DPO, evaluations, etc.)
make ablation-suite
```

### 2. Evaluate the Models
Once the suite finishes, DVC triggers the evaluation script which generates a Weave leaderboard and local markdown summaries.

### 3. Promote a Model to Production (W&B Web UI)
1. Go to your **W&B Web Dashboard** and open your project.
2. Navigate to the **Model Registry** portfolio (`AskBeforeAnswer-Models`).
3. Find the specific model variant that won the ablation suite (e.g., `Clarifier-sft_only`).
4. Click on the model version and add the **`production`** alias to it.

### 4. Deploy to Hugging Face
With the alias set in W&B, you can now seamlessly deploy directly from your remote server!

```bash
make deploy-hf
```

**What this script does dynamically:**
1. Authenticates with the W&B API and queries your `AskBeforeAnswer-Models` registry.
2. Identifies the model tagged with `production` (e.g., `sft_only`).
3. Maps that alias directly to the existing local checkpoint cache (`models/sft_only/final`), saving gigabytes of cloud download bandwidth.
4. Dynamically reads `results/leaderboard.md` and injects the Weave Evaluation Metrics directly into the Hugging Face `README.md` Model Card.
5. Pushes the dataset and the exact winning model weights to your public Hugging Face Hub.

### 5. Update the GitHub Repository
Now, save the pointer (`dvc.lock`) to your GitHub repository so your codebase knows about the finalized experiment.

```bash
git add dvc.lock
git commit -m "feat: deployed new state-of-the-art clarification model"
git push origin main
```
