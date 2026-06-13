# End-to-End Deployment & Version Control Workflow

This document outlines the architecture and step-by-step instructions for managing datasets and model weights in the AskBeforeAnswer project. It explains how we seamlessly bridge local experimentation with public Hugging Face deployment using Data Version Control (DVC) and GitHub Actions.

## Architectural Overview: The "Messy Kitchen" vs. The "Dining Room"

When training machine learning models, a common problem is managing hundreds of intermediate, experimental datasets and model checkpoints. You do not want to pollute your public repository or public Hugging Face Hub with broken or intermediate files.

We solve this using a two-tiered architecture:

### 1. The "Messy Kitchen" (DVC + Internal Cloud Storage)
During development, we use **Data Version Control (DVC)**. DVC acts like Git for large binary files. It silently tracks every dataset generation and model training run you execute locally, mapping them to your Git commits. 
When you run `dvc push`, it uploads these heavy files to a private, internal storage bucket (e.g., a Hugging Face Storage Bucket, AWS S3, or Google Drive) in a compressed, hidden format. This keeps your local laptop clean and ensures your team has private backups of every experiment.

### 2. The "Dining Room" (GitHub Actions + Hugging Face Hub)
Once an experiment is successful and finalized, we want to serve it to the public. This is handled entirely by **GitHub Actions** (`.github/workflows/release_hf.yml`) and our custom deployment script (`scripts/publish_to_hf.py`).
By triggering a **Git Release Tag**, the CI/CD pipeline automatically authenticates with DVC, downloads the heavy files from the private "Messy Kitchen," and cleanly publishes the uncompressed datasets and final model weights to the public Hugging Face Datasets and Model Hubs.

---

## Step-by-Step Workflow

Follow these instructions to execute the end-to-end pipeline from local development to public Hugging Face deployment.

### 1. One-Time Setup: Configure DVC Remote
You must configure DVC to know where your internal "Messy Kitchen" storage bucket is located. In this project, we use a Hugging Face Dataset repository acting as a WebDAV storage bucket.

```bash
# Add your HF bucket as the default DVC remote using WebDAV
dvc remote add -d origin webdavs://huggingface.co/datasets/chrisjcc/ask-before-answer-dataset/tree/main

# Authenticate DVC with your HF Token (Requires Write Access)
dvc remote modify --local origin user chrisjcc
dvc remote modify --local origin password YOUR_HF_TOKEN_HERE
```
*(Note: The `--local` flag ensures your password is saved to `.dvc/config.local` and is never committed to Git).*

### 2. The Development Phase (Locally)
When you want to process data or train the model, do not run the Python scripts directly. Let DVC orchestrate the pipeline so it can automatically track the outputs.

```bash
# This automatically runs preprocess_data.py, train_sft.py, and train_dpo.py
make run-pipeline
```

### 3. Backup to the "Messy Kitchen" (DVC Push)
After the pipeline finishes, DVC updates the `dvc.lock` file. You now need to upload the actual heavy datasets and model weights to your internal storage bucket.

```bash
dvc push
```
*(This securely uploads the heavy binary data into your private HF bucket in a compressed format).*

### 4. Update the GitHub Repository
Now, save the pointer (`dvc.lock`) to your GitHub repository so your codebase knows about the new experiment.

```bash
git add dvc.lock
git commit -m "feat: generated new v1.5 dataset and trained model"
git push origin main
```
*(At this point, your GitHub repository is updated, but nothing has been formally published to the public Hugging Face Model or Dataset Hubs).*

### 5. Publish to the "Dining Room" (GitHub Release)
When you decide a specific commit is ready for public release:

1. Go to your **GitHub Repository** in your web browser.
2. Navigate to the **Releases** tab on the right side.
3. Click **Draft a new release**.
4. Type a version tag (e.g., `v1.5.0`).
5. Write your release notes in the description box (e.g., *"Added 5,000 new clarification examples!"*).
6. Click **Publish release**.

**Automated Deployment Execution:**
The moment you click publish, the `.github/workflows/release_hf.yml` GitHub Action triggers. It spins up a cloud runner, automatically authenticates with DVC using the `HF_TOKEN` repository secret, runs `dvc pull` to fetch your heavy files, and executes `scripts/publish_to_hf.py`. 

This script dynamically generates professional READMEs containing your release notes and publishes the final `sft_data.jsonl`, `dpo_data.jsonl`, and model weights directly to your public Hugging Face repositories!
