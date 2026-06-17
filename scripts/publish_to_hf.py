import os
import sys
import tempfile

from huggingface_hub import HfApi, login

DATASET_REPO = "chrisjcc/ask-before-answer-dataset"
MODEL_REPO = "chrisjcc/ask-before-answer-Qwen2.5-1.5B-Instruct"


def generate_dataset_readme(release_tag, release_name, release_body):
    return f"""---
license: apache-2.0
task_categories:
- question-answering
language:
- en
---

# AskBeforeAnswer Dataset 🤖
> **Release:** {release_tag} ({release_name})

This dataset contains the processed Supervised Fine-Tuning (SFT) and 
Direct Preference Optimization (DPO) data for the AskBeforeAnswer 
clarification model.

## Release Notes
{release_body}
"""


def generate_model_readme(release_tag, release_name, release_body):
    return f"""---
license: apache-2.0
base_model: Qwen/Qwen2.5-1.5B-Instruct
tags:
- alignment
- clarification
- rlhf
- dpo
---

# AskBeforeAnswer Qwen2.5 1.5B Instruct 🤖
> **Release:** {release_tag} ({release_name})

This model has been fine-tuned to detect ambiguity and ask clarification 
questions before answering. It contains both the Supervised Fine-Tuning 
(SFT) and Direct Preference Optimization (DPO) artifacts.

## Release Notes
{release_body}
"""


def main():
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        print("Error: HF_TOKEN environment variable is not set.", file=sys.stderr)
        sys.exit(1)

    release_tag = os.environ.get("RELEASE_TAG", "v1.0.0")
    release_name = os.environ.get("RELEASE_NAME", "Initial Release")
    release_body = os.environ.get(
        "RELEASE_BODY", "Automated deployment via GitHub Actions."
    )

    # Authenticate
    login(token=hf_token)
    api = HfApi()

    # 1. Ensure Repositories Exist
    print(f"Ensuring Dataset Repository exists: {DATASET_REPO}")
    api.create_repo(repo_id=DATASET_REPO, repo_type="dataset", exist_ok=True)

    print(f"Ensuring Model Repository exists: {MODEL_REPO}")
    api.create_repo(repo_id=MODEL_REPO, repo_type="model", exist_ok=True)

    # 2. Upload Dataset Files & README
    print("Uploading to Dataset Hub...")
    if os.path.exists("data/sft_data.jsonl"):
        api.upload_file(
            path_or_fileobj="data/sft_data.jsonl",
            path_in_repo="sft_data.jsonl",
            repo_id=DATASET_REPO,
            repo_type="dataset",
        )
    if os.path.exists("data/dpo_data.jsonl"):
        api.upload_file(
            path_or_fileobj="data/dpo_data.jsonl",
            path_in_repo="dpo_data.jsonl",
            repo_id=DATASET_REPO,
            repo_type="dataset",
        )

    with tempfile.NamedTemporaryFile("w", delete=False) as f:
        f.write(generate_dataset_readme(release_tag, release_name, release_body))
        ds_readme_path = f.name
    api.upload_file(
        path_or_fileobj=ds_readme_path,
        path_in_repo="README.md",
        repo_id=DATASET_REPO,
        repo_type="dataset",
    )
    os.unlink(ds_readme_path)

    # 3. Upload Model Files & README
    print("Uploading to Model Hub...")
    if os.path.exists("models/sft/final"):
        print("Uploading SFT final model...")
        api.upload_folder(
            folder_path="models/sft/final",
            path_in_repo="sft_final",
            repo_id=MODEL_REPO,
            repo_type="model",
        )

    if os.path.exists("models/dpo/final"):
        print("Uploading DPO final model...")
        api.upload_folder(
            folder_path="models/dpo/final",
            path_in_repo="dpo_final",
            repo_id=MODEL_REPO,
            repo_type="model",
        )

    with tempfile.NamedTemporaryFile("w", delete=False) as f:
        f.write(generate_model_readme(release_tag, release_name, release_body))
        mdl_readme_path = f.name
    api.upload_file(
        path_or_fileobj=mdl_readme_path,
        path_in_repo="README.md",
        repo_id=MODEL_REPO,
        repo_type="model",
    )
    os.unlink(mdl_readme_path)

    print("Successfully published Release to Hugging Face! 🚀")


if __name__ == "__main__":
    main()
