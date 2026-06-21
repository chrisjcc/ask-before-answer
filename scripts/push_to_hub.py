import argparse
import logging
import os

from datasets import DatasetDict, load_dataset
from huggingface_hub import HfApi

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Upload AskBeforeAnswer to Hugging Face Hub"
    )
    parser.add_argument("--model-repo", type=str, default="chrisjcc/ask-before-answer")
    parser.add_argument(
        "--dataset-repo", type=str, default="chrisjcc/ask-before-answer-data"
    )
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--model-dir", type=str, default="models/dpo/final")
    args = parser.parse_args()

    api = HfApi()

    # 1. Push Dataset
    logger.info(f"Loading datasets from {args.data_dir}...")
    try:
        ds_dict = DatasetDict(
            {
                "sft_train": load_dataset(
                    "json",
                    data_files=os.path.join(args.data_dir, "sft_train.jsonl"),
                    split="train",
                ),
                "sft_val": load_dataset(
                    "json",
                    data_files=os.path.join(args.data_dir, "sft_val.jsonl"),
                    split="train",
                ),
                "dpo_train": load_dataset(
                    "json",
                    data_files=os.path.join(args.data_dir, "dpo_train.jsonl"),
                    split="train",
                ),
                "dpo_val": load_dataset(
                    "json",
                    data_files=os.path.join(args.data_dir, "dpo_val.jsonl"),
                    split="train",
                ),
            }
        )
        logger.info(f"Pushing dataset to {args.dataset_repo}...")
        ds_dict.push_to_hub(args.dataset_repo)
        logger.info("Dataset push complete.")
    except Exception as e:
        logger.error(f"Failed to push dataset: {e}")
        return

    # 2. Generate Model Card
    readme_content = f"""---
language: en
license: mit
tags:
  - reinforcement-learning
  - dpo
  - sft
  - qwen2.5
  - clarification
datasets:
  - {args.dataset_repo}
---

# AskBeforeAnswer 🤖

This model is a Qwen 2.5 7B Instruct model fine-tuned using a two-stage \
pipeline (Supervised Fine-Tuning followed by Direct Preference Optimization) \
on the AmbigNQ dataset.

## Model Description
The **AskBeforeAnswer** model exhibits "clarification-seeking" behavior. \
When presented with an ambiguous question, rather than hallucinating or blindly \
assuming an intent, the model:
1. Detects the ambiguity.
2. Explains the reasoning behind the ambiguity.
3. Identifies the missing facets of information.
4. Asks a targeted clarification question to the user.

## Pipeline
- **Base Model:** Qwen/Qwen2.5-7B-Instruct
- **Stage 1 (SFT):** Aligned to output structured JSON indicating \
`Action: Clarify` or `Action: Answer`.
- **Stage 2 (DPO):** Preference optimized to strongly penalize \
hallucinations on ambiguous queries, using `{args.dataset_repo}`.

## Usage
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base_model_name = "Qwen/Qwen2.5-7B-Instruct"
adapter_model_name = "{args.model_repo}"

# Load Base
model = AutoModelForCausalLM.from_pretrained(base_model_name)
tokenizer = AutoTokenizer.from_pretrained(base_model_name)

# Attach AskBeforeAnswer Adapters
model = PeftModel.from_pretrained(model, adapter_model_name)
```
"""
    readme_path = os.path.join(args.model_dir, "README.md")
    logger.info("Writing Model Card to README.md...")
    os.makedirs(args.model_dir, exist_ok=True)
    with open(readme_path, "w") as f:
        f.write(readme_content)

    # 3. Push Model
    logger.info(f"Creating/Checking Model Repo {args.model_repo}...")
    api.create_repo(repo_id=args.model_repo, repo_type="model", exist_ok=True)

    logger.info(f"Uploading model folder {args.model_dir} to {args.model_repo}...")
    api.upload_folder(
        folder_path=args.model_dir, repo_id=args.model_repo, repo_type="model"
    )
    logger.info("Model upload complete!")


if __name__ == "__main__":
    main()
