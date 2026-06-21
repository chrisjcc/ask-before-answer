import logging
import os

import hydra
from datasets import DatasetDict, load_dataset
from dotenv import load_dotenv
from huggingface_hub import HfApi
from omegaconf import DictConfig

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@hydra.main(version_base="1.3", config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    api = HfApi()

    # Extract config values
    model_repo = cfg.deployment.model_repo
    dataset_repo = cfg.deployment.dataset_repo
    data_dir = cfg.data_dir
    model_dir = os.path.join(cfg.models_dir, "dpo", "final")

    # 1. Push Dataset
    logger.info(f"Loading datasets from {data_dir}...")
    try:
        # SFT Dataset (instruction, input, output)
        sft_ds = DatasetDict(
            {
                "train": load_dataset(
                    "json",
                    data_files=os.path.join(data_dir, "sft_train.jsonl"),
                    split="train",
                ),
                "validation": load_dataset(
                    "json",
                    data_files=os.path.join(data_dir, "sft_val.jsonl"),
                    split="train",
                ),
            }
        )
        logger.info(f"Pushing SFT subset to {dataset_repo}...")
        sft_ds.push_to_hub(dataset_repo, config_name="sft")

        # DPO Dataset (prompt, chosen, rejected)
        dpo_ds = DatasetDict(
            {
                "train": load_dataset(
                    "json",
                    data_files=os.path.join(data_dir, "dpo_train.jsonl"),
                    split="train",
                ),
                "validation": load_dataset(
                    "json",
                    data_files=os.path.join(data_dir, "dpo_val.jsonl"),
                    split="train",
                ),
            }
        )
        logger.info(f"Pushing DPO subset to {dataset_repo}...")
        dpo_ds.push_to_hub(dataset_repo, config_name="dpo")
        # Upload Dataset Card
        dataset_card_content = f"""
# AskBeforeAnswer Dataset

This dataset contains the training and validation splits for the \
**AskBeforeAnswer** clarification-seeking model.

## Subsets (Configurations)
This repository contains two subsets which must be loaded separately \
depending on the training stage:

### 1. `sft` (Supervised Fine-Tuning)
Contains the structured JSON responses for initial alignment.
- **Features:** `instruction`, `input`, `output` (JSON dict containing \
`action`, `reasoning`, `facets`, `response`)

```python
from datasets import load_dataset
sft_dataset = load_dataset("{dataset_repo}", "sft")
```

### 2. `dpo` (Direct Preference Optimization)
Contains the preference pairs used to penalize hallucinations.
- **Features:** `prompt`, `chosen`, `rejected`

```python
from datasets import load_dataset
dpo_dataset = load_dataset("{dataset_repo}", "dpo")
```
"""
        logger.info("Uploading Dataset Card...")
        from huggingface_hub import DatasetCard

        try:
            card = DatasetCard.load(dataset_repo)
            # Only append if we haven't already
            if "AskBeforeAnswer Dataset" not in card.text:
                card.text = card.text + dataset_card_content
                card.push_to_hub(dataset_repo)
        except Exception as e:
            logger.warning(f"Could not load/update DatasetCard: {e}")

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
  - {dataset_repo}
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
hallucinations on ambiguous queries, using `{dataset_repo}`.

## Usage
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base_model_name = "Qwen/Qwen2.5-7B-Instruct"
adapter_model_name = "{model_repo}"

# Load Base
model = AutoModelForCausalLM.from_pretrained(base_model_name)
tokenizer = AutoTokenizer.from_pretrained(base_model_name)

# Attach AskBeforeAnswer Adapters
model = PeftModel.from_pretrained(model, adapter_model_name)
```
"""
    readme_path = os.path.join(model_dir, "README.md")
    logger.info("Writing Model Card to README.md...")
    os.makedirs(model_dir, exist_ok=True)
    with open(readme_path, "w") as f:
        f.write(readme_content)

    # 3. Push Model
    logger.info(f"Creating/Checking Model Repo {model_repo}...")
    api.create_repo(repo_id=model_repo, repo_type="model", exist_ok=True)

    logger.info(f"Uploading model folder {model_dir} to {model_repo}...")
    api.upload_folder(folder_path=model_dir, repo_id=model_repo, repo_type="model")
    logger.info("Model upload complete!")


if __name__ == "__main__":
    main()
