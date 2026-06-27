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

    # ---------------------------------------------------------
    # W&B Model Registry Resolution
    # ---------------------------------------------------------
    registry_alias = cfg.deployment.get("registry_alias", "production")
    wandb_entity = os.environ.get("WANDB_ENTITY")
    wandb_project = os.environ.get("WANDB_PROJECT")

    winning_model_name = "dpo"  # Fallback

    if wandb_entity and wandb_project:
        import wandb

        logger.info(f"Querying W&B Model Registry for alias: '{registry_alias}'...")
        try:
            api = wandb.Api()
            # The artifact is stored in the Model Registry portfolio
            artifact_path = f"{wandb_entity}/{wandb_project}/AskBeforeAnswer-Models:{registry_alias}"
            artifact = api.artifact(artifact_path)

            # The artifact name format is Clarifier-{model_name}
            artifact_basename = artifact.name.split(":")[0]  # e.g. Clarifier-sft_only
            if artifact_basename.startswith("Clarifier-"):
                winning_model_name = artifact_basename.replace("Clarifier-", "")
                logger.info(f"🏆 W&B resolved winning model: '{winning_model_name}'!")
            else:
                logger.warning(
                    f"Unexpected artifact name format: {artifact.name}. Falling back to 'dpo'."
                )

        except Exception as e:
            logger.warning(
                f"Could not fetch artifact from W&B Registry: {e}. Falling back to 'dpo'."
            )
    else:
        logger.warning(
            "WANDB_ENTITY or WANDB_PROJECT not set. Skipping W&B resolution and falling back to 'dpo'."
        )

    # Construct dynamic local directory path
    local_dir_mapping = {
        "sft_only": "models/sft_only/final",
        "dpo_only": "models/dpo_only/final",
        "sft": "models/sft/final",
        "dpo": "models/dpo/final",
        "base": "Qwen/Qwen2.5-7B-Instruct",
    }

    if winning_model_name not in local_dir_mapping:
        logger.error(
            f"Resolved model '{winning_model_name}' is not mapped to a local directory! Aborting."
        )
        return

    model_dir = (
        os.path.join(
            cfg.models_dir, local_dir_mapping[winning_model_name].split("/", 1)[-1]
        )
        if "models/" in local_dir_mapping[winning_model_name]
        else local_dir_mapping[winning_model_name]
    )
    logger.info(f"Targeting local model directory: {model_dir}")

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
        release_text = (
            f"**GitHub Release:** [{cfg.deployment.release_tag}](https://github.com/chrisjcc/ask-before-answer/releases/tag/{cfg.deployment.release_tag})\n"
            if cfg.deployment.get("release_tag")
            else ""
        )

        dataset_card_content = f"""
# AskBeforeAnswer Dataset

This dataset contains the training and validation splits for the \
**AskBeforeAnswer** clarification-seeking model.

{release_text}
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
            # Remove old appended description if it exists so we can cleanly replace it
            if "# AskBeforeAnswer Dataset" in card.text:
                card.text = card.text.split("# AskBeforeAnswer Dataset")[0]

            card.text = (
                card.text.rstrip() + "\n\n" + dataset_card_content.strip() + "\n"
            )
            card.push_to_hub(dataset_repo)
        except Exception as e:
            logger.warning(f"Could not load/update DatasetCard: {e}")

        logger.info("Dataset push complete.")
    except Exception as e:
        logger.error(f"Failed to push dataset: {e}")
        return

    # 2. Generate Model Card
    leaderboard_content = ""
    leaderboard_path = os.path.join(cfg.project_dir, "results", "leaderboard.md")
    if os.path.exists(leaderboard_path):
        with open(leaderboard_path, "r") as f:
            leaderboard_content = f.read()

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
- **Ablation Winner:** The model variant promoted to Production via W&B Registry is: `{winning_model_name}`.

{leaderboard_content}

{release_text}
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
