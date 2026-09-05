"""DVC Stage: Data Preprocessing.

This script executes the data processing step of the DVC pipeline, saving
the outputs to the paths defined in the Hydra configuration.
"""

import json
import logging
import os

import hydra
import weave
from omegaconf import DictConfig

from ask_before_answer.data.preprocess import (
    extract_qa_data,
    prepare_dpo_dataset,
    prepare_sft_dataset,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@hydra.main(version_base="1.3", config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    logger.info("Starting data preprocessing...")

    os.makedirs(cfg.data_dir, exist_ok=True)

    df_train = extract_qa_data(
        dataset_name=cfg.data.name,
        split="train",
        max_samples=cfg.data.max_samples,
        synthetic_cfg=cfg.data.get("synthetic_generation", None),
    )
    prepare_sft_dataset(df_train, cfg.data.output_sft_train_file)
    prepare_dpo_dataset(
        df_train,
        cfg.data.output_dpo_train_file,
        cfg.data.get("synthetic_generation", None),
    )

    df_val = extract_qa_data(
        dataset_name=cfg.data.name,
        split="validation",
        max_samples=cfg.data.max_samples,
        synthetic_cfg=cfg.data.get("synthetic_generation", None),
    )
    prepare_sft_dataset(df_val, cfg.data.output_sft_val_file)
    prepare_dpo_dataset(
        df_val, cfg.data.output_dpo_val_file, cfg.data.get("synthetic_generation", None)
    )

    logger.info("Data preprocessing complete. Publishing to W&B Weave...")

    # Initialize Weave to the project specified in .env (or default)
    project_name = os.environ.get("WANDB_PROJECT", "ask-before-answer")
    weave.init(project_name)

    def publish_weave_dataset(jsonl_path: str, dataset_name: str):
        """Helper to load a JSONL file and publish it as a Weave dataset."""
        if not os.path.exists(jsonl_path):
            logger.warning(f"File {jsonl_path} not found. Skipping publish.")
            return

        with open(jsonl_path, "r") as f:
            rows = [json.loads(line) for line in f if line.strip()]

        # For Weave, the name should be a clean identifier without slashes
        dataset = weave.Dataset(name=dataset_name, rows=rows)
        weave.publish(dataset)
        logger.info(f"Successfully published Weave dataset: {dataset_name}")

    # Publish all generated splits
    publish_weave_dataset(cfg.data.output_sft_train_file, "sft-train")
    publish_weave_dataset(cfg.data.output_dpo_train_file, "dpo-train")
    publish_weave_dataset(cfg.data.output_sft_val_file, "sft-val")
    publish_weave_dataset(cfg.data.output_dpo_val_file, "dpo-val")

    logger.info("All datasets published successfully!")


if __name__ == "__main__":
    main()
