import logging
import os

import hydra
from omegaconf import DictConfig

from src.data.preprocess import (
    extract_qa_data,
    prepare_dpo_dataset,
    prepare_sft_dataset,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@hydra.main(version_base="1.3", config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    logger.info("Starting data preprocessing...")

    os.makedirs(cfg.data_dir, exist_ok=True)

    df = extract_qa_data(
        dataset_name=cfg.data.name,
        split=cfg.data.split,
        max_samples=cfg.data.max_samples,
    )

    prepare_sft_dataset(df, cfg.data.output_sft_file)
    prepare_dpo_dataset(df, cfg.data.output_dpo_file)

    logger.info("Data preprocessing complete.")


if __name__ == "__main__":
    main()
