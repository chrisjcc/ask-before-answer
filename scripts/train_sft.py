"""DVC Stage: SFT Training.

This script executes the Supervised Fine-Tuning stage of the DVC pipeline.
"""

import logging
import os

import hydra
from dotenv import load_dotenv
from omegaconf import DictConfig

from src.training.trainer import run_sft_training


load_dotenv()

logger = logging.getLogger(__name__)


@hydra.main(version_base="1.3", config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    logger.info("Starting SFT training pipeline")

    # Explicitly group experiments for W&B.
    if os.environ.get("WANDB_SWEEP_ID"):
        os.environ["WANDB_RUN_GROUP"] = "sft_sweeps"
    else:
        os.environ["WANDB_RUN_GROUP"] = "sft_baseline"

    logger.info("W&B run ID: %s", os.environ.get("WANDB_RUN_ID"))
    logger.info("W&B sweep ID: %s", os.environ.get("WANDB_SWEEP_ID"))
    logger.info("W&B project: %s", os.environ.get("WANDB_PROJECT"))
    logger.info("W&B entity: %s", os.environ.get("WANDB_ENTITY"))
    logger.info("W&B run group: %s", os.environ.get("WANDB_RUN_GROUP"))

    os.makedirs(cfg.training.output_dir, exist_ok=True)
    run_sft_training(cfg)

    logger.info("SFT training pipeline complete.")


if __name__ == "__main__":
    main()
