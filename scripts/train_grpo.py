"""DVC Stage: GRPO Training.

This script executes the Group Relative Policy Optimization stage of the DVC pipeline.
"""

import logging
import os

import hydra
from dotenv import load_dotenv
from omegaconf import DictConfig

from src.training.trainer import run_grpo_training

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@hydra.main(version_base="1.3", config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    logger.info("Starting GRPO training pipeline...")

    # Explicitly group experiments for W&B
    if os.environ.get("WANDB_SWEEP_ID"):
        os.environ["WANDB_RUN_GROUP"] = "grpo_sweeps"
    else:
        os.environ["WANDB_RUN_GROUP"] = "grpo_baseline"

    os.makedirs(cfg.training.output_dir, exist_ok=True)
    run_grpo_training(cfg)
    logger.info("GRPO training pipeline complete.")


if __name__ == "__main__":
    main()
