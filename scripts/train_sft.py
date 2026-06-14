import logging
import os

import hydra
from dotenv import load_dotenv
from omegaconf import DictConfig

from src.training.trainer import run_sft_training

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@hydra.main(version_base="1.3", config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    logger.info("Starting SFT training pipeline...")
    os.makedirs(cfg.training.output_dir, exist_ok=True)
    run_sft_training(cfg)
    logger.info("SFT training pipeline complete.")


if __name__ == "__main__":
    main()
