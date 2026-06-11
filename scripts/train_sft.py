import hydra
from omegaconf import DictConfig
import logging
import os
from src.training.trainer import run_sft_training

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
