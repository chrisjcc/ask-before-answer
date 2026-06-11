import hydra
from omegaconf import DictConfig
import logging
import os
import subprocess

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@hydra.main(version_base="1.3", config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    logger.info("Starting ablation study pipelines...")
    
    # Preprocess Data
    logger.info("1. Preprocessing data...")
    subprocess.run(["python", "scripts/preprocess_data.py"], check=True)
    
    # Base Evaluation (Base model without FT)
    logger.info("2. Evaluating Base Model...")
    subprocess.run(["python", "scripts/evaluate.py", "model_name=base"], check=False)
    
    # SFT Training
    logger.info("3. Running SFT Training...")
    subprocess.run(["python", "scripts/train_sft.py"], check=True)
    
    # SFT Evaluation
    logger.info("4. Evaluating SFT Model...")
    subprocess.run(["python", "scripts/evaluate.py", "model_name=sft"], check=False)
    
    # DPO Training (from Base)
    # logger.info("5. Running DPO Training from Base Model...")
    # subprocess.run(["python", "scripts/train_dpo.py", "model.name=Qwen/Qwen2.5-7B-Instruct"], check=True)
    
    # DPO Training (from SFT)
    logger.info("6. Running DPO Training from SFT Model...")
    # Point the model to the output of SFT
    sft_model_path = os.path.join(cfg.models_dir, "sft_output", "final")
    subprocess.run(["python", "scripts/train_dpo.py", f"model.name={sft_model_path}"], check=True)
    
    # SFT+DPO Evaluation
    logger.info("7. Evaluating SFT+DPO Model...")
    subprocess.run(["python", "scripts/evaluate.py", "model_name=sft_dpo"], check=False)
    
    logger.info("Ablation study complete.")

if __name__ == "__main__":
    main()
