import os
import subprocess
import wandb
from dotenv import load_dotenv
from omegaconf import OmegaConf

def main():
    # 1. Load environment variables (e.g., WANDB_API_KEY from .env)
    load_dotenv()

    # 2. Initialize W&B to get the sweep parameters for this trial
    run = wandb.init()
    cfg = wandb.config

    # 3. Update the Hydra configuration directly
    sft_cfg_path = "configs/training/sft.yaml"
    sft_cfg = OmegaConf.load(sft_cfg_path)
    
    # We apply the specific hyperparameters defined in the sweep config
    if "learning_rate" in cfg:
        sft_cfg.learning_rate = cfg.learning_rate
        
    OmegaConf.save(sft_cfg, sft_cfg_path)
    print(f"Updated {sft_cfg_path} with new learning_rate: {sft_cfg.learning_rate}")

    # 4. Trigger DVC to track and execute the run
    # Because DVC is tracking sft.yaml as a dependency, it will notice the file 
    # changed and automatically run the 'train_sft' pipeline stage.
    print(f"Triggering DVC Experiment for Sweep Run: {run.id}")
    cmd = ["dvc", "exp", "run", "-n", f"sweep_{run.id}"]
    
    # We use subprocess.run to execute the DVC CLI command
    subprocess.run(cmd, check=True)

if __name__ == "__main__":
    main()
