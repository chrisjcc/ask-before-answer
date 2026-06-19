import argparse
import os
import subprocess

import wandb
from dotenv import load_dotenv
from omegaconf import OmegaConf

CONFIG_MAP = {
    "train_sft": "configs/training/sft.yaml",
    "train_dpo": "configs/training/dpo.yaml",
    "train_sft_only": "configs/training/sft.yaml",
    "train_dpo_only": "configs/training/dpo.yaml",
}


def main():
    # 1. Load environment variables (e.g., WANDB_API_KEY from .env)
    load_dotenv()

    # 2. Parse arguments to determine the stage
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage", type=str, default="train_sft", help="DVC stage to sweep"
    )

    # We use parse_known_args because W&B also passes hyperparameters as CLI args
    args, unknown = parser.parse_known_args()
    stage = args.stage

    if stage not in CONFIG_MAP:
        raise ValueError(
            f"Unknown stage: {stage}. Must be one of {list(CONFIG_MAP.keys())}"
        )

    # 3. Initialize W&B to get the sweep parameters for this trial
    run = wandb.init()
    cfg = wandb.config

    # 4. Update the mapped Hydra configuration directly
    cfg_path = CONFIG_MAP[stage]
    hydra_cfg = OmegaConf.load(cfg_path)

    # We apply the specific hyperparameters defined in the sweep config
    if "learning_rate" in cfg:
        hydra_cfg.learning_rate = cfg.learning_rate
    if "beta" in cfg:
        hydra_cfg.beta = cfg.beta

    OmegaConf.save(hydra_cfg, cfg_path)
    print(f"Updated {cfg_path} with new hyperparameters: {dict(cfg)}")

    # 5. Trigger DVC to track and execute the run for the specific stage
    print(f"Triggering DVC Experiment for Sweep Run: {run.id} targeting stage: {stage}")
    cmd = ["dvc", "exp", "run", stage, "-n", f"sweep_{run.id}"]

    # We use subprocess.run to execute the DVC CLI command
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
