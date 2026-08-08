import argparse
import logging
import os
import subprocess

import wandb
from dotenv import load_dotenv
from omegaconf import OmegaConf

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CONFIG_MAP = {
    "train_sft": "configs/training/sft.yaml",
    "train_dpo": "configs/training/dpo.yaml",
    "train_sft_only": "configs/training/sft.yaml",
    "train_dpo_only": "configs/training/dpo.yaml",
}


def main():
    # 1. Load environment variables (e.g., WANDB_API_KEY from .env)
    load_dotenv()
    
    # 1.1 Start wandb early to prevent timeout crashes while DVC preprocesses
    run_id = os.environ.get("WANDB_RUN_ID")
    if run_id:
        wandb.init(id=run_id, resume="allow")

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

    # 3. Extract W&B hyperparameters manually from unknown args
    # W&B agent passes parameters as: --learning_rate=0.0001
    sweep_params = {}
    for arg in unknown:
        if arg.startswith("--") and "=" in arg:
            key, value = arg.lstrip("-").split("=", 1)
            try:
                if "." in value or "e" in value.lower():
                    value = float(value)
                else:
                    value = int(value)
            except ValueError:
                pass
            sweep_params[key] = value

    # 4. Update the mapped Hydra configuration directly
    cfg_path = CONFIG_MAP[stage]
    hydra_cfg = OmegaConf.load(cfg_path)

    # We apply the specific hyperparameters defined in the sweep config
    if "learning_rate" in sweep_params:
        hydra_cfg.learning_rate = sweep_params["learning_rate"]
    if "beta" in sweep_params:
        hydra_cfg.beta = sweep_params["beta"]

    OmegaConf.save(hydra_cfg, cfg_path)
    logger.info(f"Updated {cfg_path} with new hyperparameters: {sweep_params}")

    # Grab the run ID dynamically injected by the W&B agent environment
    run_id = os.environ.get("WANDB_RUN_ID", "local")
    logger.info(
        f"Triggering DVC Experiment for Sweep Run: {run_id} targeting stage: {stage}"
    )

    # 5.1 Force clean any stale DVC locks left behind by W&B early-stopping kills
    dvc_lock_file = ".dvc/tmp/rwlock"
    if os.path.exists(dvc_lock_file):
        logger.warning(
            f"Found stale DVC lock at {dvc_lock_file}. Removing it to prevent deadlock."
        )
        try:
            os.remove(dvc_lock_file)
        except OSError:
            pass

    # Note the '-f' flag to forcefully overwrite the experiment name
    # if W&B replays a run ID
    cmd = ["dvc", "exp", "run", stage, "-n", f"sweep_{run_id}", "-f"]

    # We use subprocess.run to execute the DVC CLI command
    try:
        subprocess.run(cmd, check=True)
    except Exception as e:
        logger.error(f"DVC Experiment failed: {e}")
        if os.path.exists(dvc_lock_file):
            try:
                os.remove(dvc_lock_file)
            except OSError:
                pass
        raise


if __name__ == "__main__":
    main()
