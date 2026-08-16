import argparse
import logging
import os
import subprocess

import wandb
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# DVC parameter namespace corresponding to each training stage.
PARAM_MAP = {
    "train_sft": "training.sft",
    "train_dpo": "training.dpo",
    "train_sft_only": "training.sft",
    "train_dpo_only": "training.dpo",
    "train_orpo": "training.orpo",
    "train_grpo": "training.grpo",
}


def main():
    # 1. Load environment variables, including WANDB_API_KEY.
    load_dotenv()

    # 1.1 Capture the W&B sweep-run identity.
    #
    # The W&B sweep agent creates and owns the run before launching this
    # script. We intentionally do not call wandb.init() here because the
    # actual training process (train_sft.py) uses the Hugging Face/TRL
    # W&B integration and inherits WANDB_RUN_ID from the sweep agent.
    run_id = os.environ.get("WANDB_RUN_ID")

    if not run_id:
        raise RuntimeError(
            "WANDB_RUN_ID is not set. "
            "This script is intended to be launched by a W&B sweep agent."
        )

    print("=== BEFORE TRAINING ===")
    print(f"WANDB_RUN_ID={os.environ.get('WANDB_RUN_ID')}")
    print("=======================")

    # 2. Parse arguments.
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--stage",
        type=str,
        default="train_sft",
        help="DVC stage to sweep",
    )

    # W&B supplies sweep parameters as additional CLI arguments.
    args, unknown = parser.parse_known_args()

    stage = args.stage

    if stage not in PARAM_MAP:
        raise ValueError(
            f"Unknown stage: {stage}. Must be one of {list(PARAM_MAP.keys())}"
        )

    # 3. Log the relevant W&B environment.
    #
    # Do NOT print WANDB_API_KEY.
    logger.info("========== W&B ENVIRONMENT ==========")

    for key in [
        "WANDB_RUN_ID",
        "WANDB_SWEEP_ID",
        "WANDB_ENTITY",
        "WANDB_PROJECT",
        "WANDB_DIR",
    ]:
        logger.info(
            "ENV %s=%s",
            key,
            os.environ.get(key),
        )

    logger.info("====================================")

    # 4. Parse W&B sweep parameters.
    #
    # Example:
    #   --learning_rate=7.632829697182058e-05
    #
    # becomes:
    #   {"learning_rate": 7.632829697182058e-05}
    sweep_params = {}

    for arg in unknown:
        if not arg.startswith("--") or "=" not in arg:
            continue

        key, value = arg.lstrip("-").split("=", 1)

        try:
            if "." in value or "e" in value.lower():
                value = float(value)
            else:
                value = int(value)
        except ValueError:
            pass

        sweep_params[key] = value

    logger.info(
        "Received W&B sweep parameters for %s: %s",
        stage,
        sweep_params,
    )

    if not sweep_params:
        raise RuntimeError(f"No sweep parameters were received for stage '{stage}'.")

    # 5. Construct DVC parameter overrides.
    #
    # W&B:
    #   learning_rate=7.632829697182058e-05
    #
    # becomes:
    #   -S training.sft.learning_rate=7.632829697182058e-05
    #
    # DVC then translates that parameter override into the Hydra
    # command-line override defined in dvc.yaml:
    #
    #   training.learning_rate=7.632829697182058e-05
    #
    # IMPORTANT:
    # We do NOT modify configs/training/*.yaml.
    param_namespace = PARAM_MAP[stage]

    dvc_param_overrides = []

    for key, value in sweep_params.items():
        dvc_param_overrides.extend(
            [
                "-S",
                f"{param_namespace}.{key}={value}",
            ]
        )

    # 6. Construct the DVC experiment command.
    run_name = f"sweep_{run_id}"

    cmd = [
        "dvc",
        "exp",
        "run",
        stage,
        "-n",
        run_name,
        *dvc_param_overrides,
    ]

    logger.info("=== W&B ENVIRONMENT BEFORE DVC ===")
    logger.info("WANDB_RUN_ID=%s", os.environ.get("WANDB_RUN_ID"))
    logger.info("WANDB_SWEEP_ID=%s", os.environ.get("WANDB_SWEEP_ID"))
    logger.info("WANDB_PROJECT=%s", os.environ.get("WANDB_PROJECT"))
    logger.info("WANDB_ENTITY=%s", os.environ.get("WANDB_ENTITY"))
    logger.info("==================================")

    logger.info("DVC parameter overrides: %s", dvc_param_overrides)
    logger.info("DVC command: %s", " ".join(cmd))
    logger.info(
        "Triggering DVC Experiment for Sweep Run: %s targeting stage: %s",
        run_id,
        stage,
    )

    # 7. Handle stale DVC locks.
    #
    # This remains from the previous implementation because Hyperband
    # can terminate a trial while DVC still has its lock.
    dvc_lock_file = ".dvc/tmp/rwlock"

    if os.path.exists(dvc_lock_file):
        logger.warning(
            "Found stale DVC lock at %s. Removing it to prevent deadlock.",
            dvc_lock_file,
        )

        try:
            os.remove(dvc_lock_file)
        except OSError:
            pass

    # 8. Execute DVC.
    try:
        subprocess.run(cmd, check=True)

    except Exception as e:
        logger.error("DVC Experiment failed: %s", e)

        if os.path.exists(dvc_lock_file):
            try:
                os.remove(dvc_lock_file)
            except OSError:
                pass

        raise


if __name__ == "__main__":
    main()
