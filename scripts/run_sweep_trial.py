import argparse
import logging
import os
import subprocess

import wandb
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Map each DVC stage to the corresponding parameter namespace in params.yaml.
PARAM_MAP = {
    "train_sft": {
        "learning_rate": "training.sft.learning_rate",
    },
    "train_dpo": {
        "learning_rate": "training.dpo.learning_rate",
        "beta": "training.dpo.beta",
    },
    "train_sft_only": {
        "learning_rate": "training.sft.learning_rate",
    },
    "train_dpo_only": {
        "learning_rate": "training.dpo.learning_rate",
        "beta": "training.dpo.beta",
    },
    "train_orpo": {
        "learning_rate": "training.orpo.learning_rate",
        "beta": "training.orpo.beta",
    },
    "train_grpo": {
        "learning_rate": "training.grpo.learning_rate",
        "beta": "training.grpo.beta",
    },
}


def parse_sweep_params(unknown_args):
    """Extract W&B sweep parameters from CLI arguments.

    W&B passes sweep parameters in the form:

        --learning_rate=0.0001
        --beta=0.1
    """
    sweep_params = {}

    for arg in unknown_args:
        if not arg.startswith("--") or "=" not in arg:
            continue

        key, value = arg.lstrip("-").split("=", 1)

        # Convert numeric values to int/float where appropriate.
        try:
            if "." in value or "e" in value.lower():
                value = float(value)
            else:
                value = int(value)
        except ValueError:
            pass

        sweep_params[key] = value

    return sweep_params


def main():
    load_dotenv()

    # ---------------------------------------------------------
    # 1. Parse arguments
    # ---------------------------------------------------------
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--stage",
        type=str,
        default="train_sft",
        help="DVC training stage to sweep",
    )

    args, unknown = parser.parse_known_args()

    stage = args.stage

    if stage not in PARAM_MAP:
        raise ValueError(
            f"Unknown stage: {stage}. "
            f"Supported stages: {list(PARAM_MAP.keys())}"
        )

    # ---------------------------------------------------------
    # 2. Extract W&B sweep parameters
    # ---------------------------------------------------------
    sweep_params = parse_sweep_params(unknown)

    if not sweep_params:
        raise ValueError(
            "No W&B sweep parameters were provided. "
            f"Received unknown arguments: {unknown}"
        )

    logger.info(
        "Received W&B sweep parameters for %s: %s",
        stage,
        sweep_params,
    )

    # ---------------------------------------------------------
    # 3. Resolve W&B run ID
    # ---------------------------------------------------------
    run_id = os.environ.get("WANDB_RUN_ID")

    if not run_id:
        raise RuntimeError(
            "WANDB_RUN_ID is not set. "
            "This script is intended to be launched by a W&B sweep agent."
        )

    logger.info(
        "Processing W&B Sweep Run: %s",
        run_id,
    )

    # ---------------------------------------------------------
    # 4. Initialize/resume W&B run
    # ---------------------------------------------------------
    wandb.init(
        id=run_id,
        resume="allow",
    )

    # ---------------------------------------------------------
    # 5. Convert W&B parameters into DVC -S overrides
    # ---------------------------------------------------------
    dvc_overrides = []

    stage_param_map = PARAM_MAP[stage]

    for sweep_key, sweep_value in sweep_params.items():

        if sweep_key not in stage_param_map:
            logger.warning(
                "Ignoring unsupported sweep parameter '%s' for stage '%s'.",
                sweep_key,
                stage,
            )
            continue

        dvc_param = stage_param_map[sweep_key]

        override = f"{dvc_param}={sweep_value}"

        dvc_overrides.extend(["-S", override])

        logger.info(
            "Mapping W&B parameter '%s=%s' -> DVC parameter '%s'",
            sweep_key,
            sweep_value,
            dvc_param,
        )

    if not dvc_overrides:
        raise ValueError(
            f"No valid sweep parameters were found for stage '{stage}'. "
            f"Received: {sweep_params}"
        )

    # ---------------------------------------------------------
    # 6. Construct DVC experiment name
    # ---------------------------------------------------------
    experiment_name = f"sweep_{run_id}"

    # ---------------------------------------------------------
    # 7. Construct DVC command
    # ---------------------------------------------------------
    #
    # Example for SFT:
    #
    # dvc exp run train_sft \
    #     -n sweep_jmmerfsj \
    #     -f \
    #     -S training.sft.learning_rate=9.36e-05
    #
    # DVC will then apply the parameter override and execute
    # the stage command with the corresponding Hydra override.
    #
    cmd = [
        "dvc",
        "exp",
        "run",
        stage,
        "-n",
        experiment_name,
        "-f",
        *dvc_overrides,
    ]

    logger.info(
        "Executing DVC experiment:\n%s",
        " ".join(map(str, cmd)),
    )

    # ---------------------------------------------------------
    # 8. Execute DVC experiment
    # ---------------------------------------------------------
    try:
        subprocess.run(cmd, check=True)

    except subprocess.CalledProcessError as e:
        logger.error(
            "DVC experiment failed for W&B run %s: %s",
            run_id,
            e,
        )
        raise

    except Exception:
        logger.exception(
            "Unexpected error while executing DVC experiment "
            "for W&B run %s.",
            run_id,
        )
        raise

    finally:
        # Finish the W&B run cleanly when this script owns the run.
        if wandb.run is not None:
            wandb.finish()


if __name__ == "__main__":
    main()
