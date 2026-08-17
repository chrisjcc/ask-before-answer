import argparse
import logging
import os
import subprocess

from dotenv import load_dotenv
import wandb


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


def get_git_experiment_sha(experiment_name):
    """
    Find the Git commit SHA associated with a DVC experiment.

    DVC stores experiments as Git refs under refs/exps/.
    The experiment name is the final path component.
    """
    result = subprocess.run(
        [
            "git",
            "for-each-ref",
            "--format=%(objectname) %(refname)",
            "refs/exps/",
        ],
        capture_output=True,
        text=True,
        check=True,
    )

    for line in result.stdout.splitlines():
        if not line.strip():
            continue

        sha, ref = line.split(" ", 1)

        if ref.endswith(f"/{experiment_name}"):
            return sha

    return None


def update_wandb_provenance(
    run_id,
    sweep_id,
    project,
    entity,
    dvc_experiment,
    dvc_experiment_sha,
    stage,
    sweep_params,
):
    """
    Record the explicit W&B <-> DVC provenance relationship
    in the existing W&B sweep run.
    """

    api = wandb.Api()

    run = api.run(f"{entity}/{project}/{run_id}")

    provenance = {
        # W&B identity
        "provenance/wandb_run_id": run_id,
        "provenance/wandb_sweep_id": sweep_id,

        # DVC identity
        "provenance/dvc_experiment": dvc_experiment,
        "provenance/dvc_experiment_sha": dvc_experiment_sha,
        "provenance/dvc_stage": stage,
    }

    # Record the exact sweep parameters used by this DVC experiment.
    for key, value in sweep_params.items():
        provenance[f"provenance/dvc_param/{key}"] = value

    logger.info("=== W&B/DVC PROVENANCE ===")

    for key, value in provenance.items():
        logger.info("%s=%s", key, value)

    logger.info("==========================")

    # Update the existing W&B run's summary.
    run.summary.update(provenance)

    # Explicitly persist the API update.
    run.update()

    logger.info(
        "Recorded DVC provenance in W&B run %s",
        run_id,
    )


def verify_wandb_provenance(
    run_id,
    sweep_id,
    project,
    entity,
    dvc_experiment,
    dvc_experiment_sha,
):
    """
    Re-read the W&B run through the API and verify that the provenance
    fields were actually persisted.
    """

    api = wandb.Api()

    run = api.run(f"{entity}/{project}/{run_id}")

    expected = {
        "provenance/wandb_run_id": run_id,
        "provenance/wandb_sweep_id": sweep_id,
        "provenance/dvc_experiment": dvc_experiment,
        "provenance/dvc_experiment_sha": dvc_experiment_sha,
    }

    logger.info("=== VERIFY W&B PROVENANCE ===")

    failed = False

    for key, expected_value in expected.items():
        actual_value = run.summary.get(key)

        logger.info(
            "%s: expected=%s actual=%s",
            key,
            expected_value,
            actual_value,
        )

        if actual_value != expected_value:
            failed = True

    logger.info("=============================")

    if failed:
        raise RuntimeError(
            f"W&B provenance verification failed for run {run_id}."
        )

    logger.info(
        "W&B provenance verification PASSED for run %s",
        run_id,
    )


def main():
    load_dotenv()

    # ---------------------------------------------------------------
    # 1. Capture W&B sweep-run identity
    # ---------------------------------------------------------------

    run_id = os.environ.get("WANDB_RUN_ID")

    if not run_id:
        raise RuntimeError(
            "WANDB_RUN_ID is not set. "
            "This script is intended to be launched by a W&B sweep agent."
        )

    sweep_id = os.environ.get("WANDB_SWEEP_ID")
    entity = os.environ.get("WANDB_ENTITY", "rl4aa")
    project = os.environ.get("WANDB_PROJECT", "ask-before-answer")

    print("=== BEFORE TRAINING ===")
    print(f"WANDB_RUN_ID={run_id}")
    print(f"WANDB_SWEEP_ID={sweep_id}")
    print(f"WANDB_ENTITY={entity}")
    print(f"WANDB_PROJECT={project}")
    print("=======================")

    # ---------------------------------------------------------------
    # 2. Parse arguments
    # ---------------------------------------------------------------

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--stage",
        type=str,
        default="train_sft",
        help="DVC stage to sweep",
    )

    args, unknown = parser.parse_known_args()

    stage = args.stage

    if stage not in PARAM_MAP:
        raise ValueError(
            f"Unknown stage: {stage}. "
            f"Must be one of {list(PARAM_MAP.keys())}"
        )

    # ---------------------------------------------------------------
    # 3. Log W&B environment
    # ---------------------------------------------------------------

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

    # ---------------------------------------------------------------
    # 4. Parse W&B sweep parameters
    # ---------------------------------------------------------------

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
        raise RuntimeError(
            f"No sweep parameters were received for stage '{stage}'."
        )

    # ---------------------------------------------------------------
    # 5. Construct DVC parameter overrides
    # ---------------------------------------------------------------

    param_namespace = PARAM_MAP[stage]

    dvc_param_overrides = []

    for key, value in sweep_params.items():
        dvc_param_overrides.extend(
            [
                "-S",
                f"{param_namespace}.{key}={value}",
            ]
        )

    # ---------------------------------------------------------------
    # 6. Construct deterministic W&B <-> DVC identity
    # ---------------------------------------------------------------

    # This is the W&B -> DVC link.
    #
    # Example:
    #
    #   W&B run:
    #       oglusx4l
    #
    #   DVC experiment:
    #       sweep_oglusx4l
    #
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
    logger.info("WANDB_RUN_ID=%s", run_id)
    logger.info("WANDB_SWEEP_ID=%s", sweep_id)
    logger.info("WANDB_PROJECT=%s", project)
    logger.info("WANDB_ENTITY=%s", entity)
    logger.info("==================================")

    logger.info("DVC parameter overrides: %s", dvc_param_overrides)
    logger.info("DVC experiment name: %s", run_name)
    logger.info("DVC command: %s", " ".join(cmd))

    # ---------------------------------------------------------------
    # 7. Handle stale DVC locks
    # ---------------------------------------------------------------

    dvc_lock_file = ".dvc/tmp/rwlock"

    if os.path.exists(dvc_lock_file):
        logger.warning(
            "Found stale DVC lock at %s. Removing it.",
            dvc_lock_file,
        )

        try:
            os.remove(dvc_lock_file)
        except OSError:
            pass

    # ---------------------------------------------------------------
    # 8. Execute DVC experiment
    # ---------------------------------------------------------------

    try:
        subprocess.run(cmd, check=True)

    except Exception as e:
        logger.error("DVC experiment failed: %s", e)

        if os.path.exists(dvc_lock_file):
            try:
                os.remove(dvc_lock_file)
            except OSError:
                pass

        raise

    # ---------------------------------------------------------------
    # 9. Resolve immutable DVC experiment SHA
    # ---------------------------------------------------------------

    dvc_experiment_sha = get_git_experiment_sha(run_name)

    if dvc_experiment_sha is None:
        raise RuntimeError(
            f"DVC experiment '{run_name}' completed, but its Git "
            "experiment reference could not be found."
        )

    logger.info(
        "DVC experiment '%s' resolved to SHA %s",
        run_name,
        dvc_experiment_sha,
    )

    # ---------------------------------------------------------------
    # 10. Write DVC -> W&B provenance
    # ---------------------------------------------------------------

    update_wandb_provenance(
        run_id=run_id,
        sweep_id=sweep_id,
        project=project,
        entity=entity,
        dvc_experiment=run_name,
        dvc_experiment_sha=dvc_experiment_sha,
        stage=stage,
        sweep_params=sweep_params,
    )

    # ---------------------------------------------------------------
    # 11. Verify provenance persisted in W&B
    # ---------------------------------------------------------------

    verify_wandb_provenance(
        run_id=run_id,
        sweep_id=sweep_id,
        project=project,
        entity=entity,
        dvc_experiment=run_name,
        dvc_experiment_sha=dvc_experiment_sha,
    )

    # ---------------------------------------------------------------
    # 12. Final provenance report
    # ---------------------------------------------------------------

    logger.info("=== PROVENANCE COMPLETE ===")
    logger.info("W&B entity:   %s", entity)
    logger.info("W&B project:  %s", project)
    logger.info("W&B sweep:    %s", sweep_id)
    logger.info("W&B run:      %s", run_id)
    logger.info("DVC experiment: %s", run_name)
    logger.info("DVC SHA:        %s", dvc_experiment_sha)
    logger.info("DVC stage:      %s", stage)
    logger.info("============================")


if __name__ == "__main__":
    main()
