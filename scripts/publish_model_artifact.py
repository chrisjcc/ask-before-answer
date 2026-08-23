"""Publish a DVC-promoted model to the W&B Model Registry.

This script is the bridge between DVC-level promotion and W&B-level
promotion.

Workflow:

    DVC experiment
        -> models/<variant>/final
        -> W&B project artifact
        -> W&B Registry collection
        -> production alias
        -> provenance/model_promotion.json

The DVC-managed training output is never modified.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import hydra
import wandb
from dotenv import load_dotenv
from omegaconf import DictConfig

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Model variants
# ---------------------------------------------------------------------------

MODEL_PATHS = {
    "sft_only": "models/sft_only/final",
    "dpo_only": "models/dpo_only/final",
    "sft": "models/sft/final",
    "dpo": "models/dpo/final",
    "sft_dpo": "models/dpo/final",
    "grpo": "models/grpo/final",
    "orpo": "models/orpo/final",
}


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PROVENANCE_DIR = Path("provenance")
PROMOTION_FILE = PROVENANCE_DIR / "model_promotion.json"


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


def get_registry_config(
    cfg: DictConfig,
) -> tuple[str, str, str]:
    """Read W&B Registry configuration from Hydra config."""

    deployment = cfg.deployment

    registry_name = str(
        deployment.get("registry_name", "")
    ).strip()

    registry_collection = str(
        deployment.get("registry_collection", "")
    ).strip()

    registry_alias = str(
        deployment.get("registry_alias", "")
    ).strip()

    if not registry_name:
        raise RuntimeError(
            "deployment.registry_name is not configured."
        )

    if not registry_collection:
        raise RuntimeError(
            "deployment.registry_collection is not configured."
        )

    if not registry_alias:
        raise RuntimeError(
            "deployment.registry_alias is not configured."
        )

    return (
        registry_name,
        registry_collection,
        registry_alias,
    )


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------


def get_wandb_config() -> tuple[str, str]:
    """Resolve the W&B entity and project from the environment."""

    entity = os.environ.get("WANDB_ENTITY")
    project = os.environ.get("WANDB_PROJECT")

    if not entity:
        raise RuntimeError(
            "WANDB_ENTITY environment variable is not set."
        )

    if not project:
        raise RuntimeError(
            "WANDB_PROJECT environment variable is not set."
        )

    return entity, project


# ---------------------------------------------------------------------------
# DVC helpers
# ---------------------------------------------------------------------------


def get_git_experiment_sha(
    experiment_name: str,
) -> str:
    """Resolve a DVC experiment name to its immutable Git SHA."""

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

    raise RuntimeError(
        f"Could not resolve DVC experiment '{experiment_name}' "
        "to a Git experiment SHA."
    )


def validate_model_directory(
    model: str,
) -> Path:
    """Validate that the DVC-promoted model output exists."""

    if model not in MODEL_PATHS:
        supported = ", ".join(sorted(MODEL_PATHS))

        raise ValueError(
            f"Unsupported model '{model}'. "
            f"Supported models: {supported}"
        )

    model_dir = Path(MODEL_PATHS[model])

    if not model_dir.is_dir():
        raise FileNotFoundError(
            f"Promoted model directory does not exist: {model_dir}"
        )

    files = [
        path
        for path in model_dir.rglob("*")
        if path.is_file()
    ]

    if not files:
        raise RuntimeError(
            f"Promoted model directory is empty: {model_dir}"
        )

    logger.info(
        "Validated DVC model output: %s",
        model_dir,
    )

    logger.info(
        "Model contains %d files.",
        len(files),
    )

    return model_dir


# ---------------------------------------------------------------------------
# W&B artifact publication
# ---------------------------------------------------------------------------


def publish_artifact(
    model: str,
    model_dir: Path,
    dvc_experiment: str,
    dvc_experiment_sha: str,
    stage: str,
    entity: str,
    project: str,
    registry_name: str,
    registry_collection: str,
    registry_alias: str,
) -> dict[str, Any]:
    """Publish the DVC-promoted model and link it to the W&B Registry."""

    artifact_name = f"Clarifier-{model}"

    metadata = {
        "model_variant": model,
        "dvc_experiment": dvc_experiment,
        "dvc_experiment_sha": dvc_experiment_sha,
        "dvc_stage": stage,
        "source_directory": str(model_dir),
        "promotion_timestamp": datetime.now(
            timezone.utc
        ).isoformat(),
    }

    logger.info(
        "Creating W&B artifact: %s",
        artifact_name,
    )

    with wandb.init(
        entity=entity,
        project=project,
        job_type="model-promotion",
        name=f"promote-{model}-{dvc_experiment}",
        config={
            "model_variant": model,
            "dvc_experiment": dvc_experiment,
            "dvc_experiment_sha": dvc_experiment_sha,
            "dvc_stage": stage,
        },
    ) as run:

        artifact = wandb.Artifact(
            name=artifact_name,
            type="model",
            description=(
                f"AskBeforeAnswer {model} model promoted "
                f"from DVC experiment {dvc_experiment}."
            ),
            metadata=metadata,
        )

        logger.info(
            "Adding DVC model output to W&B artifact..."
        )

        artifact.add_dir(
            local_path=str(model_dir),
        )

        logger.info(
            "Logging W&B artifact '%s'...",
            artifact_name,
        )

        logged_artifact = run.log_artifact(
            artifact,
        )

        # log_artifact() queues the artifact for upload. Calling wait()
        # on the artifact is sufficient here. There is no Run.wait_until_finish()
        # method in the installed W&B SDK.
        logger.info(
            "Waiting for W&B artifact upload to complete..."
        )

        logged_artifact.wait()

        source_version = getattr(
            logged_artifact,
            "version",
            None,
        )

        if not source_version:
            raise RuntimeError(
                "W&B did not return an artifact version after upload."
            )

        source_artifact_ref = (
            f"{entity}/{project}/{artifact_name}:{source_version}"
        )

        logger.info(
            "Source W&B artifact: %s",
            source_artifact_ref,
        )

        registry_target = (
            f"wandb-registry-{registry_name}/"
            f"{registry_collection}"
        )

        logger.info(
            "Linking artifact to W&B Registry collection: %s",
            registry_target,
        )

        linked_artifact = run.link_artifact(
            artifact=logged_artifact,
            target_path=registry_target,
            aliases=[registry_alias],
        )

        if linked_artifact is None:
            raise RuntimeError(
                "W&B did not return a linked Registry artifact."
            )

        logger.info(
            "Artifact linked to W&B Registry collection."
        )

        run_id = run.id

    # ------------------------------------------------------------------
    # Resolve the Registry artifact after the run has completed.
    # ------------------------------------------------------------------

    linked_name = getattr(
        linked_artifact,
        "name",
        None,
    )

    if not linked_name:
        raise RuntimeError(
            "W&B linked artifact did not expose a Registry artifact name."
        )

    logger.info(
        "W&B linked artifact reference: %s",
        linked_name,
    )

    api = wandb.Api()

    try:
        registry_artifact = api.artifact(
            linked_name,
        )
    except Exception as exc:
        raise RuntimeError(
            f"Failed to resolve linked W&B Registry artifact "
            f"'{linked_name}'."
        ) from exc

    registry_digest = getattr(
        registry_artifact,
        "digest",
        None,
    )

    if not registry_digest:
        raise RuntimeError(
            f"Registry artifact '{linked_name}' did not expose a digest."
        )

    logger.info(
        "W&B Registry artifact: %s",
        linked_name,
    )

    logger.info(
        "W&B Registry digest: %s",
        registry_digest,
    )

    return {
        "wandb_entity": entity,
        "wandb_project": project,
        "wandb_run_id": run_id,
        "source_artifact_ref": source_artifact_ref,
        "registry_artifact_ref": linked_name,
        "artifact_name": artifact_name,
        "artifact_digest": registry_digest,
        "registry_name": registry_name,
        "registry_collection": registry_collection,
        "registry_alias": registry_alias,
    }


# ---------------------------------------------------------------------------
# Promotion provenance
# ---------------------------------------------------------------------------


def write_promotion_provenance(
    model: str,
    stage: str,
    dvc_experiment: str,
    dvc_experiment_sha: str,
    artifact_info: dict[str, Any],
) -> Path:
    """Write the exact W&B promotion record consumed by deployment."""

    PROVENANCE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    promotion = {
        "schema_version": 1,
        "model_variant": model,
        "dvc_stage": stage,
        "dvc_experiment": dvc_experiment,
        "dvc_experiment_sha": dvc_experiment_sha,
        "wandb": {
            "entity": artifact_info["wandb_entity"],
            "project": artifact_info["wandb_project"],
            "run_id": artifact_info["wandb_run_id"],
            "source_artifact_ref": artifact_info[
                "source_artifact_ref"
            ],
            "registry_artifact_ref": artifact_info[
                "registry_artifact_ref"
            ],
            "artifact_name": artifact_info[
                "artifact_name"
            ],
            "artifact_digest": artifact_info[
                "artifact_digest"
            ],
            "registry_name": artifact_info[
                "registry_name"
            ],
            "registry_collection": artifact_info[
                "registry_collection"
            ],
            "registry_alias": artifact_info[
                "registry_alias"
            ],
        },
        "promoted_at": datetime.now(
            timezone.utc
        ).isoformat(),
    }

    with PROMOTION_FILE.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            promotion,
            file,
            indent=2,
        )
        file.write("\n")

    logger.info(
        "Wrote promotion provenance: %s",
        PROMOTION_FILE,
    )

    return PROMOTION_FILE


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Publish a DVC-promoted model to the W&B Model Registry."
        ),
    )

    parser.add_argument(
        "--model",
        required=True,
        choices=sorted(MODEL_PATHS),
        help="Model variant to publish.",
    )

    parser.add_argument(
        "--experiment",
        required=True,
        help="DVC experiment name that was promoted.",
    )

    parser.add_argument(
        "--stage",
        required=True,
        help="DVC training stage associated with the model.",
    )

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


@hydra.main(
    version_base="1.3",
    config_path="../configs",
    config_name="config",
)
def main(cfg: DictConfig) -> None:
    """Publish the DVC-promoted model to W&B Registry."""

    args = parse_args()

    entity, project = get_wandb_config()

    (
        registry_name,
        registry_collection,
        registry_alias,
    ) = get_registry_config(cfg)

    logger.info(
        "=========================================================="
    )
    logger.info(
        "W&B MODEL ARTIFACT PUBLICATION"
    )
    logger.info(
        "=========================================================="
    )
    logger.info(
        "Model:              %s",
        args.model,
    )
    logger.info(
        "Experiment:         %s",
        args.experiment,
    )
    logger.info(
        "Stage:              %s",
        args.stage,
    )
    logger.info(
        "W&B entity:         %s",
        entity,
    )
    logger.info(
        "W&B project:        %s",
        project,
    )
    logger.info(
        "Registry:            %s",
        registry_name,
    )
    logger.info(
        "Registry collection: %s",
        registry_collection,
    )
    logger.info(
        "Registry alias:      %s",
        registry_alias,
    )
    logger.info(
        "=========================================================="
    )

    # ------------------------------------------------------------------
    # 1. Validate the already-promoted DVC artifact.
    # ------------------------------------------------------------------

    model_dir = validate_model_directory(
        args.model,
    )

    # ------------------------------------------------------------------
    # 2. Resolve the immutable DVC experiment SHA.
    # ------------------------------------------------------------------

    dvc_experiment_sha = get_git_experiment_sha(
        args.experiment,
    )

    logger.info(
        "DVC experiment SHA: %s",
        dvc_experiment_sha,
    )

    # ------------------------------------------------------------------
    # 3. Publish exact DVC-promoted model to W&B Registry.
    # ------------------------------------------------------------------

    artifact_info = publish_artifact(
        model=args.model,
        model_dir=model_dir,
        dvc_experiment=args.experiment,
        dvc_experiment_sha=dvc_experiment_sha,
        stage=args.stage,
        entity=entity,
        project=project,
        registry_name=registry_name,
        registry_collection=registry_collection,
        registry_alias=registry_alias,
    )

    # ------------------------------------------------------------------
    # 4. Write promotion-level provenance.
    # ------------------------------------------------------------------

    promotion_file = write_promotion_provenance(
        model=args.model,
        stage=args.stage,
        dvc_experiment=args.experiment,
        dvc_experiment_sha=dvc_experiment_sha,
        artifact_info=artifact_info,
    )

    logger.info(
        "=========================================================="
    )
    logger.info(
        "W&B MODEL PROMOTION SUCCESSFUL"
    )
    logger.info(
        "=========================================================="
    )
    logger.info(
        "Model:             %s",
        args.model,
    )
    logger.info(
        "DVC experiment:    %s",
        args.experiment,
    )
    logger.info(
        "DVC SHA:           %s",
        dvc_experiment_sha,
    )
    logger.info(
        "Source artifact:   %s",
        artifact_info["source_artifact_ref"],
    )
    logger.info(
        "Registry artifact: %s",
        artifact_info["registry_artifact_ref"],
    )
    logger.info(
        "Artifact digest:   %s",
        artifact_info["artifact_digest"],
    )
    logger.info(
        "Production alias:  %s",
        registry_alias,
    )
    logger.info(
        "Provenance file:   %s",
        promotion_file,
    )
    logger.info(
        "=========================================================="
    )


if __name__ == "__main__":
    main()
