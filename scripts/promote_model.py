import argparse
import json
import logging
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import wandb
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Verify and promote an exact W&B model artifact "
            "to the AskBeforeAnswer Model Registry."
        )
    )

    parser.add_argument(
        "--artifact-ref",
        required=True,
        help=(
            "Exact source W&B artifact reference, including version. "
            "Example: rl4aa/ask-before-answer/Clarifier-grpo:v17"
        ),
    )

    parser.add_argument(
        "--verification-command",
        nargs=argparse.REMAINDER,
        help=(
            "Optional command used to verify the candidate model. "
            "The command must exit with status 0 for verification to pass."
        ),
    )

    parser.add_argument(
        "--registry-name",
        default="Model",
        help="W&B Registry name.",
    )

    parser.add_argument(
        "--registry-collection",
        default="AskBeforeAnswer-Models",
        help="W&B Registry collection name.",
    )

    parser.add_argument(
        "--production-alias",
        default="production",
        help="Registry alias assigned to the verified artifact.",
    )

    parser.add_argument(
        "--provenance-file",
        default="provenance/model_promotion.json",
        help="Path for the promotion provenance record.",
    )

    return parser.parse_args()


def validate_artifact_ref(artifact_ref: str) -> None:
    """Reject mutable or ambiguous artifact references."""

    if ":" not in artifact_ref:
        raise ValueError(
            "The artifact reference must include an explicit version. "
            "For example: rl4aa/ask-before-answer/Clarifier-grpo:v17"
        )

    alias = artifact_ref.rsplit(":", 1)[1]

    forbidden_aliases = {
        "latest",
        "production",
        "staging",
        "development",
        "dev",
        "test",
    }

    if alias in forbidden_aliases:
        raise ValueError(
            f"Artifact reference '{artifact_ref}' uses mutable alias "
            f"'{alias}'. Promotion requires an immutable artifact version."
        )


def resolve_artifact(
    api: wandb.Api,
    artifact_ref: str,
) -> Any:
    """Resolve the exact source artifact."""

    logger.info(
        "Resolving candidate artifact: %s",
        artifact_ref,
    )

    try:
        artifact = api.artifact(artifact_ref)
    except Exception as exc:
        raise RuntimeError(
            f"Could not resolve candidate artifact '{artifact_ref}'."
        ) from exc

    if artifact.is_link:
        raise RuntimeError(
            f"Candidate artifact '{artifact_ref}' is already a linked "
            "Registry artifact. Promotion requires the original source "
            "artifact."
        )

    if not artifact.digest:
        raise RuntimeError(
            f"Candidate artifact '{artifact_ref}' does not expose a digest."
        )

    logger.info(
        "Candidate artifact resolved successfully."
    )
    logger.info(
        "Artifact: %s",
        artifact.qualified_name,
    )
    logger.info(
        "Digest: %s",
        artifact.digest,
    )

    return artifact


def verify_artifact(
    artifact: Any,
    verification_command: list[str] | None,
) -> None:
    """
    Verify the candidate artifact.

    The artifact must first be resolved by exact version and digest.
    An optional project-specific verification command can then perform
    model evaluation.
    """

    logger.info("=== MODEL VERIFICATION ===")

    logger.info(
        "Verifying artifact: %s",
        artifact.qualified_name,
    )

    logger.info(
        "Artifact digest: %s",
        artifact.digest,
    )

    # Verify the artifact manifest/content integrity.
    try:
        verified = artifact.verify()
    except Exception as exc:
        raise RuntimeError(
            f"W&B artifact integrity verification failed for "
            f"'{artifact.qualified_name}'."
        ) from exc

    if verified is False:
        raise RuntimeError(
            f"W&B artifact integrity verification failed for "
            f"'{artifact.qualified_name}'."
        )

    logger.info("W&B artifact integrity verification: PASSED")

    if verification_command:
        logger.info(
            "Running model verification command: %s",
            " ".join(verification_command),
        )

        try:
            subprocess.run(
                verification_command,
                check=True,
            )
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(
                "Model verification command FAILED. "
                "Artifact will not be promoted."
            ) from exc

        logger.info("Model verification command: PASSED")

    logger.info("MODEL VERIFICATION: PASSED")
    logger.info("==========================")


def promote_artifact(
    artifact: Any,
    registry_name: str,
    registry_collection: str,
    production_alias: str,
) -> tuple[Any, str]:
    """
    Link the exact verified artifact to the W&B Model Registry.

    Returns:
        linked_artifact:
            The Registry-linked artifact.
        registry_ref:
            Immutable Registry reference such as:
            wandb-registry-Model/AskBeforeAnswer-Models:v17
    """

    target_path = (
        f"wandb-registry-{registry_name}/{registry_collection}"
    )

    logger.info(
        "Promoting verified artifact to Registry collection: %s",
        target_path,
    )

    logger.info(
        "Assigning Registry alias: %s",
        production_alias,
    )

    try:
        linked_artifact = artifact.link(
            target_path=target_path,
            aliases=[production_alias],
        )
    except Exception as exc:
        raise RuntimeError(
            f"Failed to promote artifact '{artifact.qualified_name}' "
            f"to '{target_path}'."
        ) from exc

    registry_name_with_version = linked_artifact.name

    # W&B's linked artifact name should contain a concrete version.
    if ":" not in registry_name_with_version:
        raise RuntimeError(
            "W&B returned a Registry artifact without an explicit "
            f"version: {registry_name_with_version}"
        )

    registry_ref = (
        f"{target_path}:{registry_name_with_version.rsplit(':', 1)[1]}"
    )

    logger.info(
        "Registry artifact created: %s",
        registry_ref,
    )

    logger.info(
        "Production alias assigned to verified artifact."
    )

    return linked_artifact, registry_ref


def write_promotion_provenance(
    path: str,
    source_artifact: Any,
    registry_artifact: Any,
    source_artifact_ref: str,
    registry_ref: str,
    registry_name: str,
    registry_collection: str,
    production_alias: str,
) -> Path:
    """Write an immutable record of the promotion operation."""

    output_path = Path(path)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    provenance = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "source_artifact_ref": source_artifact_ref,
        "source_artifact_digest": source_artifact.digest,
        "source_artifact_name": source_artifact.name,
        "source_artifact_qualified_name": (
            source_artifact.qualified_name
        ),
        "registry_name": registry_name,
        "registry_collection": registry_collection,
        "registry_alias": production_alias,
        "registry_artifact_ref": registry_ref,
        "registry_artifact_name": registry_artifact.name,
        "registry_artifact_digest": registry_artifact.digest,
    }

    output_path.write_text(
        json.dumps(
            provenance,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    logger.info(
        "Wrote model promotion provenance: %s",
        output_path,
    )

    return output_path


def main() -> None:
    args = parse_args()

    wandb_entity = os.environ.get("WANDB_ENTITY")
    wandb_project = os.environ.get("WANDB_PROJECT")

    if not wandb_entity:
        raise RuntimeError(
            "WANDB_ENTITY environment variable is not set."
        )

    if not wandb_project:
        raise RuntimeError(
            "WANDB_PROJECT environment variable is not set."
        )

    validate_artifact_ref(args.artifact_ref)

    logger.info("==========================================")
    logger.info("AskBeforeAnswer Model Promotion")
    logger.info("==========================================")
    logger.info("W&B entity: %s", wandb_entity)
    logger.info("W&B project: %s", wandb_project)
    logger.info("Candidate artifact: %s", args.artifact_ref)
    logger.info("Registry: %s", args.registry_name)
    logger.info(
        "Collection: %s",
        args.registry_collection,
    )
    logger.info(
        "Production alias: %s",
        args.production_alias,
    )
    logger.info("==========================================")

    api = wandb.Api()

    # ---------------------------------------------------------------
    # 1. Resolve the exact candidate artifact.
    # ---------------------------------------------------------------

    artifact = resolve_artifact(
        api=api,
        artifact_ref=args.artifact_ref,
    )

    # Capture the digest BEFORE doing anything with the Registry.
    source_digest = artifact.digest

    # ---------------------------------------------------------------
    # 2. Verify the candidate artifact.
    # ---------------------------------------------------------------

    verify_artifact(
        artifact=artifact,
        verification_command=args.verification_command,
    )

    # Ensure the object we promote is still the exact artifact we
    # originally resolved.
    if artifact.digest != source_digest:
        raise RuntimeError(
            "Candidate artifact digest changed during verification. "
            "Refusing to promote."
        )

    # ---------------------------------------------------------------
    # 3. Promote the exact verified artifact.
    # ---------------------------------------------------------------

    linked_artifact, registry_ref = promote_artifact(
        artifact=artifact,
        registry_name=args.registry_name,
        registry_collection=args.registry_collection,
        production_alias=args.production_alias,
    )

    # ---------------------------------------------------------------
    # 4. Verify Registry promotion.
    # ---------------------------------------------------------------

    logger.info("=== VERIFYING PROMOTION ===")

    promoted_digest = linked_artifact.digest

    logger.info(
        "Source digest:   %s",
        source_digest,
    )
    logger.info(
        "Registry digest: %s",
        promoted_digest,
    )

    if promoted_digest != source_digest:
        raise RuntimeError(
            "Registry promotion digest does not match the verified "
            "source artifact digest."
        )

    logger.info(
        "Registry artifact: %s",
        registry_ref,
    )

    logger.info(
        "Promotion verification: PASSED"
    )

    # ---------------------------------------------------------------
    # 5. Record exact promotion provenance.
    # ---------------------------------------------------------------

    write_promotion_provenance(
        path=args.provenance_file,
        source_artifact=artifact,
        registry_artifact=linked_artifact,
        source_artifact_ref=args.artifact_ref,
        registry_ref=registry_ref,
        registry_name=args.registry_name,
        registry_collection=args.registry_collection,
        production_alias=args.production_alias,
    )

    logger.info("==========================================")
    logger.info("MODEL PROMOTION SUCCESSFUL")
    logger.info("==========================================")
    logger.info(
        "Verified source artifact: %s",
        args.artifact_ref,
    )
    logger.info(
        "Artifact digest: %s",
        source_digest,
    )
    logger.info(
        "Registry artifact: %s",
        registry_ref,
    )
    logger.info(
        "Production alias: %s",
        args.production_alias,
    )
    logger.info("==========================================")


if __name__ == "__main__":
    main()
