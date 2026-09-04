#!/usr/bin/env python3

"""
Publish a promoted DVC model artifact to Weights & Biases.

Workflow:

    DVC experiment
        |
        v
    models/<model>/final
        |
        v
    W&B project artifact
        |
        v
    W&B Registry collection
        |
        v
    provenance/model_promotion.json

The Registry artifact is the immutable deployment artifact.

Important:

    - Model files are uploaded directly from the local DVC output.
    - No W&B Weave references are created.
    - The source project artifact is linked to the W&B Registry.
    - The Registry artifact's own digest is stored in provenance.
    - The source artifact digest is stored separately for provenance.
    - The provenance artifact_ref is ALWAYS the fully qualified
      W&B Registry reference:

          wandb-registry-<registry_name>/<collection>:<version>

    - push_to_hub.py must verify the Registry artifact against
      artifact_digest, never against the source artifact digest.

    - Publication is idempotent. If the exact DVC experiment has
      already been published and the recorded Registry artifact still
      passes integrity verification, the script exits successfully
      without creating another W&B run, artifact version, or Registry
      link.
"""

from __future__ import annotations

import json
import logging
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import hydra
import wandb
from omegaconf import DictConfig, OmegaConf

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s][%(name)s][%(levelname)s] - %(message)s",
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PROMOTION_FILE = "provenance/model_promotion.json"
DEFAULT_WANDB_ENTITY = "rl4aa"
DEFAULT_WANDB_PROJECT = "ask-before-answer"
DEFAULT_REGISTRY_ENTITY = "rl4aa-org"
DEFAULT_REGISTRY_NAME = "Model"
DEFAULT_REGISTRY_COLLECTION = "AskBeforeAnswer-Models"
DEFAULT_REGISTRY_ALIAS = "production"
MODEL_ARTIFACT_PREFIX = "Clarifier"


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------


def run_command(
    command: list[str],
    *,
    cwd: Path | None = None,
) -> str:
    """Run a command and return stdout."""
    logger.debug(
        "Running command: %s",
        " ".join(command),
    )

    result = subprocess.run(
        command,
        cwd=str(cwd) if cwd else None,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed ({result.returncode}): "
            f"{' '.join(command)}\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )

    return result.stdout.strip()


def project_root(cfg: DictConfig) -> Path:
    """Return the project root."""
    return Path(
        OmegaConf.select(
            cfg,
            "project_dir",
            default=".",
        )
    ).resolve()


def get_wandb_config(cfg: DictConfig) -> dict[str, str]:
    """
    Read W&B configuration with safe fallbacks.

    This deliberately does not require cfg.wandb to exist.
    """
    wandb_cfg = OmegaConf.select(
        cfg,
        "wandb",
        default=None,
    )

    if wandb_cfg is None:
        return {
            "entity": DEFAULT_WANDB_ENTITY,
            "project": DEFAULT_WANDB_PROJECT,
            "registry_entity": DEFAULT_REGISTRY_ENTITY,
            "registry_name": DEFAULT_REGISTRY_NAME,
            "registry_collection": DEFAULT_REGISTRY_COLLECTION,
            "registry_alias": DEFAULT_REGISTRY_ALIAS,
        }

    return {
        "entity": str(
            OmegaConf.select(
                cfg,
                "wandb.entity",
                default=DEFAULT_WANDB_ENTITY,
            )
        ),
        "project": str(
            OmegaConf.select(
                cfg,
                "wandb.project",
                default=DEFAULT_WANDB_PROJECT,
            )
        ),
        "registry_entity": str(
            OmegaConf.select(
                cfg,
                "wandb.registry_entity",
                default=DEFAULT_REGISTRY_ENTITY,
            )
        ),
        "registry_name": str(
            OmegaConf.select(
                cfg,
                "wandb.registry_name",
                default=DEFAULT_REGISTRY_NAME,
            )
        ),
        "registry_collection": str(
            OmegaConf.select(
                cfg,
                "wandb.registry_collection",
                default=DEFAULT_REGISTRY_COLLECTION,
            )
        ),
        "registry_alias": str(
            OmegaConf.select(
                cfg,
                "wandb.registry_alias",
                default=DEFAULT_REGISTRY_ALIAS,
            )
        ),
    }


# ---------------------------------------------------------------------------
# W&B Registry reference helpers
# ---------------------------------------------------------------------------


def build_registry_ref(
    *,
    registry_name: str,
    registry_collection: str,
    version: str,
) -> str:
    """
    Construct the fully qualified W&B Registry artifact reference.

    Example:

        wandb-registry-Model/AskBeforeAnswer-Models:v0

    This fully qualified namespace is important. A short reference such as

        AskBeforeAnswer-Models:v0

    can be resolved ambiguously by the W&B API.
    """
    return f"wandb-registry-{registry_name}/{registry_collection}:{version}"


def build_registry_alias_ref(
    *,
    registry_name: str,
    registry_collection: str,
    registry_alias: str,
) -> str:
    """
    Construct the fully qualified W&B Registry alias reference.

    Example:

        wandb-registry-Model/AskBeforeAnswer-Models:production
    """
    return f"wandb-registry-{registry_name}/{registry_collection}:{registry_alias}"


def get_git_commit() -> str | None:
    """Return the current Git commit when available."""

    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None

    return result.stdout.strip() or None


# ---------------------------------------------------------------------------
# DVC
# ---------------------------------------------------------------------------


def validate_model_output(
    cfg: DictConfig,
    model_variant: str,
) -> Path:
    """Validate the local DVC model output."""
    root = project_root(cfg)

    model_path = root / "models" / model_variant / "final"

    if not model_path.is_dir():
        raise RuntimeError(f"DVC model output does not exist: {model_path}")

    files = [path for path in model_path.rglob("*") if path.is_file()]

    if not files:
        raise RuntimeError(f"DVC model output is empty: {model_path}")

    logger.info(
        "Validated DVC model output: %s",
        model_path,
    )

    logger.info(
        "Model contains %d files.",
        len(files),
    )

    return model_path


def resolve_dvc_experiment_sha(
    cfg: DictConfig,
    experiment: str,
) -> str:
    """
    Resolve a DVC experiment name to its Git SHA.

    DVC versions differ in the structure of `dvc exp list` and
    `dvc exp show` output. The most reliable representation for
    the installed DVC version is the Git experiment ref:

        refs/exps/<...>/<experiment>

    We therefore inspect Git refs first and fall back to DVC output.
    """
    root = project_root(cfg)

    logger.info(
        "Resolving DVC experiment: %s",
        experiment,
    )

    if experiment.upper() == "HEAD":
        logger.info("Using current Git HEAD as experiment SHA.")
        commit = get_git_commit()
        if not commit:
            raise RuntimeError("Could not resolve Git HEAD. Is this a Git repository?")
        return commit

    # ------------------------------------------------------------------
    # Method 1: Git refs
    # ------------------------------------------------------------------

    try:
        output = run_command(
            [
                "git",
                "for-each-ref",
                "--format=%(refname) %(objectname)",
                "refs/exps",
            ],
            cwd=root,
        )

        for line in output.splitlines():
            parts = line.split(maxsplit=1)

            if len(parts) != 2:
                continue

            ref_name, sha = parts

            if ref_name.endswith(f"/{experiment}"):
                logger.info(
                    "Resolved DVC experiment '%s' through Git ref:",
                    experiment,
                )

                logger.info(
                    "  Git ref: %s",
                    ref_name,
                )

                logger.info(
                    "  SHA:     %s",
                    sha,
                )

                return sha

    except RuntimeError:
        pass

    # ------------------------------------------------------------------
    # Method 2: dvc exp list --json
    # ------------------------------------------------------------------

    try:
        output = run_command(
            [
                "dvc",
                "exp",
                "list",
                "--json",
            ],
            cwd=root,
        )

        data = json.loads(output)
        experiments = data.get("experiments", {})

        if isinstance(experiments, dict):
            value = experiments.get(experiment)

            if isinstance(value, str):
                logger.info(
                    "DVC experiment SHA: %s",
                    value,
                )
                return value

            if isinstance(value, dict):
                for key in (
                    "sha",
                    "rev",
                    "baseline_rev",
                    "commit",
                ):
                    candidate = value.get(key)

                    if isinstance(candidate, str) and candidate:
                        logger.info(
                            "DVC experiment SHA: %s",
                            candidate,
                        )
                        return candidate

        if isinstance(experiments, list):
            for item in experiments:
                if not isinstance(item, dict):
                    continue

                name = (
                    item.get("name") or item.get("experiment") or item.get("exp_name")
                )

                if name != experiment:
                    continue

                for key in (
                    "sha",
                    "rev",
                    "baseline_rev",
                    "commit",
                ):
                    candidate = item.get(key)

                    if isinstance(candidate, str) and candidate:
                        logger.info(
                            "DVC experiment SHA: %s",
                            candidate,
                        )
                        return candidate

    except (
        RuntimeError,
        json.JSONDecodeError,
    ):
        pass

    # ------------------------------------------------------------------
    # Method 3: dvc exp show --json
    # ------------------------------------------------------------------

    try:
        output = run_command(
            [
                "dvc",
                "exp",
                "show",
                "--json",
            ],
            cwd=root,
        )

        data = json.loads(output)

        def search_json(
            obj: Any,
        ) -> str | None:
            if isinstance(obj, dict):
                if experiment in obj:
                    value = obj[experiment]

                    if isinstance(value, str):
                        return value

                    if isinstance(value, dict):
                        for key in (
                            "sha",
                            "rev",
                            "baseline_rev",
                            "commit",
                        ):
                            candidate = value.get(key)

                            if isinstance(candidate, str) and candidate:
                                return candidate

                name = obj.get("name") or obj.get("experiment") or obj.get("exp_name")

                if name == experiment:
                    for key in (
                        "sha",
                        "rev",
                        "baseline_rev",
                        "commit",
                    ):
                        candidate = obj.get(key)

                        if isinstance(candidate, str) and candidate:
                            return candidate

                for value in obj.values():
                    result = search_json(value)

                    if result:
                        return result

            elif isinstance(obj, list):
                for value in obj:
                    result = search_json(value)

                    if result:
                        return result

            return None

        sha = search_json(data)

        if sha:
            logger.info(
                "DVC experiment SHA: %s",
                sha,
            )
            return sha

    except (
        RuntimeError,
        json.JSONDecodeError,
    ):
        pass

    # ------------------------------------------------------------------
    # Method 4: dvc exp list
    # ------------------------------------------------------------------

    try:
        output = run_command(
            [
                "dvc",
                "exp",
                "list",
            ],
            cwd=root,
        )

        for line in output.splitlines():
            if experiment not in line:
                continue

            tokens = line.split()

            for token in tokens:
                if len(token) >= 7 and all(
                    char in "0123456789abcdef" for char in token.lower()
                ):
                    logger.info(
                        "DVC experiment SHA: %s",
                        token,
                    )
                    return token

    except RuntimeError:
        pass

    raise RuntimeError(
        f"Could not determine DVC SHA for experiment "
        f"'{experiment}'. "
        "Verify it with 'dvc exp list' and ensure the "
        "experiment exists locally."
    )


# ---------------------------------------------------------------------------
# W&B source artifact
# ---------------------------------------------------------------------------


def create_source_artifact(
    *,
    entity: str,
    project: str,
    model_variant: str,
    model_path: Path,
    dvc_experiment: str,
    dvc_sha: str,
) -> tuple[str, str, str]:
    """
    Create or reuse the project-level W&B artifact.

    Returns:

        source_artifact_ref
        source_artifact_digest
        run_id

    The source artifact digest is deliberately kept separate from
    the Registry artifact digest.
    """
    artifact_name = f"{MODEL_ARTIFACT_PREFIX}-{model_variant}"

    logger.info(
        "Creating W&B artifact: %s",
        artifact_name,
    )

    run = wandb.init(
        entity=entity,
        project=project,
        job_type="model-promotion",
        name=f"promote-{model_variant}-{dvc_experiment}",
        config={
            "model_variant": model_variant,
            "dvc_experiment": dvc_experiment,
            "dvc_experiment_sha": dvc_sha,
        },
    )

    try:
        artifact = wandb.Artifact(
            name=artifact_name,
            type="model",
            metadata={
                "model_variant": model_variant,
                "dvc_experiment": dvc_experiment,
                "dvc_experiment_sha": dvc_sha,
            },
        )

        logger.info("Adding DVC model output to W&B artifact...")

        artifact.add_dir(
            local_path=str(model_path),
        )

        logger.info(
            "Logging W&B artifact '%s'...",
            artifact_name,
        )

        logged_artifact = run.log_artifact(
            artifact,
            aliases=[
                f"dvc-{dvc_experiment}",
            ],
        )

        logger.info("Waiting for W&B artifact upload to complete...")

        try:
            logged_artifact.wait()
        except AttributeError:
            pass

        # --------------------------------------------------------------
        # Resolve the actual project artifact explicitly.
        # --------------------------------------------------------------

        source_latest_ref = f"{entity}/{project}/{artifact_name}:latest"

        logger.info(
            "Resolving project artifact: %s",
            source_latest_ref,
        )

        api = wandb.Api()

        resolved = api.artifact(
            source_latest_ref,
            type="model",
        )

        if not resolved.version:
            raise RuntimeError(
                "W&B project artifact resolved successfully, "
                "but no immutable artifact version was returned."
            )

        source_ref = f"{entity}/{project}/{artifact_name}:{resolved.version}"

        source_digest = resolved.digest

        if not source_digest:
            raise RuntimeError(
                "W&B project artifact resolved successfully, "
                "but no source artifact digest was returned."
            )

        logger.info(
            "Source W&B artifact: %s",
            source_ref,
        )

        logger.info(
            "Source W&B artifact digest: %s",
            source_digest,
        )

        return (
            source_ref,
            source_digest,
            run.id,
        )

    finally:
        run.finish()


# ---------------------------------------------------------------------------
# W&B Registry
# ---------------------------------------------------------------------------


def link_to_registry(
    *,
    source_artifact_ref: str,
    entity: str,
    registry_entity: str,
    registry_name: str,
    registry_collection: str,
    registry_alias: str,
) -> tuple[str, str]:
    """
    Link the exact source artifact to the W&B Registry.

    Returns:

        fully-qualified immutable Registry artifact reference
        Registry artifact digest

    CRITICAL:

        The returned digest belongs to the Registry artifact.

        The returned reference is fully qualified:

            wandb-registry-<name>/<collection>:<version>

        It must NOT be shortened to:

            <collection>:<version>

    The fully-qualified reference is what downstream deployment code
    must use to prevent W&B from resolving an unrelated artifact.
    """
    logger.info(
        "Linking artifact to W&B Registry collection: %s",
        registry_collection,
    )

    api = wandb.Api()

    # ------------------------------------------------------------------
    # Resolve exact source artifact.
    # ------------------------------------------------------------------

    source_artifact = api.artifact(
        source_artifact_ref,
        type="model",
    )

    logger.info(
        "Source artifact resolved: %s",
        source_artifact_ref,
    )

    logger.info(
        "Source artifact digest: %s",
        source_artifact.digest,
    )

    if not source_artifact.digest:
        raise RuntimeError("Source W&B artifact has no digest.")

    # ------------------------------------------------------------------
    # Link source artifact to Registry.
    # ------------------------------------------------------------------

    # W&B registries exist in the organization's entity, not the personal user's entity.
    # We explicitly prepend the org entity (registry_entity)
    # and the special registry prefix.
    registry_project = f"wandb-registry-{registry_name}"

    target_path = f"{registry_entity}/{registry_project}/{registry_collection}"
    registry_base = f"{registry_entity}/{registry_project}/{registry_collection}"

    logger.info("Registry target path: %s", target_path)

    # 1. Proactively detach alias from old version (if it exists)
    try:
        old_artifact = api.artifact(f"{registry_base}:{registry_alias}", type="model")
        if old_artifact.digest != source_artifact.digest:
            logger.info(
                f"Proactively detaching alias '{registry_alias}' "
                f"from older Registry artifact '{old_artifact.version}'..."
            )
            old_artifact.aliases.remove(registry_alias)
            old_artifact.save()
    except Exception:
        pass

    # 2. Perform the link WITHOUT aliases because W&B link() is buggy at applying them
    source_artifact.link(target_path=target_path)
    logger.info("Artifact linked to W&B Registry collection.")

    # Add a short delay to ensure W&B backend has indexed the new link
    import time

    time.sleep(2)

    # ------------------------------------------------------------------
    # Explicitly attach alias to the newly created v0 registry artifact
    # ------------------------------------------------------------------
    # Since this is a dedicated collection, we KNOW this model is v0.
    v0_ref = f"{registry_base}:v0"
    logger.info(f"Explicitly resolving {v0_ref} to apply alias...")

    resolved_alias = api.artifact(v0_ref, type="model")

    if registry_alias not in resolved_alias.aliases:
        logger.info(f"Appending '{registry_alias}' to {v0_ref} and saving...")
        resolved_alias.aliases.append(registry_alias)
        resolved_alias.save()
        time.sleep(1)  # wait for save to propagate

    if not resolved_alias.version:
        raise RuntimeError(
            "W&B Registry alias resolved successfully, "
            "but no immutable version was returned."
        )

    if not resolved_alias.digest:
        raise RuntimeError(
            "W&B Registry alias resolved successfully, "
            "but no Registry digest was returned."
        )

    # ------------------------------------------------------------------
    # Construct the FULLY QUALIFIED immutable Registry reference.
    # ------------------------------------------------------------------

    registry_ref = build_registry_ref(
        registry_name=registry_name,
        registry_collection=registry_collection,
        version=resolved_alias.version,
    )

    registry_digest = resolved_alias.digest

    logger.info(
        "W&B Registry artifact: %s",
        registry_ref,
    )

    logger.info(
        "W&B Registry digest: %s",
        registry_digest,
    )

    # ------------------------------------------------------------------
    # IMPORTANT:
    #
    # Re-resolve the exact immutable reference we are about to persist.
    #
    # This guarantees that:
    #
    #     provenance artifact_ref
    #
    # and
    #
    #     provenance artifact_digest
    #
    # refer to the same W&B object that downstream deployment will
    # resolve.
    # ------------------------------------------------------------------

    logger.info(
        "Verifying immutable Registry artifact: %s",
        registry_ref,
    )

    verified = api.artifact(
        registry_ref,
        type="model",
    )

    verified_digest = verified.digest

    if not verified_digest:
        raise RuntimeError(
            "Immutable Registry artifact resolved successfully, "
            "but no digest was returned during verification."
        )

    logger.info(
        "Verified Registry digest: %s",
        verified_digest,
    )

    if verified_digest != registry_digest:
        raise RuntimeError(
            "W&B Registry integrity check failed immediately after "
            "linking. The alias resolved to digest "
            f"'{registry_digest}', but immutable reference "
            f"'{registry_ref}' resolved to '{verified_digest}'."
        )

    logger.info("Registry artifact reference and digest verified.")

    return registry_ref, verified_digest


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------


def verify_existing_promotion(
    cfg: DictConfig,
    *,
    model_variant: str,
    stage: str,
    experiment: str,
    dvc_sha: str,
    entity: str,
    registry_name: str,
    registry_collection: str,
    registry_alias: str,
) -> Path | None:
    """
    Check whether this exact DVC promotion has already been completed.

    Returns:

        The existing provenance path if a valid identical promotion exists.
        None if no provenance file exists.

    Raises:

        RuntimeError:

            If provenance exists but does not describe the requested
            promotion, or if the recorded Registry artifact fails
            integrity verification.
    """
    root = project_root(cfg)
    provenance_path = root / PROMOTION_FILE

    if not provenance_path.exists():
        return None

    logger.info(
        "Existing promotion provenance found: %s",
        provenance_path,
    )

    try:
        record = json.loads(provenance_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"Could not read existing promotion provenance: {provenance_path}"
        ) from exc

    # ------------------------------------------------------------------
    # Verify that the provenance belongs to this exact promotion request.
    # ------------------------------------------------------------------

    expected = {
        "model_variant": model_variant,
        "dvc_stage": stage,
        "dvc_experiment": experiment,
        "dvc_experiment_sha": dvc_sha,
    }

    mismatches: list[str] = []
    source_record = record.get("source", {})

    for key, expected_value in expected.items():
        actual_value = source_record.get(key)

        if actual_value != expected_value:
            mismatches.append(
                f"{key}: expected '{expected_value}', found '{actual_value}'"
            )

    if mismatches:
        raise RuntimeError(
            "Promotion provenance already exists, but it does not "
            "match the requested promotion:\n"
            + "\n".join(f"  - {mismatch}" for mismatch in mismatches)
            + "\nRefusing to overwrite existing provenance."
        )

    # ------------------------------------------------------------------
    # Extract the canonical immutable Registry reference and digest.
    # ------------------------------------------------------------------

    registry_record = record.get("registry", {})
    registry_artifact_ref = registry_record.get("artifact_ref")
    registry_artifact_digest = registry_record.get("digest")

    if not registry_artifact_ref:
        raise RuntimeError(
            "Existing promotion provenance does not contain "
            "'artifact_ref' in registry block."
        )

    if not registry_artifact_digest:
        raise RuntimeError(
            "Existing promotion provenance does not contain 'digest' in registry block."
        )

    expected_prefix = f"wandb-registry-{registry_name}/{registry_collection}:"

    if not registry_artifact_ref.startswith(expected_prefix):
        raise RuntimeError(
            "Existing promotion provenance contains an unexpected "
            f"Registry artifact reference:\n"
            f"  {registry_artifact_ref}\n"
            f"Expected prefix:\n"
            f"  {expected_prefix}"
        )

    # ------------------------------------------------------------------
    # Verify the immutable Registry artifact.
    # ------------------------------------------------------------------

    logger.info(
        "Verifying existing immutable Registry artifact: %s",
        registry_artifact_ref,
    )

    api = wandb.Api()

    try:
        registry_artifact = api.artifact(
            registry_artifact_ref,
            type="model",
        )
    except Exception as exc:
        raise RuntimeError(
            "Existing promotion provenance references a Registry "
            f"artifact that could not be resolved:\n"
            f"  {registry_artifact_ref}"
        ) from exc

    actual_digest = registry_artifact.digest

    if not actual_digest:
        raise RuntimeError(
            f"Existing Registry artifact has no digest:\n  {registry_artifact_ref}"
        )

    logger.info(
        "Recorded Registry digest: %s",
        registry_artifact_digest,
    )

    logger.info(
        "Resolved Registry digest: %s",
        actual_digest,
    )

    if actual_digest != registry_artifact_digest:
        raise RuntimeError(
            "Existing promotion provenance failed its Registry "
            "integrity check.\n"
            f"  Recorded digest: {registry_artifact_digest}\n"
            f"  Actual digest:   {actual_digest}"
        )

    # ------------------------------------------------------------------
    # Verify that the production alias still points to the same artifact.
    # ------------------------------------------------------------------

    registry_alias_ref = build_registry_alias_ref(
        registry_name=registry_name,
        registry_collection=registry_collection,
        registry_alias=registry_alias,
    )

    logger.info(
        "Verifying production alias: %s",
        registry_alias_ref,
    )

    try:
        aliased_artifact = api.artifact(
            registry_alias_ref,
            type="model",
        )
    except Exception as exc:
        raise RuntimeError(
            "Existing promotion provenance is valid, but the "
            f"'{registry_alias}' Registry alias could not be resolved."
        ) from exc

    alias_digest = aliased_artifact.digest

    if not alias_digest:
        raise RuntimeError(
            f"Registry alias '{registry_alias}' resolved without a digest."
        )

    logger.info(
        "Production alias digest: %s",
        alias_digest,
    )

    if alias_digest != registry_artifact_digest:
        raise RuntimeError(
            "The production alias no longer points to the artifact "
            "recorded in promotion provenance.\n"
            f"  Provenance artifact: {registry_artifact_ref}\n"
            f"  Provenance digest:   {registry_artifact_digest}\n"
            f"  Alias digest:        {alias_digest}\n"
            "\n"
            "Refusing to silently republish or overwrite the "
            "existing promotion."
        )

    logger.info("Existing promotion verified successfully.")

    return provenance_path


def write_promotion_provenance(
    cfg: DictConfig,
    *,
    model_variant: str,
    stage: str,
    experiment: str,
    dvc_sha: str,
    entity: str,
    project: str,
    source_artifact_ref: str,
    source_artifact_digest: str,
    registry_artifact_ref: str,
    registry_artifact_digest: str,
    artifact_name: str,
    registry_name: str,
    registry_collection: str,
    registry_alias: str,
    run_id: str,
) -> Path:
    """
    Write the deployment provenance record.

    CRITICAL SCHEMA RULE:

        artifact_ref
            = fully qualified immutable W&B Registry reference

        artifact_digest
            = immutable W&B Registry artifact digest

    The source project artifact reference and digest are retained
    separately under wandb.source_* fields.

    push_to_hub.py consumes artifact_ref and artifact_digest directly.
    """
    root = project_root(cfg)

    provenance_dir = root / "provenance"

    provenance_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    path = provenance_dir / "model_promotion.json"

    now = datetime.now(timezone.utc).isoformat()

    record = {
        "timestamp": now,
        "operation": "model_promotion",
        "status": "verified",
        "source": {
            "artifact_ref": source_artifact_ref,
            "artifact_name": artifact_name,
            "qualified_name": f"{entity}/{project}/{artifact_name}:latest",
            "digest": source_artifact_digest,
            # Preserve DVC metadata from publish_model_artifact
            "model_variant": model_variant,
            "dvc_stage": stage,
            "dvc_experiment": experiment,
            "dvc_experiment_sha": dvc_sha,
        },
        "registry": {
            "name": registry_name,
            "collection": registry_collection,
            "alias": registry_alias,
            "artifact_ref": registry_artifact_ref,
            "artifact_name": artifact_name,
            "digest": registry_artifact_digest,
        },
        "verification": {
            "artifact_integrity": True,
            "verification_command": "make publish-model-artifact",
            "digest_match": (source_artifact_digest == registry_artifact_digest),
        },
        "git": {
            "commit": get_git_commit(),
        },
    }

    # ------------------------------------------------------------------
    # Final provenance sanity checks.
    # ------------------------------------------------------------------

    if not registry_artifact_ref.startswith(f"wandb-registry-{registry_name}/"):
        raise RuntimeError(
            "Refusing to write provenance because the Registry "
            "artifact reference is not fully qualified: "
            f"{registry_artifact_ref}"
        )

    if not registry_artifact_digest:
        raise RuntimeError(
            "Refusing to write provenance because the Registry "
            "artifact digest is empty."
        )

    path.write_text(
        json.dumps(
            record,
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    logger.info(
        "Wrote promotion provenance: %s",
        path,
    )

    return path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


@hydra.main(
    version_base="1.3",
    config_path="../configs",
    config_name="config",
)
def main(cfg: DictConfig) -> None:
    """Publish a DVC experiment model to W&B Registry."""

    model_variant = str(
        OmegaConf.select(
            cfg,
            "publication_model",
            default="sft",
        )
    )

    experiment = str(
        OmegaConf.select(
            cfg,
            "experiment",
            default="",
        )
    )

    stage = str(
        OmegaConf.select(
            cfg,
            "stage",
            default="",
        )
    )

    if not experiment:
        raise RuntimeError(
            "No DVC experiment was supplied. Use experiment=<experiment-name>."
        )

    if not stage:
        raise RuntimeError("No DVC stage was supplied. Use stage=<stage-name>.")

    wandb_cfg = get_wandb_config(cfg)

    entity = wandb_cfg["entity"]
    project = wandb_cfg["project"]
    registry_name = wandb_cfg["registry_name"]
    registry_collection = wandb_cfg["registry_collection"]
    registry_alias = wandb_cfg["registry_alias"]

    artifact_name = f"{MODEL_ARTIFACT_PREFIX}-{model_variant}"

    logger.info("=" * 58)
    logger.info("W&B MODEL ARTIFACT PUBLICATION")
    logger.info("=" * 58)

    logger.info(
        "Model:              %s",
        model_variant,
    )

    logger.info(
        "Experiment:         %s",
        experiment,
    )

    logger.info(
        "Stage:              %s",
        stage,
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
        "Registry:           %s",
        registry_name,
    )

    logger.info(
        "Registry collection: %s",
        registry_collection,
    )

    logger.info(
        "Registry alias:     %s",
        registry_alias,
    )

    logger.info("=" * 58)

    # ------------------------------------------------------------------
    # 1. Validate local DVC output.
    # ------------------------------------------------------------------

    model_path = validate_model_output(
        cfg,
        model_variant,
    )

    # ------------------------------------------------------------------
    # 2. Resolve DVC experiment SHA.
    # ------------------------------------------------------------------

    dvc_sha = resolve_dvc_experiment_sha(
        cfg,
        experiment,
    )

    logger.info(
        "DVC experiment SHA: %s",
        dvc_sha,
    )

    # ------------------------------------------------------------------
    # 3. Check for an existing identical promotion.
    #
    # This makes publication idempotent. If the exact DVC experiment
    # has already been promoted and its Registry artifact remains
    # valid, do not create another W&B run, artifact version, or
    # Registry link.
    # ------------------------------------------------------------------

    existing_provenance = verify_existing_promotion(
        cfg,
        model_variant=model_variant,
        stage=stage,
        experiment=experiment,
        dvc_sha=dvc_sha,
        entity=entity,
        registry_name=registry_name,
        registry_collection=registry_collection,
        registry_alias=registry_alias,
    )

    if existing_provenance is not None:
        logger.info("=" * 58)
        logger.info("W&B MODEL ARTIFACT ALREADY PUBLISHED")
        logger.info("=" * 58)

        logger.info(
            "Model:              %s",
            model_variant,
        )

        logger.info(
            "DVC experiment:     %s",
            experiment,
        )

        logger.info(
            "DVC SHA:            %s",
            dvc_sha,
        )

        logger.info(
            "Provenance file:    %s",
            existing_provenance,
        )

        logger.info("No new W&B artifact or Registry version was created.")

        logger.info("Existing promotion is valid and remains the deployment target.")

        logger.info("=" * 58)

        return

    # ------------------------------------------------------------------
    # 4. Create/reuse project-level W&B artifact.
    #
    # This is the block that must exist after the idempotency check.
    #
    # It produces the three values needed by the Registry and
    # provenance stages:
    #
    #     source_artifact_ref
    #     source_artifact_digest
    #     run_id
    # ------------------------------------------------------------------

    (
        source_artifact_ref,
        source_artifact_digest,
        run_id,
    ) = create_source_artifact(
        entity=entity,
        project=project,
        model_variant=model_variant,
        model_path=model_path,
        dvc_experiment=experiment,
        dvc_sha=dvc_sha,
    )

    # ------------------------------------------------------------------
    # 5. Link exact source artifact to W&B Registry.
    #
    # This returns the FULLY QUALIFIED Registry artifact reference
    # and the Registry artifact's own digest.
    # ------------------------------------------------------------------

    (
        registry_artifact_ref,
        registry_artifact_digest,
    ) = link_to_registry(
        source_artifact_ref=source_artifact_ref,
        entity=entity,
        registry_entity=wandb_cfg["registry_entity"],
        registry_name=registry_name,
        registry_collection=registry_collection,
        registry_alias=registry_alias,
    )

    # ------------------------------------------------------------------
    # 6. Write immutable promotion provenance.
    #
    # IMPORTANT:
    #
    # artifact_digest receives registry_artifact_digest.
    #
    # artifact_ref receives the FULLY QUALIFIED Registry reference.
    #
    # Neither value comes from the source project artifact.
    # ------------------------------------------------------------------

    provenance_path = write_promotion_provenance(
        cfg,
        model_variant=model_variant,
        stage=stage,
        experiment=experiment,
        dvc_sha=dvc_sha,
        entity=entity,
        project=project,
        source_artifact_ref=source_artifact_ref,
        source_artifact_digest=source_artifact_digest,
        registry_artifact_ref=registry_artifact_ref,
        registry_artifact_digest=registry_artifact_digest,
        artifact_name=artifact_name,
        registry_name=registry_name,
        registry_collection=registry_collection,
        registry_alias=registry_alias,
        run_id=run_id,
    )

    # ------------------------------------------------------------------
    # 7. Final summary.
    # ------------------------------------------------------------------

    logger.info("=" * 58)
    logger.info("W&B MODEL PROMOTION SUCCESSFUL")
    logger.info("=" * 58)

    logger.info(
        "Model:              %s",
        model_variant,
    )

    logger.info(
        "DVC experiment:     %s",
        experiment,
    )

    logger.info(
        "DVC SHA:            %s",
        dvc_sha,
    )

    logger.info(
        "Source artifact:    %s",
        source_artifact_ref,
    )

    logger.info(
        "Source digest:      %s",
        source_artifact_digest,
    )

    logger.info(
        "Registry artifact:  %s",
        registry_artifact_ref,
    )

    logger.info(
        "Artifact digest:    %s",
        registry_artifact_digest,
    )

    logger.info(
        "Production alias:   %s",
        registry_alias,
    )

    logger.info(
        "Provenance file:    %s",
        provenance_path,
    )

    logger.info("=" * 58)


if __name__ == "__main__":
    main()
