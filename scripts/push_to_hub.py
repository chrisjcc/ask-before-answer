import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any

import hydra
import wandb
from datasets import DatasetDict, load_dataset
from dotenv import load_dotenv
from huggingface_hub import HfApi
from omegaconf import DictConfig

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL_PATHS = {
    "base": "Qwen/Qwen2.5-7B-Instruct",
    "sft_only": "models/sft_only/final",
    "dpo_only": "models/dpo_only/final",
    "sft": "models/sft/final",
    "dpo": "models/dpo/final",
    "sft_dpo": "models/dpo/final",
    "grpo": "models/grpo/final",
    "orpo": "models/orpo/final",
}

PROMOTION_FILE = "model_promotion.json"


# ---------------------------------------------------------------------------
# Promotion record
# ---------------------------------------------------------------------------


def load_promotion_record(
    cfg: DictConfig,
) -> dict[str, Any]:
    """Load and validate the immutable model promotion record.

    The promotion record is produced by promote_model.py and is the
    deployment source of truth.

    push_to_hub.py deliberately does not resolve the W&B production alias.
    Instead, it consumes the exact artifact reference and digest recorded
    during promotion.
    """

    promotion_path = Path(cfg.project_dir) / "provenance"/ PROMOTION_FILE

    if not promotion_path.is_file():
        raise FileNotFoundError(
            f"Promotion record does not exist: {promotion_path}. "
            "Run promote_model.py before publishing to Hugging Face."
        )

    logger.info(
        "Loading promotion record: %s",
        promotion_path,
    )

    try:
        promotion = json.loads(
            promotion_path.read_text(encoding="utf-8")
        )
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"Invalid JSON in promotion record: {promotion_path}"
        ) from exc

    required_fields = (
        "artifact_ref",
        "artifact_digest",
        "model_variant",
    )

    missing = [
        field
        for field in required_fields
        if not promotion.get(field)
    ]

    if missing:
        raise RuntimeError(
            "Promotion record is incomplete. Missing required fields: "
            + ", ".join(missing)
        )

    artifact_ref = promotion["artifact_ref"]

    # A promotion record must contain an immutable artifact version.
    # Explicitly reject aliases as deployment references.
    rejected_aliases = {
        "production",
        "staging",
        "latest",
        "development",
        "dev",
    }

    if artifact_ref in rejected_aliases:
        raise RuntimeError(
            f"Promotion record contains mutable W&B alias '{artifact_ref}'. "
            "Deployment requires an exact versioned artifact reference."
        )

    if ":" not in artifact_ref:
        raise RuntimeError(
            f"Invalid W&B artifact reference '{artifact_ref}'. "
            "Expected a versioned reference containing ':', for example "
            "'wandb-registry-Model/AskBeforeAnswer-Models:v17'."
        )

    model_variant = promotion["model_variant"]

    if model_variant not in MODEL_PATHS:
        supported = ", ".join(sorted(MODEL_PATHS))
        raise RuntimeError(
            f"Unsupported model variant '{model_variant}' in promotion "
            f"record. Supported models: {supported}"
        )

    artifact_digest = promotion["artifact_digest"]

    logger.info(
        "Promotion record validated successfully."
    )
    logger.info(
        "Promoted model variant: %s",
        model_variant,
    )
    logger.info(
        "Promoted W&B artifact: %s",
        artifact_ref,
    )
    logger.info(
        "Promoted W&B digest: %s",
        artifact_digest,
    )

    return promotion


# ---------------------------------------------------------------------------
# W&B artifact resolution and verification
# ---------------------------------------------------------------------------


def resolve_and_verify_artifact(
    promotion: dict[str, Any],
) -> tuple[Any, str]:
    """Resolve the exact promoted W&B artifact and verify its digest.

    The artifact reference comes from model_promotion.json, not from a
    mutable registry alias.

    Returns:
        artifact: The exact W&B Artifact object.
        artifact_ref: The exact versioned artifact reference.
    """

    artifact_ref = promotion["artifact_ref"]
    expected_digest = promotion["artifact_digest"]

    wandb_entity = os.environ.get("WANDB_ENTITY")
    wandb_project = os.environ.get("WANDB_PROJECT")

    if not wandb_entity or not wandb_project:
        raise RuntimeError(
            "WANDB_ENTITY and WANDB_PROJECT must be set."
        )

    logger.info(
        "Resolving promoted W&B artifact: %s",
        artifact_ref,
    )

    try:
        wandb_api = wandb.Api()

        artifact = wandb_api.artifact(
            artifact_ref,
        )
    except Exception as exc:
        raise RuntimeError(
            f"Failed to resolve promoted W&B artifact "
            f"'{artifact_ref}'."
        ) from exc

    actual_digest = getattr(
        artifact,
        "digest",
        None,
    )

    if not actual_digest:
        raise RuntimeError(
            f"W&B artifact '{artifact_ref}' did not expose a digest. "
            "Deployment cannot verify artifact identity."
        )

    logger.info(
        "Expected W&B digest: %s",
        expected_digest,
    )
    logger.info(
        "Resolved W&B digest: %s",
        actual_digest,
    )

    if actual_digest != expected_digest:
        raise RuntimeError(
            "W&B artifact integrity check failed. "
            f"Promotion record expects digest '{expected_digest}', "
            f"but artifact '{artifact_ref}' resolved to "
            f"'{actual_digest}'. Deployment aborted."
        )

    logger.info(
        "W&B artifact digest verification PASSED."
    )

    return artifact, artifact_ref


# ---------------------------------------------------------------------------
# Dataset publication
# ---------------------------------------------------------------------------


def push_datasets(
    cfg: DictConfig,
) -> None:
    """Push SFT and DPO datasets to the Hugging Face dataset repository."""

    dataset_repo = cfg.deployment.dataset_repo
    data_dir = Path(cfg.data_dir)

    logger.info(
        "Loading datasets from %s...",
        data_dir,
    )

    required_files = (
        "sft_train.jsonl",
        "sft_val.jsonl",
        "dpo_train.jsonl",
        "dpo_val.jsonl",
    )

    for filename in required_files:
        path = data_dir / filename

        if not path.is_file():
            raise FileNotFoundError(
                f"Required dataset file does not exist: {path}"
            )

    # ------------------------------------------------------------------
    # SFT
    # ------------------------------------------------------------------

    sft_ds = DatasetDict(
        {
            "train": load_dataset(
                "json",
                data_files=str(
                    data_dir / "sft_train.jsonl"
                ),
                split="train",
            ),
            "validation": load_dataset(
                "json",
                data_files=str(
                    data_dir / "sft_val.jsonl"
                ),
                split="train",
            ),
        }
    )

    logger.info(
        "Pushing SFT dataset configuration to %s...",
        dataset_repo,
    )

    sft_ds.push_to_hub(
        dataset_repo,
        config_name="sft",
    )

    # ------------------------------------------------------------------
    # DPO
    # ------------------------------------------------------------------

    dpo_ds = DatasetDict(
        {
            "train": load_dataset(
                "json",
                data_files=str(
                    data_dir / "dpo_train.jsonl"
                ),
                split="train",
            ),
            "validation": load_dataset(
                "json",
                data_files=str(
                    data_dir / "dpo_val.jsonl"
                ),
                split="train",
            ),
        }
    )

    logger.info(
        "Pushing DPO dataset configuration to %s...",
        dataset_repo,
    )

    dpo_ds.push_to_hub(
        dataset_repo,
        config_name="dpo",
    )

    logger.info("Dataset upload complete.")


# ---------------------------------------------------------------------------
# Model card
# ---------------------------------------------------------------------------


def generate_model_card(
    cfg: DictConfig,
    promotion: dict[str, Any],
) -> str:
    """Generate the Hugging Face model card from the promotion record."""

    dataset_repo = cfg.deployment.dataset_repo
    model_repo = cfg.deployment.model_repo

    model_variant = promotion["model_variant"]
    artifact_ref = promotion["artifact_ref"]
    artifact_digest = promotion["artifact_digest"]

    registry_alias = promotion.get(
        "registry_alias",
        cfg.deployment.get(
            "registry_alias",
            "production",
        ),
    )

    release_tag = cfg.deployment.get(
        "release_tag",
        "",
    )

    release_text = (
        f"- **Release:** `{release_tag}`\n"
        if release_tag
        else ""
    )

    training_descriptions = {
        "sft_only": "Supervised Fine-Tuning (SFT)",
        "dpo_only": "Direct Preference Optimization (DPO)",
        "sft": "Supervised Fine-Tuning (SFT)",
        "dpo": "Direct Preference Optimization (DPO)",
        "sft_dpo": (
            "Supervised Fine-Tuning followed by "
            "Direct Preference Optimization"
        ),
        "grpo": "Group Relative Policy Optimization (GRPO)",
        "orpo": "Odds Ratio Preference Optimization (ORPO)",
    }

    training_method = training_descriptions.get(
        model_variant,
        model_variant.upper(),
    )

    leaderboard_content = ""

    leaderboard_path = (
        Path(cfg.project_dir)
        / "results"
        / "leaderboard.md"
    )

    if leaderboard_path.is_file():
        leaderboard_content = leaderboard_path.read_text(
            encoding="utf-8"
        ).strip()

    return f"""---
language:
  - en
license: apache-2.0
library_name: transformers
pipeline_tag: text-generation
base_model:
  - Qwen/Qwen2.5-7B-Instruct
datasets:
  - {dataset_repo}
tags:
  - clarification
  - ambiguity-detection
  - question-answering
  - qwen2.5
  - {model_variant}
  - reinforcement-learning
---

# AskBeforeAnswer

The **AskBeforeAnswer** model is a clarification-seeking language model
based on **Qwen2.5-7B-Instruct**.

Instead of immediately answering an ambiguous question, the model is trained
to determine whether clarification is required and, when necessary, identify
the missing information and ask a targeted clarification question.

## Production Model

This repository contains the exact W&B artifact approved by the project's
model promotion procedure.

- **Training method:** {training_method}
- **Model variant:** `{model_variant}`
- **W&B Registry alias at promotion:** `{registry_alias}`
- **W&B artifact:** `{artifact_ref}`
- **W&B artifact digest:** `{artifact_digest}`
{release_text}- **Base model:** `Qwen/Qwen2.5-7B-Instruct`
- **Dataset:** `{dataset_repo}`

The model was obtained directly from the immutable W&B artifact recorded in
the promotion record.

The DVC-managed training output is not modified during deployment.

## Behavior

The model produces structured responses containing:

1. **Action**: whether to `Clarify` or `Answer`
2. **Reasoning**: why clarification is or is not required
3. **Facets**: missing information required to disambiguate the question
4. **Response**: either a clarification question or a direct answer

Expected format:

```text
Action: Clarify|Answer
Reasoning: <reasoning>
Facets: <list of missing facets>
Response: <clarifying question or direct answer>
```

## Training

The production winner was selected through the project's experiment
selection, verification, and promotion procedure.

The training pipeline supports:

- Supervised Fine-Tuning (SFT)
- Direct Preference Optimization (DPO)
- Group Relative Policy Optimization (GRPO)
- Odds Ratio Preference Optimization (ORPO)

This repository corresponds specifically to the model variant recorded in
the promotion record.

## Evaluation

{
leaderboard_content
if leaderboard_content
else
"Evaluation results are maintained in the project evaluation artifacts."
}

## Usage
```
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base_model_name = "Qwen/Qwen2.5-7B-Instruct"
adapter_model_name = "{model_repo}"

tokenizer = AutoTokenizer.from_pretrained(
    base_model_name,
)

model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
)

model = PeftModel.from_pretrained(
    model,
    adapter_model_name,
)
```

## Dataset

The training datasets are available from:

{dataset_repo}

The repository contains separate sft and dpo configurations.

## Intended Use

This model is intended for research into clarification-seeking behavior in
language models, particularly for systems that should distinguish between
answerable and underspecified user questions.

## Limitations

The model may incorrectly classify questions as ambiguous or unambiguous.

Its generated reasoning and answers should not be treated as authoritative.

The model was trained on English-language data and may not generalize
reliably to other languages or domains.

## Reproducibility

The model published here was selected through the project's experiment
selection, verification, and promotion procedure.

The promotion record identifies the exact W&B artifact version and digest
used for deployment.

The local DVC training artifact is treated as immutable during deployment.
"""


# ---------------------------------------------------------------------------
# Model publication
# ---------------------------------------------------------------------------


def push_model(
    cfg: DictConfig,
    api: HfApi,
    artifact: Any,
    promotion: dict[str, Any],
) -> None:
    """Download the promoted W&B artifact and publish it to Hugging Face.
    The artifact is downloaded into a temporary staging directory.

    README.md is added only to that temporary staging directory.

    The DVC training artifact and W&B artifact are never modified.
    """

    model_repo = cfg.deployment.model_repo

    logger.info(
        "Creating/checking Hugging Face model repository: %s",
        model_repo,
    )

    api.create_repo(
        repo_id=model_repo,
        repo_type="model",
        exist_ok=True,
    )

    model_variant = promotion["model_variant"]
    artifact_ref = promotion["artifact_ref"]

    with tempfile.TemporaryDirectory(
        prefix="ask-before-answer-hf-",
    ) as tmp_dir:
        staging_dir = Path(tmp_dir) / "model"

        logger.info(
            "Downloading exact promoted W&B artifact '%s' "
            "into temporary staging directory...",
            artifact_ref,
        )

        downloaded_dir = Path(
            artifact.download(
                root=str(staging_dir),
            )
        )

        if not downloaded_dir.is_dir():
            raise RuntimeError(
                "W&B artifact download did not produce a valid "
                f"directory: {downloaded_dir}"
            )

        logger.info(
            "Generating Hugging Face model card for '%s'...",
            model_variant,
        )

        readme_path = staging_dir / "README.md"

        readme_path.write_text(
            generate_model_card(
                cfg,
                promotion,
            ),
            encoding="utf-8",
        )

        logger.info(
            "Uploading verified W&B artifact to %s...",
            model_repo,
        )

        api.upload_folder(
            folder_path=str(staging_dir),
            repo_id=model_repo,
            repo_type="model",
        )

    logger.info(
        "Model upload complete: %s",
        model_repo,
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

@hydra.main(
    version_base="1.3",
    config_path="../configs",
    config_name="config",
)
def main(cfg: DictConfig) -> None:
    """Publish the promoted production model to Hugging Face."""
    hf_token = os.environ.get("HF_TOKEN")

    if not hf_token:
        raise RuntimeError(
            "HF_TOKEN environment variable is not set."
        )

    logger.info(
        "Starting Hugging Face deployment."
    )

    api = HfApi(
        token=hf_token,
    )

    # ------------------------------------------------------------------
    # 1. Load promotion record.
    # ------------------------------------------------------------------

    promotion = load_promotion_record(cfg)

    # ------------------------------------------------------------------
    # 2. Resolve the exact W&B artifact and verify its digest.
    # ------------------------------------------------------------------

    artifact, _ = resolve_and_verify_artifact(
        promotion,
    )

    # ------------------------------------------------------------------
    # 3. Publish datasets.
    #
    # Dataset publication is independent of model artifact resolution.
    # ------------------------------------------------------------------

    push_datasets(cfg)

    # ------------------------------------------------------------------
    # 4. Publish the exact promoted W&B artifact.
    #
    # IMPORTANT:
    #
    # We deliberately do NOT:
    #
    #   - resolve the mutable production alias
    #   - select a local DVC model directory
    #   - modify models/<variant>/final/
    #
    # The promotion record identifies the deployment artifact.
    # ------------------------------------------------------------------

    push_model(
        cfg,
        api,
        artifact,
        promotion,
     )

    logger.info(
        "🚀 Successfully published promoted model '%s' "
        "(W&B artifact '%s') to Hugging Face.",
        promotion["model_variant"],
        promotion["artifact_ref"],
    )


if __name__ == "__main__":
   main()
