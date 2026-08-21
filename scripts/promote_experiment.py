#!/usr/bin/env python3
"""Promote a selected DVC experiment into the current Git workspace.

The promotion process:
1. Resolve the named DVC experiment.
2. Inspect the experiment's dvc.lock metadata.
3. Verify that the requested stage/output matches the experiment.
4. Verify that the expected model artifact exists locally.
5. Update only the selected model's parameters in params.yaml while
   preserving comments and existing YAML structure.
6. Use DVC to apply the experiment's pipeline state, including dvc.lock.
7. Verify the resulting DVC stage.
8. Show the promotion diff.
9. Commit only the intended tracked files.

Important:
- params.yaml is edited with ruamel.yaml so comments are preserved.
- dvc.lock is NEVER parsed and re-serialized by this script.
- Untracked files are deliberately left untouched.
"""

from __future__ import annotations

import argparse
import copy
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedMap

ROOT = Path(__file__).resolve().parents[1]
PARAMS_FILE = ROOT / "params.yaml"

MODEL_CONFIG = {
    "sft": {
        "stage": "train_sft",
        "output": "models/sft/final",
        "params_path": ("training", "sft"),
    },
    "dpo": {
        "stage": "train_dpo",
        "output": "models/dpo/final",
        "params_path": ("training", "dpo"),
    },
    "grpo": {
        "stage": "train_grpo",
        "output": "models/grpo/final",
        "params_path": ("training", "grpo"),
    },
    "orpo": {
        "stage": "train_orpo",
        "output": "models/orpo/final",
        "params_path": ("training", "orpo"),
    },
}


def run(
    command: list[str],
    *,
    check: bool = True,
    capture_output: bool = True,
) -> subprocess.CompletedProcess[str]:
    """Run a command from the repository root."""
    return subprocess.run(
        command,
        cwd=ROOT,
        text=True,
        check=check,
        capture_output=capture_output,
    )


def git_output(*args: str) -> str:
    """Run a Git command and return stdout."""
    result = run(["git", *args])
    return result.stdout.strip()


def dvc_output(*args: str) -> str:
    """Run a DVC command and return stdout."""
    result = run(["dvc", *args])
    return result.stdout.strip()


def fail(message: str) -> None:
    """Print an error and exit."""
    print(f"ERROR: {message}", file=sys.stderr)
    raise SystemExit(1)


def get_experiment_ref(experiment: str) -> str:
    """Resolve a DVC experiment name to its Git ref."""
    result = run(
        [
            "git",
            "for-each-ref",
            "--format=%(refname)",
            "refs/exps",
        ]
    )

    refs = [
        line.strip()
        for line in result.stdout.splitlines()
        if line.strip().endswith(f"/{experiment}")
    ]

    if not refs:
        fail(f"Could not resolve DVC experiment '{experiment}'.")

    if len(refs) > 1:
        fail(
            f"Multiple experiment refs found for '{experiment}':\n"
            + "\n".join(refs)
        )

    return refs[0]


def get_experiment_sha(experiment_ref: str) -> str:
    """Resolve an experiment ref to its Git commit SHA."""
    result = run(
        [
            "git",
            "rev-parse",
            experiment_ref,
        ]
    )
    return result.stdout.strip()


def get_experiment_dvc_lock(experiment_sha: str) -> dict[str, Any]:
    """Read dvc.lock from the experiment without modifying it."""
    result = run(
        [
            "git",
            "show",
            f"{experiment_sha}:dvc.lock",
        ]
    )

    yaml = YAML(typ="safe")
    data = yaml.load(result.stdout)

    if not isinstance(data, dict):
        fail("Experiment dvc.lock does not contain a valid YAML mapping.")

    return data


def get_nested_value(
    data: dict[str, Any],
    path: tuple[str, ...],
) -> Any:
    """Get a nested YAML value."""
    current: Any = data

    for key in path:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]

    return current


def verify_experiment_metadata(
    lock_data: dict[str, Any],
    *,
    stage: str,
    output: str,
) -> dict[str, Any]:
    """Verify that the experiment contains the requested stage/output."""
    stages = lock_data.get("stages")

    if not isinstance(stages, dict):
        fail("Experiment dvc.lock does not contain a 'stages' mapping.")

    stage_data = stages.get(stage)

    if not isinstance(stage_data, dict):
        fail(
            f"Experiment does not contain expected stage '{stage}'."
        )

    outputs = stage_data.get("outs", [])

    if not isinstance(outputs, list):
        fail(f"Stage '{stage}' has an invalid 'outs' section.")

    output_entry = None

    for entry in outputs:
        if isinstance(entry, dict) and output in entry:
            output_entry = entry
            break

    if output_entry is None:
        fail(
            f"Stage '{stage}' does not contain expected output "
            f"'{output}'."
        )

    return stage_data


def get_output_hash(
    stage_data: dict[str, Any],
    output: str,
) -> str | None:
    """Return the DVC hash for a stage output, if available."""
    outputs = stage_data.get("outs", [])

    if not isinstance(outputs, list):
        return None

    for entry in outputs:
        if not isinstance(entry, dict):
            continue

        metadata = entry.get(output)

        if isinstance(metadata, dict):
            hash_type = metadata.get("hash", "md5")
            digest = metadata.get(hash_type)

            if digest:
                return f"{digest}.{hash_type}" if hash_type != "md5" else digest

    return None


def load_params_yaml() -> YAML:
    """Create a ruamel YAML instance configured for round-trip editing."""
    yaml = YAML()
    yaml.preserve_quotes = True
    yaml.indent(mapping=2, sequence=4, offset=2)
    yaml.width = 4096
    yaml.default_flow_style = False
    return yaml


def update_model_params(
    *,
    model: str,
    learning_rate: float,
) -> bool:
    """Update only the selected model learning rate in params.yaml.

    Returns True when the file was changed.
    """
    if not PARAMS_FILE.exists():
        fail(f"Missing {PARAMS_FILE}.")

    yaml = load_params_yaml()

    with PARAMS_FILE.open("r", encoding="utf-8") as file:
        params = yaml.load(file)

    if not isinstance(params, CommentedMap):
        fail("params.yaml must contain a YAML mapping.")

    config = MODEL_CONFIG[model]
    training = params.get("training")

    if not isinstance(training, CommentedMap):
        fail("params.yaml does not contain a 'training' mapping.")

    model_params = training.get(model)

    if not isinstance(model_params, CommentedMap):
        fail(
            f"params.yaml does not contain "
            f"'training.{model}' mapping."
        )

    old_value = model_params.get("learning_rate")

    # Use a normal Python float so ruamel writes the value as a scalar
    # without introducing additional YAML structure.
    new_value = float(learning_rate)

    if old_value == new_value:
        return False

    model_params["learning_rate"] = new_value

    with PARAMS_FILE.open("w", encoding="utf-8") as file:
        yaml.dump(params, file)

    return True


def check_tracked_worktree_clean() -> None:
    """Ensure no tracked files have uncommitted changes.

    Untracked files are intentionally ignored.
    """
    status = git_output("status", "--porcelain")

    tracked_changes = []

    for line in status.splitlines():
        if not line:
            continue

        # Git porcelain format:
        # XY path
        #
        # For untracked files it is "??".
        if line[:2] != "??":
            tracked_changes.append(line)

    if tracked_changes:
        print("Tracked working tree is not clean:")
        print("\n".join(tracked_changes))
        fail(
            "Commit or stash tracked changes before promoting "
            "an experiment."
        )

    print("Tracked working tree is clean.")

    if status:
        print("Untracked files will be left untouched.")


def get_experiment_learning_rate(
    stage_data: dict[str, Any],
    model: str,
) -> float | None:
    """Extract the experiment's learning rate from dvc.lock."""
    params = stage_data.get("params")

    if not isinstance(params, dict):
        return None

    params_yaml = params.get("params.yaml")

    if not isinstance(params_yaml, dict):
        return None

    key = f"training.{model}.learning_rate"
    value = params_yaml.get(key)

    if value is None:
        return None

    try:
        return float(value)
    except (TypeError, ValueError):
        fail(
            f"Invalid learning rate in experiment dvc.lock: {value!r}"
        )

    return None


def apply_experiment(experiment: str) -> None:
    """Apply the DVC experiment to the workspace.

    This is intentionally delegated to DVC. DVC updates dvc.lock and
    restores the experiment state without us re-serializing dvc.lock.
    """
    result = run(
        [
            "dvc",
            "exp",
            "apply",
            experiment,
        ],
        check=False,
    )

    if result.returncode != 0:
        print(result.stdout, end="")
        print(result.stderr, end="", file=sys.stderr)
        fail(
            f"Failed to apply DVC experiment '{experiment}'."
        )

    if result.stdout:
        print(result.stdout, end="")


def verify_dvc_stage(stage: str) -> None:
    """Verify the resulting DVC stage."""
    result = run(
        [
            "dvc",
            "status",
            stage,
        ],
        check=False,
    )

    if result.returncode != 0:
        print(result.stdout, end="")
        print(result.stderr, end="", file=sys.stderr)
        fail(
            f"DVC status failed for stage '{stage}'."
        )

    output = result.stdout.strip()

    if output:
        print(output)
    else:
        print("DVC verification successful.")


def show_promotion_diff() -> str:
    """Return the tracked Git diff after promotion."""
    return git_output("diff", "--", "params.yaml", "dvc.lock")


def stage_and_commit(
    *,
    model: str,
    experiment: str,
) -> str | None:
    """Commit only params.yaml and dvc.lock if they changed."""
    diff = show_promotion_diff()

    if not diff:
        print("Promotion diff:")
        print("No metadata changes were produced by the promotion.")
        print("Nothing to commit.")
        return None

    print("Promotion diff:")
    print(diff)

    # Stage only the two files that promotion is allowed to modify.
    run(
        [
            "git",
            "add",
            "--",
            "params.yaml",
            "dvc.lock",
        ]
    )

    staged = git_output(
        "diff",
        "--cached",
        "--name-only",
    )

    allowed = {"params.yaml", "dvc.lock"}
    staged_files = {
        line.strip()
        for line in staged.splitlines()
        if line.strip()
    }

    unexpected = staged_files - allowed

    if unexpected:
        run(["git", "reset"])
        fail(
            "Promotion attempted to stage unexpected files:\n"
            + "\n".join(sorted(unexpected))
        )

    if not staged_files:
        print("Nothing to commit.")
        return None

    commit_message = (
        f"dvc: promote {model} experiment {experiment}"
    )

    print("Creating Git commit:")
    print(f"  {commit_message}")

    run(
        [
            "git",
            "commit",
            "-m",
            commit_message,
        ],
        capture_output=False,
    )

    return git_output("rev-parse", "--short", "HEAD")


def promote(
    *,
    model: str,
    experiment: str,
) -> None:
    """Promote a DVC experiment."""
    config = MODEL_CONFIG[model]
    stage = config["stage"]
    output = config["output"]

    print("=" * 58)
    print()
    print("Promoting DVC experiment")
    print()
    print("=" * 58)
    print()
    print(f"Model:       {model}")
    print(f"Experiment:  {experiment}")
    print()
    print("=" * 58)
    print()

    print("Checking tracked working tree...")
    check_tracked_worktree_clean()
    print()

    print(f"Resolving experiment {experiment}...")

    experiment_ref = get_experiment_ref(experiment)
    experiment_sha = get_experiment_sha(experiment_ref)

    print(f"Experiment ref: {experiment_ref}")
    print(f"Experiment SHA: {experiment_sha}")

    current_head = git_output("rev-parse", "HEAD")
    print(f"Current HEAD:   {current_head}")
    print()

    print("Reading experiment metadata...")

    lock_data = get_experiment_dvc_lock(experiment_sha)

    stage_data = verify_experiment_metadata(
        lock_data,
        stage=stage,
        output=output,
    )

    experiment_learning_rate = get_experiment_learning_rate(
        stage_data,
        model,
    )

    print(f"Experiment stage:  {stage}")
    print(f"Experiment output: {output}")

    output_hash = get_output_hash(
        stage_data,
        output,
    )

    if output_hash:
        print(f"Experiment hash:   {output_hash}")
    else:
        print("Experiment hash:   unavailable")

    if experiment_learning_rate is not None:
        print(
            "Experiment learning rate: "
            f"{experiment_learning_rate:.17g}"
        )

    print()
    print("Checking local model artifact...")

    local_output = ROOT / output

    if not local_output.exists():
        fail(
            f"Expected local model artifact does not exist: "
            f"{local_output}"
        )

    print(f"  Local output exists: {output}")

    if output_hash:
        print(f"  Expected DVC hash:   {output_hash}")

    print()
    print(f"Promoting stage '{stage}'...")
    print()

    # DVC is responsible for restoring the experiment's pipeline state.
    # This updates dvc.lock without us reformatting the entire file.
    apply_experiment(experiment)

    print()
    print(f"Promoting parameters under 'training.{model}'...")

    if experiment_learning_rate is None:
        fail(
            f"Experiment dvc.lock does not contain "
            f"training.{model}.learning_rate."
        )

    update_model_params(
        model=model,
        learning_rate=experiment_learning_rate,
    )

    print()
    print("Verifying promoted DVC stage...")
    verify_dvc_stage(stage)

    print()
    commit = stage_and_commit(
        model=model,
        experiment=experiment,
    )

    print()
    print("=" * 58)
    print()
    print("Promotion successful")
    print()
    print("=" * 58)
    print()
    print(f"Model:       {model}")
    print(f"Experiment:  {experiment}")
    print(f"Stage:       {stage}")
    print(f"Output:      {output}")

    if commit:
        print(f"Commit:      {commit}")
    else:
        print("Commit:      none")

    print()
    print("=" * 58)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Promote a DVC experiment into the current workspace."
    )

    parser.add_argument(
        "--model",
        required=True,
        choices=sorted(MODEL_CONFIG),
        help="Model/training algorithm to promote.",
    )

    parser.add_argument(
        "--experiment",
        required=True,
        help="DVC experiment name to promote.",
    )

    return parser.parse_args()


def main() -> None:
    """CLI entry point."""
    args = parse_args()

    promote(
        model=args.model,
        experiment=args.experiment,
    )


if __name__ == "__main__":
    main()
