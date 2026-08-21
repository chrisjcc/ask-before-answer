#!/usr/bin/env python3
"""Promote a DVC experiment into the current working tree.

The promotion process:
1. Resolve a DVC experiment name to its Git ref/SHA.
2. Read the experiment's dvc.lock.
3. Identify the requested DVC stage and output.
4. Verify that the corresponding local artifact exists.
5. Update only the selected model's parameters in params.yaml.
6. Preserve comments and existing YAML structure using ruamel.yaml.
7. Verify the resulting DVC stage.
8. Commit only the intended params.yaml change.

Example:
    python scripts/promote_experiment.py \
        --model sft \
        --experiment sweep_epl5w24i
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Any

from ruamel.yaml import YAML


MODEL_CONFIG = {
    "sft": {
        "stage": "train_sft",
        "output": "models/sft/final",
        "param_key": "training.sft.learning_rate",
        "param_path": ("training", "sft"),
        "parameter": "learning_rate",
    },
    "dpo": {
        "stage": "train_dpo",
        "output": "models/dpo/final",
        "param_key": "training.dpo.learning_rate",
        "param_path": ("training", "dpo"),
        "parameter": "learning_rate",
    },
    "grpo": {
        "stage": "train_grpo",
        "output": "models/grpo/final",
        "param_key": "training.grpo.learning_rate",
        "param_path": ("training", "grpo"),
        "parameter": "learning_rate",
    },
    "orpo": {
        "stage": "train_orpo",
        "output": "models/orpo/final",
        "param_key": "training.orpo.learning_rate",
        "param_path": ("training", "orpo"),
        "parameter": "learning_rate",
    },
}


ROOT = Path(__file__).resolve().parents[1]
DVC_LOCK = ROOT / "dvc.lock"
PARAMS_FILE = ROOT / "params.yaml"


def run_git(*args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    """Run a git command from the repository root."""
    return subprocess.run(
        ["git", *args],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=check,
    )


def run_dvc(*args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    """Run a dvc command from the repository root."""
    return subprocess.run(
        ["dvc", *args],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=check,
    )


def print_header(title: str) -> None:
    """Print a consistent section header."""
    print("=" * 58)
    print(title)
    print("=" * 58)


def get_git_status() -> str:
    """Return short Git status output."""
    result = run_git("status", "--short")
    return result.stdout.strip()


def ensure_tracked_worktree_clean() -> None:
    """Ensure tracked files have no modifications.

    Untracked files are intentionally allowed because model training,
    W&B, logs, caches, and other generated artifacts may exist locally.
    """
    status = get_git_status()

    if not status:
        print("Tracked working tree is clean.")
        return

    tracked_changes: list[str] = []

    for line in status.splitlines():
        if len(line) < 3:
            continue

        index_status = line[0]
        worktree_status = line[1]

        # Ignore purely untracked files: "?? filename"
        if index_status == "?" and worktree_status == "?":
            continue

        tracked_changes.append(line)

    if tracked_changes:
        print("ERROR: Tracked working tree is not clean.")
        print()
        print("\n".join(tracked_changes))
        print()
        print("Commit or stash your existing tracked changes before promoting.")
        raise SystemExit(1)

    print("Tracked working tree is clean.")
    print("Untracked files will be left untouched.")


def resolve_experiment(experiment: str) -> tuple[str, str]:
    """Resolve an experiment name to its ref and commit SHA."""
    result = run_git(
        "for-each-ref",
        "--format=%(refname) %(objectname)",
        "refs/exps",
    )

    matches = []

    for line in result.stdout.splitlines():
        if line.endswith(f"/{experiment}"):
            ref, sha = line.split(maxsplit=1)
            matches.append((ref, sha))

    if not matches:
        raise RuntimeError(
            f"Could not resolve DVC experiment '{experiment}'."
        )

    if len(matches) > 1:
        refs = "\n".join(ref for ref, _ in matches)
        raise RuntimeError(
            f"Multiple DVC experiment refs found for '{experiment}':\n{refs}"
        )

    return matches[0]


def read_experiment_lock(experiment_sha: str) -> str:
    """Read dvc.lock from the experiment commit."""
    result = run_git("show", f"{experiment_sha}:dvc.lock")
    return result.stdout


def load_yaml(text: str) -> Any:
    """Load YAML using ruamel.yaml round-trip mode."""
    yaml = YAML()
    yaml.preserve_quotes = True
    yaml.width = 120
    yaml.indent(mapping=2, sequence=4, offset=2)

    return yaml.load(text)


def load_params() -> tuple[YAML, Any]:
    """Load params.yaml while preserving comments and structure."""
    yaml = YAML()
    yaml.preserve_quotes = True
    yaml.width = 120
    yaml.indent(mapping=2, sequence=4, offset=2)

    with PARAMS_FILE.open("r", encoding="utf-8") as handle:
        data = yaml.load(handle)

    return yaml, data


def get_stage_metadata(
    experiment_lock: Any,
    stage_name: str,
    expected_output: str,
) -> tuple[Any, str | None]:
    """Extract stage metadata and the promoted parameter value."""
    stages = experiment_lock.get("stages")

    if not stages or stage_name not in stages:
        raise RuntimeError(
            f"Stage '{stage_name}' was not found in the experiment dvc.lock."
        )

    stage = stages[stage_name]

    outputs = stage.get("outs", [])
    output_entry = None

    for entry in outputs:
        if isinstance(entry, dict) and expected_output in entry:
            output_entry = entry[expected_output]
            break

        if isinstance(entry, str) and entry == expected_output:
            output_entry = {}
            break

    if output_entry is None:
        raise RuntimeError(
            f"Output '{expected_output}' was not found in stage "
            f"'{stage_name}'."
        )

    params = stage.get("params", {})
    params_file = params.get("params.yaml", {})

    learning_rate = params_file.get("training.sft.learning_rate")

    return stage, learning_rate


def extract_stage_parameter(
    stage: Any,
    param_key: str,
) -> Any:
    """Extract a parameter value recorded by DVC for a stage."""
    params = stage.get("params", {})
    params_file = params.get("params.yaml", {})

    if param_key not in params_file:
        raise RuntimeError(
            f"Parameter '{param_key}' was not recorded in the experiment "
            "dvc.lock."
        )

    return params_file[param_key]


def update_nested_parameter(
    data: Any,
    path: tuple[str, ...],
    parameter_name: str,
    value: Any,
) -> None:
    """Update a nested YAML parameter without replacing its parent mapping."""
    current = data

    for key in path:
        if key not in current:
            raise RuntimeError(
                f"Expected YAML section '{key}' was not found in params.yaml."
            )

        current = current[key]

    if parameter_name not in current:
        print(
            f"Warning: adding missing parameter "
            f"{'.'.join((*path, parameter_name))}."
        )

    current[parameter_name] = value


def write_params(yaml: YAML, data: Any) -> None:
    """Write params.yaml using ruamel.yaml round-trip serialization."""
    with PARAMS_FILE.open("w", encoding="utf-8") as handle:
        yaml.dump(data, handle)


def verify_dvc_stage(stage_name: str) -> None:
    """Verify that DVC considers the promoted stage clean."""
    result = run_dvc("status", stage_name, check=False)

    if result.returncode != 0:
        print(
            f"ERROR: DVC status failed for stage '{stage_name}':",
            file=sys.stderr,
        )
        print(result.stderr or result.stdout, file=sys.stderr)
        raise SystemExit(1)

    output = result.stdout.strip()

    if output and "Data and pipelines are up to date." not in output:
        print("DVC verification output:")
        print(output)

        raise RuntimeError(
            f"DVC stage '{stage_name}' is not clean after promotion."
        )

    print("DVC verification successful.")


def git_diff() -> str:
    """Return the current Git diff."""
    result = run_git("diff")
    return result.stdout


def commit_params(experiment: str, model: str) -> str | None:
    """Commit params.yaml if the promotion changed it."""
    diff = git_diff()

    if not diff.strip():
        print("No metadata changes were produced by the promotion.")
        print("Nothing to commit.")
        return None

    print("Promotion diff:")
    print(diff)

    run_git("add", "params.yaml")

    commit_message = f"dvc: promote {model} experiment {experiment}"

    print()
    print("Files staged for promotion commit:")
    print("params.yaml")

    print()
    print("Creating Git commit:")
    print(f"  {commit_message}")

    result = run_git("commit", "-m", commit_message)

    print(result.stdout)

    sha = run_git("rev-parse", "--short", "HEAD").stdout.strip()

    return sha


def promote(model: str, experiment: str) -> None:
    """Promote a DVC experiment."""
    if model not in MODEL_CONFIG:
        valid = ", ".join(MODEL_CONFIG)
        raise ValueError(
            f"Unsupported model '{model}'. Valid models: {valid}"
        )

    config = MODEL_CONFIG[model]

    stage_name = config["stage"]
    expected_output = config["output"]
    param_path = config["param_path"]

    print_header("Promoting DVC experiment")
    print()
    print(f"Model:       {model}")
    print(f"Experiment:  {experiment}")
    print()
    print_header("")

    print(f"Stage:       {stage_name}")
    print(f"Output:      {expected_output}")
    print("=" * 58)

    print("Checking tracked working tree...")
    ensure_tracked_worktree_clean()
    print()

    print(f"Resolving experiment {experiment}...")

    experiment_ref, experiment_sha = resolve_experiment(experiment)

    print(f"Experiment ref: {experiment_ref}")
    print(f"Experiment SHA: {experiment_sha}")

    current_head = run_git("rev-parse", "HEAD").stdout.strip()
    print(f"Current HEAD:   {current_head}")
    print()

    print("Reading experiment metadata...")

    lock_text = read_experiment_lock(experiment_sha)
    experiment_lock = load_yaml(lock_text)

    stage, _ = get_stage_metadata(
        experiment_lock,
        stage_name,
        expected_output,
    )

    print(f"Experiment stage:  {stage_name}")
    print(f"Experiment output: {expected_output}")

    output_entry = next(
        entry[expected_output]
        for entry in stage["outs"]
        if isinstance(entry, dict) and expected_output in entry
    )

    experiment_hash = output_entry.get("md5")

    if experiment_hash:
        print(f"Experiment hash:   {experiment_hash}")
    else:
        print("Experiment hash:   unavailable")

    print()
    print("Checking local model artifact...")

    local_output = ROOT / expected_output

    if not local_output.exists():
        raise RuntimeError(
            f"Local output does not exist: {expected_output}"
        )

    print(f"  Local output exists: {expected_output}")

    if experiment_hash:
        print(f"  Expected DVC hash:   {experiment_hash}")

    print()
    print(f"Promoting stage '{stage_name}'...")

    param_key = ".".join((*param_path, "learning_rate"))

    promoted_value = extract_stage_parameter(
        stage,
        config["param_key"]
    )

    print(f"Promoting parameters under '{'.'.join(param_path)}'...")
    print(f"  learning_rate: {promoted_value}")

    yaml, params = load_params()

    if params is None:
        raise RuntimeError("params.yaml is empty.")

    update_nested_parameter(
        params,
        config["param_path"],
        config["parameter"],
        promoted_value,
    )

    write_params(yaml, params)

    print("Verifying promoted DVC stage...")
    verify_dvc_stage(stage_name)

    print()
    commit_sha = commit_params(experiment, model)

    print()
    print("=" * 58)
    print("Promotion successful")
    print("=" * 58)
    print(f"Model:       {model}")
    print(f"Experiment:  {experiment}")
    print(f"Stage:       {stage_name}")
    print(f"Output:      {expected_output}")

    if commit_sha:
        print(f"Commit:      {commit_sha}")
    else:
        print("Commit:      none (already promoted)")

    print("=" * 58)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Promote a DVC experiment."
    )

    parser.add_argument(
        "--model",
        required=True,
        choices=sorted(MODEL_CONFIG),
        help="Model/training stage to promote.",
    )

    parser.add_argument(
        "--experiment",
        required=True,
        help="DVC experiment name.",
    )

    return parser.parse_args()


def main() -> None:
    """CLI entry point."""
    args = parse_args()

    try:
        promote(args.model, args.experiment)
    except subprocess.CalledProcessError as exc:
        print(
            f"ERROR: command failed: {' '.join(exc.cmd)}",
            file=sys.stderr,
        )
        if exc.stdout:
            print(exc.stdout, file=sys.stderr)
        if exc.stderr:
            print(exc.stderr, file=sys.stderr)
        raise SystemExit(exc.returncode) from exc
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
