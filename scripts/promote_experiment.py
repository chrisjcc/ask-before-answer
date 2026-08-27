#!/usr/bin/env python3

"""Promote a DVC experiment's parameter into the current working tree.

The promotion process:

1. Resolve a DVC experiment name to its Git ref/SHA.
2. Read the experiment's dvc.lock.
3. Identify the requested DVC stage and output.
4. Extract the parameter recorded by DVC for that stage.
5. Verify that the corresponding local artifact exists.
6. Update only the selected model's parameter in params.yaml.
7. Preserve comments and existing YAML structure using ruamel.yaml.
8. Verify that the promoted parameter matches the experiment metadata.
9. Report DVC status for the promoted stage.
10. Commit only the intended params.yaml change.

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
PARAMS_FILE = ROOT / "params.yaml"


def run_git(
    *args: str,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    """Run a Git command from the repository root."""
    return subprocess.run(
        ["git", *args],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=check,
    )


def run_dvc(
    *args: str,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    """Run a DVC command from the repository root."""
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

        # Ignore purely untracked files.
        if index_status == "?" and worktree_status == "?":
            continue

        tracked_changes.append(line)

    if tracked_changes:
        print("ERROR: Tracked working tree is not clean.")
        print()
        print("\n".join(tracked_changes))
        print()
        print("Commit or stash your existing tracked changes before promoting.")
        raise RuntimeError("Tracked working tree is not clean.")

    print("Tracked working tree is clean.")
    print("Untracked files will be left untouched.")


def resolve_experiment(experiment: str) -> tuple[str, str]:
    """Resolve a DVC experiment name to its Git ref and commit SHA."""
    result = run_git(
        "for-each-ref",
        "--format=%(refname) %(objectname)",
        "refs/exps",
    )

    matches: list[tuple[str, str]] = []

    for line in result.stdout.splitlines():
        if not line.strip():
            continue

        ref, sha = line.split(maxsplit=1)

        if ref.rsplit("/", 1)[-1] == experiment:
            matches.append((ref, sha))

    if not matches:
        raise RuntimeError(
            f"Could not resolve DVC experiment '{experiment}'.\n"
            "Verify the experiment exists locally with:\n"
            f"  git for-each-ref "
            "--format='%(refname) %(objectname)' refs/exps "
            f"| grep '{experiment}'"
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


def create_yaml() -> YAML:
    """Create a ruamel.yaml instance configured for round-trip editing."""
    yaml = YAML()
    yaml.preserve_quotes = True
    yaml.width = 120
    yaml.indent(mapping=2, sequence=4, offset=2)
    return yaml


def load_yaml(text: str) -> Any:
    """Load YAML using ruamel.yaml round-trip mode."""
    yaml = create_yaml()
    return yaml.load(text)


def load_params() -> tuple[YAML, Any]:
    """Load params.yaml while preserving comments and structure."""
    yaml = create_yaml()

    with PARAMS_FILE.open("r", encoding="utf-8") as handle:
        data = yaml.load(handle)

    return yaml, data


def get_stage_metadata(
    experiment_lock: Any,
    stage_name: str,
    expected_output: str,
) -> Any:
    """Return a DVC stage after validating its expected output."""
    if not isinstance(experiment_lock, dict):
        raise RuntimeError("Experiment dvc.lock does not contain a valid mapping.")

    stages = experiment_lock.get("stages")

    if not stages or stage_name not in stages:
        raise RuntimeError(
            f"Stage '{stage_name}' was not found in the experiment dvc.lock."
        )

    stage = stages[stage_name]

    outputs = stage.get("outs", [])

    for entry in outputs:
        if not isinstance(entry, dict):
            continue

        # DVC lock files normally represent outputs as:
        #
        #   - path: models/sft/final
        #     md5: ...
        #
        if entry.get("path") == expected_output:
            return stage

    raise RuntimeError(
        f"Output '{expected_output}' was not found in stage '{stage_name}'."
    )


def get_output_metadata(
    stage: Any,
    expected_output: str,
) -> dict[str, Any]:
    """Return metadata for a specific DVC output."""
    for entry in stage.get("outs", []):
        if not isinstance(entry, dict):
            continue

        if entry.get("path") == expected_output:
            return dict(entry)

    raise RuntimeError(f"Output '{expected_output}' was not found in the DVC stage.")


def extract_stage_parameter(
    stage: Any,
    param_key: str,
) -> Any:
    """Extract a parameter value recorded by DVC for a stage."""
    params = stage.get("params", {})
    params_file = params.get("params.yaml", {})

    if param_key not in params_file:
        raise RuntimeError(
            f"Parameter '{param_key}' was not recorded in the experiment dvc.lock."
        )

    return params_file[param_key]


def get_nested_parameter(
    data: Any,
    path: tuple[str, ...],
    parameter_name: str,
) -> Any:
    """Read a nested parameter from params.yaml."""
    current = data

    for key in path:
        if key not in current:
            raise RuntimeError(
                f"Expected YAML section '{key}' was not found in params.yaml."
            )

        current = current[key]

    if parameter_name not in current:
        raise RuntimeError(
            f"Parameter '{'.'.join((*path, parameter_name))}' "
            "was not found in params.yaml."
        )

    return current[parameter_name]


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
        raise RuntimeError(
            f"Parameter '{'.'.join((*path, parameter_name))}' "
            "was not found in params.yaml."
        )

    current[parameter_name] = value


def write_params(yaml: YAML, data: Any) -> None:
    """Write params.yaml using ruamel.yaml round-trip serialization."""
    with PARAMS_FILE.open("w", encoding="utf-8") as handle:
        yaml.dump(data, handle)


def verify_promoted_parameter(
    path: tuple[str, ...],
    parameter_name: str,
    expected_value: Any,
) -> None:
    """Verify params.yaml contains the promoted value."""
    _, params = load_params()

    actual_value = get_nested_parameter(
        params,
        path,
        parameter_name,
    )

    if actual_value != expected_value:
        raise RuntimeError(
            "Promotion verification failed: "
            f"expected {parameter_name}={expected_value!r}, "
            f"found {actual_value!r}."
        )

    print(
        "Parameter verification successful: "
        f"{'.'.join((*path, parameter_name))}={actual_value}"
    )


def report_dvc_status(stage_name: str) -> None:
    """Report DVC status for the promoted stage."""
    result = run_dvc("status", stage_name, check=False)

    if result.returncode != 0:
        print(
            f"WARNING: unable to determine DVC status for stage '{stage_name}'.",
            file=sys.stderr,
        )
        print(result.stderr or result.stdout, file=sys.stderr)
        return

    output = result.stdout.strip()

    if not output:
        print("DVC status: no output.")
        return

    print("DVC status after promotion:")

    for line in output.splitlines():
        print(f"  {line}")


def git_diff() -> str:
    """Return the current Git diff."""
    result = run_git("diff", "--", "params.yaml")
    return result.stdout


def verify_only_params_changed() -> None:
    """Verify that promotion changed only params.yaml."""
    result = run_git(
        "diff",
        "--name-only",
    )

    changed_files = [
        line.strip() for line in result.stdout.splitlines() if line.strip()
    ]

    unexpected = [path for path in changed_files if path != "params.yaml"]

    if unexpected:
        print("ERROR: Promotion changed unexpected tracked files:")
        print("\n".join(unexpected))
        raise RuntimeError("Promotion must modify only params.yaml.")


def commit_params(
    experiment: str,
    model: str,
) -> str | None:
    """Commit params.yaml if the promotion changed it."""
    diff = git_diff()

    if not diff.strip():
        print("No params.yaml changes were produced.")
        print("Nothing to commit.")
        return None

    print("Promotion diff:")
    print(diff)

    verify_only_params_changed()

    run_git("add", "--", "params.yaml")

    commit_message = f"dvc: promote {model} experiment {experiment}"

    print()
    print("Files staged for promotion commit:")
    print("  params.yaml")
    print()
    print("Creating Git commit:")
    print(f"  {commit_message}")

    result = run_git(
        "commit",
        "-m",
        commit_message,
    )

    print(result.stdout)

    sha = run_git(
        "rev-parse",
        "--short",
        "HEAD",
    ).stdout.strip()

    return sha


def promote(
    model: str,
    experiment: str,
) -> None:
    """Promote a DVC experiment."""
    config = MODEL_CONFIG[model]

    stage_name = config["stage"]
    expected_output = config["output"]
    param_key = config["param_key"]
    param_path = config["param_path"]
    parameter_name = config["parameter"]

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

    current_head = run_git(
        "rev-parse",
        "HEAD",
    ).stdout.strip()

    print(f"Current HEAD:   {current_head}")
    print()

    print("Reading experiment metadata...")

    lock_text = read_experiment_lock(experiment_sha)
    experiment_lock = load_yaml(lock_text)

    stage = get_stage_metadata(
        experiment_lock,
        stage_name,
        expected_output,
    )

    print(f"Experiment stage:  {stage_name}")
    print(f"Experiment output: {expected_output}")

    output_metadata = get_output_metadata(
        stage,
        expected_output,
    )

    experiment_hash = output_metadata.get("md5")

    if experiment_hash:
        print(f"Experiment hash:   {experiment_hash}")
    else:
        print("Experiment hash:   unavailable")

    print()

    print("Checking local model artifact...")

    local_output = ROOT / expected_output

    if not local_output.exists():
        raise RuntimeError(f"Local output does not exist: {expected_output}")

    if not local_output.is_dir():
        raise RuntimeError(
            f"Expected model output to be a directory: {expected_output}"
        )

    print(f"  Local output exists: {expected_output}")

    if experiment_hash:
        print(f"  Expected DVC hash:   {experiment_hash}")

    print()

    print("Reading experiment parameter...")

    promoted_value = extract_stage_parameter(
        stage,
        param_key,
    )

    print(f"  Parameter: {param_key}")
    print(f"  Value:     {promoted_value}")

    print()

    print("Loading current params.yaml...")

    yaml, params = load_params()

    if params is None:
        raise RuntimeError("params.yaml is empty.")

    current_value = get_nested_parameter(
        params,
        param_path,
        parameter_name,
    )

    print(f"  Current value:    {current_value}")
    print(f"  Experiment value: {promoted_value}")

    if current_value == promoted_value:
        print()
        print("Parameter already matches experiment.")
        print("No params.yaml update is necessary.")
    else:
        print()
        print(f"Promoting {'.'.join((*param_path, parameter_name))}:")
        print(f"  {current_value} -> {promoted_value}")

        update_nested_parameter(
            params,
            param_path,
            parameter_name,
            promoted_value,
        )

        write_params(
            yaml,
            params,
        )

        print("params.yaml updated successfully.")

    print()

    print("Verifying promoted parameter...")

    verify_promoted_parameter(
        param_path,
        parameter_name,
        promoted_value,
    )

    print()

    print("Checking resulting DVC status...")

    report_dvc_status(stage_name)

    print()

    commit_sha = commit_params(
        experiment,
        model,
    )

    print()

    print_header("Promotion successful")
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
    parser = argparse.ArgumentParser(description="Promote a DVC experiment.")

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
        promote(
            args.model,
            args.experiment,
        )

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
