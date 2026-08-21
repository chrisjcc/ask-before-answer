#!/usr/bin/env python3

"""Promote a selected DVC experiment into the workspace.

The promotion process:

1. Verify that the tracked Git working tree is clean.
2. Resolve the DVC experiment name to its Git SHA.
3. Read the experiment's dvc.lock and params.yaml.
4. Verify that the expected model artifact exists locally.
5. Promote the selected DVC stage from the experiment.
6. Promote only the selected model's parameters.
7. Preserve comments and formatting in params.yaml.
8. Verify that DVC considers the promoted stage up to date.
9. Commit only the files changed by the promotion.

Examples:
    python scripts/promote_experiment.py \
        --model sft \
        --experiment sweep_epl5w24i

    python scripts/promote_experiment.py \
        --model dpo \
        --experiment sweep_abc123
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
        "param_prefix": "training.sft",
    },
    "dpo": {
        "stage": "train_dpo",
        "output": "models/dpo/final",
        "param_prefix": "training.dpo",
    },
    "grpo": {
        "stage": "train_grpo",
        "output": "models/grpo/final",
        "param_prefix": "training.grpo",
    },
    "orpo": {
        "stage": "train_orpo",
        "output": "models/orpo/final",
        "param_prefix": "training.orpo",
    },
}


yaml = YAML()
yaml.preserve_quotes = True
yaml.default_flow_style = False
yaml.indent(mapping=2, sequence=4, offset=2)


def run(
    command: list[str],
    *,
    capture: bool = False,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    """Run a command and optionally capture stdout/stderr."""
    return subprocess.run(
        command,
        check=check,
        text=True,
        capture_output=capture,
    )


def git_output(*args: str) -> str:
    """Run Git and return stripped stdout."""
    return run(["git", *args], capture=True).stdout.strip()


def verify_tracked_working_tree() -> None:
    """Ensure that no tracked files have uncommitted changes.

    Untracked files are intentionally ignored. This allows local artifacts
    such as W&B logs, generated reports, caches, and DVC credentials to
    remain untouched while still preventing the promotion from overwriting
    tracked user changes.
    """
    print("Checking tracked working tree...")

    result = run(
        ["git", "status", "--short"],
        capture=True,
    )

    tracked_changes: list[str] = []
    untracked_files: list[str] = []

    for line in result.stdout.splitlines():
        if not line.strip():
            continue

        status = line[:2]
        path = line[3:] if len(line) > 3 else ""

        if status == "??":
            untracked_files.append(path)
        else:
            tracked_changes.append(line)

    if tracked_changes:
        print("\nTracked working tree changes detected:")
        for change in tracked_changes:
            print(f"  {change}")

        raise RuntimeError(
            "Working tree is not clean.\n\n"
            "Commit or stash your existing tracked changes before "
            "promoting an experiment."
        )

    print("Tracked working tree is clean.")

    if untracked_files:
        print("\nUntracked files will be left untouched.")


def resolve_experiment(experiment: str) -> str:
    """Resolve a DVC experiment name to its Git commit SHA.

    DVC experiment names are stored as Git refs under refs/exps and are
    not necessarily valid arguments to `git rev-parse <experiment>`.
    """
    print(f"\nResolving experiment {experiment}...")

    result = run(
        [
            "git",
            "for-each-ref",
            "--format=%(refname) %(objectname)",
            "refs/exps",
        ],
        capture=True,
    )

    matches: list[tuple[str, str]] = []

    for line in result.stdout.splitlines():
        parts = line.split()

        if len(parts) != 2:
            continue

        refname, sha = parts

        if refname.endswith(f"/{experiment}") or experiment in refname:
            matches.append((refname, sha))

    if len(matches) == 1:
        refname, sha = matches[0]

        print(f"Experiment ref: {refname}")
        print(f"Experiment SHA: {sha}")

        return sha

    if len(matches) > 1:
        raise RuntimeError(
            f"Multiple Git refs were found for DVC experiment "
            f"'{experiment}':\n"
            + "\n".join(
                f"  {refname} -> {sha}"
                for refname, sha in matches
            )
        )

    raise RuntimeError(
        f"Could not resolve DVC experiment '{experiment}' "
        "to a Git experiment ref."
    )


def load_yaml_from_git(commit: str, path: str) -> dict[str, Any]:
    """Load YAML content from a Git commit."""
    content = git_output("show", f"{commit}:{path}")

    data = yaml.load(content)

    if data is None:
        return {}

    return data


def load_current_yaml(path: Path) -> dict[str, Any]:
    """Load YAML from the current workspace while preserving comments."""
    with path.open() as file:
        data = yaml.load(file)

    if data is None:
        return {}

    return data


def write_yaml(path: Path, data: dict[str, Any]) -> None:
    """Write YAML while preserving comments and formatting."""
    with path.open("w") as file:
        yaml.dump(data, file)


def get_nested_value(
    data: dict[str, Any],
    dotted_path: str,
) -> Any:
    """Retrieve a nested YAML value using a dotted path."""
    current: Any = data

    for key in dotted_path.split("."):
        if not isinstance(current, dict) or key not in current:
            raise RuntimeError(
                f"YAML path '{dotted_path}' does not exist."
            )

        current = current[key]

    return current


def set_nested_value(
    data: dict[str, Any],
    dotted_path: str,
    value: Any,
) -> None:
    """Set a nested YAML value using a dotted path."""
    keys = dotted_path.split(".")

    current: Any = data

    for key in keys[:-1]:
        if not isinstance(current, dict):
            raise RuntimeError(
                f"Cannot descend through YAML path "
                f"'{dotted_path}'."
            )

        if key not in current:
            current[key] = {}

        current = current[key]

    current[keys[-1]] = value


def promote_stage(
    current_lock: dict[str, Any],
    experiment_lock: dict[str, Any],
    stage: str,
) -> dict[str, Any]:
    """Replace one stage in dvc.lock with the experiment version."""
    current_stages = current_lock.setdefault("stages", {})
    experiment_stages = experiment_lock.get("stages", {})

    if stage not in experiment_stages:
        raise RuntimeError(
            f"Stage '{stage}' does not exist in the experiment "
            "dvc.lock."
        )

    current_stages[stage] = experiment_stages[stage]

    return current_lock


def promote_parameter(
    current_params: dict[str, Any],
    experiment_params: dict[str, Any],
    param_path: str,
) -> dict[str, Any]:
    """Promote one model's parameter subtree.

    Only the selected parameter subtree is replaced. The rest of
    params.yaml remains untouched.

    Because ruamel.yaml is used, comments associated with the existing
    YAML structure are preserved.
    """
    experiment_value = get_nested_value(
        experiment_params,
        param_path,
    )

    set_nested_value(
        current_params,
        param_path,
        experiment_value,
    )

    return current_params


def verify_output(model: str) -> None:
    """Verify that the promoted model output exists locally."""
    output = Path(MODEL_CONFIG[model]["output"])

    print("\nChecking local model artifact...")

    if not output.exists():
        raise RuntimeError(
            f"Promoted output does not exist locally: {output}\n"
            "The experiment cannot be promoted without the "
            "model artifact."
        )

    if output.is_dir() and not any(output.iterdir()):
        raise RuntimeError(
            f"Promoted output directory is empty: {output}"
        )

    print(f"  Local output exists: {output}")


def verify_stage(stage: str) -> None:
    """Verify that DVC considers a stage up to date."""
    result = run(
        ["dvc", "status", stage],
        capture=True,
        check=False,
    )

    output = f"{result.stdout}\n{result.stderr}"

    if result.returncode != 0:
        raise RuntimeError(
            f"DVC status failed for stage '{stage}':\n"
            f"{output}"
        )

    if "Data and pipelines are up to date." not in output:
        raise RuntimeError(
            f"DVC stage '{stage}' is not up to date after "
            f"promotion:\n{output}"
        )


def print_experiment_metadata(
    experiment_lock: dict[str, Any],
    model: str,
) -> None:
    """Print useful metadata about the selected experiment stage."""
    stage = MODEL_CONFIG[model]["stage"]
    output = MODEL_CONFIG[model]["output"]

    stage_data = experiment_lock.get("stages", {}).get(stage)

    if not stage_data:
        raise RuntimeError(
            f"Stage '{stage}' was not found in experiment dvc.lock."
        )

    experiment_outputs = stage_data.get("outs", [])

    output_hash = None

    for item in experiment_outputs:
        if isinstance(item, dict) and output in item:
            output_data = item[output]

            if isinstance(output_data, dict):
                output_hash = output_data.get("md5")

            break

    print(f"Experiment stage: {stage}")
    print(f"Experiment output: {output}")

    if output_hash:
        print(f"Experiment hash:   {output_hash}")
    else:
        print("Experiment hash:   unavailable")


def main() -> int:
    """Run the promotion workflow."""
    parser = argparse.ArgumentParser(
        description="Promote a model from a DVC experiment."
    )

    parser.add_argument(
        "--model",
        required=True,
        choices=sorted(MODEL_CONFIG),
        help="Model type to promote.",
    )

    parser.add_argument(
        "--experiment",
        required=True,
        help="DVC experiment name to promote.",
    )

    args = parser.parse_args()

    model = args.model
    experiment = args.experiment
    config = MODEL_CONFIG[model]

    repo_root = Path.cwd()

    stage = config["stage"]
    output = config["output"]
    param_path = config["param_prefix"]

    print("=" * 58)
    print("Promoting DVC experiment")
    print("=" * 58)
    print(f"Model:       {model}")
    print(f"Experiment:  {experiment}")
    print(f"Stage:       {stage}")
    print(f"Output:      {output}")
    print("=" * 58)

    # ---------------------------------------------------------
    # 1. Verify tracked Git working tree
    # ---------------------------------------------------------

    verify_tracked_working_tree()

    # ---------------------------------------------------------
    # 2. Resolve experiment
    # ---------------------------------------------------------

    experiment_sha = resolve_experiment(experiment)

    current_head = git_output(
        "rev-parse",
        "HEAD",
    )

    print(f"Current HEAD:   {current_head}")

    # ---------------------------------------------------------
    # 3. Read experiment metadata
    # ---------------------------------------------------------

    print("\nReading experiment metadata...")

    experiment_lock = load_yaml_from_git(
        experiment_sha,
        "dvc.lock",
    )

    experiment_params = load_yaml_from_git(
        experiment_sha,
        "params.yaml",
    )

    print_experiment_metadata(
        experiment_lock,
        model,
    )

    # ---------------------------------------------------------
    # 4. Load current workspace metadata
    # ---------------------------------------------------------

    current_lock_path = repo_root / "dvc.lock"
    current_params_path = repo_root / "params.yaml"

    current_lock = load_current_yaml(
        current_lock_path,
    )

    current_params = load_current_yaml(
        current_params_path,
    )

    # ---------------------------------------------------------
    # 5. Verify local artifact
    # ---------------------------------------------------------

    verify_output(model)

    # ---------------------------------------------------------
    # 6. Promote DVC stage
    # ---------------------------------------------------------

    print(f"\nPromoting stage '{stage}'...")

    current_lock = promote_stage(
        current_lock,
        experiment_lock,
        stage,
    )

    # ---------------------------------------------------------
    # 7. Promote selected model parameters
    # ---------------------------------------------------------

    print(
        f"Promoting parameters under '{param_path}'..."
    )

    current_params = promote_parameter(
        current_params,
        experiment_params,
        param_path,
    )

    # ---------------------------------------------------------
    # 8. Write metadata
    # ---------------------------------------------------------

    write_yaml(
        current_lock_path,
        current_lock,
    )

    write_yaml(
        current_params_path,
        current_params,
    )

    # ---------------------------------------------------------
    # 9. Verify DVC
    # ---------------------------------------------------------

    print("\nVerifying promoted DVC stage...")

    verify_stage(stage)

    print("DVC verification successful.")

    # ---------------------------------------------------------
    # 10. Show promotion diff
    # ---------------------------------------------------------

    print("\nPromotion diff:")

    run(
        [
            "git",
            "diff",
            "--",
            "dvc.lock",
            "params.yaml",
        ]
    )

    # ---------------------------------------------------------
    # 11. Stage only promotion-owned files
    # ---------------------------------------------------------

    run(
        [
            "git",
            "add",
            "dvc.lock",
            "params.yaml",
        ]
    )

    staged = git_output(
        "diff",
        "--cached",
        "--name-only",
    )

    if not staged:
        print(
            "\nNo metadata changes were produced by the "
            "promotion."
        )
        print("Nothing to commit.")

        return 0

    print("\nFiles staged for promotion commit:")

    for path in staged.splitlines():
        print(f"  {path}")

    # ---------------------------------------------------------
    # 12. Commit
    # ---------------------------------------------------------

    commit_message = (
        f"dvc: promote {model} experiment {experiment}"
    )

    print("\nCreating Git commit:")
    print(f"  {commit_message}")

    run(
        [
            "git",
            "commit",
            "-m",
            commit_message,
        ]
    )

    new_commit = git_output(
        "rev-parse",
        "--short",
        "HEAD",
    )

    # ---------------------------------------------------------
    # 13. Final result
    # ---------------------------------------------------------

    print("\n" + "=" * 58)
    print("Promotion successful")
    print("=" * 58)
    print(f"Model:       {model}")
    print(f"Experiment:  {experiment}")
    print(f"Stage:       {stage}")
    print(f"Commit:      {new_commit}")
    print("=" * 58)

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())

    except subprocess.CalledProcessError as exc:
        print(
            f"\nERROR: Command failed with exit code "
            f"{exc.returncode}:",
            file=sys.stderr,
        )

        if exc.stdout:
            print(exc.stdout, file=sys.stderr)

        if exc.stderr:
            print(exc.stderr, file=sys.stderr)

        sys.exit(exc.returncode)

    except Exception as exc:
        print(
            f"\nERROR: {exc}",
            file=sys.stderr,
        )

        sys.exit(1)
