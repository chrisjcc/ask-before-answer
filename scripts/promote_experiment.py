#!/usr/bin/env python3

"""Promote a selected DVC experiment stage into the current workspace."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml


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


def run(
    command: list[str],
    *,
    capture: bool = False,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    """Run a subprocess."""
    return subprocess.run(
        command,
        check=check,
        text=True,
        capture_output=capture,
    )


def git_output(*args: str) -> str:
    """Run git and return stripped stdout."""
    return run(["git", *args], capture=True).stdout.strip()


def resolve_experiment(experiment: str) -> str:
    """Resolve a DVC experiment name to its Git commit SHA."""

    # DVC experiment names are refs under refs/exps/.
    result = run(
        [
            "git",
            "for-each-ref",
            "--format=%(refname) %(objectname)",
            "refs/exps",
        ],
        capture=True,
    )

    for line in result.stdout.splitlines():
        ref, sha = line.split(maxsplit=1)

        if ref.endswith(f"/{experiment}"):
            return sha

    # Also support passing a SHA directly.
    result = run(
        ["git", "rev-parse", "--verify", experiment],
        capture=True,
        check=False,
    )

    if result.returncode == 0:
        return result.stdout.strip()

    raise RuntimeError(
        f"Could not resolve DVC experiment '{experiment}'.\n"
        "Use 'git for-each-ref refs/exps' to inspect available experiments."
    )


def load_yaml_from_git(commit: str, path: str) -> dict[str, Any]:
    """Load a YAML file from a Git commit."""
    content = git_output("show", f"{commit}:{path}")
    return yaml.safe_load(content) or {}


def load_yaml(path: Path) -> dict[str, Any]:
    """Load YAML from the current workspace."""
    with path.open() as file:
        return yaml.safe_load(file) or {}


def write_yaml(path: Path, data: dict[str, Any]) -> None:
    """Write YAML using standard project formatting."""
    with path.open("w") as file:
        yaml.safe_dump(
            data,
            file,
            sort_keys=False,
            default_flow_style=False,
        )


def get_nested(data: dict[str, Any], dotted_path: str) -> Any:
    """Get a nested YAML value using a dotted path."""
    current: Any = data

    for key in dotted_path.split("."):
        if not isinstance(current, dict) or key not in current:
            raise RuntimeError(
                f"Missing parameter '{dotted_path}' in experiment params.yaml."
            )

        current = current[key]

    return current


def set_nested(
    data: dict[str, Any],
    dotted_path: str,
    value: Any,
) -> None:
    """Set a nested YAML value using a dotted path."""
    keys = dotted_path.split(".")
    current = data

    for key in keys[:-1]:
        current = current.setdefault(key, {})

    current[keys[-1]] = value


def get_stage(
    lock: dict[str, Any],
    stage: str,
) -> dict[str, Any]:
    """Return a stage from a DVC lockfile."""
    stages = lock.get("stages", {})

    if stage not in stages:
        raise RuntimeError(
            f"Stage '{stage}' does not exist in the experiment dvc.lock."
        )

    return stages[stage]


def get_output_info(
    stage: dict[str, Any],
    output_path: str,
) -> dict[str, Any]:
    """Find an output entry in a DVC stage."""
    for output in stage.get("outs", []):
        if output.get("path") == output_path:
            return output

    raise RuntimeError(
        f"Output '{output_path}' is not present in the experiment stage."
    )


def promote_stage(
    current_lock: dict[str, Any],
    experiment_stage: dict[str, Any],
    stage_name: str,
) -> None:
    """Replace exactly one stage in the current lockfile."""
    stages = current_lock.setdefault("stages", {})
    stages[stage_name] = experiment_stage


def verify_workspace_output(
    output_path: str,
    expected_hash: str,
) -> None:
    """
    Verify that the promoted model exists and matches the experiment hash.

    DVC's directory outputs are represented by a .dir hash in the cache.
    We ask DVC to calculate the workspace hash rather than trying to
    reconstruct it ourselves.
    """
    output = Path(output_path)

    if not output.exists():
        raise RuntimeError(
            f"Required model output does not exist locally:\n"
            f"  {output_path}\n\n"
            "The promotion metadata cannot be committed because the model "
            "artifact is not present in the workspace."
        )

    if output.is_dir() and not any(output.iterdir()):
        raise RuntimeError(
            f"Model output directory is empty:\n  {output_path}"
        )

    # Use `dvc status` after installing the experiment stage metadata.
    # The caller performs the final DVC verification.
    print(f"  Local output exists: {output_path}")
    print(f"  Expected DVC hash:   {expected_hash}")


def tracked_worktree_is_clean() -> bool:
    """
    Check only tracked changes.

    Untracked files are intentionally allowed. This repository generates
    logs, reports, caches, W&B artifacts, outputs, and other temporary files.
    The promotion command only owns dvc.lock and params.yaml.
    """
    unstaged = git_output("diff", "--name-only")
    staged = git_output("diff", "--cached", "--name-only")

    if unstaged or staged:
        print("Tracked files have uncommitted changes:")

        if unstaged:
            print("\nUnstaged:")
            print(unstaged)

        if staged:
            print("\nStaged:")
            print(staged)

        return False

    return True


def verify_stage(stage: str) -> None:
    """Verify that DVC considers the promoted stage up to date."""
    result = run(
        ["dvc", "status", stage],
        capture=True,
        check=False,
    )

    output = f"{result.stdout}\n{result.stderr}"

    if result.returncode != 0:
        raise RuntimeError(
            f"DVC status failed for stage '{stage}':\n{output}"
        )

    if "Data and pipelines are up to date." not in output:
        raise RuntimeError(
            f"DVC stage '{stage}' is not up to date after promotion:\n"
            f"{output}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Promote a model from a DVC experiment."
    )

    parser.add_argument(
        "--model",
        required=True,
        choices=sorted(MODEL_CONFIG),
    )

    parser.add_argument(
        "--experiment",
        required=True,
    )

    args = parser.parse_args()

    model = args.model
    experiment = args.experiment
    config = MODEL_CONFIG[model]

    repo_root = Path.cwd()
    stage_name = config["stage"]
    output_path = config["output"]
    param_prefix = config["param_prefix"]

    print("=" * 58)
    print("Promoting DVC experiment")
    print("=" * 58)
    print(f"Model:       {model}")
    print(f"Experiment:  {experiment}")
    print(f"Stage:       {stage_name}")
    print(f"Output:      {output_path}")
    print("=" * 58)

    # ------------------------------------------------------------
    # 1. Verify tracked working tree is clean.
    #
    # Untracked files are explicitly allowed.
    # ------------------------------------------------------------

    print("\nChecking tracked working tree...")

    if not tracked_worktree_is_clean():
        raise RuntimeError(
            "Tracked working-tree changes detected.\n\n"
            "Commit or stash those changes before promoting an experiment."
        )

    print("Tracked working tree is clean.")
    print("Untracked files will be left untouched.")

    # ------------------------------------------------------------
    # 2. Resolve experiment.
    # ------------------------------------------------------------

    print(f"\nResolving experiment {experiment}...")

    experiment_sha = resolve_experiment(experiment)

    print(f"Experiment SHA: {experiment_sha}")

    current_head = git_output("rev-parse", "HEAD")

    print(f"Current HEAD:   {current_head}")

    # ------------------------------------------------------------
    # 3. Load experiment metadata directly from Git.
    #
    # IMPORTANT:
    # We deliberately do NOT run:
    #
    #     dvc exp apply
    #
    # because that attempts to restore unrelated experiment outputs.
    # ------------------------------------------------------------

    print("\nReading experiment metadata...")

    experiment_lock = load_yaml_from_git(
        experiment_sha,
        "dvc.lock",
    )

    experiment_params = load_yaml_from_git(
        experiment_sha,
        "params.yaml",
    )

    experiment_stage = get_stage(
        experiment_lock,
        stage_name,
    )

    experiment_output = get_output_info(
        experiment_stage,
        output_path,
    )

    expected_hash = experiment_output.get("md5")

    if not expected_hash:
        raise RuntimeError(
            f"No DVC hash found for '{output_path}' in experiment."
        )

    print(f"Experiment stage: {stage_name}")
    print(f"Experiment output: {output_path}")
    print(f"Experiment hash:   {expected_hash}")

    # ------------------------------------------------------------
    # 4. Verify the model artifact exists locally.
    # ------------------------------------------------------------

    print("\nChecking local model artifact...")

    verify_workspace_output(
        output_path,
        expected_hash,
    )

    # ------------------------------------------------------------
    # 5. Load current metadata.
    # ------------------------------------------------------------

    current_lock_path = repo_root / "dvc.lock"
    current_params_path = repo_root / "params.yaml"

    current_lock = load_yaml(current_lock_path)
    current_params = load_yaml(current_params_path)

    # ------------------------------------------------------------
    # 6. Promote exactly one DVC stage.
    # ------------------------------------------------------------

    print(f"\nPromoting stage '{stage_name}'...")

    promote_stage(
        current_lock,
        experiment_stage,
        stage_name,
    )

    # ------------------------------------------------------------
    # 7. Promote only the relevant model parameters.
    # ------------------------------------------------------------

    print(f"Promoting parameters under '{param_prefix}'...")

    promoted_params = get_nested(
        experiment_params,
        param_prefix,
    )

    set_nested(
        current_params,
        param_prefix,
        promoted_params,
    )

    # ------------------------------------------------------------
    # 8. Write metadata.
    # ------------------------------------------------------------

    write_yaml(
        current_lock_path,
        current_lock,
    )

    write_yaml(
        current_params_path,
        current_params,
    )

    # ------------------------------------------------------------
    # 9. Verify DVC.
    # ------------------------------------------------------------

    print("\nVerifying promoted DVC stage...")

    verify_stage(stage_name)

    print("DVC verification successful.")

    # ------------------------------------------------------------
    # 10. Show exact changes.
    # ------------------------------------------------------------

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

    # ------------------------------------------------------------
    # 11. Stage ONLY promotion-owned files.
    # ------------------------------------------------------------

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
            "\nNo metadata changes were produced by the promotion.\n"
            "Nothing to commit."
        )
        return 0

    print("\nFiles staged for promotion commit:")
    print(staged)

    # Safety check: only expected files may be staged.
    allowed = {"dvc.lock", "params.yaml"}
    staged_files = set(staged.splitlines())

    unexpected = staged_files - allowed

    if unexpected:
        raise RuntimeError(
            "Unexpected files are staged:\n"
            + "\n".join(sorted(unexpected))
        )

    # ------------------------------------------------------------
    # 12. Automatically commit.
    # ------------------------------------------------------------

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

    commit_sha = git_output(
        "rev-parse",
        "--short",
        "HEAD",
    )

    print("\n" + "=" * 58)
    print("Promotion successful")
    print("=" * 58)
    print(f"Model:       {model}")
    print(f"Experiment:  {experiment}")
    print(f"Stage:       {stage_name}")
    print(f"Output:      {output_path}")
    print(f"Commit:      {commit_sha}")
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
        print(f"\nERROR: {exc}", file=sys.stderr)
        sys.exit(1)
