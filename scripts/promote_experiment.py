#!/usr/bin/env python3

"""Promote a selected DVC experiment model into the current workspace.

This intentionally does NOT use `dvc exp apply`.

DVC `exp apply` restores the complete experiment workspace, including
unrelated stages and outputs. For model promotion we only want to promote
the requested model stage and its corresponding parameters.

Supported models:
    sft
    dpo
    grpo
    orpo
"""

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
    """Run a subprocess command."""
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
    """Resolve a DVC experiment name to its Git SHA.

    DVC experiments are stored under refs/exps/<...>/<experiment>.
    The experiment name itself is not necessarily a Git revision.
    """

    output = git_output(
        "for-each-ref",
        "--format=%(refname) %(objectname)",
        "refs/exps",
    )

    matches = []

    for line in output.splitlines():
        if not line.strip():
            continue

        ref, sha = line.split(maxsplit=1)

        if ref.endswith(f"/{experiment}"):
            matches.append((ref, sha))

    if not matches:
        raise RuntimeError(
            f"Could not resolve DVC experiment '{experiment}'.\n"
            "Available experiment refs can be inspected with:\n"
            "  git for-each-ref --format='%(refname) %(objectname)' refs/exps"
        )

    if len(matches) > 1:
        refs = "\n".join(f"  {ref} -> {sha}" for ref, sha in matches)
        raise RuntimeError(
            f"Multiple Git refs found for experiment '{experiment}':\n{refs}"
        )

    ref, sha = matches[0]

    print(f"Experiment ref: {ref}")
    print(f"Experiment SHA:  {sha}")

    return sha


def load_yaml_from_git(commit: str, path: str) -> dict[str, Any]:
    """Load a YAML file from a Git commit."""
    content = git_output("show", f"{commit}:{path}")
    return yaml.safe_load(content) or {}


def load_current_yaml(path: Path) -> dict[str, Any]:
    """Load YAML from the current workspace."""
    with path.open() as file:
        return yaml.safe_load(file) or {}


def write_yaml(path: Path, data: dict[str, Any]) -> None:
    """Write YAML using normal project formatting."""
    with path.open("w") as file:
        yaml.safe_dump(
            data,
            file,
            sort_keys=False,
            default_flow_style=False,
        )


def get_nested_value(data: dict[str, Any], dotted_path: str) -> Any:
    """Get a nested dictionary value using a dotted path."""
    current: Any = data

    for key in dotted_path.split("."):
        if not isinstance(current, dict) or key not in current:
            raise RuntimeError(
                f"Parameter '{dotted_path}' not found in experiment params.yaml."
            )

        current = current[key]

    return current


def set_nested_value(
    data: dict[str, Any],
    dotted_path: str,
    value: Any,
) -> None:
    """Set a nested dictionary value using a dotted path."""
    keys = dotted_path.split(".")
    current = data

    for key in keys[:-1]:
        current = current.setdefault(key, {})

    current[keys[-1]] = value


def promote_stage(
    current_lock: dict[str, Any],
    experiment_lock: dict[str, Any],
    stage: str,
) -> None:
    """Replace one stage in the current dvc.lock with the experiment stage."""

    experiment_stages = experiment_lock.get("stages", {})

    if stage not in experiment_stages:
        raise RuntimeError(f"Stage '{stage}' does not exist in experiment dvc.lock.")

    current_lock.setdefault("stages", {})[stage] = experiment_stages[stage]


def get_stage_output(
    experiment_lock: dict[str, Any],
    stage: str,
    output: str,
) -> dict[str, Any]:
    """Return metadata for a specific stage output."""

    stage_data = experiment_lock["stages"][stage]

    for out in stage_data.get("outs", []):
        if isinstance(out, str):
            path = out
            metadata = {}
        else:
            path = out.get("path")
            metadata = out

        if path == output:
            return metadata

    raise RuntimeError(
        f"Output '{output}' was not found in stage '{stage}' of experiment dvc.lock."
    )


def cache_path_for_hash(cache_dir: Path, md5: str) -> Path:
    """Return the DVC 3 cache path for an MD5 object."""

    return cache_dir / md5[:2] / md5[2:]


def verify_cache_object(
    output_metadata: dict[str, Any],
    output: str,
) -> None:
    """Verify that the promoted output exists in the local DVC cache.

    DVC directory outputs have a `.dir` object in the cache.
    """

    md5 = output_metadata.get("md5")

    if not md5:
        raise RuntimeError(f"No MD5 hash recorded for promoted output '{output}'.")

    cache_dir = Path(
        run(
            ["dvc", "cache", "dir"],
            capture=True,
        ).stdout.strip()
    )

    cache_object = cache_path_for_hash(cache_dir, md5)

    print("\nDVC cache:")
    print(f"  {cache_dir}")

    print("Required cache object:")
    print(f"  {md5}")

    print("Expected cache path:")
    print(f"  {cache_object}")

    if not cache_object.exists():
        raise RuntimeError(
            f"The experiment's model artifact is not present in the "
            f"local DVC cache:\n\n"
            f"  {output}\n"
            f"  md5: {md5}\n"
            f"  cache: {cache_object}\n\n"
            "The experiment cannot be promoted from this machine until "
            "the artifact is available in the local DVC cache.\n\n"
            "If this artifact was never uploaded to the DVC remote, "
            "run the promotion on the machine where the experiment "
            "artifact exists and use `dvc push` afterward."
        )

    print("Cache object: FOUND")


def checkout_output(output: str) -> None:
    """Checkout only the requested model output."""

    print(f"\nChecking out promoted output: {output}")

    result = run(
        [
            "dvc",
            "checkout",
            "--force",
            output,
        ],
        check=False,
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"DVC checkout failed for '{output}'.\nExit code: {result.returncode}"
        )


def verify_output(output: str) -> None:
    """Verify the promoted model exists and is non-empty."""

    path = Path(output)

    if not path.exists():
        raise RuntimeError(f"Promoted model output does not exist locally:\n  {output}")

    if path.is_dir() and not any(path.iterdir()):
        raise RuntimeError(f"Promoted model output directory is empty:\n  {output}")

    print(f"Verified model output: {output}")


def verify_dvc_stage(stage: str) -> None:
    """Verify that the promoted stage is up to date."""

    result = run(
        ["dvc", "status", stage],
        capture=True,
        check=False,
    )

    output = f"{result.stdout}\n{result.stderr}"

    if result.returncode != 0:
        raise RuntimeError(f"DVC status failed for stage '{stage}':\n{output}")

    if "Data and pipelines are up to date." not in output:
        raise RuntimeError(
            f"DVC stage '{stage}' is not up to date after promotion:\n{output}"
        )

    print(f"DVC stage verified: {stage}")


def ensure_clean_workspace() -> None:
    """Require a clean Git worktree before promotion."""

    status = git_output("status", "--short")

    if status:
        print("ERROR: Working tree is not clean.", file=sys.stderr)
        print(file=sys.stderr)
        print(status, file=sys.stderr)
        print(file=sys.stderr)
        print(
            "Commit or stash your existing changes before promoting an experiment.",
            file=sys.stderr,
        )
        raise RuntimeError("Working tree is not clean.")


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

    stage = config["stage"]
    output = config["output"]
    param_prefix = config["param_prefix"]

    print("=" * 58)
    print("Promoting DVC experiment")
    print("=" * 58)
    print(f"Model:       {model}")
    print(f"Experiment:  {experiment}")
    print(f"Stage:       {stage}")
    print(f"Output:      {output}")
    print("=" * 58)

    # ---------------------------------------------------------
    # Safety check
    # ---------------------------------------------------------

    ensure_clean_workspace()

    # ---------------------------------------------------------
    # Resolve experiment
    # ---------------------------------------------------------

    print(f"\nResolving experiment {experiment}...")

    experiment_sha = resolve_experiment(experiment)

    current_head = git_output("rev-parse", "HEAD")

    print(f"Current HEAD:   {current_head}")
    print(f"Experiment SHA: {experiment_sha}")

    # ---------------------------------------------------------
    # Load experiment metadata
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

    current_lock_path = repo_root / "dvc.lock"
    current_params_path = repo_root / "params.yaml"

    current_lock = load_current_yaml(current_lock_path)
    current_params = load_current_yaml(current_params_path)

    # ---------------------------------------------------------
    # Inspect target stage
    # ---------------------------------------------------------

    print("\nExperiment stage:")

    experiment_stage = experiment_lock["stages"][stage]

    print(f"  Stage: {stage}")
    print(f"  Command: {experiment_stage.get('cmd', '').strip()}")

    output_metadata = get_stage_output(
        experiment_lock,
        stage,
        output,
    )

    print("\nExperiment output:")

    for key, value in output_metadata.items():
        print(f"  {key}: {value}")

    # ---------------------------------------------------------
    # Verify artifact exists in local cache
    # ---------------------------------------------------------

    verify_cache_object(
        output_metadata,
        output,
    )

    # ---------------------------------------------------------
    # Get experiment parameter
    # ---------------------------------------------------------

    experiment_parameter = get_nested_value(
        experiment_params,
        param_prefix,
    )

    print("\nExperiment parameters:")

    print(f"  {param_prefix}: {experiment_parameter}")

    # ---------------------------------------------------------
    # Promote ONLY the requested stage
    # ---------------------------------------------------------

    print("\nPromoting stage metadata...")

    promote_stage(
        current_lock,
        experiment_lock,
        stage,
    )

    # Promote the corresponding model parameter section.
    set_nested_value(
        current_params,
        param_prefix,
        experiment_parameter,
    )

    # ---------------------------------------------------------
    # Write metadata
    # ---------------------------------------------------------

    write_yaml(
        current_lock_path,
        current_lock,
    )

    write_yaml(
        current_params_path,
        current_params,
    )

    print("Updated:")
    print("  dvc.lock")
    print("  params.yaml")

    # ---------------------------------------------------------
    # Checkout ONLY the promoted model
    # ---------------------------------------------------------

    checkout_output(output)

    # ---------------------------------------------------------
    # Verify
    # ---------------------------------------------------------

    verify_output(output)
    verify_dvc_stage(stage)

    # ---------------------------------------------------------
    # Show diff
    # ---------------------------------------------------------

    print("\nGit diff:")
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
    # Stage promotion files
    # ---------------------------------------------------------

    run(
        [
            "git",
            "add",
            "dvc.lock",
            "params.yaml",
            output,
        ]
    )

    staged = git_output(
        "diff",
        "--cached",
        "--name-only",
    )

    if not staged:
        print("\nNo changes were produced by the promotion. Nothing to commit.")
        return 0

    # ---------------------------------------------------------
    # Commit
    # ---------------------------------------------------------

    commit_message = f"dvc: promote {model} experiment {experiment}"

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

    print("\n" + "=" * 58)
    print("Promotion successful")
    print("=" * 58)
    print(f"Model:       {model}")
    print(f"Experiment:  {experiment}")
    print(f"Stage:       {stage}")
    print(f"Output:      {output}")
    print(f"Commit:      {git_output('rev-parse', '--short', 'HEAD')}")
    print("=" * 58)

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())

    except subprocess.CalledProcessError as exc:
        print(
            f"\nERROR: Command failed with exit code {exc.returncode}:",
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
