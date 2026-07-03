"""Plot training dynamics.

This script parses HuggingFace trainer_state.json files generated during SFT, DPO,
and GRPO training runs, and plots comparative loss and metric convergence curves.
"""

import json
import logging
import os
from pathlib import Path

import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Paths to the model output directories where trainer_state.json is expected
MODEL_DIRS = {"SFT": "models/sft", "DPO": "models/dpo", "GRPO": "models/grpo"}

OUTPUT_DIR = Path("docs/figures")


def extract_metrics(state_path: str):
    """Extracts train and eval loss from a trainer_state.json file."""
    if not os.path.exists(state_path):
        return None

    with open(state_path, "r") as f:
        try:
            state = json.load(f)
        except json.JSONDecodeError:
            logger.warning(f"Failed to decode JSON from {state_path}")
            return None

    log_history = state.get("log_history", [])

    train_loss = []
    eval_loss = []

    for entry in log_history:
        step = entry.get("step")
        if step is None:
            continue

        if "loss" in entry:
            train_loss.append((step, entry["loss"]))
        if "eval_loss" in entry:
            eval_loss.append((step, entry["eval_loss"]))

    return {
        "train_loss": train_loss,
        "eval_loss": eval_loss,
        "log_history": log_history,
    }


def plot_comparisons(results: dict):
    """Plots training and evaluation loss comparisons."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Plot Training Loss Comparison
    plt.figure(figsize=(10, 6))
    for model_name, metrics in results.items():
        if metrics and metrics["train_loss"]:
            steps, losses = zip(*metrics["train_loss"])
            plt.plot(steps, losses, label=f"{model_name} Train Loss", linewidth=2)

    plt.title("Training Loss Comparison")
    plt.xlabel("Training Steps")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)

    train_out = OUTPUT_DIR / "train_loss_comparison.png"
    plt.savefig(train_out, bbox_inches="tight", dpi=300)
    plt.close()
    logger.info(f"Saved {train_out}")

    # Plot Evaluation Loss Comparison
    plt.figure(figsize=(10, 6))
    for model_name, metrics in results.items():
        if metrics and metrics["eval_loss"]:
            steps, losses = zip(*metrics["eval_loss"])
            plt.plot(
                steps,
                losses,
                label=f"{model_name} Eval Loss",
                linewidth=2,
                linestyle="--",
            )

    plt.title("Evaluation Loss Comparison")
    plt.xlabel("Training Steps")
    plt.ylabel("Eval Loss")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)

    eval_out = OUTPUT_DIR / "eval_loss_comparison.png"
    plt.savefig(eval_out, bbox_inches="tight", dpi=300)
    plt.close()
    logger.info(f"Saved {eval_out}")


def plot_grpo_specifics(grpo_metrics: dict):
    """Plots GRPO-specific metrics like reward components and KL divergence."""
    if not grpo_metrics:
        return

    log_history = grpo_metrics.get("log_history", [])

    # Track metrics like format_reward, routing_reward, kl, etc.
    #metrics_to_plot = {
    #    "KL Divergence": "objective/kl",
    #    "Format Reward": "objective/scores_margin",  # Adjust based on actual TRL logs
    #    "Routing Reward": "objective/scores",
    #}

    # In HuggingFace TRL GRPO, custom rewards are typically logged as env/reward_name
    # We will search for common keys
    keys_found = set()
    for entry in log_history:
        for k in entry.keys():
            if "reward" in k.lower() or "kl" in k.lower():
                keys_found.add(k)

    if not keys_found:
        logger.info("No GRPO-specific reward/kl metrics found in logs.")
        return

    for key in keys_found:
        data = [
            (entry["step"], entry[key])
            for entry in log_history
            if key in entry and "step" in entry
        ]
        if not data:
            continue

        steps, vals = zip(*data)
        plt.figure(figsize=(8, 5))
        plt.plot(steps, vals, label=key, color="purple")
        plt.title(f"GRPO Metric: {key}")
        plt.xlabel("Steps")
        plt.ylabel("Value")
        plt.grid(True)
        plt.legend()

        safe_key = key.replace("/", "_").replace(" ", "_")
        out_path = OUTPUT_DIR / f"{safe_key}.png"
        plt.savefig(out_path, bbox_inches="tight", dpi=300)
        plt.close()
        logger.info(f"Saved {out_path}")


def main():
    logger.info("Extracting metrics from trainer_state.json files...")

    results = {}
    for model_name, model_dir in MODEL_DIRS.items():
        # trainer_state.json can be in the root of the model dir
        # or inside checkpoint dirs
        state_path = os.path.join(model_dir, "trainer_state.json")
        if not os.path.exists(state_path):
            # Check for the latest checkpoint
            checkpoints = (
                [d for d in os.listdir(model_dir) if d.startswith("checkpoint-")]
                if os.path.exists(model_dir)
                else []
            )
            if checkpoints:
                checkpoints.sort(key=lambda x: int(x.split("-")[-1]))
                state_path = os.path.join(
                    model_dir, checkpoints[-1], "trainer_state.json"
                )

        metrics = extract_metrics(state_path)
        if metrics:
            logger.info(f"Found metrics for {model_name} at {state_path}")
            results[model_name] = metrics
        else:
            logger.warning(f"No metrics found for {model_name}")

    if any(results.values()):
        plot_comparisons(results)
        plot_grpo_specifics(results.get("GRPO"))
    else:
        logger.error("No trainer_state.json files found. Cannot generate plots.")


if __name__ == "__main__":
    main()
