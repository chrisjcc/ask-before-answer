import logging
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import wandb

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_sweep_report():
    logger.info("Connecting to Weights & Biases API...")
    api = wandb.Api()

    ENTITY = "rl4aa"
    PROJECT = "ask-before-answer"

    logger.info(f"Fetching sweep runs for {ENTITY}/{PROJECT}...")
    try:
        runs = api.runs(f"{ENTITY}/{PROJECT}")
    except Exception as e:
        logger.error(f"Error fetching runs: {e}")
        return

    sweep_groups = {}

    for run in runs:
        # Only process sweep runs
        if not run.sweep:
            continue

        # Only process completed runs with eval/loss
        if run.state != "finished" or "eval/loss" not in run.summary:
            continue

        sweep_id = run.sweep.id
        if sweep_id not in sweep_groups:
            sweep_groups[sweep_id] = {
                "runs": [],
                "name": (
                    run.sweep.name
                    if hasattr(run.sweep, "name") and run.sweep.name
                    else f"Sweep {sweep_id}"
                ),
            }

        sweep_groups[sweep_id]["runs"].append(
            {
                "Run ID": run.id,
                "Name": run.name,
                "Learning Rate": run.config.get("learning_rate", "N/A"),
                "Batch Size": run.config.get("per_device_train_batch_size", "N/A"),
                "Beta": run.config.get("beta", "N/A"),
                "Eval Loss": run.summary.get("eval/loss", float("inf")),
                "URL": run.url,
            }
        )

    if not sweep_groups:
        logger.info("No completed sweep runs found.")
        return

    os.makedirs("docs/plots", exist_ok=True)
    sns.set_theme(style="whitegrid")

    report_content = [
        "# Parameter Sweep Report",
        "",
        "This report is dynamically generated from Weights & Biases telemetry.",
        "It groups all completed sweep trials by their Sweep ID.",
        "",
    ]

    for sweep_id, group_data in sweep_groups.items():
        sweep_name = group_data["name"]
        df = pd.DataFrame(group_data["runs"]).sort_values("Eval Loss")

        report_content.extend(
            [
                f"## {sweep_name} (`{sweep_id}`)",
                "",
                "### Top Performing Configurations",
                "",
                df.to_markdown(index=False),
                "",
            ]
        )

        # Generate Validation Curve
        valid_runs = df[df["Learning Rate"] != "N/A"].copy()
        if not valid_runs.empty:
            valid_runs["Learning Rate"] = pd.to_numeric(valid_runs["Learning Rate"])
            plt.figure(figsize=(8, 6))
            sns.scatterplot(data=valid_runs, x="Learning Rate", y="Eval Loss", s=100)
            plt.xscale("log")
            plt.title(f"Validation Curve: {sweep_name}")
            plt.xlabel("Learning Rate (log scale)")
            plt.ylabel("Final Eval Loss")
            plt.tight_layout()

            plot_path = f"docs/plots/sweep_val_curve_{sweep_id}.png"
            plt.savefig(plot_path)
            plt.close()

            report_content.extend(
                [
                    "### Validation Curve",
                    "",
                    f"![Validation Curve](plots/sweep_val_curve_{sweep_id}.png)",
                    "",
                ]
            )

    with open("docs/sweep_report.md", "w") as f:
        f.write("\n".join(report_content))

    logger.info("Sweep report successfully generated at docs/sweep_report.md")


if __name__ == "__main__":
    generate_sweep_report()
