import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import wandb


def generate_report():
    print("Connecting to Weights & Biases API...")
    api = wandb.Api()

    # Update with actual entity and project
    ENTITY = "rl4aa"
    PROJECT = "ask-before-answer"

    print(f"Fetching runs for {ENTITY}/{PROJECT}...")
    try:
        runs = api.runs(f"{ENTITY}/{PROJECT}")
    except Exception as e:
        print(f"Error fetching runs: {e}")
        return

    summary_list = []
    run_histories = {}

    for run in runs:
        # Debug printing
        print(
            f"Run: {run.id}, state: {run.state}, keys: {list(run.summary.keys())[:5]}..."
        )
        # Only process completed runs that logged eval/loss
        if run.state != "finished" or "eval/loss" not in run.summary:
            continue

        summary_list.append(
            {
                "Run ID": run.id,
                "Name": run.name,
                "Sweep ID": run.sweep.id if run.sweep else "N/A",
                "Learning Rate": run.config.get("learning_rate", "N/A"),
                "Batch Size": run.config.get("per_device_train_batch_size", "N/A"),
                "Eval Loss": run.summary.get("eval/loss", float("inf")),
                "URL": run.url,
            }
        )

        # Download history for plotting (loss and eval/loss)
        history = run.history(keys=["_step", "train/loss", "eval/loss", "epoch"])
        if not history.empty:
            run_histories[run.name] = history

    if not summary_list:
        print("No completed runs with eval_loss found.")
        return

    # Convert to DataFrame and sort by Eval Loss
    df_summary = pd.DataFrame(summary_list).sort_values("Eval Loss")

    # Setup directories
    os.makedirs("docs/plots", exist_ok=True)

    print("Generating plots...")
    sns.set_theme(style="whitegrid")

    # Plot 1: Eval Loss Comparison
    plt.figure(figsize=(10, 6))
    for name, hist in run_histories.items():
        if "eval/loss" in hist.columns:
            # Drop NaN values for eval/loss
            valid_hist = hist.dropna(subset=["eval/loss"])
            if not valid_hist.empty:
                sns.lineplot(data=valid_hist, x="_step", y="eval/loss", label=name)

    plt.title("Evaluation Loss vs Steps")
    plt.xlabel("Training Steps")
    plt.ylabel("Eval Loss")
    if plt.gca().get_legend_handles_labels()[0]:
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plot_path = "docs/plots/eval_loss_comparison.png"
    plt.savefig(plot_path)
    plt.close()

    print("Generating Markdown Report...")
    report_content = [
        "# Ablation Experiment Report",
        "",
        "This report was automatically generated from Weights & Biases metrics.",
        "",
        "## Top Performing Configurations",
        "",
    ]

    # Format table
    report_content.append(df_summary.to_markdown(index=False))

    report_content.extend(
        [
            "",
            "## Learning Curves",
            "",
            "![Evaluation Loss](plots/eval_loss_comparison.png)",
            "",
        ]
    )

    with open("docs/ablation_report.md", "w") as f:
        f.write("\n".join(report_content))

    print("Report successfully generated at docs/ablation_report.md")


if __name__ == "__main__":
    generate_report()
