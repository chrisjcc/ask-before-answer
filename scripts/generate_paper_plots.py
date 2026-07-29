#!/usr/bin/env python3
"""Script to extract training metrics from W&B and generate plots for the research paper."""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import wandb

# Use seaborn style for paper-ready plots
sns.set_theme(style="whitegrid", context="paper")
plt.rcParams.update({'font.size': 12})

def get_run_history(api, project_path, run_name):
    """Fetch the latest successful run with a given name and return its history."""
    try:
        runs = api.runs(project_path, filters={"display_name": run_name, "state": "finished"})
        if not runs:
            print(f"Warning: No finished runs found for {run_name}")
            return None
            
        # Get the most recent run
        run = sorted(runs, key=lambda r: r.created_at, reverse=True)[0]
        print(f"Fetching history for {run_name} (ID: {run.id})...")
        
        # Download history as pandas DataFrame
        history = run.history(samples=1000)
        return history
    except Exception as e:
        print(f"Error fetching {run_name}: {e}")
        return None

def plot_loss_comparison(sft_hist, dpo_hist, out_dir):
    """Plot Training and Evaluation Loss for SFT and DPO."""
    if sft_hist is None and dpo_hist is None:
        return
        
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot SFT
    if sft_hist is not None:
        if 'train/loss' in sft_hist.columns:
            sft_train = sft_hist.dropna(subset=['train/loss'])
            axes[0].plot(sft_train['_step'], sft_train['train/loss'], label='SFT Train Loss', color='blue', alpha=0.7)
        if 'eval/loss' in sft_hist.columns:
            sft_eval = sft_hist.dropna(subset=['eval/loss'])
            axes[0].plot(sft_eval['_step'], sft_eval['eval/loss'], label='SFT Eval Loss', color='orange', marker='o')
        
        axes[0].set_title("SFT Loss Convergence")
        axes[0].set_xlabel("Training Step")
        axes[0].set_ylabel("Cross Entropy Loss")
        axes[0].legend()
        
    # Plot DPO
    if dpo_hist is not None:
        if 'train/loss' in dpo_hist.columns:
            dpo_train = dpo_hist.dropna(subset=['train/loss'])
            axes[1].plot(dpo_train['_step'], dpo_train['train/loss'], label='DPO Train Loss', color='green', alpha=0.7)
        if 'eval/loss' in dpo_hist.columns:
            dpo_eval = dpo_hist.dropna(subset=['eval/loss'])
            axes[1].plot(dpo_eval['_step'], dpo_eval['eval/loss'], label='DPO Eval Loss', color='red', marker='o')
            
        axes[1].set_title("DPO Loss Convergence")
        axes[1].set_xlabel("Training Step")
        axes[1].set_ylabel("DPO Loss")
        axes[1].legend()

    plt.tight_layout()
    out_path = os.path.join(out_dir, "loss_comparison.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Saved {out_path}")
    plt.close()

def plot_dpo_metrics(dpo_hist, out_dir):
    """Plot DPO specific metrics: Rewards and Logprobs."""
    if dpo_hist is None:
        return
        
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Plot 1: Reward Margins
    if 'eval/rewards/margins' in dpo_hist.columns:
        df = dpo_hist.dropna(subset=['eval/rewards/margins'])
        axes[0].plot(df['_step'], df['eval/rewards/margins'], label='Margin', color='purple', marker='o')
        axes[0].set_title("DPO Reward Margin (Chosen - Rejected)")
        axes[0].set_xlabel("Training Step")
        axes[0].set_ylabel("Margin")
        axes[0].legend()
        
    # Plot 2: Chosen vs Rejected Rewards
    has_chosen = 'eval/rewards/chosen' in dpo_hist.columns
    has_rejected = 'eval/rewards/rejected' in dpo_hist.columns
    if has_chosen and has_rejected:
        df = dpo_hist.dropna(subset=['eval/rewards/chosen', 'eval/rewards/rejected'])
        axes[1].plot(df['_step'], df['eval/rewards/chosen'], label='Chosen Reward', color='green', marker='o')
        axes[1].plot(df['_step'], df['eval/rewards/rejected'], label='Rejected Reward', color='red', marker='x')
        axes[1].set_title("DPO Implicit Rewards")
        axes[1].set_xlabel("Training Step")
        axes[1].set_ylabel("Reward")
        axes[1].legend()
        
    # Plot 3: Reward Accuracies
    if 'eval/rewards/accuracies' in dpo_hist.columns:
        df = dpo_hist.dropna(subset=['eval/rewards/accuracies'])
        axes[2].plot(df['_step'], df['eval/rewards/accuracies'], label='Accuracy', color='blue', marker='o')
        axes[2].set_title("DPO Preference Accuracy")
        axes[2].set_xlabel("Training Step")
        axes[2].set_ylabel("Accuracy")
        axes[2].legend()
        
    plt.tight_layout()
    out_path = os.path.join(out_dir, "dpo_training_metrics.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Saved {out_path}")
    plt.close()
    
    # Separate Plot for Logprobs
    has_lp_chosen = 'eval/logps/chosen' in dpo_hist.columns
    has_lp_rejected = 'eval/logps/rejected' in dpo_hist.columns
    
    if has_lp_chosen and has_lp_rejected:
        plt.figure(figsize=(8, 5))
        df = dpo_hist.dropna(subset=['eval/logps/chosen', 'eval/logps/rejected'])
        plt.plot(df['_step'], df['eval/logps/chosen'], label='Chosen Logps', color='green', marker='o')
        plt.plot(df['_step'], df['eval/logps/rejected'], label='Rejected Logps', color='red', marker='x')
        plt.title("DPO Log Probabilities")
        plt.xlabel("Training Step")
        plt.ylabel("Log Probability")
        plt.legend()
        
        out_path = os.path.join(out_dir, "dpo_logprobs.png")
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        print(f"Saved {out_path}")
        plt.close()

def plot_grpo_metrics(grpo_hist, out_dir):
    """Plot GRPO specific metrics (Placeholder)."""
    if grpo_hist is None:
        print("GRPO data not available. Skipping GRPO plots (Reward Convergence, KL Divergence).")
        return
        
    # Example logic for when GRPO runs are available
    # Reward Convergence
    reward_cols = [c for c in grpo_hist.columns if 'reward' in c.lower()]
    if reward_cols:
        plt.figure(figsize=(10, 6))
        for col in reward_cols:
            df = grpo_hist.dropna(subset=[col])
            plt.plot(df['_step'], df[col], label=col)
        plt.title("GRPO Reward Component Convergence")
        plt.xlabel("Training Step")
        plt.ylabel("Reward Value")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "reward_convergence.png"), dpi=300, bbox_inches="tight")
        plt.close()
        
    # KL Divergence
    kl_cols = [c for c in grpo_hist.columns if 'kl' in c.lower()]
    if kl_cols:
        plt.figure(figsize=(8, 5))
        for col in kl_cols:
            df = grpo_hist.dropna(subset=[col])
            plt.plot(df['_step'], df[col], label=col)
        plt.title("GRPO KL Divergence from Reference Policy")
        plt.xlabel("Training Step")
        plt.ylabel("KL Divergence")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "kl_divergence.png"), dpi=300, bbox_inches="tight")
        plt.close()


def main():
    api = wandb.Api()
    
    # Detect Project
    wandb_entity = os.environ.get("WANDB_ENTITY")
    wandb_project = os.environ.get("WANDB_PROJECT", "ask-before-answer")
    
    if wandb_entity:
        project_path = f"{wandb_entity}/{wandb_project}"
    else:
        # Infer entity from current user if not provided
        try:
            user = api.default_entity
            project_path = f"{user}/{wandb_project}"
        except Exception:
            project_path = wandb_project
            
    # For testing, we found rl4aa is the entity from the scratch script.
    if project_path == "ask-before-answer":
        project_path = "rl4aa/ask-before-answer"
            
    print(f"Connecting to W&B Project: {project_path}")
    
    out_dir = "docs/plots"
    os.makedirs(out_dir, exist_ok=True)
    
    sft_hist = get_run_history(api, project_path, "sft_training")
    dpo_hist = get_run_history(api, project_path, "dpo_training")
    grpo_hist = get_run_history(api, project_path, "grpo_training")
    
    plot_loss_comparison(sft_hist, dpo_hist, out_dir)
    plot_dpo_metrics(dpo_hist, out_dir)
    plot_grpo_metrics(grpo_hist, out_dir)
    
    print("Done generating plots.")

if __name__ == "__main__":
    main()
