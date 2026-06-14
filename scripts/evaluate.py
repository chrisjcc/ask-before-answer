import json
import logging
import os

import hydra
import wandb
from datasets import load_dataset
from dotenv import load_dotenv
from omegaconf import DictConfig

from src.evaluation.judge import run_evaluation_suite
from src.inference.pipeline import ClarificationPipeline

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@hydra.main(version_base="1.3", config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    model_name = cfg.get("model_name", "sft")
    logger.info(f"Starting evaluation pipeline for {model_name}...")

    wandb.init(
        project=os.environ.get("WANDB_PROJECT", "ask-before-answer"),
        name=f"eval_{model_name}",
    )

    # Determine model path
    if model_name == "base":
        model_path = "Qwen/Qwen2.5-7B-Instruct"
        is_peft = False
    elif model_name == "sft":
        model_path = os.path.join(cfg.models_dir, "sft_output", "final")
        is_peft = True
    elif model_name == "sft_dpo":
        model_path = os.path.join(cfg.models_dir, "dpo_output", "final")
        is_peft = True
    else:
        logger.error(f"Unknown model_name: {model_name}")
        return

    pipeline = ClarificationPipeline(model_path, is_peft=is_peft)

    # Run evaluation on the test set
    dataset_name = cfg.evaluation.dataset_name
    split_name = cfg.evaluation.split
    logger.info(f"Loading evaluation dataset: {dataset_name} ({split_name} split)")
    dataset = load_dataset(dataset_name, split=split_name)

    # Slice to max_samples
    max_samples = cfg.evaluation.get("max_samples", 20)
    dataset = dataset.select(range(min(max_samples, len(dataset))))

    outputs = []
    for example in dataset:
        q = example["question"]
        resp = pipeline.generate(q)
        logger.info(f"Q: {q}\\nA: {resp}\\n---")
        outputs.append({"question": q, "response": resp})

    # Evaluate with Gemini
    if os.environ.get("GEMINI_API_KEY"):
        logger.info("Running Gemini evaluation suite...")
        metrics = run_evaluation_suite(outputs)
        logger.info(f"Evaluation Metrics: {json.dumps(metrics, indent=2)}")
        wandb.log(metrics)

        # Save metrics
        os.makedirs(os.path.join(cfg.project_dir, "results"), exist_ok=True)
        with open(
            os.path.join(cfg.project_dir, f"results/{model_name}_metrics.json"), "w"
        ) as f:
            json.dump(metrics, f, indent=2)
    else:
        logger.warning("GEMINI_API_KEY not found. Skipping Gemini evaluation.")

    logger.info("Evaluation complete.")
    wandb.finish()


if __name__ == "__main__":
    main()
