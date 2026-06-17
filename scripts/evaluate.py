import asyncio
import json
import logging
import os

import hydra
import weave
from datasets import load_dataset
from dotenv import load_dotenv
from omegaconf import DictConfig

from src.evaluation.judge import GeminiJudge
from src.inference.pipeline import ClarificationPipeline

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@hydra.main(version_base="1.3", config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    model_name = cfg.get("model_name", "sft")
    logger.info(f"Starting evaluation pipeline for {model_name}...")

    weave.init(os.environ.get("WANDB_PROJECT", "ask-before-answer"))

    # Determine model path
    if model_name == "base":
        model_path = "Qwen/Qwen2.5-7B-Instruct"
        is_peft = False
    elif model_name == "sft":
        model_path = os.path.join(cfg.models_dir, "sft", "final")
        is_peft = True
    elif model_name == "sft_dpo":
        model_path = os.path.join(cfg.models_dir, "dpo", "final")
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

    # Prepare Weave Dataset
    weave_dataset = []
    for row in dataset:
        weave_dataset.append(
            {"question": row["question"], "ground_truth": row.get("ground_truth", "")}
        )

    # Evaluate with Gemini
    if os.environ.get("GEMINI_API_KEY"):
        logger.info("Running Weave evaluation suite...")
        judge = GeminiJudge()

        @weave.op()
        def model_predict(question: str) -> str:
            return pipeline.generate(question)

        evaluation = weave.Evaluation(dataset=weave_dataset, scorers=[judge.score])

        results = asyncio.run(evaluation.evaluate(model_predict))

        # Save metrics
        os.makedirs(os.path.join(cfg.project_dir, "results"), exist_ok=True)
        metrics = results.get("score", {})
        logger.info(f"Evaluation Metrics: {json.dumps(metrics, indent=2)}")
        with open(
            os.path.join(cfg.project_dir, f"results/{model_name}_metrics.json"), "w"
        ) as f:
            json.dump(metrics, f, indent=2)
    else:
        logger.warning("GEMINI_API_KEY not found. Skipping Gemini evaluation.")

    logger.info("Evaluation complete.")


if __name__ == "__main__":
    main()
