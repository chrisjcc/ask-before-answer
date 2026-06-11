import json
import logging
import os

import hydra
from omegaconf import DictConfig

from src.evaluation.judge import run_evaluation_suite
from src.inference.pipeline import ClarificationPipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@hydra.main(version_base="1.3", config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    model_name = cfg.get("model_name", "sft")
    logger.info(f"Starting evaluation pipeline for {model_name}...")

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

    # Run evaluation on a small test set (e.g. 10 examples)
    # Ideally load this from cfg.data.test_file
    test_questions = [
        "When did The Simpsons first air?",
        "Who won the US Open?",
        "How do I make pasta?",
        "What is the capital of France?",
    ]

    outputs = []
    for q in test_questions:
        resp = pipeline.generate(q)
        logger.info(f"Q: {q}\\nA: {resp}\\n---")
        outputs.append({"question": q, "response": resp})

    # Evaluate with Gemini
    if os.environ.get("GEMINI_API_KEY"):
        logger.info("Running Gemini evaluation suite...")
        metrics = run_evaluation_suite(outputs)
        logger.info(f"Evaluation Metrics: {json.dumps(metrics, indent=2)}")

        # Save metrics
        os.makedirs(os.path.join(cfg.project_dir, "results"), exist_ok=True)
        with open(
            os.path.join(cfg.project_dir, f"results/{model_name}_metrics.json"), "w"
        ) as f:
            json.dump(metrics, f, indent=2)
    else:
        logger.warning("GEMINI_API_KEY not found. Skipping Gemini evaluation.")

    logger.info("Evaluation complete.")


if __name__ == "__main__":
    main()
