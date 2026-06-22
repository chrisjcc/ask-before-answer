import asyncio
import json
import logging
import os
import threading

import hydra
import weave
from datasets import load_dataset
from dotenv import load_dotenv
from omegaconf import DictConfig

from src.evaluation.judge import LocalGemmaJudge
from src.inference.pipeline import ClarificationPipeline

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


_PIPELINE_CACHE = {}
_PIPELINE_LOCK = threading.Lock()
_INFERENCE_LOCK = threading.Lock()


def get_cached_pipeline(model_path: str, is_peft: bool):
    with _PIPELINE_LOCK:
        if model_path not in _PIPELINE_CACHE:
            # Clear old models to free VRAM
            _PIPELINE_CACHE.clear()
            import gc

            import torch

            gc.collect()
            torch.cuda.empty_cache()

            _PIPELINE_CACHE[model_path] = ClarificationPipeline(model_path, is_peft)
        return _PIPELINE_CACHE[model_path]


class ClarificationModel(weave.Model):
    model_name: str
    model_path: str
    is_peft: bool

    @weave.op()
    def predict(self, question: str) -> str:
        pipeline = get_cached_pipeline(self.model_path, self.is_peft)
        with _INFERENCE_LOCK:
            return pipeline.generate(question)


@hydra.main(version_base="1.3", config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    logger.info("Starting systematic evaluation pipeline...")
    weave.init(os.environ.get("WANDB_PROJECT", "ask-before-answer"))

    # Load and publish dataset
    dataset_name = cfg.evaluation.dataset_name
    split_name = cfg.evaluation.split
    logger.info(f"Loading evaluation dataset: {dataset_name} ({split_name} split)")
    dataset = load_dataset(dataset_name, split=split_name)

    max_samples = cfg.evaluation.get("max_samples", 50)
    dataset = dataset.select(range(min(max_samples, len(dataset))))

    # Preprocess dataset to the format expected by Weave
    weave_dataset_rows = []
    for row in dataset:
        weave_dataset_rows.append(
            {
                "question": row["question"],
                "target": row.get(
                    "ground_truth", ""
                ),  # 'target' is the standard key for Weave Scorers
            }
        )

    # Publish the dataset once so all models evaluate against the exact same version
    eval_dataset = weave.Dataset(
        name=f"{dataset_name.replace('/', '_')}_eval", rows=weave_dataset_rows
    )
    weave.publish(eval_dataset)

    # Setup Scorer
    judge = LocalGemmaJudge(model_id="google/gemma-4-e4b-it")

    # Store results for reporting
    all_results = {}

    # Loop over models to evaluate
    models_to_eval = cfg.evaluation.get("models_to_evaluate", [])
    if not models_to_eval:
        logger.warning("No models_to_evaluate found in config.")
        return

    for model_cfg in models_to_eval:
        model_name = model_cfg.name
        model_path = model_cfg.path
        is_peft = model_cfg.is_peft

        # For non-base models, prepend the project directory to the path
        if is_peft and not os.path.isabs(model_path):
            model_path = os.path.join(cfg.project_dir, model_path)

        logger.info(f"Evaluating model: {model_name} from {model_path}")

        # Instantiate Weave Model
        model = ClarificationModel(
            model_name=model_name, model_path=model_path, is_peft=is_peft
        )

        # Run Evaluation
        evaluation = weave.Evaluation(
            name=f"eval_{model_name}",
            dataset=eval_dataset,
            scorers=[judge],
        )

        logger.info(f"Running weave.Evaluation for {model_name}...")
        results = asyncio.run(evaluation.evaluate(model))

        # Format the metric summary nicely
        metrics = results.get("LocalGemmaJudge") or results.get("GeminiJudge") or {}
        all_results[model_name] = {
            "ambiguity_detection": metrics.get("ambiguity_detection", {}).get(
                "mean", 0.0
            ),
            "clarification_quality": metrics.get("clarification_quality", {}).get(
                "mean", 0.0
            ),
            "usefulness": metrics.get("usefulness", {}).get("mean", 0.0),
        }

        # Cleanup model from GPU memory to make room for the next one
        if hasattr(model, "_pipeline"):
            del model._pipeline
        import torch

        torch.cuda.empty_cache()
        import gc

        gc.collect()

    # Save summary results to JSON for the report generator
    os.makedirs(os.path.join(cfg.project_dir, "results"), exist_ok=True)
    results_path = os.path.join(cfg.project_dir, "results", "weave_eval_summary.json")
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)

    logger.info(f"All evaluations complete. Summary saved to {results_path}")
    logger.info("Check your W&B Weave dashboard for the dynamic leaderboard!")


if __name__ == "__main__":
    main()
