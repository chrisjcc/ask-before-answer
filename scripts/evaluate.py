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

from src.evaluation.judge import GeminiJudge, LocalGemmaJudge
from src.evaluation.metrics import ActionScorer
from src.inference.pipeline import ClarificationPipeline

load_dotenv()
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["WEAVE_PARALLELISM"] = "1"

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
        # Determine the true action based on AmbigQA schema
        ann = row.get("annotations", {})
        if isinstance(ann, list):
            ann_type = ann[0].get("type", "") if ann else ""
        elif isinstance(ann, dict):
            type_val = ann.get("type", "")
            ann_type = type_val[0] if isinstance(type_val, list) and len(type_val) > 0 else type_val
        else:
            ann_type = ""
            
        is_ambiguous = ann_type == "multipleQAs"
        expected_action = "Clarify" if is_ambiguous else "Answer"

        weave_dataset_rows.append(
            {
                "question": row["question"],
                "target": f"Action: {expected_action}\n"
                + str(row.get("ground_truth", "")),
            }
        )

    # Publish the dataset once so all models evaluate against the exact same version
    eval_dataset = weave.Dataset(
        name=f"{dataset_name.replace('/', '_')}_eval", rows=weave_dataset_rows
    )
    weave.publish(eval_dataset)

    # Setup Scorer
    judge_model_id = cfg.evaluation.get("judge_model", "gemini-2.0-flash")
    if "gemini" in judge_model_id.lower():
        judge = GeminiJudge(model_name=judge_model_id)
    else:
        judge = LocalGemmaJudge(model_id=judge_model_id)

    action_scorer = ActionScorer()

    all_results = {}
    evaluations_list = []

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
        # if the path exists locally, otherwise assume it's a HuggingFace hub path
        if is_peft and not os.path.isabs(model_path):
            local_path = os.path.join(cfg.project_dir, model_path)
            if os.path.exists(local_path):
                model_path = local_path

        logger.info(f"Evaluating model: {model_name} from {model_path}")

        # Instantiate Weave Model
        model = ClarificationModel(
            model_name=model_name, model_path=model_path, is_peft=is_peft
        )

        # Run Evaluation
        evaluation = weave.Evaluation(
            name=f"eval_{model_name}",
            dataset=eval_dataset,
            scorers=[judge, action_scorer],
        )

        logger.info(f"Running weave.Evaluation for {model_name}...")
        results = asyncio.run(
            evaluation.evaluate(model, __weave={"display_name": f"{model_name}_eval"})
        )
        evaluations_list.append(evaluation)

        # Format the metric summary nicely
        metrics = results.get("LocalGemmaJudge") or results.get("GeminiJudge") or {}
        action_metrics = results.get("ActionScorer") or {}

        all_results[model_name] = {
            "ambiguity_detection": metrics.get("ambiguity_detection", {}).get(
                "mean", 0.0
            ),
            "clarification_quality": metrics.get("clarification_quality", {}).get(
                "mean", 0.0
            ),
            "usefulness": metrics.get("usefulness", {}).get("mean", 0.0),
            "model_accuracy": action_metrics.get("correct_action", {}).get(
                "true_fraction", 0.0
            ),
        }

        # ---------------------------------------------------------
        # W&B Model Registry Integration
        # ---------------------------------------------------------
        try:
            model_ref = weave.publish(model)
            ENTITY = os.environ.get("WANDB_ENTITY")
            PROJECT = os.environ.get("WANDB_PROJECT")

            if ENTITY and PROJECT:
                import wandb

                models_object_name = model_ref.name
                models_object_version = model_ref.digest

                models_url = f"https://wandb.ai/{ENTITY}/{PROJECT}/weave/objects/{models_object_name}/versions/{models_object_version}"
                models_link = f"weave://{ENTITY}/{PROJECT}/object/{models_object_name}:{models_object_version}"

                with wandb.init(
                    project=PROJECT,
                    entity=ENTITY,
                    job_type="model-registry",
                    reinit=True,
                ) as run:
                    artifact_model = wandb.Artifact(
                        name=f"Clarifier-{model_name}",
                        type="model",
                        description=f"Weave Model Link for {model_name}",
                        metadata={"url": models_url},
                    )
                    artifact_model.add_reference(
                        models_link, name="model", checksum=False
                    )
                    run.log_artifact(artifact_model, aliases=[models_object_version])
                    run.link_artifact(
                        artifact_model,
                        target_path=f"{ENTITY}/{PROJECT}/AskBeforeAnswer-Models",
                    )
                logger.info(f"Registered {model_name} to W&B Model Registry.")
        except Exception as e:
            logger.warning(f"Could not register model to W&B Registry: {e}")

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

    # Publish native Weave Leaderboard
    logger.info("Publishing Weave Leaderboard...")
    try:
        from weave.flow import leaderboard
        from weave.trace.ref_util import get_ref

        # We assume the scorer name is the class name of the judge used
        scorer_name = judge.__class__.__name__

        columns = []
        for eval_obj in evaluations_list:
            try:
                eval_ref = get_ref(eval_obj).uri()
                columns.extend(
                    [
                        leaderboard.LeaderboardColumn(
                            evaluation_object_ref=eval_ref,
                            scorer_name=scorer_name,
                            summary_metric_path="ambiguity_detection.mean",
                        ),
                        leaderboard.LeaderboardColumn(
                            evaluation_object_ref=eval_ref,
                            scorer_name=scorer_name,
                            summary_metric_path="clarification_quality.mean",
                        ),
                        leaderboard.LeaderboardColumn(
                            evaluation_object_ref=eval_ref,
                            scorer_name=scorer_name,
                            summary_metric_path="usefulness.mean",
                        ),
                        leaderboard.LeaderboardColumn(
                            evaluation_object_ref=eval_ref,
                            scorer_name="ActionScorer",
                            summary_metric_path="correct_action.true_fraction",
                        ),
                    ]
                )
            except Exception as e:
                logger.warning(f"Could not get ref for evaluation {eval_obj.name}: {e}")

        if columns:
            spec = leaderboard.Leaderboard(
                name="Clarify-or-Act Ablation Leaderboard",
                description="Model ambiguity and clarify quality comparison.",
                columns=columns,
            )
            weave.publish(spec)
            logger.info("Check your W&B Weave dashboard for the dynamic leaderboard!")
    except ImportError:
        logger.warning(
            "weave.flow.leaderboard not found. Update weave to publish leaderboards."
        )


if __name__ == "__main__":
    main()
