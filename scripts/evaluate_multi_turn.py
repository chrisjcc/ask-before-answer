"""Multi-turn evaluation script for ClarQ-LLM inspired framework."""

import logging
import os

import hydra
from datasets import load_dataset
from dotenv import load_dotenv
from omegaconf import DictConfig
from tqdm import tqdm

from ask_before_answer.evaluation.multi_turn import (
    ProviderAgent,
    SeekerAgent,
    simulate_conversation,
)
from ask_before_answer.inference.pipeline import ClarifyOrActPipeline

load_dotenv()
os.environ["TOKENIZERS_PARALLELISM"] = "false"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@hydra.main(version_base="1.3", config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Execute the multi-turn evaluation loop."""
    dataset_name = cfg.evaluation.dataset_name
    config_name = cfg.evaluation.get("config_name", "light")
    split_name = cfg.evaluation.split
    max_turns = cfg.evaluation.get("multi_turn_max_turns", 3)

    logger.info(f"Loading dataset: {dataset_name}")
    dataset = load_dataset(
        dataset_name, config_name, split=split_name, trust_remote_code=False
    )
    
    # We only want to evaluate ambiguous questions for the multi-turn loop
    ambiguous_samples = []
    for row in dataset:
        ann = row.get("annotations", {})
        ann_type = ""
        if isinstance(ann, list) and len(ann) > 0:
            ann_type = ann[0].get("type", "")
        elif isinstance(ann, dict):
            type_val = ann.get("type", "")
            ann_type = (
                type_val[0]
                if isinstance(type_val, list) and len(type_val) > 0
                else type_val
            )
            
        if ann_type == "multipleQAs":
            # Extract the hidden interpretations
            qa_pairs = []
            if isinstance(ann, list):
                qa_pairs = ann[0].get("qaPairs", [])
            elif isinstance(ann, dict):
                qa_pairs = ann.get("qaPairs", [])
            
            if qa_pairs:
                ambiguous_samples.append({
                    "question": row["question"],
                    "disambiguations": qa_pairs
                })
                
    max_samples = min(cfg.evaluation.get("max_samples", 50), len(ambiguous_samples))
    ambiguous_samples = ambiguous_samples[:max_samples]
    logger.info(
        f"Filtered to {len(ambiguous_samples)} ambiguous samples for multi-turn eval."
    )
    
    models_to_eval = cfg.evaluation.get("models_to_evaluate", [])
    if not models_to_eval:
        logger.warning("No models_to_evaluate found in config.")
        return

    for model_cfg in models_to_eval:
        model_name = model_cfg.name
        model_path = model_cfg.path
        is_peft = model_cfg.is_peft
        
        if not os.path.isabs(model_path):
            local_path = os.path.join(cfg.project_dir, model_path)
            if os.path.exists(local_path):
                model_path = local_path
            elif model_path.startswith("models/") or model_path.startswith("./"):
                logger.warning(
                    f"Local model path {local_path} does not exist. "
                    f"Skipping {model_name}..."
                )
                continue
                
        logger.info(
            f"Evaluating model in multi-turn mode: {model_name} from {model_path}"
        )
        pipeline = ClarifyOrActPipeline(
            model_path, is_peft, base_model_id=cfg.evaluation.base_model_id
        )
        seeker = SeekerAgent(pipeline)
        
        success_count = 0
        total_turns = 0
        
        for sample in tqdm(ambiguous_samples, desc=f"Evaluating {model_name}"):
            provider = ProviderAgent(sample["question"], sample["disambiguations"])
            result = simulate_conversation(
                seeker, provider, sample["question"], max_turns=max_turns
            )
            
            if result["success"]:
                success_count += 1
            total_turns += result["turns"]
            
            logger.info(
                f"Q: '{sample['question']}' -> Success: {result['success']} "
                f"(Turns: {result['turns']})"
            )
            
        sr = (success_count / max(1, len(ambiguous_samples))) * 100
        avg_turns = total_turns / max(1, len(ambiguous_samples))
        
        logger.info(f"--- Model: {model_name} Multi-Turn Results ---")
        logger.info(f"Task Success Rate: {sr:.2f}%")
        logger.info(f"Average Turns: {avg_turns:.2f}\n")
        
        # Explicit memory clean up to avoid OOM when iterating across models
        import gc

        import torch
        del pipeline
        del seeker
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
