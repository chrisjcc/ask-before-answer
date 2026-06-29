"""Interactive CLI for inference.

This script allows you to chat with the trained ClarifyOrAct models in
your terminal to manually evaluate their clarification-seeking behaviors.
"""

import logging
import os

import hydra
from omegaconf import DictConfig

from src.inference.pipeline import ClarifyOrActPipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@hydra.main(version_base="1.3", config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    model_name = cfg.get("model_name", "sft")
    logger.info(f"Starting interactive inference for {model_name}...")

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

    pipeline = ClarifyOrActPipeline(model_path, is_peft=is_peft)

    print("\\nInteractive inference started. Type 'quit' or 'exit' to stop.")
    while True:
        try:
            q = input("\\nEnter a question: ")
            if q.lower() in ["quit", "exit"]:
                break
            if not q.strip():
                continue

            resp = pipeline.generate(q)
            print(f"\\nAssistant: {resp}")
        except KeyboardInterrupt:
            break

    logger.info("Interactive inference complete.")


if __name__ == "__main__":
    main()
