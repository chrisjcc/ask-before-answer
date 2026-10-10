import logging
import re
from typing import Any, Dict, List

import torch

from ask_before_answer.rewards.base import Criterion

logger = logging.getLogger(__name__)


class JudgeRubric(Criterion):
    """GPU-native Hugging Face local judge implementation."""

    def __init__(
        self, name: str, weight: float, judge_model_path: str, judge_prompt: str
    ):
        super().__init__(name=name, weight=weight, description="Local LLM-as-a-judge")
        self.judge_model_path = judge_model_path
        self.judge_prompt = judge_prompt

        self.model = None
        self.tokenizer = None

    def _load_model(self):
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            logger.info(f"Loading Local Judge Model: {self.judge_model_path}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.judge_model_path)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.judge_model_path,
                device_map="auto",
                torch_dtype=torch.bfloat16,
                load_in_4bit=True,
            )
            # Ensure padding token is set
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
        except ImportError:
            logger.error(
                "transformers or bitsandbytes is missing. Cannot load local judge."
            )
            raise

    @torch.inference_mode()
    def evaluate_batch(
        self, prompts: List[str], completions: List[str], kwargs: Dict[str, Any]
    ) -> List[float]:
        if self.model is None:
            self._load_model()

        rewards = []
        judge_inputs = []

        for prompt, completion in zip(prompts, completions):
            # Format the grading prompt
            eval_text = self.judge_prompt.format(prompt=prompt, response=completion)
            judge_inputs.append(eval_text)

        # Tokenize batch
        inputs = self.tokenizer(
            judge_inputs, return_tensors="pt", padding=True, truncation=True
        ).to(self.model.device)

        # Generate scores
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=20,
            temperature=0.0,
            do_sample=False,
            pad_token_id=self.tokenizer.pad_token_id,
        )

        # Decode and parse
        for i, out_tokens in enumerate(outputs):
            # Slice off the prompt tokens to just get the generation
            gen_tokens = out_tokens[inputs["input_ids"][i].shape[0] :]
            response_text = self.tokenizer.decode(gen_tokens, skip_special_tokens=True)

            try:
                # Find the first floating point number or integer
                match = re.search(r"[-+]?\d*\.\d+|\d+", response_text.strip())
                if match:
                    score = float(match.group())
                    # Bound score to [-1, 1]
                    score = max(-1.0, min(1.0, score))
                    rewards.append(score)
                else:
                    rewards.append(0.0)
            except Exception as e:
                logger.warning(
                    f"Failed to parse judge output: {response_text}. Error: {e}"
                )
                rewards.append(0.0)

        return rewards
