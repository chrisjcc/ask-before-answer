"""Inference pipeline for clarification generation.

This module provides the core inference class for the ClarifyOrAct architecture,
routing ambiguous questions to clarification requests and clear questions to direct answers.
"""

import logging
from typing import List, Optional

import torch
import weave
from peft import AutoPeftModelForCausalLM
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)


class ClarifyOrActPipeline:
    """A dual-routing inference pipeline for ambiguity resolution.

    This class wraps the trained models (either base or LoRA-adapter) and provides
    methods to generate clarification-seeking questions or direct answers based on
    the input question's inherent ambiguity.
    """

    def __init__(self, model_path: str, is_peft: bool = True) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Loading inference model from {model_path} on {self.device}...")

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True
        )
        if self.tokenizer.chat_template is None:
            logger.warning(
                f"Tokenizer {model_path} missing chat_template. Falling back to Qwen."
            )
            base_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
            self.tokenizer.chat_template = base_tokenizer.chat_template

        if is_peft:
            self.model = AutoPeftModelForCausalLM.from_pretrained(
                model_path, torch_dtype=torch.bfloat16, device_map="auto"
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path, torch_dtype=torch.bfloat16, device_map="auto"
            )

        self.model.eval()

    @weave.op()
    def generate(self, question: str, system_prompt: Optional[str] = None) -> str:
        """Run single-turn inference for clarification seeking.

        Args:
            question (str): The user's input query.
            system_prompt (Optional[str]): A custom system prompt overriding the default.

        Returns:
            str: The raw generated string from the model.
        """
        if system_prompt is None:
            system_prompt = (
                "You are a helpful assistant. "
                "Given a question, you must decide whether it is ambiguous or not. "
                "Output MUST follow this format:\n"
                "Action: Clarify|Answer\n"
                "Reasoning: <your reasoning>\n"
                "Facets: <list of facets if ambiguous, else empty>\n"
                "Response: <clarifying question or direct answer>"
            )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ]

        input_text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = self.tokenizer(input_text, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=300,
                do_sample=False,
                temperature=None,
                top_p=None,
                top_k=None,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            )

        # Decode only the generated response
        gen_tokens = outputs[0][inputs["input_ids"].shape[1] :]
        return self.tokenizer.decode(gen_tokens, skip_special_tokens=True)

    @weave.op()
    def batch_generate(self, questions: List[str]) -> List[str]:
        """Run batch inference."""
        return [self.generate(q) for q in questions]
