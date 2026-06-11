import logging
from typing import List

import torch
from peft import AutoPeftModelForCausalLM
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)


class ClarificationPipeline:
    def __init__(self, model_path: str, is_peft: bool = True):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Loading inference model from {model_path} on {self.device}...")

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True
        )

        if is_peft:
            self.model = AutoPeftModelForCausalLM.from_pretrained(
                model_path, torch_dtype=torch.bfloat16, device_map="auto"
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path, torch_dtype=torch.bfloat16, device_map="auto"
            )

        self.model.eval()

    def generate(self, question: str, system_prompt: str = None) -> str:
        """Run single-turn inference for clarification seeking."""
        if system_prompt is None:
            system_prompt = (
                "You are a question understanding agent. For each user question:\\n"
                "1) Decide if it is ambiguous.\\n"
                "2) Explain your reasoning.\\n"
                "3) List facets if ambiguous.\\n"
                "4) Ask a clarifying question OR give a direct answer."
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
                **inputs, max_new_tokens=300, temperature=0.7, top_p=0.9, do_sample=True
            )

        # Decode only the generated response
        gen_tokens = outputs[0][inputs["input_ids"].shape[1] :]
        return self.tokenizer.decode(gen_tokens, skip_special_tokens=True)

    def batch_generate(self, questions: List[str]) -> List[str]:
        """Run batch inference."""
        return [self.generate(q) for q in questions]
