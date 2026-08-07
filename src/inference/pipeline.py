"""Inference pipeline for clarification generation.

This module provides the core inference class for the ClarifyOrAct architecture,
routing ambiguous questions to clarification requests and clear questions
to direct answers using vLLM for high-throughput inference.
"""

import logging
from typing import List, Optional

import torch

try:
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
except ImportError:
    LLM, SamplingParams, LoRARequest = None, None, None

logger = logging.getLogger(__name__)

# =====================================================================
# vLLM Singleton Engine
# =====================================================================
_VLLM_ENGINE = None


def get_vllm_engine():
    """Initialize the vLLM engine exactly once to save massive VRAM overhead."""
    global _VLLM_ENGINE
    if _VLLM_ENGINE is None:
        if LLM is None:
            raise ImportError("vLLM is not installed. Please pip install vllm.")

        base_model_id = "unsloth/Qwen2.5-7B-Instruct"
        logger.info(f"Initializing vLLM base engine with {base_model_id}...")

        # Load the base model with LoRA support enabled for dynamic swapping
        _VLLM_ENGINE = LLM(
            model=base_model_id,
            enable_lora=True,
            max_lora_rank=64,
            trust_remote_code=True,
            dtype=(
                "bfloat16"
                if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
                else "float16"
            ),
        )
    return _VLLM_ENGINE


# =====================================================================


class ClarifyOrActPipeline:
    """A dual-routing inference pipeline for ambiguity resolution via vLLM.

    This class wraps the vLLM engine to dynamically swap LoRA adapters
    and process massive batches of inference requests concurrently using PagedAttention.
    """

    def __init__(self, model_path: str, is_peft: bool = True) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.is_peft = is_peft
        self.model_path = model_path

        if self.device == "cpu":
            logger.warning(
                "vLLM requires a GPU! CPU fallback is not supported. This will crash."
            )

        logger.info(f"Binding vLLM pipeline to model: {model_path} (is_peft={is_peft})")

        # Ensure the global engine is initialized
        self.llm = get_vllm_engine()

        # Initialize standard sampling parameters
        self.sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=300,
            skip_special_tokens=True,
        )

    def generate(self, question: str, system_prompt: Optional[str] = None) -> str:
        """Run single-turn inference (Not recommended for high throughput)."""
        return self.batch_generate([question], system_prompt)[0]

    def batch_generate(
        self, questions: List[str], system_prompt: Optional[str] = None
    ) -> List[str]:
        """Run high-throughput continuous batch inference using vLLM PagedAttention."""
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

        tokenizer = self.llm.get_tokenizer()

        prompts = []
        for q in questions:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": q},
            ]
            prompts.append(
                tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            )

        if self.is_peft:
            # Dynamically attach the LoRA adapter just for this batch!
            lora_request = LoRARequest(
                lora_name="current_adapter",
                lora_int_id=1,
                lora_local_path=self.model_path,
            )
            outputs = self.llm.generate(
                prompts,
                sampling_params=self.sampling_params,
                lora_request=lora_request,
                use_tqdm=False,
            )
        else:
            # Raw base model generation
            outputs = self.llm.generate(
                prompts, sampling_params=self.sampling_params, use_tqdm=False
            )

        return [output.outputs[0].text.strip() for output in outputs]
