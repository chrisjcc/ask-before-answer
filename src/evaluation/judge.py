import json
import logging
import os
import threading
from typing import Any

import torch
import weave
from google import genai
from google.genai import types
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)


class GeminiJudge(weave.Scorer):
    model_name: str = "gemini-2.0-flash"

    @weave.op()
    def score(self, target: Any, output: str, question: str = "") -> dict:
        """Evaluation scorer that returns scalar metrics."""
        # Using the existing evaluate_response method internally.
        # Weave passes target, output automatically,
        # but we need to fetch the question.
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            return {
                "ambiguity_detection": 0.0,
                "clarification_quality": 0.0,
                "usefulness": 0.0,
                "justification": "GEMINI_API_KEY not set",
            }

        client = genai.Client(api_key=api_key)

        prompt = (
            "You are an expert judge evaluating clarification-seeking "
            "behavior in an AI agent.\\n\\n"
            f"Question: {question}\\n"
            f"Agent Response: {output}\\n"
            f"Ground Truth: {target}\\n\\n"
            "Evaluate the Agent Response on the following criteria:\\n"
            "1. Ambiguity Detection F1\\n"
            "2. Clarification Quality F1\\n"
            "3. Clarification Usefulness\\n\\n"
            "Return a JSON object with scores between 0.0 and 1.0 for each "
            "metric, and a short justification.\\n"
            "Format:\\n"
            "{\\n"
            '    "ambiguity_detection": 1.0,\\n'
            '    "clarification_quality": 0.8,\\n'
            '    "usefulness": 0.9,\\n'
            '    "justification": "..."\\n'
            "}"
        )
        max_retries = 5
        base_delay = 10  # seconds

        for attempt in range(max_retries):
            try:
                response = client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                    ),
                )
                res = json.loads(response.text)
                break  # Success
            except Exception as e:
                error_str = str(e)
                if "429" in error_str or "503" in error_str:
                    if attempt < max_retries - 1:
                        # Exponential backoff with some jitter
                        import random

                        delay = base_delay * (2**attempt) + random.uniform(0, 2)
                        logger.warning(
                            f"Gemini API rate limit. Retrying in {delay:.1f}s..."
                        )
                        import time

                        time.sleep(delay)
                        continue
                logger.error(f"Error calling Gemini API: {e}")
                return {
                    "ambiguity_detection": 0.0,
                    "clarification_quality": 0.0,
                    "usefulness": 0.0,
                    "justification": error_str,
                }
        else:
            return {
                "ambiguity_detection": 0.0,
                "clarification_quality": 0.0,
                "usefulness": 0.0,
                "justification": "Max retries exceeded due to rate limits",
            }

        return {
            "ambiguity_detection": float(res.get("ambiguity_detection", 0.0)),
            "clarification_quality": float(res.get("clarification_quality", 0.0)),
            "usefulness": float(res.get("usefulness", 0.0)),
            "justification": res.get("justification", ""),
        }


_LOCAL_JUDGE_CACHE = {}
_LOCAL_JUDGE_LOCK = threading.Lock()
_LOCAL_INFERENCE_LOCK = threading.Lock()


def get_local_judge(model_id: str):
    with _LOCAL_JUDGE_LOCK:
        if model_id not in _LOCAL_JUDGE_CACHE:
            _LOCAL_JUDGE_CACHE.clear()
            import gc

            gc.collect()
            torch.cuda.empty_cache()

            logger.info(f"Loading local judge model {model_id}...")
            tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True,
            )
            model.eval()
            _LOCAL_JUDGE_CACHE[model_id] = (model, tokenizer)
        return _LOCAL_JUDGE_CACHE[model_id]


class LocalGemmaJudge(weave.Scorer):
    model_id: str = "google/gemma-4-12b-it"

    @weave.op()
    def score(self, target: Any, output: str, question: str = "") -> dict:
        prompt = (
            "You are an expert judge evaluating clarification-seeking "
            "behavior in an AI agent.\\n\\n"
            f"Question: {question}\\n"
            f"Agent Response: {output}\\n"
            f"Ground Truth: {target}\\n\\n"
            "Evaluate the Agent Response on the following criteria:\\n"
            "1. Ambiguity Detection F1\\n"
            "2. Clarification Quality F1\\n"
            "3. Clarification Usefulness\\n\\n"
            "Return ONLY a valid JSON object with scores between 0.0 and 1.0 for each "
            "metric, and a short justification. Escape any double quotes inside "
            "the justification.\\n"
            "Format:\\n"
            "{\\n"
            '    "ambiguity_detection": 1.0,\\n'
            '    "clarification_quality": 0.8,\\n'
            '    "usefulness": 0.9,\\n'
            '    "justification": "..."\\n'
            "}"
        )

        model, tokenizer = get_local_judge(self.model_id)

        messages = [{"role": "user", "content": prompt}]
        input_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

        pad_token_id = tokenizer.pad_token_id
        if pad_token_id is None:
            eos = tokenizer.eos_token_id
            pad_token_id = eos[0] if isinstance(eos, list) else eos

        with _LOCAL_INFERENCE_LOCK:
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=300,
                    do_sample=False,
                    pad_token_id=pad_token_id,
                )

        gen_tokens = outputs[0][inputs["input_ids"].shape[1] :]
        response_text = tokenizer.decode(gen_tokens, skip_special_tokens=True)

        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0].strip()
        elif "```" in response_text:
            response_text = response_text.split("```")[1].split("```")[0].strip()

        try:
            res = json.loads(response_text)
        except json.JSONDecodeError:
            import re

            res = {}
            for key in ["ambiguity_detection", "clarification_quality", "usefulness"]:
                match = re.search(rf'"{key}"\s*:\s*([0-9.]+)', response_text)
                if match:
                    res[key] = float(match.group(1))
            just_match = re.search(
                r'"justification"\s*:\s*"(.*?)"\s*\}?', response_text, re.DOTALL
            )
            if just_match:
                res["justification"] = just_match.group(1)

        try:
            return {
                "ambiguity_detection": float(res.get("ambiguity_detection", 0.0)),
                "clarification_quality": float(res.get("clarification_quality", 0.0)),
                "usefulness": float(res.get("usefulness", 0.0)),
                "justification": res.get("justification", ""),
            }
        except Exception as e:
            logger.error(
                f"Failed to parse Local Judge JSON: {e} \\nRaw output: {response_text}"
            )
            return {
                "ambiguity_detection": 0.0,
                "clarification_quality": 0.0,
                "usefulness": 0.0,
                "justification": f"Parse error: {str(e)}",
            }
