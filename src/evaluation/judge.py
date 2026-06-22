import json
import logging
import os
from typing import Any

import weave
from google import genai
from google.genai import types

logger = logging.getLogger(__name__)


class GeminiJudge(weave.Scorer):
    model_name: str = "gemini-2.5-flash"

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
