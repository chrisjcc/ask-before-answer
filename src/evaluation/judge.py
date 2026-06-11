import os
import json
import logging
from typing import List, Dict, Any
from google import genai
from google.genai import types

logger = logging.getLogger(__name__)

class GeminiJudge:
    def __init__(self, api_key: str = None, model_name: str = "gemini-2.5-flash"):
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY is not set.")
        self.client = genai.Client(api_key=self.api_key)
        self.model_name = model_name

    def evaluate_response(self, question: str, response: str, ground_truth: Any = None) -> Dict[str, Any]:
        """Use Gemini to evaluate the model's clarification response."""
        prompt = f"""
        You are an expert judge evaluating clarification-seeking behavior in an AI agent.
        
        Question: {question}
        Agent Response: {response}
        Ground Truth: {ground_truth}
        
        Evaluate the Agent Response on the following criteria:
        1. Ambiguity Detection F1
        2. Clarification Quality F1
        3. Clarification Usefulness
        
        Return a JSON object with scores between 0.0 and 1.0 for each metric, and a short justification.
        Format:
        {{
            "ambiguity_detection": 1.0,
            "clarification_quality": 0.8,
            "usefulness": 0.9,
            "justification": "..."
        }}
        """
        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                ),
            )
            return json.loads(response.text)
        except Exception as e:
            logger.error(f"Error calling Gemini API: {e}")
            return {
                "ambiguity_detection": 0.0,
                "clarification_quality": 0.0,
                "usefulness": 0.0,
                "justification": str(e)
            }

def run_evaluation_suite(model_outputs: List[Dict[str, Any]]) -> Dict[str, float]:
    """Run full evaluation suite on a list of model outputs."""
    judge = GeminiJudge()
    results = []
    
    for item in model_outputs:
        score = judge.evaluate_response(
            question=item["question"],
            response=item["response"],
            ground_truth=item.get("ground_truth")
        )
        results.append(score)
        
    avg_scores = {
        "ambiguity_detection": sum(r["ambiguity_detection"] for r in results) / len(results) if results else 0,
        "clarification_quality": sum(r["clarification_quality"] for r in results) / len(results) if results else 0,
        "usefulness": sum(r["usefulness"] for r in results) / len(results) if results else 0,
    }
    
    return avg_scores
