import re
import ast
from typing import Dict, Any

from ask_before_answer.rewards.base import Criterion


class FormatCriterion(Criterion):
    def __init__(self, weight: float = 1.0, penalty: float = -2.0):
        super().__init__(name="format", weight=weight, description="Checks for exact structural blocks.")
        self.penalty = penalty

    def evaluate_single(self, prompt: str, completion: str, kwargs: Dict[str, Any]) -> float:
        has_action = "Action:" in completion
        has_reasoning = "Reasoning:" in completion
        has_facets = "Facets:" in completion
        has_response = "Response:" in completion

        if has_action and has_reasoning and has_facets and has_response:
            return 1.0  # Multiplying by weight is handled by the Rubric adapter
        else:
            return self.penalty


class ActionCriterion(Criterion):
    def __init__(self, weight: float = 1.0, penalty: float = -1.0):
        super().__init__(name="action", weight=weight, description="Matches predicted action against target.")
        self.penalty = penalty

    def evaluate_single(self, prompt: str, completion: str, kwargs: Dict[str, Any]) -> float:
        # In batch execution, we need to extract the target for THIS specific completion.
        # However, TRL passes kwargs as batched lists. The Rubric adapter passes kwargs entirely, 
        # but since TRL calls the function with batched prompts/completions, kwargs["target_action"] is a list.
        # We need evaluate_single to handle the target action string directly.
        # Wait, the base class evaluate_batch zips prompts and completions but passes kwargs as is.
        raise NotImplementedError("ActionCriterion requires batch-aware evaluation due to TRL kwargs.")

    def evaluate_batch(self, prompts: list[str], completions: list[str], kwargs: Dict[str, Any]) -> list[float]:
        rewards = []
        target_actions = kwargs.get("target_action", [])
        
        for i, text in enumerate(completions):
            target = target_actions[i] if i < len(target_actions) else None
            action_match = re.search(r"Action:\s*(Clarify|Answer)", text)
            
            if action_match and target:
                pred_action = action_match.group(1)
                if pred_action == target:
                    rewards.append(1.0)
                else:
                    rewards.append(self.penalty)
            else:
                rewards.append(self.penalty)
                
        return rewards


class FacetLogicCriterion(Criterion):
    def __init__(self, weight: float = 0.5, penalty: float = -0.5):
        super().__init__(name="facet_logic", weight=weight, description="Checks facet presence based on action.")
        self.penalty = penalty

    def evaluate_single(self, prompt: str, completion: str, kwargs: Dict[str, Any]) -> float:
        action_match = re.search(r"Action:\s*(Clarify|Answer)", completion)
        facets_match = re.search(r"Facets:\s*(\[.*?\])", completion, re.DOTALL)

        if not action_match or not facets_match:
            return self.penalty

        pred_action = action_match.group(1)
        facets_str = facets_match.group(1)

        try:
            facets = ast.literal_eval(facets_str)
            if not isinstance(facets, list):
                facets = []
        except Exception:
            facets = []

        if pred_action == "Clarify":
            return 1.0 if len(facets) > 0 else self.penalty
        else:
            return 1.0 if len(facets) == 0 else self.penalty


class AccuracyCriterion(Criterion):
    def __init__(
        self, 
        weight: float = 1.0, 
        scale: float = 1.5, 
        shift: float = -0.5, 
        miss_penalty: float = -1.0, 
        format_penalty: float = -0.5
    ):
        super().__init__(name="accuracy", weight=weight, description="Factual accuracy token overlap.")
        self.scale = scale
        self.shift = shift
        self.miss_penalty = miss_penalty
        self.format_penalty = format_penalty

    def evaluate_batch(self, prompts: list[str], completions: list[str], kwargs: Dict[str, Any]) -> list[float]:
        rewards = []
        target_responses = kwargs.get("target_response", [])
        target_actions = kwargs.get("target_action", [])

        for i, text in enumerate(completions):
            target_resp = target_responses[i] if i < len(target_responses) else ""
            target_act = target_actions[i] if i < len(target_actions) else ""

            if target_act != "Answer":
                rewards.append(0.0)
                continue

            response_match = re.search(r"Response:\s*(.*)", text, re.DOTALL)
            if not response_match or not target_resp:
                rewards.append(self.format_penalty)
                continue

            pred_resp = response_match.group(1).strip()
            pred_words = set(re.findall(r"\b\w+\b", pred_resp.lower()))
            target_words = set(re.findall(r"\b\w+\b", target_resp.lower()))

            if not target_words:
                rewards.append(0.0)
                continue

            intersection = pred_words.intersection(target_words)
            if not intersection:
                rewards.append(self.miss_penalty)
                continue

            recall = len(intersection) / len(target_words)
            precision = len(intersection) / len(pred_words)
            f1 = 2 * (precision * recall) / (precision + recall)
            
            # The base reward calculation before applying Criterion weight
            reward = (f1 * self.scale) + self.shift
            rewards.append(reward)

        return rewards
