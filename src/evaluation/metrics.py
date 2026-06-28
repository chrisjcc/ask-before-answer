"""Syntactic and deterministic evaluation metrics.

This module provides the ActionScorer class, which evaluates whether the agent
correctly adhered to the required `Action: Clarify|Answer` output syntax format.
"""

import re
from typing import Any, Dict

import weave


class ActionScorer(weave.Scorer):
    """A Weave Scorer that validates deterministic structural formatting.

    This scorer parses the agent's raw string output to ensure it matches
    the strict `Action: [Clarify|Answer]` format and tracks true/false positive
    rates for clarification actions across the dataset.
    """

    @weave.op()
    def score(self, target: Any, output: str, **kwargs) -> Dict[str, Any]:
        # Extract true action from ground truth
        true_action_match = re.search(
            r"Action:\s*(Clarify|Answer)", str(target), re.IGNORECASE
        )
        true_action = true_action_match.group(1).title() if true_action_match else None

        # Extract predicted action from model output
        pred_action_match = re.search(
            r"Action:\s*(Clarify|Answer)", str(output), re.IGNORECASE
        )
        pred_action = pred_action_match.group(1).title() if pred_action_match else None

        # Determine correctness
        correct_action = (true_action == pred_action) and (true_action is not None)

        is_true_clarify = true_action == "Clarify"
        is_pred_clarify = pred_action == "Clarify"

        return {
            "correct_action": correct_action,
            "true_action": true_action or "Unknown",
            "pred_action": pred_action or "Unknown",
            "is_true_positive": is_true_clarify and is_pred_clarify,
            "is_false_positive": (not is_true_clarify) and is_pred_clarify,
            "is_false_negative": is_true_clarify and (not is_pred_clarify),
        }

    @weave.op()
    def summarize(self, score_rows: list[dict]) -> dict:
        tp = sum(row.get("is_true_positive", False) for row in score_rows)
        fp = sum(row.get("is_false_positive", False) for row in score_rows)
        fn = sum(row.get("is_false_negative", False) for row in score_rows)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        correct = sum(row.get("correct_action", False) for row in score_rows)
        accuracy = correct / len(score_rows) if score_rows else 0.0

        return {
            "accuracy": accuracy,
            "clarify_precision": precision,
            "clarify_recall": recall,
            "clarify_f1": f1,
        }
