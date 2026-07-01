"""Syntactic and deterministic evaluation metrics.

This module provides the ActionScorer class, which evaluates whether the agent
correctly adhered to the required `Action: Clarify|Answer` output syntax format.
"""

import ast
import difflib
import re
import string
from typing import Any, Dict

import weave


def semantic_match(a: str, b: str) -> bool:
    """
    Lenient answer comparison:
    - lowercase
    - remove punctuation
    - fuzzy token match
    """
    if not a or not b:
        return False

    table = str.maketrans("", "", string.punctuation)
    a_clean = a.lower().translate(table).strip()
    b_clean = b.lower().translate(table).strip()

    if a_clean == b_clean:
        return True

    ratio = difflib.SequenceMatcher(None, a_clean, b_clean).ratio()
    return ratio > 0.75


class ActionScorer(weave.Scorer):
    """A Weave Scorer that validates deterministic structural formatting.

    This scorer parses the agent's raw string output to ensure it matches
    the strict `Action: [Clarify|Answer]` format and tracks true/false positive
    rates for clarification actions across the dataset.
    """

    @weave.op()
    def score(self, target: Any, output: str, **kwargs) -> Dict[str, Any]:
        # Extract target action and response from ground truth
        target_str = str(target)
        target_action_match = re.search(
            r"Action:\s*(Clarify|Answer)", target_str, re.IGNORECASE
        )
        target_action = (
            target_action_match.group(1).title() if target_action_match else None
        )

        target_response_match = re.search(
            r"Response:\s*(.*)", target_str, re.IGNORECASE
        )
        target_response = (
            target_response_match.group(1).strip() if target_response_match else ""
        )

        # Extract predicted action, response, and facets from model output
        output_str = str(output)
        pred_action_match = re.search(
            r"Action:\s*(Clarify|Answer)", output_str, re.IGNORECASE
        )
        pred_action = pred_action_match.group(1).title() if pred_action_match else None

        pred_response_match = re.search(r"Response:\s*(.*)", output_str, re.IGNORECASE)
        pred_response = (
            pred_response_match.group(1).strip() if pred_response_match else ""
        )

        pred_facets_match = re.search(
            r"Facets:\s*(\[.*?\])", output_str, re.DOTALL | re.IGNORECASE
        )
        has_facets = False
        if pred_facets_match:
            facet_str = pred_facets_match.group(1).strip()
            try:
                facets = ast.literal_eval(facet_str)
                if isinstance(facets, list) and len(facets) > 0:
                    has_facets = True
            except (SyntaxError, ValueError):
                if len(facet_str.strip("[] \n\r")) > 0:
                    has_facets = True

        # Determine correctness
        correct_action = (target_action == pred_action) and (target_action is not None)

        is_target_clarify = target_action == "Clarify"
        is_pred_clarify = pred_action == "Clarify"

        is_answer_case = target_action == "Answer"
        is_semantic_match = False
        if is_answer_case:
            is_semantic_match = semantic_match(pred_response, target_response)

        return {
            "correct_action": correct_action,
            "target_action": target_action or "Unknown",
            "pred_action": pred_action or "Unknown",
            "is_true_positive": is_target_clarify and is_pred_clarify,
            "is_false_positive": (not is_target_clarify) and is_pred_clarify,
            "is_false_negative": is_target_clarify and (not is_pred_clarify),
            "is_answer_case": is_answer_case,
            "is_semantic_match": is_semantic_match,
            "is_pred_clarify": is_pred_clarify,
            "is_target_clarify": is_target_clarify,
            "has_facets": has_facets,
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

        # Answer metrics (treating Answer as the positive class)
        # For a binary Clarify/Answer split:
        # TP(Answer) = True Negatives (for Clarify)
        # FP(Answer) = False Negatives (for Clarify)
        # FN(Answer) = False Positives (for Clarify)
        tn = len(score_rows) - tp - fp - fn
        ans_precision = tn / (tn + fn) if (tn + fn) > 0 else 0.0
        ans_recall = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        ans_f1 = (
            2 * ans_precision * ans_recall / (ans_precision + ans_recall)
            if (ans_precision + ans_recall) > 0
            else 0.0
        )

        macro_f1 = (f1 + ans_f1) / 2.0

        correct = sum(row.get("correct_action", False) for row in score_rows)
        accuracy = correct / len(score_rows) if score_rows else 0.0

        # Semantic Answer Accuracy
        answer_cases = sum(row.get("is_answer_case", False) for row in score_rows)
        semantic_matches = sum(
            row.get("is_semantic_match", False) for row in score_rows
        )
        answer_accuracy = semantic_matches / answer_cases if answer_cases > 0 else 0.0

        # Facet Generation Rate
        pred_clarify = sum(row.get("is_pred_clarify", False) for row in score_rows)
        with_facets = sum(
            row.get("has_facets", False)
            for row in score_rows
            if row.get("is_pred_clarify", False)
        )
        facet_generation_rate = with_facets / pred_clarify if pred_clarify > 0 else 0.0

        # Clarify Ratio
        target_clarify = sum(row.get("is_target_clarify", False) for row in score_rows)
        clarify_ratio = pred_clarify / target_clarify if target_clarify > 0 else 0.0

        return {
            "accuracy": accuracy,
            "clarify_precision": precision,
            "clarify_recall": recall,
            "clarify_f1": f1,
            "answer_f1": ans_f1,
            "macro_f1": macro_f1,
            "answer_accuracy": answer_accuracy,
            "facet_generation_rate": facet_generation_rate,
            "clarify_ratio": clarify_ratio,
        }
