import re
from typing import Any

import weave


class ActionScorer(weave.Scorer):
    @weave.op()
    def score(self, target: Any, output: str, **kwargs) -> dict:
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

        # Determine correctness (must match exactly and not be None)
        correct_action = (true_action == pred_action) and (true_action is not None)

        return {
            "correct_action": correct_action,
            "true_action": true_action or "Unknown",
            "pred_action": pred_action or "Unknown",
        }
