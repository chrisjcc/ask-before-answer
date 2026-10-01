from dataclasses import dataclass
from typing import Callable, Any, Dict, List
import logging

logger = logging.getLogger(__name__)


@dataclass
class Criterion:
    """A base class standardizing how an individual heuristic or model grades a completion."""
    name: str
    weight: float
    description: str

    def evaluate_single(self, prompt: str, completion: str, kwargs: Dict[str, Any]) -> float:
        """Override for simple regex/heuristic logic."""
        raise NotImplementedError("Criteria must implement evaluate_single or evaluate_batch")

    def evaluate_batch(self, prompts: List[str], completions: List[str], kwargs: Dict[str, Any]) -> List[float]:
        """Override for optimized batched execution (e.g. GPU inference)."""
        return [self.evaluate_single(p, c, kwargs) for p, c in zip(prompts, completions)]


class Rubric:
    """A collection of grading criteria."""
    def __init__(self, criteria: List[Criterion]):
        self.criteria = criteria

    def as_trl_reward_funcs(self) -> List[Callable]:
        """Wraps each Criterion into its own TRL-compliant function for W&B tracking."""
        adapters = []
        for crit in self.criteria:
            def adapter_fn(prompts, completions, crit=crit, **kwargs):
                extracted_completions = [c[0]["content"] if isinstance(c, list) else c for c in completions]
                scores = crit.evaluate_batch(prompts, extracted_completions, kwargs)
                return [s * crit.weight for s in scores]
            
            adapter_fn.__name__ = f"{crit.name}_reward_func"
            adapters.append(adapter_fn)
        return adapters


class SingleTurnEnv:
    """The TRL Bridge linking datasets, prompts, and the grading rubric."""
    def __init__(self, train_dataset, eval_dataset, rubric: Rubric, system_prompt: str):
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.rubric = rubric
        self.system_prompt = system_prompt
        
    def _format_dataset(self, dataset, tokenizer):
        import re
        def format_chatml(example: dict) -> dict:
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": example["prompt"]},
            ]
            prompt_str = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            
            target_action = "Answer"
            target_response = ""
            
            if "chosen" in example:
                if "Action: Clarify" in example["chosen"]:
                    target_action = "Clarify"
                response_match = re.search(r"Response:\s*(.*)", example["chosen"], re.DOTALL)
                if response_match:
                    target_response = response_match.group(1).strip()
            
            return {
                "prompt": prompt_str,
                "target_action": target_action,
                "target_response": target_response,
            }
            
        return dataset.map(format_chatml, remove_columns=dataset.column_names)
        
    def get_train_dataset(self, tokenizer):
        if self.train_dataset is None:
            return None
        return self._format_dataset(self.train_dataset, tokenizer)
        
    def get_eval_dataset(self, tokenizer):
        if self.eval_dataset is None:
            return None
        return self._format_dataset(self.eval_dataset, tokenizer)
        
    def get_reward_funcs(self) -> List[Callable]:
        return self.rubric.as_trl_reward_funcs()
