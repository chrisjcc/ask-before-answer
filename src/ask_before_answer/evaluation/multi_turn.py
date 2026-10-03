import logging
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


class Environment:
    """Base class for interactive evaluation environments."""

    def __init__(self, dataset, max_turns: int = 3):
        self.dataset = dataset
        self.max_turns = max_turns

    def env_response(self, state: Dict[str, Any], seeker_message: str) -> str:
        raise NotImplementedError

    def evaluate(self, seeker_model):
        raise NotImplementedError


class ProviderAgent:
    """Simulates a human holding the ground truth information."""

    def __init__(self, model_name: str = "gemini-2.5-flash"):
        self.model_name = model_name

        import os

        from google import genai

        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            logger.warning(
                "GEMINI_API_KEY is missing. ProviderAgent will fail if called."
            )

        # Initialize the GenAI client
        self.client = genai.Client(api_key=api_key) if api_key else None

    def reply(self, seeker_message: str, ground_truth_facets: Any) -> str:
        """
        Uses an LLM to generate a user response based on the hidden facets.
        """
        if not self.client:
            return "Error: GEMINI_API_KEY not set. Cannot simulate user response."

        system_prompt = (
            "You are a human user answering a clarifying question from an AI "
            "assistant. You asked an initial question, and the assistant needs "
            "more information to answer it. "
            f"Here is your hidden knowledge (the ground truth facets): "
            f"{ground_truth_facets}\n\n"
            "Instructions:\n"
            "1. Answer the assistant's clarifying question truthfully using "
            "ONLY the hidden knowledge.\n"
            "2. Be concise and natural, as a human would be.\n"
            "3. Do NOT reveal information that the assistant didn't "
            "specifically ask for."
        )

        try:
            from google.genai import types

            response = self.client.models.generate_content(
                model=self.model_name,
                contents=seeker_message,
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    temperature=0.3,
                ),
            )
            return response.text
        except Exception as e:
            logger.error(f"ProviderAgent failed to generate reply: {e}")
            return f"User provides clarifying info: {ground_truth_facets}"


class LocalProviderAgent:
    """Simulates a human using a local PyTorch model (e.g., Gemma)."""

    def __init__(self, model_name: str = "google/gemma-2-2b-it"):
        self.model_name = model_name
        # We reuse the cache and lock from the judge to avoid loading it twice
        from ask_before_answer.evaluation.judge import (
            _LOCAL_INFERENCE_LOCK,
            get_local_judge,
        )

        self.get_local_judge = get_local_judge
        self.inference_lock = _LOCAL_INFERENCE_LOCK

    def reply(self, seeker_message: str, ground_truth_facets: Any) -> str:
        import torch

        system_prompt = (
            "You are a human user answering a clarifying question from an AI "
            "assistant. You asked an initial question, and the assistant needs "
            "more information to answer it. "
            f"Here is your hidden knowledge (the ground truth facets): "
            f"{ground_truth_facets}\n\n"
            "Instructions:\n"
            "1. Answer the assistant's clarifying question truthfully using "
            "ONLY the hidden knowledge.\n"
            "2. Be concise and natural, as a human would be.\n"
            "3. Do NOT reveal information that the assistant didn't "
            "specifically ask for."
        )

        model, tokenizer = self.get_local_judge(self.model_name)

        messages = [
            {
                "role": "user",
                "content": system_prompt + "\n\nAssistant asks:\n" + seeker_message,
            }
        ]

        input_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

        with self.inference_lock:
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=150,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                )

        gen_tokens = outputs[0][inputs["input_ids"].shape[1] :]
        return tokenizer.decode(gen_tokens, skip_special_tokens=True)


class MultiTurnEnv(Environment):
    """The interactive evaluation environment for AskBeforeAnswer."""

    def __init__(
        self, dataset, provider_model: str = "google/gemma-2-2b-it", max_turns: int = 3
    ):
        super().__init__(dataset, max_turns)
        if "gemini" in provider_model.lower():
            self.provider = ProviderAgent(model_name=provider_model)
        else:
            self.provider = LocalProviderAgent(model_name=provider_model)

    def env_response(self, state: Dict[str, Any], seeker_message: str) -> str:
        """
        The environment's reaction to the model's turn.
        The Provider Agent answers the clarification!
        """
        ground_truth_facets = state.get("disambiguations", [])
        return self.provider.reply(seeker_message, ground_truth_facets)

    def is_action_taken(self, seeker_message: str) -> bool:
        """Stop condition: The Seeker outputs 'Action: Answer'."""
        return "Action: Answer" in seeker_message

    def _calculate_metrics(self, history: List[Dict[str, str]]) -> Dict[str, Any]:
        """Calculates success metrics for a single trajectory."""
        last_message = history[-1]["content"]
        success = self.is_action_taken(last_message)

        # Calculate how many turns the Seeker took (excluding system/provider messages)
        seeker_turns = sum(1 for msg in history if msg["role"] == "assistant")

        return {"success": success, "turns_taken": seeker_turns, "history": history}

    def evaluate(self, seeker_model) -> List[Dict[str, Any]]:
        """The core multi-turn rollout loop."""
        results = []
        for example in self.dataset:
            state = {"disambiguations": example.get("disambiguations", [])}

            # Use standard ChatML history
            history = [{"role": "user", "content": example["prompt"]}]

            turn = 0
            while turn < self.max_turns:
                # 1. Seeker acts (predicts action or asks a clarification question)
                seeker_reply = seeker_model.generate(history)
                history.append({"role": "assistant", "content": seeker_reply})

                # 2. Check Stop Conditions
                if self.is_action_taken(seeker_reply):
                    break

                # 3. Environment (Provider) responds with hidden knowledge
                env_reply = self.env_response(state, seeker_reply)
                history.append({"role": "user", "content": env_reply})

                turn += 1

            results.append(self._calculate_metrics(history))

        return results


class SeekerAgent:
    """Wraps the ClarifyOrActPipeline to handle multi-turn dialogue history."""

    def __init__(self, pipeline):
        self.pipeline = pipeline

    def generate(self, dialogue_history: List[Dict[str, str]]) -> str:
        """
        Converts the standard ChatML list into the format the pipeline expects.
        Since pipeline.generate() expects a single question, we format the history
        into a dialogue string using the pipeline's internal tokenizer.
        """
        # If the pipeline supports passing raw chat history, we could pass it directly.
        # But if pipeline.generate() expects a string:

        # We can use the pipeline's tokenizer to format the chat template
        prompt_str = self.pipeline.tokenizer.apply_chat_template(
            dialogue_history, tokenize=False, add_generation_prompt=True
        )

        # Bypass pipeline.generate(question) since it adds its own template.
        # Directly use the model to generate from the pre-formatted prompt_str.
        import torch

        inputs = self.pipeline.tokenizer(prompt_str, return_tensors="pt").to(
            self.pipeline.model.device
        )

        with torch.no_grad():
            outputs = self.pipeline.model.generate(
                **inputs,
                max_new_tokens=300,
                do_sample=False,
                pad_token_id=self.pipeline.tokenizer.pad_token_id,
            )

        gen_tokens = outputs[0][inputs["input_ids"].shape[1] :]
        return self.pipeline.tokenizer.decode(gen_tokens, skip_special_tokens=True)
