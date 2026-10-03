import logging
from typing import Dict, Any, List

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
            logger.warning("GEMINI_API_KEY is missing. ProviderAgent will fail if called.")

        # Initialize the GenAI client
        self.client = genai.Client(api_key=api_key) if api_key else None

    def reply(self, seeker_message: str, ground_truth_facets: Any) -> str:
        """
        Uses an LLM to generate a user response based on the hidden facets.
        """
        if not self.client:
            return "Error: GEMINI_API_KEY not set. Cannot simulate user response."

        system_prompt = (
            "You are a human user answering a clarifying question from an AI assistant. "
            "You asked an initial question, and the assistant needs more information to answer it. "
            f"Here is your hidden knowledge (the ground truth facets): {ground_truth_facets}\n\n"
            "Instructions:\n"
            "1. Answer the assistant's clarifying question truthfully using ONLY the hidden knowledge.\n"
            "2. Be concise and natural, as a human would be.\n"
            "3. Do NOT reveal information that the assistant didn't specifically ask for."
        )

        try:
            from google.genai import types
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=seeker_message,
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    temperature=0.3,
                )
            )
            return response.text
        except Exception as e:
            logger.error(f"ProviderAgent failed to generate reply: {e}")
            return f"User provides clarifying info: {ground_truth_facets}"



class MultiTurnEnv(Environment):
    """The interactive evaluation environment for AskBeforeAnswer."""

    def __init__(self, dataset, provider_model: str = "gemini-2.5-flash", max_turns: int = 3):
        super().__init__(dataset, max_turns)
        self.provider = ProviderAgent(model_name=provider_model)

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

        return {
            "success": success,
            "turns_taken": seeker_turns,
            "history": history
        }

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
