import json
import logging

from ask_before_answer.evaluation.multi_turn import MultiTurnEnv

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DummySeekerAgent:
    """A dummy Seeker that always asks for more info twice, then answers."""

    def __init__(self):
        self.turns = 0

    def generate(self, dialogue_history: list) -> str:
        self.turns += 1
        if self.turns >= 3:
            return (
                "Action: Answer\nReasoning: I have enough info.\n"
                "Facets: []\nResponse: It is 10 AM."
            )
        return (
            "Action: Clarify\nReasoning: Need more info.\n"
            "Facets: ['time']\nResponse: What time is it?"
        )


def main():
    logger.info("Initializing MultiTurnEnv test...")

    # Create a dummy dataset
    dataset = [
        {
            "prompt": "When is the meeting?",
            "disambiguations": ["time zone", "morning or afternoon"],
        }
    ]

    # Initialize the Environment with the real ProviderAgent (Gemini)
    env = MultiTurnEnv(dataset=dataset, provider_model="gemini-2.5-flash", max_turns=3)
    seeker = DummySeekerAgent()

    logger.info("Starting evaluation rollout...")
    results = env.evaluate(seeker)

    logger.info("\n--- EVALUATION RESULTS ---")
    logger.info(json.dumps(results, indent=2))

    if results[0]["success"]:
        logger.info(
            "✅ MultiTurnEnv successfully completed the trajectory loop "
            "and returned success!"
        )
    else:
        logger.warning("❌ MultiTurnEnv terminated early or failed.")


if __name__ == "__main__":
    main()
