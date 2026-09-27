"""Mock test for the multi-turn evaluation framework."""

from unittest.mock import MagicMock

from ask_before_answer.evaluation.multi_turn import (
    ProviderAgent,
    SeekerAgent,
    simulate_conversation,
)
from ask_before_answer.inference.pipeline import ClarifyOrActPipeline


def test_multi_turn_success():
    """Verify that a successful conversation terminates before max_turns."""

    # Mock the pipeline so we don't need real weights
    mock_pipeline = MagicMock(spec=ClarifyOrActPipeline)

    # We will simulate the Seeker taking 2 turns.
    # Turn 1: It asks a clarification question.
    # Turn 2: It decides to output an Answer based on the Provider's reply.
    mock_pipeline.generate_from_messages.side_effect = [
        "Action: Clarify\nReasoning: Need time.\nFacets: Time\n"
        "Response: What time is the flight?",
        "Action: Answer\nReasoning: I know the time now.\nFacets: \n"
        "Response: The flight is at 5 PM.",
    ]

    seeker = SeekerAgent(pipeline=mock_pipeline)

    # Mock the ProviderAgent to avoid hitting the Gemini API during testing
    original_question = "When is the flight?"
    disambiguations = [{"question": "What time is the flight?", "answer": "5 PM"}]
    provider = ProviderAgent(original_question, disambiguations)

    # Override the reply method to return a hardcoded facet
    provider.reply = MagicMock(return_value="The flight is at 5 PM.")

    # Run the simulation
    result = simulate_conversation(seeker, provider, original_question, max_turns=3)

    # Assertions
    assert result["success"] is True, "The conversation should have succeeded."
    assert result["turns"] == 2, f"Expected 2 turns, got {result['turns']}"
    assert result["final_answer"] == "The flight is at 5 PM."

    # Verify the history contains the Seeker and Provider back-and-forth
    history = result["history"]
    assert len(history) == 4  # Q -> Seeker1 -> Prov1 -> Seeker2 (Answer)
    assert history[0]["role"] == "user"
    assert history[1]["role"] == "assistant"
    assert history[1]["content"] == "What time is the flight?"
    assert history[2]["role"] == "user"
    assert history[2]["content"] == "The flight is at 5 PM."
    assert history[3]["role"] == "assistant"
    assert history[3]["content"] == "The flight is at 5 PM."

    print("Test passed successfully!")


if __name__ == "__main__":
    test_multi_turn_success()
