"""Mock test for the multi-turn evaluation framework."""

from unittest.mock import MagicMock

from ask_before_answer.evaluation.multi_turn import (
    MultiTurnEnv,
    ProviderAgent,
    SeekerAgent,
)


def test_multi_turn_success():
    """Verify that a successful conversation terminates before max_turns."""

    # 1. Mock the Seeker Agent completely
    mock_seeker = MagicMock(spec=SeekerAgent)
    mock_seeker.generate.side_effect = [
        (
            "Action: Clarify\nReasoning: Need time.\nFacets: ['time']\n"
            "Response: What time is the flight?"
        ),
        (
            "Action: Answer\nReasoning: I know the time.\nFacets: []\n"
            "Response: The flight is at 5 PM."
        ),
    ]

    # 2. Mock the Provider Agent
    mock_provider = MagicMock(spec=ProviderAgent)
    mock_provider.reply.return_value = "The flight is at 5 PM."

    # 3. Create a dummy dataset
    dataset = [
        {
            "prompt": "When is the flight?",
            "disambiguations": [
                {"question": "What time is the flight?", "answer": "5 PM"}
            ],
        }
    ]

    # 4. Initialize the Environment
    env = MultiTurnEnv(dataset=dataset, max_turns=3)

    # Inject our mock ProviderAgent into the environment
    env.provider = mock_provider

    # 5. Run the evaluation
    results = env.evaluate(mock_seeker)
    result = results[0]

    # 6. Assertions
    assert result["success"] is True, "The conversation should have succeeded."
    assert result["turns_taken"] == 2, f"Expected 2 turns, got {result['turns_taken']}"

    # Verify the history contains the exact sequence of turns
    # Initial Prompt -> Seeker -> Provider -> Seeker
    history = result["history"]
    assert len(history) == 4

    assert history[0]["role"] == "user"
    assert history[0]["content"] == "When is the flight?"

    assert history[1]["role"] == "assistant"
    assert "Action: Clarify" in history[1]["content"]

    assert history[2]["role"] == "user"
    assert history[2]["content"] == "The flight is at 5 PM."

    assert history[3]["role"] == "assistant"
    assert "Action: Answer" in history[3]["content"]

    # Verify Provider Agent was called correctly
    mock_provider.reply.assert_called_once()


if __name__ == "__main__":
    test_multi_turn_success()
