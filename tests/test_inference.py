from unittest.mock import MagicMock, patch

import ask_before_answer.inference.pipeline

from ask_before_answer.inference.pipeline import ClarifyOrActPipeline


@patch(
    "ask_before_answer.inference.pipeline.torch.cuda.is_bf16_supported",
    return_value=True,
)
@patch(
    "ask_before_answer.inference.pipeline.torch.cuda.is_available",
    return_value=True,
)
@patch("ask_before_answer.inference.pipeline.SamplingParams")
@patch("ask_before_answer.inference.pipeline.LLM")
def test_pipeline_initialization(
    mock_llm,
    mock_sampling_params,
    mock_cuda_available,
    mock_bf16_supported,
):
    # Reset the singleton state for the test.
    ask_before_answer.inference.pipeline._VLLM_ENGINE = None

    # Mock the vLLM engine instance.
    mock_engine = MagicMock()
    mock_llm.return_value = mock_engine

    pipeline = ClarifyOrActPipeline(
        model_path="dummy/path",
        is_peft=False,
    )

    assert pipeline is not None
    mock_llm.assert_called_once()
