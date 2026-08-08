from unittest.mock import MagicMock, patch

import src.inference.pipeline
from src.inference.pipeline import ClarifyOrActPipeline


@patch("src.inference.pipeline.SamplingParams")
@patch("src.inference.pipeline.LLM")
def test_pipeline_initialization(mock_llm, mock_sampling_params):
    # Reset the singleton state for the test
    src.inference.pipeline._VLLM_ENGINE = None

    # Mock the vLLM engine instance
    mock_engine = MagicMock()
    mock_llm.return_value = mock_engine

    pipeline = ClarifyOrActPipeline(model_path="dummy/path", is_peft=False)

    assert pipeline is not None
    mock_llm.assert_called_once()
