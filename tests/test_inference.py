from unittest.mock import patch

from src.inference.pipeline import ClarificationPipeline


@patch("src.inference.pipeline.AutoModelForCausalLM.from_pretrained")
@patch("src.inference.pipeline.AutoTokenizer.from_pretrained")
def test_pipeline_initialization(mock_tokenizer, mock_model):
    pipeline = ClarificationPipeline(model_path="dummy/path", is_peft=False)
    assert pipeline is not None
    mock_model.assert_called_once()
    mock_tokenizer.assert_called_once()
