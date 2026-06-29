from unittest.mock import patch

from src.inference.pipeline import ClarifyOrActPipeline


@patch("src.inference.pipeline.AutoModelForCausalLM.from_pretrained")
@patch("src.inference.pipeline.AutoTokenizer.from_pretrained")
def test_pipeline_initialization(mock_tokenizer, mock_model):
    pipeline = ClarifyOrActPipeline(model_path="dummy/path", is_peft=False)
    assert pipeline is not None
    mock_model.assert_called_once()
    mock_tokenizer.assert_called_once()
