from unittest.mock import patch

from ask_before_answer.inference.pipeline import ClarifyOrActPipeline


@patch("ask_before_answer.inference.pipeline.AutoModelForCausalLM.from_pretrained")
@patch("ask_before_answer.inference.pipeline.AutoTokenizer.from_pretrained")
def test_pipeline_initialization(mock_tokenizer, mock_model):
    pipeline = ClarifyOrActPipeline(model_path="dummy/path", is_peft=False)
    assert pipeline is not None
    mock_model.assert_called_once()
    mock_tokenizer.assert_called_once()
