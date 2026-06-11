import pytest
import pandas as pd
from src.data.preprocess import clean_facets, clean_response

def test_clean_facets():
    assert clean_facets("['Entity Reference']") == ["Entity Reference"]
    assert clean_facets(["Location"]) == ["Location"]
    assert clean_facets("invalid") == []
    assert clean_facets(None) == []

def test_clean_response():
    assert clean_response("['What time?']") == "What time?"
    assert clean_response(["What time?"]) == "What time?"
    assert clean_response("Could you clarify?") == "Could you clarify?"
    assert clean_response(None) == ""
