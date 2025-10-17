import joblib
import pytest
from src.clean_data import clean_text

def test_clean_text_removes_urls():
    text = "Check this out http://example.com"
    assert "http" not in clean_text(text)

def test_model_load():
    model = joblib.load("models/sentiment_model.pkl")
    pred = model.predict(["I love this product!"])
    assert pred[0] in ["Positive", "Negative", "Neutral"]
