"""
Unit tests for the baseline classifier module (src.classifier).
"""

import os
import pytest
from src.classifier import BaselineClassifier, predict_text, ID_TO_LABEL


class TestBaselineClassifier:
    """Test suite for classifier inference and prediction functionality."""

    def test_classifier_initialization(self):
        """Test that the classifier loads saved artifacts successfully."""
        classifier = BaselineClassifier()
        assert classifier.model is not None
        assert classifier.vectorizer is not None

    def test_single_prediction_neutral(self):
        """Test single sample prediction on a clearly benign/neutral comment."""
        text = "Bhai ye video bahut acchi hai, informative content banaya aapne."
        result = predict_text(text)
        assert isinstance(result, dict)
        assert "label_id" in result
        assert "label_name" in result
        assert "confidence" in result
        assert "probabilities" in result
        assert result["label_id"] in [0, 1, 2]
        assert result["label_name"] == "Neutral"
        assert result["confidence"] > 0.5

    def test_single_prediction_offensive(self):
        """Test single sample prediction on colloquial abusive/vulgar slang."""
        text = "Chup kar bhadwe kutte, dimag kharab mat kar mera."
        result = predict_text(text)
        assert isinstance(result, dict)
        assert result["label_name"] in ["Offensive", "Hate"]
        assert result["label_id"] in [1, 2]

    def test_batch_prediction(self):
        """Test batch prediction with a list of diverse texts."""
        texts = [
            "Good morning bhai, kya chal raha hai?",
            "Arey yaar ye movie kitni bekar thi.",
            "Teri maa ki chut bakwaas mat kar.",
        ]
        results = predict_text(texts)
        assert isinstance(results, list)
        assert len(results) == len(texts)
        for r in results:
            assert "label_name" in r
            assert "probabilities" in r
            assert len(r["probabilities"]) == 3
            prob_sum = sum(r["probabilities"].values())
            assert 0.99 <= prob_sum <= 1.01

    def test_empty_and_whitespace_inputs(self):
        """Test safe handling of empty or blank inputs."""
        result_empty = predict_text("")
        assert result_empty["label_id"] == 0
        assert result_empty["label_name"] == "Neutral"
        assert result_empty["confidence"] == 1.0

        result_spaces = predict_text("     \t \n ")
        assert result_spaces["label_id"] == 0
        assert result_spaces["label_name"] == "Neutral"

    def test_probabilities_sum_to_one(self):
        """Verify that class probabilities sum to approximately 1.0."""
        text = "Yeh bilkul normal comment hai."
        result = predict_text(text)
        probs = result["probabilities"]
        assert "Neutral" in probs
        assert "Offensive" in probs
        assert "Hate" in probs
        assert round(sum(probs.values()), 2) == 1.00
