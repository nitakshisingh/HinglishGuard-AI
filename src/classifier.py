"""
Classifier inference and prediction module for HinglishGuard AI.
Loads the trained baseline model and TF-IDF vectorizer, cleans input text,
and provides reusable prediction functions for single samples and batches.
"""

import os
from typing import Any, Dict, List, Union
import joblib
import numpy as np

from src.preprocessing import clean_text

DEFAULT_MODEL_PATH = "models/baseline/logistic_regression_baseline.joblib"
DEFAULT_VEC_PATH = "models/baseline/tfidf_vectorizer.joblib"

ID_TO_LABEL = {
    0: "Neutral",
    1: "Offensive",
    2: "Hate",
}

LABEL_TO_ID = {v: k for k, v in ID_TO_LABEL.items()}


class BaselineClassifier:
    """Wrapper class for the TF-IDF + Logistic Regression baseline model."""

    def __init__(self, model_path: str = DEFAULT_MODEL_PATH, vec_path: str = DEFAULT_VEC_PATH):
        if not os.path.exists(model_path) or not os.path.exists(vec_path):
            raise FileNotFoundError(
                f"Model artifacts not found at '{model_path}' or '{vec_path}'. "
                "Ensure you have trained the baseline model first."
            )
        self.model = joblib.load(model_path)
        self.vectorizer = joblib.load(vec_path)

    def predict_proba(self, text: Union[str, List[str]]) -> np.ndarray:
        """Predict class probability distribution for input text or list of texts."""
        if isinstance(text, str):
            texts = [clean_text(text)]
        else:
            texts = [clean_text(t) for t in text]

        X_vec = self.vectorizer.transform(texts)
        return self.model.predict_proba(X_vec)

    def predict(self, text: Union[str, List[str]]) -> List[Dict[str, Any]]:
        """Predict class labels, confidence scores, and probability breakdown.

        Args:
            text: A single string or a list of strings.

        Returns:
            List of dictionaries containing:
                - text: normalized text
                - label_id: integer (0, 1, or 2)
                - label_name: 'Neutral', 'Offensive', or 'Hate'
                - confidence: highest probability float
                - probabilities: dict of class name -> probability
        """
        is_single = isinstance(text, str)
        raw_texts = [text] if is_single else text
        cleaned_texts = [clean_text(t) for t in raw_texts]

        # Handle empty string case cleanly
        results = []
        for orig, cleaned in zip(raw_texts, cleaned_texts):
            if not cleaned.strip():
                results.append({
                    "original_text": orig,
                    "cleaned_text": "",
                    "label_id": 0,
                    "label_name": "Neutral",
                    "confidence": 1.0,
                    "probabilities": {"Neutral": 1.0, "Offensive": 0.0, "Hate": 0.0},
                })
            else:
                X_vec = self.vectorizer.transform([cleaned])
                probas = self.model.predict_proba(X_vec)[0]
                pred_id = int(np.argmax(probas))
                pred_label = ID_TO_LABEL[pred_id]
                conf = float(probas[pred_id])
                results.append({
                    "original_text": orig,
                    "cleaned_text": cleaned,
                    "label_id": pred_id,
                    "label_name": pred_label,
                    "confidence": round(conf, 4),
                    "probabilities": {
                        ID_TO_LABEL[i]: round(float(p), 4) for i, p in enumerate(probas)
                    },
                })

        return results


# Global singleton instance for easy functional access
_DEFAULT_CLASSIFIER = None


def get_default_classifier() -> BaselineClassifier:
    """Return singleton instance of BaselineClassifier."""
    global _DEFAULT_CLASSIFIER
    if _DEFAULT_CLASSIFIER is None:
        _DEFAULT_CLASSIFIER = BaselineClassifier()
    return _DEFAULT_CLASSIFIER


def predict_text(text: Union[str, List[str]], model_path: str = DEFAULT_MODEL_PATH, vec_path: str = DEFAULT_VEC_PATH) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
    """Reusable prediction utility for HinglishGuard AI text classification.

    Args:
        text: Single text string or list of text strings.
        model_path: Path to joblib model artifact.
        vec_path: Path to joblib vectorizer artifact.

    Returns:
        Prediction dict if single text passed, or list of prediction dicts if list passed.
    """
    if model_path == DEFAULT_MODEL_PATH and vec_path == DEFAULT_VEC_PATH:
        classifier = get_default_classifier()
    else:
        classifier = BaselineClassifier(model_path, vec_path)

    results = classifier.predict(text)
    if isinstance(text, str):
        return results[0]
    return results
