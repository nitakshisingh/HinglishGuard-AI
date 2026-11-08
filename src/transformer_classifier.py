"""
Transformer Classifier Inference Module for HinglishGuard AI.

Provides a reusable prediction interface for the fine-tuned XLM-RoBERTa model.
Accepts single strings or lists of strings, preprocesses them with `clean_text()`,
tokenizes, feeds them through the transformer classification head, applies softmax,
and returns structured predictions (label IDs, label names, and class probabilities).
"""

import os
from typing import Any, Dict, List, Optional, Union
import numpy as np

from src.preprocessing import clean_text

# Guarded imports for PyTorch & Transformers
try:
    import torch
    import torch.nn.functional as F
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    HAS_TORCH_TRANSFORMERS = True
except ImportError:
    HAS_TORCH_TRANSFORMERS = False

DEFAULT_TRANSFORMER_PATH = "models/transformers/xlm_roberta_hinglish/best_model"

ID_TO_LABEL = {
    0: "Neutral",
    1: "Offensive",
    2: "Hate",
}

LABEL_TO_ID = {v: k for k, v in ID_TO_LABEL.items()}


class TransformerClassifier:
    """Inference engine for fine-tuned XLM-RoBERTa Hinglish classifier."""

    def __init__(
        self,
        model_dir: str = DEFAULT_TRANSFORMER_PATH,
        device: Optional[str] = None,
        max_length: int = 128,
        model_instance: Any = None,
        tokenizer_instance: Any = None,
    ):
        """Initialize the transformer classifier.

        Args:
            model_dir: Directory containing saved Hugging Face model and tokenizer.
            device: 'cuda', 'cpu', or None (auto-detect).
            max_length: Maximum sequence length for tokenization.
            model_instance: Optional pre-loaded or mocked model instance (useful for unit testing).
            tokenizer_instance: Optional pre-loaded or mocked tokenizer instance.
        """
        self.max_length = max_length

        if model_instance is not None and tokenizer_instance is not None:
            self.model = model_instance
            self.tokenizer = tokenizer_instance
            self.device = device or "cpu"
            return

        if not HAS_TORCH_TRANSFORMERS:
            raise ImportError(
                "PyTorch and Hugging Face Transformers are required for TransformerClassifier. "
                "Install via 'pip install torch transformers'."
            )

        if not os.path.exists(model_dir):
            raise FileNotFoundError(
                f"Transformer model directory not found at '{model_dir}'. "
                "Ensure training has completed and the best model is saved."
            )

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading transformer model from {model_dir} on {self.device}...")

        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        self.model.to(self.device)
        self.model.eval()

    def predict_proba(self, text: Union[str, List[str]]) -> np.ndarray:
        """Compute softmax probability distribution over [Neutral, Offensive, Hate]."""
        raw_texts = [text] if isinstance(text, str) else text
        cleaned_texts = [clean_text(t) for t in raw_texts]

        if not cleaned_texts:
            return np.empty((0, 3))

        # Handle purely empty strings
        non_empty_indices = [i for i, t in enumerate(cleaned_texts) if t.strip()]
        probs = np.zeros((len(cleaned_texts), 3), dtype=np.float32)
        # Default empty strings to Neutral (class 0)
        probs[:, 0] = 1.0

        if not non_empty_indices:
            return probs

        texts_to_score = [cleaned_texts[i] for i in non_empty_indices]

        # Tokenization & Contextual Encoding
        encoded = self.tokenizer(
            texts_to_score,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt" if HAS_TORCH_TRANSFORMERS else None,
        )

        if HAS_TORCH_TRANSFORMERS and isinstance(encoded.get("input_ids"), torch.Tensor):
            encoded = {k: v.to(self.device) for k, v in encoded.items()}
            with torch.no_grad():
                outputs = self.model(**encoded)
                logits = outputs.logits
                batch_probs = F.softmax(logits, dim=-1).cpu().numpy()
        else:
            # Fallback for mocked inference in tests
            outputs = self.model(**encoded)
            logits = outputs.logits
            # Standard softmax
            exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
            batch_probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)

        for idx, orig_idx in enumerate(non_empty_indices):
            probs[orig_idx] = batch_probs[idx]

        return probs

    def predict(self, text: Union[str, List[str]]) -> List[Dict[str, Any]]:
        """Predict class labels, confidence scores, and probability breakdown.

        Args:
            text: A single string or a list of strings.

        Returns:
            List of dictionaries:
                - original_text: input string
                - cleaned_text: preprocessed text
                - label_id: integer (0, 1, or 2)
                - label_name: 'Neutral', 'Offensive', or 'Hate'
                - confidence: highest probability float
                - probabilities: dict of class name -> probability
        """
        is_single = isinstance(text, str)
        raw_texts = [text] if is_single else text
        cleaned_texts = [clean_text(t) for t in raw_texts]

        probs = self.predict_proba(raw_texts)

        results = []
        for i, (orig, cleaned) in enumerate(zip(raw_texts, cleaned_texts)):
            probas = probs[i]
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
                    ID_TO_LABEL[c]: round(float(probas[c]), 4) for c in range(3)
                },
            })

        return results


# Global singleton instance for functional access
_DEFAULT_TRANSFORMER_CLASSIFIER = None


def get_transformer_classifier(model_dir: str = DEFAULT_TRANSFORMER_PATH) -> TransformerClassifier:
    """Return singleton instance of TransformerClassifier."""
    global _DEFAULT_TRANSFORMER_CLASSIFIER
    if _DEFAULT_TRANSFORMER_CLASSIFIER is None:
        _DEFAULT_TRANSFORMER_CLASSIFIER = TransformerClassifier(model_dir=model_dir)
    return _DEFAULT_TRANSFORMER_CLASSIFIER


def predict_transformer_text(
    text: Union[str, List[str]],
    model_dir: str = DEFAULT_TRANSFORMER_PATH,
) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
    """Convenience functional interface for transformer prediction.

    Args:
        text: Single text or list of texts.
        model_dir: Checkpoint directory.

    Returns:
        Prediction dict or list of prediction dicts.
    """
    classifier = get_transformer_classifier(model_dir=model_dir)
    results = classifier.predict(text)
    if isinstance(text, str):
        return results[0]
    return results
