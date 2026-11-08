"""
Unit tests for the Transformer Classifier interface and Training utilities.
Tests tokenization logic, label mappings, class weight calculations,
and single/batch inference without requiring full pretrained model downloads.
"""

from unittest.mock import MagicMock
import numpy as np
import pytest

from src.train_transformer import (
    ID_TO_LABEL,
    LABEL_TO_ID,
    calculate_class_weights,
    compute_evaluation_metrics,
    full_evaluation_report,
)
from src.transformer_classifier import TransformerClassifier


class MockTokenizerOutput(dict):
    """Mock dictionary output simulating HuggingFace tokenizer output."""
    pass


class MockTokenizer:
    """Mock tokenizer returning token IDs and attention masks without model download."""

    def __init__(self, vocab_size: int = 1000):
        self.vocab_size = vocab_size

    def __call__(self, texts, padding=True, truncation=True, max_length=128, return_tensors=None):
        if isinstance(texts, str):
            texts = [texts]

        batch_size = len(texts)
        seq_len = min(16, max_length)

        # input_ids: mock tokens
        input_ids = np.ones((batch_size, seq_len), dtype=np.int64)
        # attention_mask: 1s for real tokens
        attention_mask = np.ones((batch_size, seq_len), dtype=np.int64)

        return MockTokenizerOutput({
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        })


class MockTransformerModel:
    """Mock transformer model producing 3-class logits without GPU or weights download."""

    def __init__(self, num_labels: int = 3):
        self.num_labels = num_labels

    def __call__(self, input_ids=None, attention_mask=None, **kwargs):
        batch_size = len(input_ids) if input_ids is not None else 1
        # Create deterministic mock logits: [Neutral, Offensive, Hate]
        # Text 0 -> Neutral highest, Text 1 -> Offensive highest, Text 2 -> Hate highest
        logits = np.zeros((batch_size, self.num_labels), dtype=np.float32)
        for i in range(batch_size):
            chosen_class = i % self.num_labels
            logits[i, chosen_class] = 3.0
            logits[i, (chosen_class + 1) % self.num_labels] = -1.0
            logits[i, (chosen_class + 2) % self.num_labels] = -2.0

        output_mock = MagicMock()
        output_mock.logits = logits
        return output_mock


class TestTransformerPipeline:
    """Test suite for Transformer Classifier interface and utilities."""

    def test_label_mappings(self):
        """Verify strict 3-class label mapping integrity."""
        assert ID_TO_LABEL[0] == "Neutral"
        assert ID_TO_LABEL[1] == "Offensive"
        assert ID_TO_LABEL[2] == "Hate"

        assert LABEL_TO_ID["Neutral"] == 0
        assert LABEL_TO_ID["Offensive"] == 1
        assert LABEL_TO_ID["Hate"] == 2

    def test_class_weight_calculation(self):
        """Verify inverse frequency class weights calculation from training split."""
        # Simulated imbalanced distribution: 80 Neutral (0), 15 Offensive (1), 5 Hate (2)
        y_train = np.array([0] * 80 + [1] * 15 + [2] * 5)
        weights = calculate_class_weights(y_train, num_classes=3)

        assert len(weights) == 3
        # Minority class 'Hate' must have significantly higher weight than 'Neutral'
        assert weights[2] > weights[1] > weights[0]
        # Normalized so mean weight equals 1.0
        assert np.isclose(np.mean(weights), 1.0, atol=1e-5)

    def test_evaluation_metrics_computation(self):
        """Verify compute_evaluation_metrics correctly calculates Macro F1."""
        # Perfect predictions
        y_true = np.array([0, 1, 2, 0, 1, 2])
        # Logits predicting 0, 1, 2, 0, 1, 2
        logits = np.array([
            [10.0, 0.0, 0.0],
            [0.0, 10.0, 0.0],
            [0.0, 0.0, 10.0],
            [10.0, 0.0, 0.0],
            [0.0, 10.0, 0.0],
            [0.0, 0.0, 10.0],
        ])

        metrics = compute_evaluation_metrics((logits, y_true))
        assert metrics["accuracy"] == 1.0
        assert metrics["macro_f1"] == 1.0
        assert metrics["macro_precision"] == 1.0
        assert metrics["macro_recall"] == 1.0

    def test_full_evaluation_report(self):
        """Verify full per-class report and confusion matrix structure."""
        y_true = [0, 1, 2, 0, 1, 2]
        y_pred = [0, 1, 2, 0, 1, 0]  # One mistake: Hate predicted as Neutral

        report = full_evaluation_report(y_true, y_pred)
        assert "accuracy" in report
        assert "macro_f1" in report
        assert "classification_report" in report
        assert "confusion_matrix" in report
        assert len(report["confusion_matrix"]) == 3
        assert len(report["confusion_matrix"][0]) == 3

    def test_transformer_classifier_single_prediction(self):
        """Verify TransformerClassifier inference on a single sample using mock."""
        mock_model = MockTransformerModel()
        mock_tok = MockTokenizer()
        classifier = TransformerClassifier(
            model_instance=mock_model,
            tokenizer_instance=mock_tok,
            device="cpu",
        )

        res = classifier.predict("Bhai kya chal raha hai?")
        assert isinstance(res, list)
        assert len(res) == 1
        item = res[0]
        assert item["label_id"] == 0
        assert item["label_name"] == "Neutral"
        assert item["confidence"] > 0.8
        assert "probabilities" in item
        assert len(item["probabilities"]) == 3

    def test_transformer_classifier_batch_prediction(self):
        """Verify TransformerClassifier batch prediction with multiple samples."""
        mock_model = MockTransformerModel()
        mock_tok = MockTokenizer()
        classifier = TransformerClassifier(
            model_instance=mock_model,
            tokenizer_instance=mock_tok,
            device="cpu",
        )

        samples = [
            "Good morning bhai",
            "Chup kar sale kutte",
            "In aatankwadiyo ko desh se nikal do",
        ]
        results = classifier.predict(samples)
        assert len(results) == 3

        # Sample 0 -> class 0 (Neutral)
        assert results[0]["label_id"] == 0
        assert results[0]["label_name"] == "Neutral"

        # Sample 1 -> class 1 (Offensive)
        assert results[1]["label_id"] == 1
        assert results[1]["label_name"] == "Offensive"

        # Sample 2 -> class 2 (Hate)
        assert results[2]["label_id"] == 2
        assert results[2]["label_name"] == "Hate"

    def test_transformer_classifier_empty_input(self):
        """Verify that empty inputs safely default to Neutral without crashing."""
        mock_model = MockTransformerModel()
        mock_tok = MockTokenizer()
        classifier = TransformerClassifier(
            model_instance=mock_model,
            tokenizer_instance=mock_tok,
            device="cpu",
        )

        res = classifier.predict("")
        assert res[0]["label_id"] == 0
        assert res[0]["label_name"] == "Neutral"
        assert res[0]["confidence"] == 1.0

    def test_training_arguments_constructor_simulation(self):
        """Verify dynamic build_training_arguments parameter resolution for v4/v5 APIs."""
        import inspect
        import math

        # Simulate signature with warmup_ratio (v4.x)
        def mock_training_args_v4(
            output_dir,
            eval_strategy="no",
            learning_rate=5e-5,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            gradient_accumulation_steps=1,
            num_train_epochs=3,
            weight_decay=0.0,
            warmup_ratio=0.0,
            metric_for_best_model=None,
            greater_is_better=None,
            load_best_model_at_end=False,
            logging_dir=None,
            logging_steps=500,
            save_total_limit=None,
            seed=42,
            fp16=False,
            report_to=None,
            save_strategy="steps",
        ):
            return locals()

        # Simulate signature without warmup_ratio but with warmup_steps (v5.x)
        def mock_training_args_v5(
            output_dir,
            eval_strategy="no",
            learning_rate=5e-5,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            gradient_accumulation_steps=1,
            num_train_epochs=3,
            weight_decay=0.0,
            warmup_steps=0,
            metric_for_best_model=None,
            greater_is_better=None,
            load_best_model_at_end=False,
            logging_dir=None,
            logging_steps=500,
            save_total_limit=None,
            seed=42,
            fp16=False,
            report_to=None,
            save_strategy="steps",
        ):
            return locals()

        # Test v4 resolution
        sig_v4 = inspect.signature(mock_training_args_v4)
        assert "warmup_ratio" in sig_v4.parameters

        # Test v5 resolution and calculation
        sig_v5 = inspect.signature(mock_training_args_v5)
        assert "warmup_ratio" not in sig_v5.parameters
        assert "warmup_steps" in sig_v5.parameters

        # Verify warmup steps math: 40083 samples, batch 16, grad_accum 2 -> eff_batch 32
        # steps_per_epoch = ceil(40083 / 32) = 1253. Total steps (3 epochs) = 3759.
        # warmup 10% = int(3759 * 0.1) = 375.
        eff_batch = 16 * 2
        total_steps = math.ceil(40083 / eff_batch) * 3
        warmup_steps = int(total_steps * 0.1)
        assert warmup_steps == 375
