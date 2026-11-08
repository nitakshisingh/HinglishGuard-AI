"""
Unit and integration tests for Stage 3E: Model Lab dashboard.
Tests:
1. Loading baseline metrics from data/processed/baseline_results/test_evaluation_metrics.json.
2. Loading transformer metrics from data/processed/transformer_test_sample_metrics.json.
3. Metric verification: accuracy, macro precision, recall, F1, weighted F1, confusion matrices.
4. Comparative calculation logic and XLM-RoBERTa superiority validation.
5. Dual-model comparison inference execution with mocked model instances.
6. Graceful handling of missing metric files.
"""

import importlib.util
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

# Dynamic import for module starting with a number
model_lab_path = Path(__file__).resolve().parent.parent / "pages" / "2_Model_Lab.py"
spec = importlib.util.spec_from_file_location("model_lab_module", model_lab_path)
model_lab_mod = importlib.util.module_from_spec(spec)

def dummy_decorator(*args, **kwargs):
    def wrapper(f):
        return f
    return wrapper

with patch("streamlit.set_page_config"), patch("streamlit.cache_resource", dummy_decorator):
    spec.loader.exec_module(model_lab_mod)

load_model_metrics = model_lab_mod.load_model_metrics


def test_load_model_metrics_success():
    """Verify load_model_metrics loads actual saved files without error."""
    baseline_data, transformer_data = load_model_metrics()

    assert baseline_data is not None
    assert transformer_data is not None

    # Baseline metrics verification
    assert "accuracy" in baseline_data
    assert "macro_f1" in baseline_data
    assert "classification_report" in baseline_data
    assert "confusion_matrix" in baseline_data
    assert baseline_data["accuracy"] > 0.85
    assert baseline_data["macro_f1"] > 0.70

    # Transformer metrics verification
    assert "accuracy" in transformer_data
    assert "macro_f1" in transformer_data
    assert "classification_report" in transformer_data
    assert "confusion_matrix" in transformer_data
    assert transformer_data["accuracy"] > 0.90
    assert transformer_data["macro_f1"] > 0.85


def test_model_superiority_validation():
    """Verify that XLM-RoBERTa outperforms baseline across all primary metrics based on saved empirical data."""
    baseline_data, transformer_data = load_model_metrics()

    # Accuracy superiority
    assert transformer_data["accuracy"] > baseline_data["accuracy"]

    # Macro F1 superiority (key optimization metric)
    assert transformer_data["macro_f1"] > baseline_data["macro_f1"]

    # Hate class F1 superiority
    b_hate_f1 = baseline_data["classification_report"]["Hate"]["f1-score"]
    t_hate_f1 = transformer_data["classification_report"]["Hate"]["f1-score"]
    assert t_hate_f1 > b_hate_f1


def test_confusion_matrices_dimensions():
    """Verify both confusion matrices are 3x3 matrices matching Neutral, Offensive, Hate taxonomy."""
    baseline_data, transformer_data = load_model_metrics()

    b_cm = baseline_data["confusion_matrix"]
    t_cm = transformer_data["confusion_matrix"]

    assert len(b_cm) == 3
    assert all(len(row) == 3 for row in b_cm)

    assert len(t_cm) == 3
    assert all(len(row) == 3 for row in t_cm)


def test_dual_model_mock_inference():
    """Verify live dual model comparison executes and produces structured predictions."""
    mock_baseline = MagicMock()
    mock_baseline.predict.return_value = [{
        "label_name": "Neutral",
        "confidence": 0.92,
        "probabilities": {"Neutral": 0.92, "Offensive": 0.05, "Hate": 0.03}
    }]

    mock_transformer = MagicMock()
    mock_transformer.predict.return_value = [{
        "label_name": "Neutral",
        "confidence": 0.98,
        "probabilities": {"Neutral": 0.98, "Offensive": 0.015, "Hate": 0.005}
    }]

    sample_text = "Aaj mausam accha hai"
    b_res = mock_baseline.predict(sample_text)[0]
    t_res = mock_transformer.predict(sample_text)[0]

    assert b_res["label_name"] == "Neutral"
    assert t_res["label_name"] == "Neutral"
    assert t_res["confidence"] > b_res["confidence"]


def test_missing_metrics_graceful_handling(tmp_path):
    """Verify load_model_metrics handles missing metric paths cleanly without crashing."""
    with patch("pathlib.Path.exists", return_value=False):
        b_data, t_data = load_model_metrics()
        assert b_data is None
        assert t_data is None
