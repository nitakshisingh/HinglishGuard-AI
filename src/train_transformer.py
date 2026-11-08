"""
Training & Fine-Tuning Pipeline for HinglishGuard AI using XLM-RoBERTa.

Fine-tunes `xlm-roberta-base` on three-class Hinglish hate speech data:
- Class 0: Neutral
- Class 1: Offensive
- Class 2: Hate

Architectural & Training Highlights:
1. Tokenization:
   - Uses XLM-RoBERTa SentencePiece BPE tokenizer (`xlm-roberta-base`).
   - Handles multilingual sub-word tokens with max_length=128.
   - Dynamic padding via DataCollatorWithPadding saves compute and memory.
2. Attention Masks & Contextual Embeddings:
   - Attention masks distinguish active token positions from pad tokens (0 vs 1).
   - Deep bi-directional self-attention outputs 768-dimensional contextual embeddings.
3. Classification Head:
   - Takes the contextual representation of the first token (<s> / CLS token representation).
   - Passes through a dense projection layer, tanh activation, dropout (0.1), and linear projection (768 -> 3).
4. Weighted Cross-Entropy Loss:
   - Mitigates class imbalance (~78% Neutral, 17.5% Offensive, 4.5% Hate).
   - Class weights are calculated strictly from the training split using inverse frequency.
5. Model Selection & Unseen Test Evaluation:
   - Tracks validation Macro F1 across epochs.
   - Saves the best checkpoint based on validation Macro F1.
   - Evaluates on the unseen test set exactly once after training completes.
6. Execution Flexibility:
   - GPU-aware (supports CUDA/MPS where available).
   - CPU-friendly flag (--smoke_test or --dry_run) for local unit testing without running full training.
   - Colab-ready out of the box.
"""

import argparse
import inspect
import json
import math
import os
import random
import sys
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

# Deep Learning / Transformers imports guarded for flexible test/environment execution
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset
    from transformers import (
        AutoConfig,
        AutoModelForSequenceClassification,
        AutoTokenizer,
        DataCollatorWithPadding,
        Trainer,
        TrainingArguments,
        set_seed,
    )
    from transformers.trainer_utils import EvalPrediction
    HAS_TORCH_TRANSFORMERS = True
except ImportError:
    HAS_TORCH_TRANSFORMERS = False

TRAIN_PATH = "data/processed/train.csv"
VAL_PATH = "data/processed/validation.csv"
TEST_PATH = "data/processed/test.csv"

DEFAULT_MODEL_NAME = "xlm-roberta-base"
DEFAULT_OUTPUT_DIR = "models/transformers/xlm_roberta_hinglish"
DEFAULT_RESULTS_DIR = "data/processed/transformer_results"

ID_TO_LABEL = {0: "Neutral", 1: "Offensive", 2: "Hate"}
LABEL_TO_ID = {v: k for k, v in ID_TO_LABEL.items()}
CLASS_NAMES = ["Neutral", "Offensive", "Hate"]


def set_reproducible_seed(seed: int = 42) -> None:
    """Set random seed across python, numpy, torch for exact reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    if HAS_TORCH_TRANSFORMERS:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        set_seed(seed)


def calculate_class_weights(y_train: np.ndarray, num_classes: int = 3) -> np.ndarray:
    """Calculate balanced class weights strictly from the training split.

    Formula:
        weight[c] = total_samples / (num_classes * count[c])

    This penalizes misclassifications on minority classes (e.g., Hate ~4.5%)
    more heavily than majority classes (Neutral ~78%).
    """
    counts = np.bincount(y_train, minlength=num_classes)
    total_samples = len(y_train)
    weights = total_samples / (num_classes * np.maximum(counts, 1).astype(float))
    # Normalize so mean weight is 1.0 for numerical stability
    weights = weights / np.mean(weights)
    return weights


def build_training_arguments(
    output_dir: str,
    num_train_epochs: int,
    per_device_train_batch_size: int,
    per_device_eval_batch_size: int,
    gradient_accumulation_steps: int,
    learning_rate: float,
    weight_decay: float,
    warmup_ratio: float,
    total_train_samples: int,
    device: str,
    seed: int = 42,
    smoke_test: bool = False,
) -> "TrainingArguments":
    """Construct TrainingArguments dynamically inspecting the installed Transformers signature.

    Guarantees cross-version compatibility across Transformers v4.x and v5.x:
    1. Inspects inspect.signature(TrainingArguments).
    2. Warmup: If 'warmup_ratio' is supported, uses it. If not, explicitly calculates
       warmup_steps = int(total_optimizer_steps * warmup_ratio), correctly accounting for
       epochs, batch size, and gradient accumulation.
    3. Evaluation Strategy: Chooses between 'eval_strategy' and legacy 'evaluation_strategy'.
    4. Mixed Precision: Sets fp16 on CUDA devices.
    """
    if not HAS_TORCH_TRANSFORMERS:
        raise ImportError("Transformers is not installed.")

    sig = inspect.signature(TrainingArguments)
    params = sig.parameters

    # Calculate exact total optimizer update steps accounting for gradient accumulation
    effective_batch_size = per_device_train_batch_size * gradient_accumulation_steps
    steps_per_epoch = math.ceil(total_train_samples / effective_batch_size) if effective_batch_size > 0 else 1
    total_optimizer_steps = steps_per_epoch * num_train_epochs
    calculated_warmup_steps = max(1, int(total_optimizer_steps * warmup_ratio))

    args_kwargs = {
        "output_dir": output_dir,
        "learning_rate": learning_rate,
        "per_device_train_batch_size": per_device_train_batch_size if device == "cuda" else min(8, per_device_train_batch_size),
        "per_device_eval_batch_size": per_device_eval_batch_size if device == "cuda" else min(16, per_device_eval_batch_size),
        "gradient_accumulation_steps": gradient_accumulation_steps if device == "cuda" else 1,
        "num_train_epochs": num_train_epochs,
        "weight_decay": weight_decay,
        "metric_for_best_model": "macro_f1",
        "greater_is_better": True,
        "load_best_model_at_end": True,
        "logging_dir": os.path.join(output_dir, "logs"),
        "logging_steps": 50 if not smoke_test else 5,
        "save_total_limit": 2,
        "seed": seed,
        "fp16": (device == "cuda"),
        "report_to": "none",
    }

    # Evaluation strategy compatibility (v5/v4.46+ uses 'eval_strategy', older uses 'evaluation_strategy')
    if "eval_strategy" in params:
        args_kwargs["eval_strategy"] = "epoch"
    elif "evaluation_strategy" in params:
        args_kwargs["evaluation_strategy"] = "epoch"

    if "save_strategy" in params:
        args_kwargs["save_strategy"] = "epoch"

    # Warmup parameter compatibility:
    # In Transformers v5.x, warmup_ratio was removed; warmup_steps accepts an int or float.
    # We pass warmup_ratio if present; otherwise explicit warmup_steps.
    if "warmup_ratio" in params:
        args_kwargs["warmup_ratio"] = warmup_ratio
    elif "warmup_steps" in params:
        args_kwargs["warmup_steps"] = calculated_warmup_steps
        print(
            f"Note: 'warmup_ratio' not in TrainingArguments signature (Transformers v5+ detected). "
            f"Using calculated warmup_steps={calculated_warmup_steps} "
            f"({warmup_ratio*100:.1f}% of {total_optimizer_steps} total optimizer steps)."
        )

    # Filter out any keys that might not exist in this specific version's constructor
    final_kwargs = {k: v for k, v in args_kwargs.items() if k in params}
    return TrainingArguments(**final_kwargs)
    """Calculate balanced class weights strictly from the training split.

    Formula:
        weight[c] = total_samples / (num_classes * count[c])

    This penalizes misclassifications on minority classes (e.g., Hate ~4.5%)
    more heavily than majority classes (Neutral ~78%).
    """
    counts = np.bincount(y_train, minlength=num_classes)
    total_samples = len(y_train)
    weights = total_samples / (num_classes * np.maximum(counts, 1).astype(float))
    # Normalize so mean weight is 1.0 for numerical stability
    weights = weights / np.mean(weights)
    return weights


if HAS_TORCH_TRANSFORMERS:
    class HinglishDataset(Dataset):
        """PyTorch Dataset wrapping cleaned Hinglish text and categorical labels."""

        def __init__(self, texts: List[str], labels: List[int], tokenizer, max_length: int = 128):
            self.texts = [str(t) for t in texts]
            self.labels = [int(l) for l in labels]
            self.tokenizer = tokenizer
            self.max_length = max_length

        def __len__(self) -> int:
            return len(self.texts)

        def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
            text = self.texts[idx]
            label = self.labels[idx]

            # Tokenization produces input_ids (token IDs) and attention_mask
            # attention_mask has 1 for real tokens and 0 for padding tokens
            encoding = self.tokenizer(
                text,
                truncation=True,
                max_length=self.max_length,
                padding=False,  # Dynamic padding handled by DataCollatorWithPadding
                return_tensors=None,
            )
            item = {k: torch.tensor(v, dtype=torch.long) for k, v in encoding.items()}
            item["labels"] = torch.tensor(label, dtype=torch.long)
            return item

    class WeightedLossTrainer(Trainer):
        """Custom Hugging Face Trainer incorporating class-weighted cross-entropy loss.

        Weighted Cross-Entropy Loss:
            Loss = - sum_{c} w_c * y_c * log(p_c)
        Where w_c is the inverse frequency weight computed strictly on train.csv.
        """

        def __init__(self, class_weights: torch.Tensor, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.class_weights = class_weights

        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            logits = outputs.logits

            # Ensure weights match model device (CUDA or CPU)
            device = logits.device
            loss_fn = nn.CrossEntropyLoss(weight=self.class_weights.to(device))
            loss = loss_fn(logits.view(-1, self.model.config.num_labels), labels.view(-1))

            return (loss, outputs) if return_outputs else loss


def compute_evaluation_metrics(p: Union[Tuple, "EvalPrediction"]) -> Dict[str, float]:
    """Compute evaluation metrics from predictions and ground-truth labels."""
    if hasattr(p, "predictions"):
        preds = p.predictions
        labels = p.label_ids
    else:
        preds, labels = p

    if isinstance(preds, tuple):
        preds = preds[0]

    preds = np.argmax(preds, axis=1)

    acc = accuracy_score(labels, preds)
    macro_p = precision_score(labels, preds, average="macro", zero_division=0)
    macro_r = recall_score(labels, preds, average="macro", zero_division=0)
    macro_f1 = f1_score(labels, preds, average="macro", zero_division=0)
    weighted_f1 = f1_score(labels, preds, average="weighted", zero_division=0)

    return {
        "accuracy": float(acc),
        "macro_precision": float(macro_p),
        "macro_recall": float(macro_r),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
    }


def full_evaluation_report(y_true: List[int], y_pred: List[int]) -> Dict[str, Union[float, Dict, List]]:
    """Produce detailed per-class metrics, confusion matrix, and summary scores."""
    acc = float(accuracy_score(y_true, y_pred))
    macro_p = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    macro_r = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

    report = classification_report(
        y_true, y_pred, target_names=CLASS_NAMES, output_dict=True, zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred).tolist()

    return {
        "accuracy": acc,
        "macro_precision": macro_p,
        "macro_recall": macro_r,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "classification_report": report,
        "confusion_matrix": cm,
    }


def train_and_evaluate(
    model_name: str = DEFAULT_MODEL_NAME,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    results_dir: str = DEFAULT_RESULTS_DIR,
    max_length: int = 128,
    num_train_epochs: int = 3,
    per_device_train_batch_size: int = 16,
    per_device_eval_batch_size: int = 32,
    gradient_accumulation_steps: int = 2,
    learning_rate: float = 2e-5,
    warmup_ratio: float = 0.1,
    weight_decay: float = 0.01,
    seed: int = 42,
    smoke_test: bool = False,
    sample_size: Optional[int] = None,
) -> None:
    """Execute complete transformer training, validation selection, and final test evaluation."""
    if not HAS_TORCH_TRANSFORMERS:
        raise ImportError(
            "PyTorch and Hugging Face Transformers are required to run training. "
            "Install them via 'pip install torch transformers datasets accelerate'."
        )

    set_reproducible_seed(seed)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device detected for training: {device}")
    if device == "cpu" and not smoke_test:
        print(
            "WARNING: Running full transformer training on CPU is extremely slow. "
            "Use Google Colab GPU or pass --smoke_test for quick verification."
        )

    # 1. Load Data
    print("Loading processed datasets...")
    train_df = pd.read_csv(TRAIN_PATH)
    val_df = pd.read_csv(VAL_PATH)
    test_df = pd.read_csv(TEST_PATH)

    # Ensure no NaN texts
    train_df["text"] = train_df["text"].fillna("").astype(str)
    val_df["text"] = val_df["text"].fillna("").astype(str)
    test_df["text"] = test_df["text"].fillna("").astype(str)

    if smoke_test:
        sample_size = sample_size or 100
        print(f"Smoke test mode enabled: subsampling to {sample_size} samples per split.")
        train_df = train_df.head(sample_size)
        val_df = val_df.head(sample_size // 2)
        test_df = test_df.head(sample_size // 2)
        num_train_epochs = 1

    print(f"Dataset sizes -> Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    # 2. Compute Class Weights from Training Split ONLY
    y_train_arr = train_df["label"].to_numpy(dtype=int)
    class_weights_arr = calculate_class_weights(y_train_arr, num_classes=3)
    class_weights_tensor = torch.tensor(class_weights_arr, dtype=torch.float)
    print("Computed Class Weights (strictly from train split):")
    for name, w in zip(CLASS_NAMES, class_weights_arr):
        print(f"  {name}: {w:.4f}")

    # 3. Load Tokenizer & Model
    print(f"Loading Tokenizer: {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    print(f"Loading Model Architecture: {model_name} for 3-class classification...")
    config = AutoConfig.from_pretrained(
        model_name,
        num_labels=3,
        id2label=ID_TO_LABEL,
        label2id=LABEL_TO_ID,
    )
    model = AutoModelForSequenceClassification.from_pretrained(model_name, config=config)

    # 4. Prepare Datasets
    train_dataset = HinglishDataset(train_df["text"].tolist(), train_df["label"].tolist(), tokenizer, max_length)
    val_dataset = HinglishDataset(val_df["text"].tolist(), val_df["label"].tolist(), tokenizer, max_length)
    test_dataset = HinglishDataset(test_df["text"].tolist(), test_df["label"].tolist(), tokenizer, max_length)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding=True)

    # 5. Training Arguments
    training_args = build_training_arguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        warmup_ratio=warmup_ratio,
        total_train_samples=len(train_df),
        device=device,
        seed=seed,
        smoke_test=smoke_test,
    )

    trainer_kwargs = {
        "class_weights": class_weights_tensor,
        "model": model,
        "args": training_args,
        "train_dataset": train_dataset,
        "eval_dataset": val_dataset,
        "data_collator": data_collator,
        "compute_metrics": compute_evaluation_metrics,
    }

    # Pass tokenizer using modern 'processing_class' if supported, fallback to 'tokenizer'
    trainer_sig = inspect.signature(Trainer.__init__)
    if "processing_class" in trainer_sig.parameters:
        trainer_kwargs["processing_class"] = tokenizer
    else:
        trainer_kwargs["tokenizer"] = tokenizer

    trainer = WeightedLossTrainer(**trainer_kwargs)

    # 6. Train with Validation Model Selection
    print("\n--- Starting Transformer Training ---")
    train_result = trainer.train()
    print("Training completed.")

    # 7. Save Best Checkpoint Artifacts
    best_checkpoint_dir = os.path.join(output_dir, "best_model")
    trainer.save_model(best_checkpoint_dir)
    tokenizer.save_pretrained(best_checkpoint_dir)
    print(f"Saved best model checkpoint to {best_checkpoint_dir}")

    # 8. Single Final Evaluation on Test Set (UNSEEN)
    print("\n--- Evaluating Best Model on UNSEEN Test Set ---")
    test_predictions = trainer.predict(test_dataset)
    y_test_pred = np.argmax(test_predictions.predictions, axis=1).tolist()
    y_test_true = test_df["label"].tolist()

    eval_report = full_evaluation_report(y_test_true, y_test_pred)
    eval_report["model_name"] = model_name
    eval_report["class_weights"] = {name: float(w) for name, w in zip(CLASS_NAMES, class_weights_arr)}
    eval_report["training_epochs"] = num_train_epochs
    eval_report["max_length"] = max_length
    eval_report["smoke_test"] = smoke_test

    # Save Results
    results_json_path = os.path.join(results_dir, "transformer_test_metrics.json")
    with open(results_json_path, "w", encoding="utf-8") as f:
        json.dump(eval_report, f, indent=2)

    cm_path = os.path.join(results_dir, "transformer_confusion_matrix.csv")
    cm_df = pd.DataFrame(
        eval_report["confusion_matrix"],
        index=[f"Actual_{c}" for c in CLASS_NAMES],
        columns=[f"Pred_{c}" for c in CLASS_NAMES],
    )
    cm_df.to_csv(cm_path)

    # Write Markdown Summary
    report_md_path = os.path.join(results_dir, "transformer_evaluation_report.md")
    write_evaluation_markdown(report_md_path, eval_report)

    print(f"Saved evaluation outputs to {results_dir}")
    print(f"Test Accuracy: {eval_report['accuracy']*100:.2f}% | Test Macro F1: {eval_report['macro_f1']*100:.2f}%")


def write_evaluation_markdown(file_path: str, report: Dict) -> None:
    """Generate markdown evaluation report for the transformer classifier."""
    rep = report["classification_report"]
    cm = report["confusion_matrix"]

    md = f"""# HinglishGuard AI — Transformer Model (XLM-RoBERTa) Evaluation Report

**Model Architecture**: `{report.get('model_name', 'xlm-roberta-base')}`
**Target Taxonomy**: 3-Class Content Moderation (`0 = Neutral`, `1 = Offensive`, `2 = Hate`)
**Loss Function**: Class-Weighted Cross-Entropy Loss
**Model Selection Criterion**: Validation Macro F1

---

## 1. Test Set Performance (Unseen Test Split)

| Global Metric | Score |
| :--- | :---: |
| **Accuracy** | **{report['accuracy']*100:.2f}%** |
| **Macro Precision** | **{report['macro_precision']*100:.2f}%** |
| **Macro Recall** | **{report['macro_recall']*100:.2f}%** |
| **Macro F1** | **{report['macro_f1']*100:.2f}%** |
| **Weighted F1** | **{report['weighted_f1']*100:.2f}%** |

### Per-Class Performance
| Class | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| **Neutral (0)** | {rep['Neutral']['precision']*100:.2f}% | {rep['Neutral']['recall']*100:.2f}% | {rep['Neutral']['f1-score']*100:.2f}% | {int(rep['Neutral']['support']):,} |
| **Offensive (1)** | {rep['Offensive']['precision']*100:.2f}% | {rep['Offensive']['recall']*100:.2f}% | {rep['Offensive']['f1-score']*100:.2f}% | {int(rep['Offensive']['support']):,} |
| **Hate (2)** | {rep['Hate']['precision']*100:.2f}% | {rep['Hate']['recall']*100:.2f}% | {rep['Hate']['f1-score']*100:.2f}% | {int(rep['Hate']['support']):,} |

### Confusion Matrix
| Actual \\ Predicted | Pred Neutral (0) | Pred Offensive (1) | Pred Hate (2) |
| :--- | :---: | :---: | :---: |
| **Actual Neutral** | **{cm[0][0]:,}** | {cm[0][1]:,} | {cm[0][2]:,} |
| **Actual Offensive** | {cm[1][0]:,} | **{cm[1][1]:,}** | {cm[1][2]:,} |
| **Actual Hate** | {cm[2][0]:,} | {cm[2][1]:,} | **{cm[2][2]:,}** |
"""
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(md)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune XLM-RoBERTa for Hinglish Content Moderation.")
    parser.add_argument("--model_name", type=str, default=DEFAULT_MODEL_NAME)
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--results_dir", type=str, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke_test", action="store_true", help="Run quick 1-epoch dry-run on 100 samples.")
    parser.add_argument("--sample_size", type=int, default=None)

    args = parser.parse_args()
    train_and_evaluate(
        model_name=args.model_name,
        output_dir=args.output_dir,
        results_dir=args.results_dir,
        max_length=args.max_length,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.lr,
        seed=args.seed,
        smoke_test=args.smoke_test,
        sample_size=args.sample_size,
    )
