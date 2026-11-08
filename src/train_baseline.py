"""
Baseline Model Training & Hyperparameter Tuning Pipeline for HinglishGuard AI.
Trains TF-IDF + LogisticRegression models (evaluating both word n-grams and char n-grams)
using only training and validation sets, selects the best configuration on validation Macro F1,
evaluates on test.csv, and saves model artifacts and reports.
"""

import json
import os
import time
import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

TRAIN_PATH = "data/processed/train.csv"
VAL_PATH = "data/processed/validation.csv"
TEST_PATH = "data/processed/test.csv"

MODELS_DIR = "models/baseline"
RESULTS_DIR = "data/processed/baseline_results"

CLASS_NAMES = ["Neutral", "Offensive", "Hate"]


def load_datasets():
    print("Loading train, validation, and test datasets...")
    train_df = pd.read_csv(TRAIN_PATH)
    val_df = pd.read_csv(VAL_PATH)
    test_df = pd.read_csv(TEST_PATH)

    # Ensure no NaN texts
    train_df["text"] = train_df["text"].fillna("").astype(str)
    val_df["text"] = val_df["text"].fillna("").astype(str)
    test_df["text"] = test_df["text"].fillna("").astype(str)

    print(f"Loaded {len(train_df)} train, {len(val_df)} validation, and {len(test_df)} test records.")
    return train_df, val_df, test_df


def evaluate_predictions(y_true, y_pred):
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


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    train_df, val_df, test_df = load_datasets()

    X_train, y_train = train_df["text"], train_df["label"]
    X_val, y_val = val_df["text"], val_df["label"]
    X_test, y_test = test_df["text"], test_df["label"]

    # 1. Define Vectorizer Configurations
    vectorizer_configs = {
        "word_ngrams_1_2": {
            "name": "Word n-grams (1,2)",
            "vectorizer": TfidfVectorizer(
                ngram_range=(1, 2),
                sublinear_tf=True,
                max_features=50000,
                min_df=2,
            ),
        },
        "char_ngrams_3_5": {
            "name": "Character n-grams (3,5)",
            "vectorizer": TfidfVectorizer(
                analyzer="char",
                ngram_range=(3, 5),
                sublinear_tf=True,
                max_features=50000,
                min_df=2,
            ),
        },
    }

    # 2. Hyperparameter Grid for C in LogisticRegression
    # Trained on X_train, tuned on X_val ONLY
    c_candidates = [0.1, 0.5, 1.0, 2.0, 5.0]

    validation_experiments = []
    best_config = None
    best_val_macro_f1 = -1.0

    print("\n--- Starting Validation Experiments & Hyperparameter Tuning ---")
    for vec_key, vec_info in vectorizer_configs.items():
        print(f"\nFitting Vectorizer: {vec_info['name']}...")
        t0 = time.time()
        vec = vec_info["vectorizer"]
        X_train_vec = vec.fit_transform(X_train)
        X_val_vec = vec.transform(X_val)
        print(f"Vectorized train shape: {X_train_vec.shape} (Time: {time.time()-t0:.2f}s)")

        for c_val in c_candidates:
            print(f"Training LogisticRegression ({vec_key}, C={c_val}, class_weight='balanced')...")
            clf = LogisticRegression(
                C=c_val,
                max_iter=1000,
                class_weight="balanced",
                random_state=42,
                solver="lbfgs",
                n_jobs=-1,
            )
            clf.fit(X_train_vec, y_train)

            y_val_pred = clf.predict(X_val_vec)
            val_metrics = evaluate_predictions(y_val, y_val_pred)

            exp_result = {
                "vectorizer_type": vec_key,
                "vectorizer_name": vec_info["name"],
                "C": c_val,
                "val_accuracy": val_metrics["accuracy"],
                "val_macro_f1": val_metrics["macro_f1"],
                "val_weighted_f1": val_metrics["weighted_f1"],
                "val_macro_precision": val_metrics["macro_precision"],
                "val_macro_recall": val_metrics["macro_recall"],
            }
            validation_experiments.append(exp_result)
            print(
                f"-> Result: Val Acc={val_metrics['accuracy']:.4f} | "
                f"Val Macro F1={val_metrics['macro_f1']:.4f} | "
                f"Val Weighted F1={val_metrics['weighted_f1']:.4f}"
            )

            if val_metrics["macro_f1"] > best_val_macro_f1:
                best_val_macro_f1 = val_metrics["macro_f1"]
                best_config = {
                    "vectorizer_key": vec_key,
                    "vectorizer_name": vec_info["name"],
                    "vectorizer_obj": vec,
                    "C": c_val,
                    "val_metrics": val_metrics,
                }

    print("\n=======================================================")
    print(f"BEST CONFIGURATION SELECTED ON VALIDATION SET:")
    print(f"Vectorizer: {best_config['vectorizer_name']}")
    print(f"C: {best_config['C']}")
    print(f"Validation Macro F1: {best_val_macro_f1:.4f}")
    print("=======================================================")

    # 3. Retrain Best Configuration on Combined Train+Val OR evaluate model on Test
    # Standard practice: Retrain best model on full train set (or train+val) or evaluate existing fit.
    # To keep strict adherence to train/val/test splits, let's retrain on Train with best C and evaluate on Test.
    print("\nRetraining Best Model on Train Set with chosen vectorizer and C...")
    best_vectorizer = vectorizer_configs[best_config["vectorizer_key"]]["vectorizer"]
    X_train_vec = best_vectorizer.fit_transform(X_train)
    X_test_vec = best_vectorizer.transform(X_test)

    best_clf = LogisticRegression(
        C=best_config["C"],
        max_iter=1000,
        class_weight="balanced",
        random_state=42,
        solver="lbfgs",
        n_jobs=-1,
    )
    best_clf.fit(X_train_vec, y_train)

    # 4. Final Evaluation on UNSEEN Test Set
    print("\nEvaluating Best Model on UNSEEN Test Set...")
    y_test_pred = best_clf.predict(X_test_vec)
    test_metrics = evaluate_predictions(y_test, y_test_pred)

    print(f"--- TEST SET RESULTS ---")
    print(f"Accuracy:        {test_metrics['accuracy']:.4f}")
    print(f"Macro Precision: {test_metrics['macro_precision']:.4f}")
    print(f"Macro Recall:    {test_metrics['macro_recall']:.4f}")
    print(f"Macro F1:        {test_metrics['macro_f1']:.4f}")
    print(f"Weighted F1:     {test_metrics['weighted_f1']:.4f}")
    print("\nClassification Report:\n", classification_report(y_test, y_test_pred, target_names=CLASS_NAMES))
    print("Confusion Matrix:\n", np.array(test_metrics["confusion_matrix"]))

    # Also evaluate the alternate vectorizer on Test for comprehensive research comparison
    alt_key = "char_ngrams_3_5" if best_config["vectorizer_key"] == "word_ngrams_1_2" else "word_ngrams_1_2"
    alt_vec = vectorizer_configs[alt_key]["vectorizer"]
    # pick best C for alt_vec from validation_experiments
    alt_exps = [e for e in validation_experiments if e["vectorizer_type"] == alt_key]
    best_alt_exp = max(alt_exps, key=lambda x: x["val_macro_f1"])
    print(f"\nEvaluating Alternate Model ({best_alt_exp['vectorizer_name']}, C={best_alt_exp['C']}) on Test...")
    X_train_alt = alt_vec.fit_transform(X_train)
    X_test_alt = alt_vec.transform(X_test)
    alt_clf = LogisticRegression(
        C=best_alt_exp["C"],
        max_iter=1000,
        class_weight="balanced",
        random_state=42,
        solver="lbfgs",
        n_jobs=-1,
    )
    alt_clf.fit(X_train_alt, y_train)
    y_test_alt_pred = alt_clf.predict(X_test_alt)
    alt_test_metrics = evaluate_predictions(y_test, y_test_alt_pred)

    # 5. Save Artifacts under models/baseline/
    model_save_path = os.path.join(MODELS_DIR, "logistic_regression_baseline.joblib")
    vec_save_path = os.path.join(MODELS_DIR, "tfidf_vectorizer.joblib")
    metadata_save_path = os.path.join(MODELS_DIR, "baseline_metadata.json")

    joblib.dump(best_clf, model_save_path)
    joblib.dump(best_vectorizer, vec_save_path)

    metadata = {
        "model_type": "LogisticRegression",
        "best_hyperparameters": {
            "C": best_config["C"],
            "max_iter": 1000,
            "class_weight": "balanced",
            "random_state": 42,
        },
        "vectorizer": {
            "type": best_config["vectorizer_key"],
            "name": best_config["vectorizer_name"],
            "ngram_range": list(best_vectorizer.ngram_range),
            "analyzer": best_vectorizer.analyzer,
            "sublinear_tf": best_vectorizer.sublinear_tf,
            "max_features": best_vectorizer.max_features,
            "min_df": best_vectorizer.min_df,
            "vocabulary_size": len(best_vectorizer.vocabulary_),
        },
        "classes": CLASS_NAMES,
        "test_metrics": {
            "accuracy": test_metrics["accuracy"],
            "macro_precision": test_metrics["macro_precision"],
            "macro_recall": test_metrics["macro_recall"],
            "macro_f1": test_metrics["macro_f1"],
            "weighted_f1": test_metrics["weighted_f1"],
        },
    }
    with open(metadata_save_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nSaved model to {model_save_path}")
    print(f"Saved vectorizer to {vec_save_path}")
    print(f"Saved metadata to {metadata_save_path}")

    # 6. Save Evaluation Metrics and Confusion Matrix under data/processed/baseline_results/
    eval_metrics_file = os.path.join(RESULTS_DIR, "test_evaluation_metrics.json")
    with open(eval_metrics_file, "w", encoding="utf-8") as f:
        json.dump({
            "selected_model": {
                "vectorizer": best_config["vectorizer_name"],
                "C": best_config["C"],
                "metrics": test_metrics,
            },
            "alternate_model": {
                "vectorizer": best_alt_exp["vectorizer_name"],
                "C": best_alt_exp["C"],
                "metrics": alt_test_metrics,
            },
            "validation_tuning_experiments": validation_experiments,
        }, f, indent=2)

    cm_file = os.path.join(RESULTS_DIR, "confusion_matrix.csv")
    cm_df = pd.DataFrame(
        test_metrics["confusion_matrix"],
        index=[f"Actual_{c}" for c in CLASS_NAMES],
        columns=[f"Pred_{c}" for c in CLASS_NAMES],
    )
    cm_df.to_csv(cm_file)

    # 7. Write Markdown Summary Report
    report_file = os.path.join(RESULTS_DIR, "baseline_evaluation_report.md")
    generate_baseline_markdown_report(
        report_file, best_config, validation_experiments, test_metrics, alt_test_metrics, best_alt_exp
    )
    print(f"Saved evaluation report to {report_file}")


def generate_baseline_markdown_report(
    report_file, best_config, val_experiments, test_metrics, alt_test_metrics, best_alt_exp
):
    val_table_rows = []
    for exp in val_experiments:
        val_table_rows.append(
            f"| {exp['vectorizer_name']} | {exp['C']} | "
            f"{exp['val_accuracy']*100:.2f}% | {exp['val_macro_precision']*100:.2f}% | "
            f"{exp['val_macro_recall']*100:.2f}% | **{exp['val_macro_f1']*100:.2f}%** | "
            f"{exp['val_weighted_f1']*100:.2f}% |"
        )
    val_table_str = "\n".join(val_table_rows)

    rep = test_metrics["classification_report"]
    cm = test_metrics["confusion_matrix"]

    report_md = f"""# HinglishGuard AI — Baseline Machine Learning Model Evaluation Report

**Model**: TF-IDF + Logistic Regression
**Dataset**: Indo-HateSpeech Processed Splits (`train.csv`, `validation.csv`, `test.csv`)
**Classes**: `0 = Neutral`, `1 = Offensive`, `2 = Hate`

---

## 1. Validation Tuning & Experimentation

Models were trained on `train.csv` (40,083 rows) and evaluated on `validation.csv` (8,589 rows).
No test data was used for model or hyperparameter selection.

| Feature Extraction | Regularization (C) | Val Accuracy | Val Macro Prec | Val Macro Recall | Val Macro F1 | Val Weighted F1 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
{val_table_str}

### Selection Decision
- **Winner**: `{best_config['vectorizer_name']}` with `C={best_config['C']}`
- **Validation Macro F1**: **{best_config['val_metrics']['macro_f1']*100:.2f}%**

---

## 2. Final Evaluation on Unseen Test Set (`test.csv`, 8,590 rows)

### Selected Best Baseline: {best_config['vectorizer_name']} (C={best_config['C']})

| Global Metric | Score |
| :--- | :---: |
| **Accuracy** | **{test_metrics['accuracy']*100:.2f}%** |
| **Macro Precision** | **{test_metrics['macro_precision']*100:.2f}%** |
| **Macro Recall** | **{test_metrics['macro_recall']*100:.2f}%** |
| **Macro F1** | **{test_metrics['macro_f1']*100:.2f}%** |
| **Weighted F1** | **{test_metrics['weighted_f1']*100:.2f}%** |

### Per-Class Detailed Performance
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

---

## 3. Comparison with Alternate Experiment: Character n-grams (3,5)

Character n-grams are effective at capturing phonetic variations in Hinglish (e.g., *'kutta'* vs *'kuttaa'*).

| Model Configuration | Test Accuracy | Test Macro F1 | Test Weighted F1 | Neutral F1 | Offensive F1 | Hate F1 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Word n-grams (1,2) [C={best_config['C']}]** | {test_metrics['accuracy']*100:.2f}% | **{test_metrics['macro_f1']*100:.2f}%** | {test_metrics['weighted_f1']*100:.2f}% | {rep['Neutral']['f1-score']*100:.2f}% | {rep['Offensive']['f1-score']*100:.2f}% | {rep['Hate']['f1-score']*100:.2f}% |
| **Char n-grams (3,5) [C={best_alt_exp['C']}]** | {alt_test_metrics['accuracy']*100:.2f}% | **{alt_test_metrics['macro_f1']*100:.2f}%** | {alt_test_metrics['weighted_f1']*100:.2f}% | {alt_test_metrics['classification_report']['Neutral']['f1-score']*100:.2f}% | {alt_test_metrics['classification_report']['Offensive']['f1-score']*100:.2f}% | {alt_test_metrics['classification_report']['Hate']['f1-score']*100:.2f}% |
"""
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(report_md)


if __name__ == "__main__":
    main()
