# HinglishGuard AI — Baseline Machine Learning Model Evaluation Report

**Model**: TF-IDF + Logistic Regression
**Dataset**: Indo-HateSpeech Processed Splits (`train.csv`, `validation.csv`, `test.csv`)
**Classes**: `0 = Neutral`, `1 = Offensive`, `2 = Hate`

---

## 1. Validation Tuning & Experimentation

Models were trained on `train.csv` (40,083 rows) and evaluated on `validation.csv` (8,589 rows).
No test data was used for model or hyperparameter selection.

| Feature Extraction | Regularization (C) | Val Accuracy | Val Macro Prec | Val Macro Recall | Val Macro F1 | Val Weighted F1 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Word n-grams (1,2) | 0.1 | 80.03% | 58.08% | 66.47% | **60.85%** | 81.37% |
| Word n-grams (1,2) | 0.5 | 82.72% | 61.30% | 67.74% | **63.84%** | 83.73% |
| Word n-grams (1,2) | 1.0 | 84.35% | 63.46% | 68.55% | **65.62%** | 85.09% |
| Word n-grams (1,2) | 2.0 | 85.25% | 64.60% | 68.65% | **66.39%** | 85.82% |
| Word n-grams (1,2) | 5.0 | 85.90% | 65.68% | 68.64% | **67.04%** | 86.32% |
| Character n-grams (3,5) | 0.1 | 84.29% | 64.77% | 74.35% | **67.89%** | 85.02% |
| Character n-grams (3,5) | 0.5 | 87.15% | 69.71% | 77.27% | **72.78%** | 87.71% |
| Character n-grams (3,5) | 1.0 | 88.25% | 71.94% | 78.05% | **74.60%** | 88.70% |
| Character n-grams (3,5) | 2.0 | 89.13% | 73.86% | 78.79% | **76.08%** | 89.47% |
| Character n-grams (3,5) | 5.0 | 90.06% | 75.99% | 78.87% | **77.36%** | 90.26% |

### Selection Decision
- **Winner**: `Character n-grams (3,5)` with `C=5.0`
- **Validation Macro F1**: **77.36%**

---

## 2. Final Evaluation on Unseen Test Set (`test.csv`, 8,590 rows)

### Selected Best Baseline: Character n-grams (3,5) (C=5.0)

| Global Metric | Score |
| :--- | :---: |
| **Accuracy** | **89.23%** |
| **Macro Precision** | **73.33%** |
| **Macro Recall** | **76.02%** |
| **Macro F1** | **74.60%** |
| **Weighted F1** | **89.49%** |

### Per-Class Detailed Performance
| Class | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| **Neutral (0)** | 96.62% | 94.27% | 95.43% | 6,700 |
| **Offensive (1)** | 68.46% | 74.47% | 71.34% | 1,504 |
| **Hate (2)** | 54.92% | 59.33% | 57.04% | 386 |

### Confusion Matrix
| Actual \ Predicted | Pred Neutral (0) | Pred Offensive (1) | Pred Hate (2) |
| :--- | :---: | :---: | :---: |
| **Actual Neutral** | **6,316** | 366 | 18 |
| **Actual Offensive** | 214 | **1,120** | 170 |
| **Actual Hate** | 7 | 150 | **229** |

---

## 3. Comparison with Alternate Experiment: Character n-grams (3,5)

Character n-grams are effective at capturing phonetic variations in Hinglish (e.g., *'kutta'* vs *'kuttaa'*).

| Model Configuration | Test Accuracy | Test Macro F1 | Test Weighted F1 | Neutral F1 | Offensive F1 | Hate F1 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Word n-grams (1,2) [C=5.0]** | 89.23% | **74.60%** | 89.49% | 95.43% | 71.34% | 57.04% |
| **Char n-grams (3,5) [C=5.0]** | 85.79% | **64.89%** | 86.15% | 94.19% | 62.76% | 37.73% |
