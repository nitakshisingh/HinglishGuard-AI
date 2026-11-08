# HinglishGuard AI — XLM-RoBERTa Training & Fine-Tuning Guide (Google Colab / Cloud GPU)

This guide provides instructions to fine-tune **XLM-RoBERTa** on the three-class HinglishGuard dataset using a free or pro **Google Colab GPU** (T4, V100, or A100).

---

## 1. Background & Architecture Details

### Objective
Train a deep multilingual transformer to detect and moderate Hinglish content across 3 target classes:
- **0 = Neutral** (benign / non-hateful)
- **1 = Offensive** (vulgar, abusive, profane without identity targeting)
- **2 = Hate** (targeted identity-based hatred, communal slurs, dehumanization)

### Architectural Mechanisms
1. **SentencePiece BPE Tokenization**:
   - Model: `xlm-roberta-base` (250,000 sub-word vocabulary covering Hindi in both Latin and Devanagari scripts alongside English).
   - `max_length = 128` captures full social media posts while optimizing memory footprint.
   - Dynamic padding via `DataCollatorWithPadding` pads batches to the longest sample in each batch rather than static 128 tokens, achieving ~30-40% throughput speedup.
2. **Contextual Representations**:
   - 12 hidden layers, 768 hidden dimensions, 12 attention heads.
   - Outputs deep bi-directional contextual embeddings sensitive to code-mixing context and word order.
3. **Classification Head**:
   - Extracts the `<s>` (CLS token) embedding.
   - Passes through a non-linear dense projection layer (`tanh`), dropout ($p=0.1$), and a linear projection head ($768 \rightarrow 3$).
4. **Class-Weighted Cross-Entropy Loss**:
   - Our training set has ~78% Neutral, 17.5% Offensive, and 4.5% Hate.
   - `src/train_transformer.py` dynamically computes balanced inverse frequency weights strictly on `train.csv`:
     $$w_c = \frac{N_{\text{total}}}{K \cdot N_c}$$
   - Penalizes misclassifying the minority Hate class proportionally, boosting Macro F1 and minority recall.
5. **Validation Model Selection**:
   - Model selection is governed exclusively by **Validation Macro F1**.
   - The test set (`test.csv`) is evaluated **strictly once** at the very end.

---

## 2. Google Colab Step-by-Step Execution

### Step 1: Open Google Colab & Select GPU Runtime
1. Open [Google Colab](https://colab.research.google.com/).
2. Click **Runtime** $\rightarrow$ **Change runtime type**.
3. Under **Hardware accelerator**, select **T4 GPU** (Standard free tier) or **A100 GPU** (Colab Pro).

### Step 2: Install Required Libraries in Colab
Run the following cell to install modern Hugging Face and PyTorch dependencies:

```python
!pip install -q torch transformers datasets accelerate scikit-learn pandas
```

> **API Compatibility Note**:
> In modern Transformers (v5.x / v4.46+), parameter names have evolved (e.g. `eval_strategy` replaced `evaluation_strategy`, `warmup_ratio` was unified into `warmup_steps`, and `Trainer` accepts `processing_class`).
> Our `src/train_transformer.py` dynamically inspects the installed constructor signature at runtime via `build_training_arguments()`, ensuring immediate compatibility with Transformers 5.22+ and 4.x.

Verify GPU availability:
```python
import torch
print("CUDA Available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Device Name:", torch.cuda.get_device_name(0))
```

### Step 3: Upload the Cleaned Split Files
You need the 3 processed CSV files from your local workspace (`data/processed/`):
- `train.csv` (40,083 rows)
- `validation.csv` (8,589 rows)
- `test.csv` (8,590 rows)

You can upload them directly in Colab via the file browser or with Python:
```python
import os
os.makedirs("data/processed", exist_ok=True)

from google.colab import files
uploaded = files.upload() # Select train.csv, validation.csv, test.csv

# Move uploaded files into data/processed/
import shutil
for fname in ["train.csv", "validation.csv", "test.csv"]:
    if os.path.exists(fname):
        shutil.move(fname, os.path.join("data/processed", fname))
```

*(Alternatively, zip the `data/processed` folder or clone your git repository).*

### Step 4: Upload Training Code
Upload `src/train_transformer.py` and `src/preprocessing.py`:
```python
os.makedirs("src", exist_ok=True)
# Upload src/train_transformer.py and src/preprocessing.py using the Colab file drawer
```

### Step 5: Launch Training
Run the training script in a Colab code cell:

```bash
!python -m src.train_transformer \
    --model_name xlm-roberta-base \
    --output_dir models/transformers/xlm_roberta_hinglish \
    --results_dir data/processed/transformer_results \
    --epochs 3 \
    --batch_size 16 \
    --gradient_accumulation_steps 2 \
    --lr 2e-5 \
    --max_length 128 \
    --seed 42
```

> **Expected Run Time**:
> - On a Google Colab T4 GPU: ~15 to 25 minutes for 3 epochs on 40,083 samples.
> - On an A100 GPU: ~5 to 8 minutes.

### Step 6: Download the Trained Artifacts Back to Local Workspace
After training completes, Colab will save:
1. Best checkpoint in `models/transformers/xlm_roberta_hinglish/best_model/`
2. Test evaluation outputs in `data/processed/transformer_results/`

Zip and download the artifacts:
```python
!zip -r xlm_roberta_best_model.zip models/transformers/xlm_roberta_hinglish/best_model/
!zip -r transformer_results.zip data/processed/transformer_results/

from google.colab import files
files.download("xlm_roberta_best_model.zip")
files.download("transformer_results.zip")
```

Extract `xlm_roberta_best_model.zip` into your local `models/transformers/xlm_roberta_hinglish/best_model/` folder.

---

## 3. Local Verification & Smoke Testing

To test that the pipeline works without downloading the full 1.1GB weights locally, run a dry-run smoke test (1 epoch on 100 samples):

```bash
# In your local activated virtual environment:
.venv\Scripts\python -m src.train_transformer --smoke_test
```

All unit tests verify the tokenization, label mappings, and inference interfaces:
```bash
.venv\Scripts\python -m pytest tests/test_transformer.py -v
```
