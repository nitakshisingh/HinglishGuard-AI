# HinglishGuard AI — Transformers API Compatibility Fix Report

**Date of Fix**: 2026-11-08
**Reported Error**: `TypeError: TrainingArguments.__init__() got an unexpected keyword argument 'warmup_ratio'`
**Colab Environment**:
- Transformers: `5.22.1`
- PyTorch: `2.14.1+cu130`
- GPU: Tesla T4

---

## 1. Root Cause Analysis

In Hugging Face `transformers` version 5.x (and starting from 4.46+ deprecation paths), several legacy parameter names in `TrainingArguments` and `Trainer` were removed:
1. **`warmup_ratio` Removed**:
   - `warmup_ratio` was deprecated and completely removed from the `TrainingArguments` constructor in Transformers v5.
   - Warmup scheduling in v5 is unified under `warmup_steps`: an integer represents absolute warmup steps, while a float less than 1.0 represents a ratio. However, passing the keyword argument `warmup_ratio` causes Python's `@dataclass` constructor to throw an immediate `TypeError`.
2. **`evaluation_strategy` vs `eval_strategy`**:
   - `evaluation_strategy` was deprecated in favor of `eval_strategy`.
3. **`Trainer.__init__` Tokenizer Parameter**:
   - In modern Transformers, passing `tokenizer` directly to `Trainer` triggers deprecation warnings in favor of `processing_class`.

---

## 2. Dynamic Signature Inspection & Fixes Implemented

Rather than guessing parameter names across environments, `src/train_transformer.py` now implements **`build_training_arguments()`**, which programmatically inspects `inspect.signature(TrainingArguments)` at runtime before instantiating arguments.

### A. Explicit Warmup Steps Calculation
When `warmup_ratio` is absent from the constructor signature:
1. The script computes the exact number of total optimizer update steps accounting for:
   - Total training samples ($N = 40,083$)
   - Micro-batch size ($B = 16$)
   - Gradient accumulation steps ($G = 2 \rightarrow \text{Effective Batch Size } = 32$)
   - Number of epochs ($E = 3$)
   $$\text{Steps Per Epoch} = \lceil 40,083 / 32 \rceil = 1,253 \text{ steps}$$
   $$\text{Total Optimizer Steps} = 1,253 \times 3 = 3,759 \text{ steps}$$
2. Explicit warmup steps are calculated for the intended 10% ratio:
   $$\text{warmup\_steps} = \text{int}(3,759 \times 0.1) = 375 \text{ steps}$$
3. Passes `warmup_steps=375` (or `warmup_ratio=0.1` if on v4.x), preserving identical learning rate warmup dynamics across all versions.

### B. Evaluation Strategy Fallback
- Dynamically checks `eval_strategy` (v5.x / v4.46+) and falls back to `evaluation_strategy` (v4.x legacy).

### C. Trainer `processing_class` Fallback
- Dynamically inspects `Trainer.__init__` and supplies `processing_class=tokenizer` if supported, falling back to `tokenizer=tokenizer`.

---

## 3. Preserved Training Behavior & Architecture

The fix strictly preserves all algorithmic specifications:
- **Architecture**: `xlm-roberta-base` (250k SentencePiece vocabulary, 12 layers, 768 hidden dim, sequence classification head).
- **Target Taxonomy**: 3 classes: Neutral (0), Offensive (1), Hate (2).
- **Class Imbalance Mitigation**: Weighted cross-entropy loss with inverse frequency weights computed **strictly from `train.csv`** ($w_c = \frac{N_{\text{total}}}{K \cdot N_c}$).
- **Optimization**:
  - Epochs: 3
  - Per-device batch size: 16
  - Gradient accumulation steps: 2 (effective batch size: 32)
  - Learning rate: 2e-5 (AdamW with weight decay 0.01)
  - Sequence length: 128 with dynamic batch padding
  - Random seed: 42
- **Model Selection**: Evaluates validation Macro F1 at each epoch and loads the best checkpoint (`load_best_model_at_end=True`).
- **Test Set Isolation**: Evaluates unseen `test.csv` strictly once after training.

---

## 4. Verification & Testing

Added a dedicated unit test in `tests/test_transformer.py` (`test_training_arguments_constructor_simulation`):
- Verifies argument resolution under both v4 (with `warmup_ratio`) and v5 (with `warmup_steps`).
- Confirms the mathematical calculation for 40,083 samples yielding 375 warmup steps.
- Full test suite passed (8/8 transformer tests passing).

---

## 5. Colab Commands

### Colab Installation Cell:
```python
!pip install -q torch transformers datasets accelerate scikit-learn pandas
```

### Colab Execution Command:
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
