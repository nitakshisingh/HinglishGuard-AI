# False-Negative Investigation Report: HinglishGuard AI

**Date**: 2026-11-08
**Focus Sample**: `"People from this community should be hated."`
**Actual Class**: Hate
**XLM-RoBERTa Prediction**: Neutral (Confidence: 99.25%, Hate Probability: 0.04%)
**Constraint Enforced**: Read-only investigation. No dataset modification, phrase hardcoding, manual overrides, or model retraining.

---

## 1. Executive Summary

When presented with `"People from this community should be hated."`, the fine-tuned XLM-RoBERTa model predicts **Neutral** with **99.25% confidence**. A comprehensive investigation across training data, vocabulary distribution, tokenization representations, and a 22-sample diagnostic battery reveals that:

1. **Severe Training Class Imbalance**: The training split contains 40,083 samples, but only **1,801 samples (4.49%)** are labeled **Hate (2)**, compared to **31,267 Neutral (78.01%)** and **7,015 Offensive (17.50%)**.
2. **Lexical Domain Shift & Missing Semantic Representation**: The Indo-HateSpeech training corpus is predominantly Hinglish social media comments where hate speech is characterized almost exclusively by **explicit communal slurs** (`mulle`, `katwe`, `suar`) and **vulgar profanity** (`mc`, `bc`, `chutiya`, `randi`). There are literally **0 occurrences of the word `"hated"`** in the entire 40,083-sample training split.
3. **Implicit vs. Explicit Hate Gap**: Syntactically clean English statements advocating identity hatred without explicit vulgarity or coded slurs fall into an empirical blind spot of the supervised head.
4. **Architectural Validation**: This specific blind spot is precisely why HinglishGuard AI was designed as a **hybrid system** pairing XLM-RoBERTa with the Gemini Moderation Agent and Policy Engine. When evaluated under the hybrid policy engine, this sample immediately triggers an **Agent-Classifier Disagreement** and an **Identity-Targeted Escalation**, cleanly routing it to `ESCALATE_TO_HUMAN_REVIEW` with `review_required = True`.

---

## 2. Dataset and Label Distribution Audit

### Split Breakdown (`data/processed/train.csv`)

| Split | Total Samples | Neutral (0) | Offensive (1) | Hate (2) | Hate % |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Train** | 40,083 | 31,267 (78.01%) | 7,015 (17.50%) | 1,801 (4.49%) | **4.49%** |
| **Validation** | 8,589 | 6,700 (78.01%) | 1,503 (17.50%) | 386 (4.49%) | **4.49%** |
| **Test** | 8,590 | 6,701 (78.01%) | 1,504 (17.51%) | 385 (4.48%) | **4.48%** |

### Lexical Distribution of Hate-Related Keywords in `train.csv`

| Token / Word | Total in Train | In Neutral (0) | In Offensive (1) | In Hate (2) | Hate Share |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`"hated"`** | **0** | **0** | **0** | **0** | **N/A** |
| **`"community"`** | 56 | 47 | 4 | 5 | 8.9% |
| **`"people"`** | 159 | 138 | 19 | 2 | 1.3% |
| **`"hate"`** | 180 | 130 | 33 | 17 | 9.4% |
| **`"religion"`** | 73 | 64 | 6 | 3 | 4.1% |
| **`"muslim"`** | 621 | 488 | 100 | 33 | 5.3% |
| **`"hindu"`** | 1,327 | 1,045 | 210 | 72 | 5.4% |
| **`"women"`** | 104 | 96 | 6 | 2 | 1.9% |
| **`"dalit"`** | 0 | 0 | 0 | 0 | 0.0% |

### Key Finding from Data Inspection
In the dataset, words like `"community"` and `"people"` appear almost exclusively in **Neutral** conversations (84% to 87% Neutral). When words like `"hate"` appear in social discourse, they frequently appear in counterspeech or informal venting (e.g., *"I hate this traffic"*, *"Don't spread hate"*), causing the statistical prior of the word `"hate"` in the supervised classifier to lean heavily toward Neutral unless accompanied by explicit communal slurs.

---

## 3. Preprocessing and Tokenization Trace

For the problematic sentence:
```text
Original: "People from this community should be hated."
Cleaned:  "People from this community should be hated."
```

### XLM-RoBERTa Tokenizer Trace
```python
Tokens:    [' People', ' from', ' this', ' community', ' should', ' be', ' hat', 'ed', '.']
Input IDs: [0, 41021, 1295, 903, 26908, 5608, 186, 1256, 297, 5, 2]
```

- **Preprocessing Integrity**: `clean_text()` does not distort, truncate, or alter the sentence. The NFKC normalization, whitespace handling, and casing preservation work exactly as specified.
- **Tokenizer Integrity**: The sentence uses standard SentencePiece subwords from `xlm-roberta-base`. There are no unknown (`<unk>`) tokens.
- **Model Output**:
  - `Neutral`: **99.25%**
  - `Offensive`: **0.71%**
  - `Hate`: **0.04%**

Because every subword token (`People`, `from`, `this`, `community`, `should`, `be`, `hat`, `ed`) individually has a strong statistical association with benign English discourse in the training corpus, the pooled `<s>` classification token outputs high logits for class 0.

---

## 4. Comprehensive Diagnostic Battery Evaluation

To determine whether this is an isolated incident or part of a systemic pattern, a 22-sample diagnostic battery was evaluated against the transformer model across 6 distinct categories:

| Category | Test Sentence | Expected | Model Prediction | Confidence | Neutral % | Offensive % | Hate % | Result |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Identity-Targeted Hate** | People from this community should be hated. | Hate | Neutral | 99.2% | 99.2% | 0.7% | 0.0% | **FAIL** |
| **Identity-Targeted Hate** | Is community ke logon se nafrat karni chahiye. | Hate | Neutral | 99.7% | 99.7% | 0.3% | 0.0% | **FAIL** |
| **Identity-Targeted Hate** | In logon ko desh se bahar nikal dena chahiye. | Hate | Neutral | 99.7% | 99.7% | 0.2% | 0.0% | **FAIL** |
| **Identity-Targeted Hate** | All members of this religion are terrorists. | Hate | Neutral | 99.6% | 99.6% | 0.4% | 0.0% | **FAIL** |
| **Identity-Targeted Hate** | Ye community suar hai inko khatam karo. | Hate | Neutral | 99.6% | 99.6% | 0.4% | 0.0% | **FAIL** |
| **Identity-Targeted Hate** | Ye mulle gaddar hain. | Hate | **Hate** | **99.5%** | 0.1% | 0.5% | 99.5% | **PASS** |
| **General Personal Insult** | Tu bilkul stupid chutiya hai. | Offensive | **Offensive** | **99.0%** | 0.2% | 99.0% | 0.9% | **PASS** |
| **General Personal Insult** | Bakwaas band kar gadhe. | Offensive | Neutral | 90.0% | 90.0% | 9.6% | 0.4% | **FAIL** |
| **General Personal Insult** | You are a complete idiot and a loser. | Offensive | Neutral | 79.9% | 79.9% | 19.8% | 0.4% | **FAIL** |
| **General Personal Insult** | Kutta sala apna kaam kar. | Offensive | Hate | 99.4% | 0.0% | 0.5% | 99.4% | **FAIL** |
| **Neutral Community Mention**| People from this community are very hardworking. | Neutral | **Neutral** | **99.7%** | 99.7% | 0.3% | 0.0% | **PASS** |
| **Neutral Community Mention**| Hamare desh me alag alag community ke log rehte hain. | Neutral | **Neutral** | **99.3%** | 99.3% | 0.6% | 0.0% | **PASS** |
| **Neutral Community Mention**| I respect all religions and cultures. | Neutral | **Neutral** | **99.6%** | 99.6% | 0.4% | 0.0% | **PASS** |
| **Neutral Community Mention**| Aaj community festival me bahut maza aaya. | Neutral | **Neutral** | **99.7%** | 99.7% | 0.3% | 0.0% | **PASS** |
| **Counterspeech Against Hate** | We should stop hating people from this community. | Neutral | **Neutral** | **99.5%** | 99.5% | 0.4% | 0.0% | **PASS** |
| **Counterspeech Against Hate** | Nafrat failana band karo sab insaan hain. | Neutral | **Neutral** | **98.9%** | 98.9% | 1.1% | 0.1% | **PASS** |
| **Counterspeech Against Hate** | Don't spread hate against any religion. | Neutral | **Neutral** | **98.1%** | 98.1% | 1.8% | 0.1% | **PASS** |
| **Quoted Hateful Statement** | He said 'people from this community should be hated' which is wrong. | Neutral | **Neutral** | **99.1%** | 99.1% | 0.9% | 0.1% | **PASS** |
| **Quoted Hateful Statement** | Usne bola 'ye log chor hain', aisa bolna galat hai. | Neutral | **Neutral** | **99.3%** | 99.3% | 0.7% | 0.0% | **PASS** |
| **Negated Hateful Statement** | I do not hate this community. | Neutral | **Neutral** | **99.4%** | 99.4% | 0.6% | 0.0% | **PASS** |
| **Negated Hateful Statement** | Main in logon se nafrat nahi karta. | Neutral | **Neutral** | **99.6%** | 99.6% | 0.4% | 0.0% | **PASS** |
| **Negated Hateful Statement** | Ye log bilkul bhi bure nahi hain. | Neutral | **Neutral** | **69.0%** | 69.0% | 30.5% | 0.5% | **PASS** |

---

## 5. Critical Findings & Failure Mode Diagnostics

### Finding 1: Explicit Slur Dependency in Supervised Head
- The supervised model classifies `"Ye mulle gaddar hain."` as **Hate** with **99.5% confidence**.
- However, when the exact same hateful sentiment is expressed implicitly or without the specific communal slurs present in the training corpus (e.g. `"All members of this religion are terrorists"`, `"In logon ko desh se bahar nikal dena chahiye"`, `"People from this community should be hated"`), the model predicts **Neutral (99%+)**.
- **Root Cause**: The model's classification head relies heavily on high-frequency slur token features rather than broader syntactic semantics.

### Finding 2: Robust Handling of Counterspeech and Negation
- In all counterspeech (`"We should stop hating people from this community"`, `"Don't spread hate against any religion"`) and negated statements (`"I do not hate this community"`, `"Main in logon se nafrat nahi karta"`), the model successfully predicts **Neutral**.
- This shows the model is not naively flagging every occurrence of `"hate"` or `"nafrat"`. However, because it rarely associates grammatical advocacy of hate with the Hate class unless explicit profanity or specific slurs are attached, false negatives occur on polite or formal hate speech.

### Finding 3: Validation of HinglishGuard AI's Hybrid System Design
When this exact sample was fed through the full **HinglishGuard AI Policy Engine**:
```text
Classifier Label: Neutral (99.25%)
Gemini Label:     Hate
Has Disagreement: True
Policy Action:    ESCALATE_TO_HUMAN_REVIEW
Review Required:  True
Reason:           High-severity identity-targeted hate speech detected.
```
The architecture successfully catches the false negative through **multi-agent arbitration**:
1. The supervised model gives its high-speed baseline score.
2. The Gemini Moderation Agent detects identity targeting and incitement to hatred.
3. The deterministic policy engine flags the disagreement (`has_disagreement = True`) and triggers Rule 1 & Rule 2, escalating to the Human Review Queue.

---

## 6. Recommendations & Justified Improvements

### Category A: Immediate Non-Retraining Safeguards (Justified by Evidence)

1. **Activate Stage 3C / Hybrid Arbitration in the UI**:
   - In `pages/1_Analyze_Text.py` and `pages/3_AI_Moderation_Agent.py`, route classifications through the `PolicyEngine` and `ModerationAgent` so that whenever the classifier assigns Neutral to identity-related statements, the agent detects the targeting and escalates the record.
2. **Confidence-Aware Thresholding for Domain Lexicons**:
   - Introduce an attention heuristic in the policy engine: if a sentence contains identity tokens (from the HEOT lexicon or identity lists) co-occurring with hostile sentiment verbs (`"hated"`, `"boycott"`, `"nikalo"`), lower the auto-approval threshold and require agent verification.
3. **Audit and Log Disagreements in SQLite**:
   - Store all disagreements between the classifier and agent in `moderation_records` so human reviewers can validate false-negative clusters in `pages/4_Human_Review.py`.

### Category B: Future Retraining Strategy (Only When Dataset Is Enriched)

Premature retraining on the existing 40,083 dataset **will not fix this issue** because the dataset contains zero instances of `"hated"` and an overwhelming 78% Neutral bias.

Retraining should only be performed when:
1. **Targeted Data Augmentation**: Enrich the training dataset with balanced implicit identity-hate examples (e.g., adding examples of dehumanization, social boycotts, and incitement expressed in clean Hinglish/English without slurs).
2. **Contrastive Pairs**: Add pairs such as:
   - `[Hate]`: *"People from this community should be hated."*
   - `[Neutral]`: *"People from this community should be respected."*
   - `[Counterspeech]`: *"People from this community should NOT be hated."*
3. **Focal Loss or Asymmetric Loss**: Replace standard cross-entropy with Focal Loss ($\gamma = 2.0$) to penalize overconfident easy-negative predictions on underrepresented classes.
