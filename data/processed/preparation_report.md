# HinglishGuard AI — Dataset Preparation & Splitting Report

**Generated Date**: 2026-11-07
**Primary Dataset**: Indo-HateSpeech Dataset (`Indo-HateSpeech_Dataset.xlsx`)
**Pipeline**: Reproducible Text Normalization, Deduplication, Conflict Resolution & Stratified Splitting

---

## 1. Dataset Attribution & Provenance
- **Dataset Title**: Indo-HateSpeech Dataset
- **Authors**: Pravin D. Kaware (Sant Gadge Baba Amravati University / HVPM's College of Engineering & Technology, Amravati, Maharashtra, India)
- **Repository**: Mendeley Data, V1 (Published Dec 2, 2024)
- **DOI**: [10.17632/snc7mxpj6t.1](https://doi.org/10.17632/snc7mxpj6t.1)
- **License**: Creative Commons Attribution 4.0 International (**CC BY 4.0**)
- **Data Source**: Authentic public comments scraped from Instagram discussion threads on Indian social, political, and communal topics.

---

## 2. Label Mapping & Academic Limitations Notice
> **Important Academic Disclosure**:
> The original Indo-HateSpeech authors categorized text on a hate speech severity scale:
> - `HS0` ("No Hateful")
> - `HS1` ("Hateful" / Moderate Hate)
> - `HSN` ("Extreme Hateful")
>
> In HinglishGuard AI, these labels are mapped to a 3-class content moderation taxonomy:
> - `HS0` $\rightarrow$ **0 (`Neutral`)**
> - `HS1` $\rightarrow$ **1 (`Offensive`)**
> - `HSN` $\rightarrow$ **2 (`Hate`)**
>
> **Documented Limitation**: In the original dataset, `HS1` contains both colloquial vulgarity and moderate communal insults, while `HSN` contains extreme vitriol and violent threats. Our downstream hybrid architecture (LLM Moderation Agent + Policy Engine) handles disambiguation for borderline cases.

---

## 3. Cleaning & Attrition Pipeline Stages

| Stage | Operation / Condition | Rows Affected / Removed | Remaining Rows |
| :--- | :--- | :---: | :---: |
| **0. Raw Ingestion** | Extracted from `Indo-HateSpeech_Dataset.xlsx` (excluding header) | 0 | **77,926** |
| **1. Label Validation** | Verified valid labels (`HS0`, `HS1`, `HSN`) | 0 | **77,926** |
| **2. Null / Blank Removal** | Removed `None`, empty strings, or whitespace-only comments | 1,360 | **76,566** |
| **3. Text Normalization** | Applied `src.preprocessing.clean_text()` (NFKC, URLs, excessive spaces, character repeats) | 24 (emptied) | **76,542** |
| **4. Label Conflict Resolution** | Removed identical normalized comments mapped to conflicting labels | 42 (8 texts) | **76,500** |
| **5. Deduplication** | Removed exact duplicate comments (Instagram spam, copypastas) | 19,238 | **57,262** |

---

## 4. Class Distribution: Before vs. After Cleaning

### Raw Dataset (77,926 rows)
- `HS0` (No Hateful): **64,194** (82.38%)
- `HS1` (Hateful): **11,034** (14.16%)
- `HSN` (Extreme Hateful): **2,698** (3.46%)

### Cleaned & Deduplicated Dataset (57,262 rows)
- **Neutral (0)**: **44,667** (78.00%)
- **Offensive (1)**: **10,022** (17.50%)
- **Hate (2)**: **2,573** (4.49%)

---

## 5. Stratified Dataset Splits (70% Train / 15% Val / 15% Test)

- **Random State**: 42
- **Stratification**: Based on target class label (`label`)
- **Data Leakage Guarantee**: 100% disjoint unique texts across splits (0 overlapping texts).

| Split | Total Rows | Neutral (0) | Offensive (1) | Hate (2) | Percentage of Total |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Train (`train.csv`)** | **40,083** | 31,267 (78.01%) | 7,015 (17.50%) | 1,801 (4.49%) | 70.0% |
| **Validation (`validation.csv`)** | **8,589** | 6,700 (78.01%) | 1,503 (17.50%) | 386 (4.49%) | 15.0% |
| **Test (`test.csv`)** | **8,590** | 6,700 (78.00%) | 1,504 (17.51%) | 386 (4.49%) | 15.0% |
| **Total Cleaned** | **57,262** | **44,667** | **10,022** | **2,573** | 100.0% |

---

## 6. Generated Output Files
All files have been generated in `data/processed/`:
1. `data/processed/hinglishguard_cleaned.csv` — Full cleaned dataset (57,262 rows)
2. `data/processed/train.csv` — Stratified training split (40,083 rows)
3. `data/processed/validation.csv` — Stratified validation split (8,589 rows)
4. `data/processed/test.csv` — Stratified test split (8,590 rows)
5. `data/processed/preparation_report.md` — This audit and verification document
