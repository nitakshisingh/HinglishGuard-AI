"""
Reproducible Dataset Preparation Pipeline for HinglishGuard AI.
Processes the primary dataset (Indo-HateSpeech) without external Excel libraries,
cleans and normalizes text, deduplicates, maps labels, resolves conflicts,
and generates stratified 70/15/15 train/val/test splits without leakage.
"""

import os
import re
import zipfile
import xml.etree.ElementTree as ET
from collections import Counter
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from src.preprocessing import clean_text

RAW_EXCEL_PATH = "data/raw/Indo_HateSpeech/Indo-HateSpeech_Dataset.xlsx"
PROCESSED_DIR = "data/processed"

LABEL_MAPPING = {
    "HS0": 0,
    "HS1": 1,
    "HSN": 2
}

LABEL_NAMES = {
    0: "Neutral",
    1: "Offensive",
    2: "Hate"
}


def load_indo_hatespeech_from_xlsx(filepath: str) -> pd.DataFrame:
    """Read Indo-HateSpeech Excel file directly from the zip container.

    This avoids requiring openpyxl or other third-party Excel dependencies.
    """
    print(f"Reading {filepath} via zipfile / XML streaming...")
    with zipfile.ZipFile(filepath, "r") as z:
        shared_strings = []
        if "xl/sharedStrings.xml" in z.namelist():
            tree = ET.fromstring(z.read("xl/sharedStrings.xml"))
            for si in tree.findall("{http://schemas.openxmlformats.org/spreadsheetml/2006/main}si"):
                t_elems = si.findall(".//{http://schemas.openxmlformats.org/spreadsheetml/2006/main}t")
                shared_strings.append("".join([t.text or "" for t in t_elems]))

        rows = []
        context = ET.iterparse(z.open("xl/worksheets/sheet1.xml"), events=("end",))
        for event, elem in context:
            if elem.tag.endswith("row"):
                row_idx = elem.get("r")
                if row_idx == "1":  # Skip header row
                    elem.clear()
                    continue

                cell_dict = {}
                for c in elem.findall("{http://schemas.openxmlformats.org/spreadsheetml/2006/main}c"):
                    ref = c.get("r")
                    col_letter = re.sub(r"[0-9]", "", ref)
                    v = c.find("{http://schemas.openxmlformats.org/spreadsheetml/2006/main}v")
                    t = c.get("t")
                    val = v.text if v is not None else None
                    if t == "s" and val is not None:
                        val = shared_strings[int(val)]
                    cell_dict[col_letter] = val

                # F is Comment, I is Label
                comment = cell_dict.get("F")
                label = cell_dict.get("I")
                rows.append({"Comment": comment, "Label": label})
                elem.clear()

    df = pd.DataFrame(rows)
    print(f"Loaded {len(df)} raw data rows from XLSX.")
    return df


def run_pipeline():
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    # Track statistics
    stats = {}

    # 1. Load data
    df_raw = load_indo_hatespeech_from_xlsx(RAW_EXCEL_PATH)
    total_raw = len(df_raw)
    stats["total_raw_rows"] = total_raw

    # Validate columns
    assert "Comment" in df_raw.columns and "Label" in df_raw.columns, "Columns Comment and Label must be present!"

    # Raw class distribution
    raw_label_counts = df_raw["Label"].astype(str).str.strip().str.strip("'").str.strip('"').value_counts().to_dict()
    stats["raw_label_counts"] = raw_label_counts

    # 2. Clean Label column
    df_raw["original_label"] = df_raw["Label"].astype(str).str.strip().str.strip("'").str.strip('"')

    # Check for invalid labels
    valid_labels = set(LABEL_MAPPING.keys())
    invalid_label_mask = ~df_raw["original_label"].isin(valid_labels)
    invalid_label_count = int(invalid_label_mask.sum())
    stats["invalid_label_count"] = invalid_label_count
    df_step1 = df_raw[~invalid_label_mask].copy()

    # 3. Check and remove null / blank / invalid comments
    null_comment_mask = df_step1["Comment"].isnull() | (df_step1["Comment"].astype(str).str.strip() == "")
    null_comment_count = int(null_comment_mask.sum())
    stats["null_comment_count"] = null_comment_count
    df_step2 = df_step1[~null_comment_mask].copy()

    # 4. Text Normalization using clean_text() from src/preprocessing.py
    print("Normalizing comments using src.preprocessing.clean_text()...")
    df_step2["cleaned_text"] = df_step2["Comment"].apply(clean_text)

    # Check for comments that became empty after cleaning (e.g. only URLs, emojis that were stripped, or whitespace)
    empty_after_clean_mask = df_step2["cleaned_text"].str.strip() == ""
    empty_after_clean_count = int(empty_after_clean_mask.sum())
    stats["empty_after_clean_count"] = empty_after_clean_count
    df_step3 = df_step2[~empty_after_clean_mask].copy()

    # 5. Check conflicting labels on identical normalized text
    print("Checking for conflicting labels on identical normalized text...")
    label_nunique = df_step3.groupby("cleaned_text")["original_label"].nunique()
    conflicting_texts = label_nunique[label_nunique > 1].index.tolist()
    conflicting_rows_count = int(df_step3["cleaned_text"].isin(conflicting_texts).sum())
    stats["conflicting_unique_texts"] = len(conflicting_texts)
    stats["conflicting_rows_count"] = conflicting_rows_count

    # Drop conflicting rows to maintain unambiguous ground truth
    df_step4 = df_step3[~df_step3["cleaned_text"].isin(conflicting_texts)].copy()

    # 6. Deduplicate exact normalized text
    # Keep the first occurrence of each unique normalized text
    total_before_dedup = len(df_step4)
    df_step5 = df_step4.drop_duplicates(subset=["cleaned_text"], keep="first").copy()
    duplicates_removed = total_before_dedup - len(df_step5)
    stats["exact_duplicates_removed"] = duplicates_removed
    stats["final_cleaned_count"] = len(df_step5)

    # 7. Apply documented provisional mapping
    df_step5["label"] = df_step5["original_label"].map(LABEL_MAPPING)
    df_step5["label_name"] = df_step5["label"].map(LABEL_NAMES)

    # Prepare clean dataset dataframe
    clean_df = df_step5[["cleaned_text", "label", "label_name", "original_label"]].rename(
        columns={"cleaned_text": "text"}
    ).reset_index(drop=True)

    cleaned_class_counts = clean_df["label_name"].value_counts().to_dict()
    stats["cleaned_class_counts"] = cleaned_class_counts

    # Save master cleaned file
    cleaned_file_path = os.path.join(PROCESSED_DIR, "hinglishguard_cleaned.csv")
    clean_df.to_csv(cleaned_file_path, index=False, encoding="utf-8")
    print(f"Saved master cleaned dataset to {cleaned_file_path} ({len(clean_df)} rows).")

    # 8. Train / Validation / Test Stratified Split (70% / 15% / 15%)
    # To guarantee 0 text leakage across splits, since clean_df already has 100% unique texts,
    # standard stratified train_test_split directly guarantees disjoint text sets.
    print("Splitting dataset into 70% Train, 15% Validation, 15% Test...")
    train_df, temp_df = train_test_split(
        clean_df,
        test_size=0.30,
        random_state=42,
        stratify=clean_df["label"]
    )

    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        random_state=42,
        stratify=temp_df["label"]
    )

    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    # Verification of disjoint sets (Zero data leakage)
    train_texts = set(train_df["text"])
    val_texts = set(val_df["text"])
    test_texts = set(test_df["text"])

    overlap_train_val = len(train_texts.intersection(val_texts))
    overlap_train_test = len(train_texts.intersection(test_texts))
    overlap_val_test = len(val_texts.intersection(test_texts))

    assert overlap_train_val == 0, "Data leakage detected between Train and Validation!"
    assert overlap_train_test == 0, "Data leakage detected between Train and Test!"
    assert overlap_val_test == 0, "Data leakage detected between Validation and Test!"
    print("Leakage verification passed: Disjoint text overlap = 0 across all splits.")

    stats["train_count"] = len(train_df)
    stats["val_count"] = len(val_df)
    stats["test_count"] = len(test_df)

    stats["train_class_counts"] = train_df["label_name"].value_counts().to_dict()
    stats["val_class_counts"] = val_df["label_name"].value_counts().to_dict()
    stats["test_class_counts"] = test_df["label_name"].value_counts().to_dict()

    # Save split files
    train_path = os.path.join(PROCESSED_DIR, "train.csv")
    val_path = os.path.join(PROCESSED_DIR, "validation.csv")
    test_path = os.path.join(PROCESSED_DIR, "test.csv")

    train_df.to_csv(train_path, index=False, encoding="utf-8")
    val_df.to_csv(val_path, index=False, encoding="utf-8")
    test_df.to_csv(test_path, index=False, encoding="utf-8")

    print(f"Saved {train_path} ({len(train_df)} rows).")
    print(f"Saved {val_path} ({len(val_df)} rows).")
    print(f"Saved {test_path} ({len(test_df)} rows).")

    # 9. Generate Detailed Markdown Report
    generate_markdown_report(stats)


def generate_markdown_report(stats: dict):
    report_path = os.path.join(PROCESSED_DIR, "preparation_report.md")

    raw_total = stats["total_raw_rows"]
    final_total = stats["final_cleaned_count"]

    report_content = f"""# HinglishGuard AI — Dataset Preparation & Splitting Report

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
> - `HS0` $\\rightarrow$ **0 (`Neutral`)**
> - `HS1` $\\rightarrow$ **1 (`Offensive`)**
> - `HSN` $\\rightarrow$ **2 (`Hate`)**
>
> **Documented Limitation**: In the original dataset, `HS1` contains both colloquial vulgarity and moderate communal insults, while `HSN` contains extreme vitriol and violent threats. Our downstream hybrid architecture (LLM Moderation Agent + Policy Engine) handles disambiguation for borderline cases.

---

## 3. Cleaning & Attrition Pipeline Stages

| Stage | Operation / Condition | Rows Affected / Removed | Remaining Rows |
| :--- | :--- | :---: | :---: |
| **0. Raw Ingestion** | Extracted from `Indo-HateSpeech_Dataset.xlsx` (excluding header) | 0 | **{raw_total:,}** |
| **1. Label Validation** | Verified valid labels (`HS0`, `HS1`, `HSN`) | {stats['invalid_label_count']:,} | **{raw_total - stats['invalid_label_count']:,}** |
| **2. Null / Blank Removal** | Removed `None`, empty strings, or whitespace-only comments | {stats['null_comment_count']:,} | **{raw_total - stats['invalid_label_count'] - stats['null_comment_count']:,}** |
| **3. Text Normalization** | Applied `src.preprocessing.clean_text()` (NFKC, URLs, excessive spaces, character repeats) | {stats['empty_after_clean_count']:,} (emptied) | **{raw_total - stats['invalid_label_count'] - stats['null_comment_count'] - stats['empty_after_clean_count']:,}** |
| **4. Label Conflict Resolution** | Removed identical normalized comments mapped to conflicting labels | {stats['conflicting_rows_count']:,} ({stats['conflicting_unique_texts']:,} texts) | **{raw_total - stats['invalid_label_count'] - stats['null_comment_count'] - stats['empty_after_clean_count'] - stats['conflicting_rows_count']:,}** |
| **5. Deduplication** | Removed exact duplicate comments (Instagram spam, copypastas) | {stats['exact_duplicates_removed']:,} | **{final_total:,}** |

---

## 4. Class Distribution: Before vs. After Cleaning

### Raw Dataset (77,926 rows)
- `HS0` (No Hateful): **{stats['raw_label_counts'].get('HS0', 0):,}** ({stats['raw_label_counts'].get('HS0', 0) / raw_total * 100:.2f}%)
- `HS1` (Hateful): **{stats['raw_label_counts'].get('HS1', 0):,}** ({stats['raw_label_counts'].get('HS1', 0) / raw_total * 100:.2f}%)
- `HSN` (Extreme Hateful): **{stats['raw_label_counts'].get('HSN', 0):,}** ({stats['raw_label_counts'].get('HSN', 0) / raw_total * 100:.2f}%)

### Cleaned & Deduplicated Dataset ({final_total:,} rows)
- **Neutral (0)**: **{stats['cleaned_class_counts'].get('Neutral', 0):,}** ({stats['cleaned_class_counts'].get('Neutral', 0) / final_total * 100:.2f}%)
- **Offensive (1)**: **{stats['cleaned_class_counts'].get('Offensive', 0):,}** ({stats['cleaned_class_counts'].get('Offensive', 0) / final_total * 100:.2f}%)
- **Hate (2)**: **{stats['cleaned_class_counts'].get('Hate', 0):,}** ({stats['cleaned_class_counts'].get('Hate', 0) / final_total * 100:.2f}%)

---

## 5. Stratified Dataset Splits (70% Train / 15% Val / 15% Test)

- **Random State**: 42
- **Stratification**: Based on target class label (`label`)
- **Data Leakage Guarantee**: 100% disjoint unique texts across splits (0 overlapping texts).

| Split | Total Rows | Neutral (0) | Offensive (1) | Hate (2) | Percentage of Total |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Train (`train.csv`)** | **{stats['train_count']:,}** | {stats['train_class_counts'].get('Neutral', 0):,} ({stats['train_class_counts'].get('Neutral', 0) / stats['train_count'] * 100:.2f}%) | {stats['train_class_counts'].get('Offensive', 0):,} ({stats['train_class_counts'].get('Offensive', 0) / stats['train_count'] * 100:.2f}%) | {stats['train_class_counts'].get('Hate', 0):,} ({stats['train_class_counts'].get('Hate', 0) / stats['train_count'] * 100:.2f}%) | 70.0% |
| **Validation (`validation.csv`)** | **{stats['val_count']:,}** | {stats['val_class_counts'].get('Neutral', 0):,} ({stats['val_class_counts'].get('Neutral', 0) / stats['val_count'] * 100:.2f}%) | {stats['val_class_counts'].get('Offensive', 0):,} ({stats['val_class_counts'].get('Offensive', 0) / stats['val_count'] * 100:.2f}%) | {stats['val_class_counts'].get('Hate', 0):,} ({stats['val_class_counts'].get('Hate', 0) / stats['val_count'] * 100:.2f}%) | 15.0% |
| **Test (`test.csv`)** | **{stats['test_count']:,}** | {stats['test_class_counts'].get('Neutral', 0):,} ({stats['test_class_counts'].get('Neutral', 0) / stats['test_count'] * 100:.2f}%) | {stats['test_class_counts'].get('Offensive', 0):,} ({stats['test_class_counts'].get('Offensive', 0) / stats['test_count'] * 100:.2f}%) | {stats['test_class_counts'].get('Hate', 0):,} ({stats['test_class_counts'].get('Hate', 0) / stats['test_count'] * 100:.2f}%) | 15.0% |
| **Total Cleaned** | **{final_total:,}** | **{stats['cleaned_class_counts'].get('Neutral', 0):,}** | **{stats['cleaned_class_counts'].get('Offensive', 0):,}** | **{stats['cleaned_class_counts'].get('Hate', 0):,}** | 100.0% |

---

## 6. Generated Output Files
All files have been generated in `data/processed/`:
1. `data/processed/hinglishguard_cleaned.csv` — Full cleaned dataset ({final_total:,} rows)
2. `data/processed/train.csv` — Stratified training split ({stats['train_count']:,} rows)
3. `data/processed/validation.csv` — Stratified validation split ({stats['val_count']:,} rows)
4. `data/processed/test.csv` — Stratified test split ({stats['test_count']:,} rows)
5. `data/processed/preparation_report.md` — This audit and verification document
"""
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_content)
    print(f"Preparation report written to {report_path}.")


if __name__ == "__main__":
    run_pipeline()
