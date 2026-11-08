# HinglishGuard AI — Comprehensive Dataset Comparison & Audit Report

**Date of Audit**: 2026-11-07
**Objective**: Evaluate three newly placed candidate datasets in `data/raw/` for the three-class Hinglish moderation objective:
- **Class 0: Neutral**
- **Class 1: Offensive**
- **Class 2: Hate**

> **Integrity Confirmation**: All original dataset files in `data/raw/` remain strictly unmodified, unmoved, and unmerged.

---

## 1. Executive Comparison Table

| Dataset | Total Records | Text Column | Label Column | Original Labels (Counts) | Language & Script | Roman Hinglish Records (Est. %) | Missing Text Values | Duplicate Records | Direct 3-Class Suitability |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **HateSpeechMix** (`data/raw/HateSpeechMix/`) | **187,315** (Train: 166,282; Test: 21,033) | `text` | `label` / `label_text` | `neutral` (114,595), `offensive` (50,873), `hate` (21,847) | Multilingual: English (36.6%), Turkish (32.2%), Chinese (20.0%), Amharic (8.0%), Hindi (3.2%) | **~300 - 450** (< 0.25% overall; ~6.4% of the Hindi subset) | **0** (0.00%) | Low | **Unsuitable for Hinglish** (Only 3.2% Hindi, 93.6% of which is Devanagari script, not Roman Hinglish) |
| **Indo-HateSpeech** (`data/raw/Indo_HateSpeech/`) | **77,926** (1 single sheet) | `Comment` | `Label` | `HS0` (64,194), `HS1` (11,034), `HSN` (2,698) | Indian social media (Instagram): Predominantly Roman Hinglish with Devanagari Hindi & English | **~68,024** (**87.3%**) | **1,360** (1.75%) | 18,418 duplicate comments (due to multi-user reposts/spam) | **High Potential with Caveat** (Matches 3-class schema: No Hate vs Hate vs Extreme Hate; requires filtering spam/nulls and aligning `HS1` vs `HSN` definition) |
| **THAR** (`data/raw/Thar/`) | **11,549** | `Comment` | `SubTask1` (Binary) & `SubTask2` (Target) | SubTask1: `Non-AntiReligion` (6,095), `AntiReligion` (5,454); SubTask2: `Islam` (3,714), `Hinduism` (1,349), `Christianity` (391) | Hinglish & Hindi (YouTube comments on religion) | **8,880** (**76.9%**) | **0** (0.00%) | 0 duplicate comments | **Incompatible for 3-class** (Strictly binary religious hostility vs non-hostility; cannot safely split into Neutral vs Offensive vs Hate) |

---

## 2. In-Depth Dataset Audit

### Dataset 1: HateSpeechMix (`data/raw/HateSpeechMix/`)

#### 1. Files & Formats
- `train-00000-of-00001.parquet` (22,749,897 bytes)
- `test-00000-of-00001.parquet` (3,010,553 bytes)
- Format: Apache Parquet (loaded using `pyarrow` via `pandas.read_parquet`).

#### 2. Shape & Features
- **Total Rows**: 187,315 (Train: 166,282; Test: 21,033)
- **Total Columns**: 12
- Columns: `['text', 'label', 'label_text', 'fine_label', 'language', 'source', 'source_id', 'source_label', 'original_split', 'topic', 'target', 'text_available']`

#### 3. Label Distribution
Three global classes:
- **0 (`neutral`)**: 114,595 total (Train: 101,460 | Test: 13,135)
- **1 (`offensive`)**: 50,873 total (Train: 45,312 | Test: 5,561)
- **2 (`hate`)**: 21,847 total (Train: 19,510 | Test: 2,337)

#### 4. Language Breakdown & Hinglish Presence
- English (`en`): 68,597 (36.6%)
- Turkish (`tr`): 60,310 (32.2%)
- Chinese (`zh`): 37,426 (20.0%)
- Amharic (`am`): 15,000 (8.0%)
- Hindi (`hi`): 5,982 (3.2%) [From source: `hasoc2019_hi`]
- **Script Analysis on Hindi Subset**:
  - Devanagari script: 93.57% (4,364 / 4,664 in train)
  - Pure Roman / Latin script: 6.43% (300 in train, ~80 in test $\rightarrow$ **~380 total samples**).

#### 5. First 5 Masked Records (Hindi Subset)
1. `[neutral]`: *"बांग्लादेश की शानदार वापसी, भारत को 314 रन पर रोका #INDvBAN #CWC19"*
2. `[offensive]`: *"सब [अश्लील शब्द] नाच देखने मे व्यस्त जैसे ही कोई #शांतीदूत के साथ कुछ होगा..."*
3. `[offensive]`: *"तुम जैसे हरामियों के लिए बस जूतों की कमी है शुक्र कर अभी तुम्हारी लिंचिंग हुई नहीं है..."*
4. `[neutral]`: *"बीजेपी MLA [नाम हटाया गया] जेल से रिहा, जमानत मिलने की खुशी में एक समर्थक ने इंदौर में हर्ष फायरिंग..."*
5. `[neutral]`: *"चमकी बुखार: विधानसभा परिसर में आरजेडी का प्रदर्शन, तेजस्वी यादव नदारद #biharencephalitisdeaths"*

#### 6. Original Label Meaning (Documentation)
Derived from unified multi-dataset compilation (Toraman et al. / HASOC 2019 / COLD / RANLP 2023):
- `neutral`: Benign, factual, or non-toxic communication.
- `offensive`: Vulgarity, insults, or profanity without targeting protected groups.
- `hate`: Hostile attacks targeting protected identities (race, ethnicity, nationality, religion).

#### 7. Suitability Assessment
- **Verdict: UNSUITABLE** for training a Hinglish moderation model.
- **Reason**: Although the 3-class schema (`neutral`, `offensive`, `hate`) is architecturally identical to our target classes, >99.7% of the dataset is non-Hinglish (English, Turkish, Chinese, Amharic, or Devanagari Hindi). The code-mixed Roman Hinglish representation is negligible (<400 records).

---

### Dataset 2: Indo-HateSpeech (`data/raw/Indo_HateSpeech/`)

#### 1. Files & Formats
- `Indo-HateSpeech_Dataset.xlsx` (5,086,704 bytes, OOXML Spreadsheet)
- `Readme.pdf` (58,965 bytes, Documentation)
- *Note on parsing*: The active environment did not have `openpyxl` installed. As per instructions (*"If a library needed to read a file is missing, report it instead of modifying the environment"*), the XLSX internal package (`xl/worksheets/sheet1.xml` and `xl/sharedStrings.xml`) was inspected directly via Python's native standard library `zipfile` and `xml.etree.ElementTree` without modifying the environment or installing packages.

#### 2. Shape & Features
- **Total Rows**: 77,926 data rows (excluding 1 header row)
- **Total Columns**: 9
- Columns: `['Source.Name', 'Column1', 'Column2', 'Date', 'Likes', 'Comment', '(view source)', 'Post ID', 'Label']`
- **Text Column**: `Comment` (Column F)
- **Label Column**: `Label` (Column I)

#### 3. Label Distribution
Three original categorical classes:
- **`HS0`**: 64,194 records (82.38%)
- **`HS1`**: 11,034 records (14.16%)
- **`HSN`**: 2,698 records (3.46%)

#### 4. Language & Script Breakdown
- Sourced from Indian social media (Instagram comments on contentious public figures, political discussions, and viral posts).
- **Roman Hinglish (Latin Script)**: **68,024 records (87.29%)**
- **Devanagari Script (Hindi)**: 8,542 records (10.96%)
- Missing / Blank Text: 1,360 records (1.75%)

#### 5. First 5 Masked Records
1. `[HS0]` (Row 2): *"Nice buro yes I am [name] is me happy owesi brothers and cold and my friend is all is well with your video"*
2. `[HS1]` (Row 3): *"@[user] teri maa [गाली हटाई गई]"*
3. `[HS1]` (Row 4): *"@[user] [धार्मिक अपशब्द हटाए गए]"*
4. `[HSN]` (Row 5): *"@[user] Teri maka [अत्यधिक अश्लील व हिंसक गाली]"*
5. `[HS0]` (Row 6): *"Send me"*

#### 6. Original Label Meaning (Documentation: Mendeley Data / Kaware, 2024)
As published on Mendeley Data (DOI: 10.17632/snc7mxpj6t.1) and documented in `Readme.pdf`:
- **`HS0` (Hate Speech 0 / No Hate)**: Neutral or positive comments free of abusive or hateful content.
- **`HS1` (Hate Speech 1 / Moderate Hate & Abuse)**: Posts displaying hostility, personal insults, vulgarity, or moderate harassment.
- **`HSN` (Hate Speech Normal / Extreme Hate & Communal Vilification)**: Severe hate speech displaying extreme profanity, identity degradation, casteist/communal slurs, or violent intent.

#### 7. Missing, Duplicates & Data Quality
- **Missing Text Entries**: 1,360 rows have empty/null comments.
- **Duplicates**: 18,418 duplicate comments (recurrent Instagram spam, copypastas, or repeated emoji strings like "🙌", "Send me", and repetitive trolling).
- **Unique Comments**: ~58,148 distinct comments.

#### 8. Suitability Assessment
- **Verdict: HIGH POTENTIAL (Strongest candidate for HinglishGuard AI)**.
- **Why**: Sourced specifically from Indian social media with **over 68,000 genuine Roman Hinglish examples**.
- **Three-Class Alignment**:
  - `HS0` $\rightarrow$ **0 (Neutral)**
  - `HS1` $\rightarrow$ **1 (Offensive)**
  - `HSN` $\rightarrow$ **2 (Hate)**
- **Required Data Hygiene**: Before training, the 1,360 nulls and bot/spam duplicates must be deduplicated, and sample validation must confirm that the boundary between `HS1` (abusive/vulgar) and `HSN` (extreme/hate) strictly reflects identity-targeted hate speech rather than subjective extremity.

---

### Dataset 3: THAR (`data/raw/Thar/`)

#### 1. Files & Formats
- `THAR-Dataset.csv` (2,409,393 bytes)
- Format: Standard CSV (loaded using `pandas.read_csv`).

#### 2. Shape & Features
- **Total Rows**: 11,549
- **Total Columns**: 4
- Columns: `['Identifier', 'Comment', 'SubTask1', 'SubTask2']`
- **Text Column**: `Comment`
- **Primary Label Column**: `SubTask1` (Binary)
- **Secondary Target Column**: `SubTask2` (Target religion name or NaN)

#### 3. Label Distribution
- **SubTask1** (Targeted Religious Hostility):
  - `Non-AntiReligion`: 6,095 (52.78%)
  - `AntiReligion`: 5,454 (47.22%)
- **SubTask2** (Targeted Religion):
  - `NaN` (Non-targeted / non-religious): 6,095 (52.78%)
  - `Islam`: 3,714 (32.16%)
  - `Hinduism`: 1,349 (11.68%)
  - `Christianity`: 391 (3.39%)

#### 4. Language & Script Breakdown
- Sourced from YouTube comments on religious debates and communal controversies in India.
- **Roman Hinglish (Latin Script)**: **8,880 records (76.89%)**
- **Devanagari Script (Hindi)**: 2,669 records (23.11%)
- Missing / Null Comments: 0 (0.00%)
- Duplicate Comments: 0 (0.00%)

#### 5. First 5 Masked Records
1. `[AntiReligion | Islam]` (ID: 3345): *"चमचों! बोल दो वहां पाकिस्तान में भी ये सब "भाजपा" करवा रही है। [धर्म] का यही असली चेहरा है..."*
2. `[Non-AntiReligion | NaN]` (ID: 9317): *"अब हम तुम जैसी औरतों पर भरोसा नहीं कर सकते हैं।और [नाम हटाया गया] तेरे धर्म की लड़की या तुम्हें..."*
3. `[Non-AntiReligion | NaN]` (ID: 8713): *"Ea log sach bolna, nhe jante, zhooth- per zhooth, ea ban gai eshwar kedlal jaiesa en ke kaan m n bol"*
4. `[Non-AntiReligion | NaN]` (ID: 3072): *"Insaniyat naam ki chez nhi inme"*
5. `[AntiReligion | Hinduism]` (ID: 5983): *"Bandar bajrang khan chale bare se muh ke sath Khan chale"*

#### 6. Original Label Meaning (Sharma et al., 2024 / ACL Anthology)
- `AntiReligion`: Hostile attacks, derogatory slurs, or vilification specifically directed at religious faiths, religious figures, rituals, or communities.
- `Non-AntiReligion`: Neutral discourse, respectful debate, or secular/political criticism not targeting a religion.

#### 7. Suitability Assessment
- **Verdict: INCOMPATIBLE for 3-Class Goal**.
- **Reason**: The dataset is strictly **binary** regarding religious animosity. A comment marked `Non-AntiReligion` frequently contains vulgar, toxic, or abusive swearing (`Offensive`) that happens not to be anti-religious. Converting `AntiReligion` into Hate and `Non-AntiReligion` into Neutral would conflate general profanity and abuse with neutral speech, which is scientifically invalid.
- **Secondary Utility**: Excellent domain benchmark for evaluating communal / religious hate speech identification in Hinglish.

---

## 3. Final Strategic Recommendation for HinglishGuard AI

1. **Top Candidate for Three-Class Moderation**:
   - **Indo-HateSpeech (`data/raw/Indo_HateSpeech/`)** is the only dataset among the three with a massive quantity of genuine Roman Hinglish (**>68,000 records**) and a tripartite distribution (`HS0`, `HS1`, `HSN`).
   - By mapping `HS0` $\rightarrow$ `Neutral (0)`, `HS1` $\rightarrow$ `Offensive (1)`, and `HSN` $\rightarrow$ `Hate (2)`, we establish a grounded, academic-scale 3-class Hinglish dataset.
2. **Preprocessing Requirements for Indo-HateSpeech**:
   - Filter out the 1,360 blank text rows.
   - Deduplicate repetitive social media spam/copypastas.
   - Run our `clean_text()` preprocessing pipeline from `src/preprocessing.py`.
3. **Environment Requirement Note**:
   - To convert `Indo-HateSpeech_Dataset.xlsx` to CSV or parquet in subsequent phases, `openpyxl` should be formally added to `requirements.txt` or converted via our standard library extraction script.
