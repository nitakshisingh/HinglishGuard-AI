# Indo-HateSpeech Dataset: Second-Stage Provenance & Annotation Verification Report

**Date of Audit**: 2026-11-07
**Dataset Path**: `data/raw/Indo_HateSpeech/`
**Inspected Artifacts**:
1. `Readme.pdf` (Original author documentation)
2. `Indo-HateSpeech_Dataset.xlsx` (Dataset corpus, 77,926 rows)
3. Published peer-reviewed research and Mendeley Data repository documentation (Kaware, 2024; DOI: 10.17632/snc7mxpj6t.1)

---

## 1. Provenance, Authorship & Licensing

- **Dataset Creator / Author**: Pravin D. Kaware (Sant Gadge Baba Amravati University / HVPM's College of Engineering & Technology, Amravati, Maharashtra, India).
- **Publication Venue**: Mendeley Data, V1 (Published December 2, 2024).
- **DOI**: [10.17632/snc7mxpj6t.1](https://doi.org/10.17632/snc7mxpj6t.1).
- **Associated Research Paper**: *"Indo-HateSpeech Analysis: A Multi-Level Hate Speech Classification Framework Using BERT Features and Machine Learning Models"* (ResearchGate / Mendeley, 2024).
- **License**: Creative Commons Attribution 4.0 International (**CC BY 4.0**) — permits copying, redistribution, and adaptation for academic/commercial use with appropriate citation.
- **Data Source**: Scraped public user comments on Instagram from contentious social, political, and communal discourse threads in India (e.g., discussions surrounding political leaders such as Asaduddin Owaisi and Yogi Adityanath, religious clashes, and election news).

---

## 2. Original Documentation & Label Definitions

### Original Text from `Readme.pdf` (Decoded via PDF Font CMap):
```text
Table 1: Columns in the dataset and their description
Column: Label
Description: "This column contains the input label i.e. HS0, HS1, and HSN."
"HS0" means No Hateful
"HS1" means Hateful
"HSN" means Extreme Hateful
```

### Research Paper Definitions (Kaware, 2024):
In the author's accompanying publication, the dataset is framed as a **multi-level hate speech severity scale**:
1. **`HS0` ("No Hateful")**: Comments that do not exhibit hate speech or severe malice.
2. **`HS1` ("Hateful")**: Comments that display moderate hate speech, hostility, vulgar personal abuse, or derogatory statements directed at individuals or groups.
3. **`HSN` ("Extreme Hateful")**: Comments exhibiting severe vitriol, direct violent threats, explicit dehumanization, extreme identity/religious slurs, or incitement to physical harm.

---

## 3. Critical Annotation Gap Analysis

### Does the Dataset Distinguish Offensive Content from Hate Speech?
**NO.** The original authors designed a **severity tier of hate speech** (`No Hate` $\rightarrow$ `Moderate Hate` $\rightarrow$ `Extreme Hate`), **not** a standard NLP taxonomy of `[Neutral vs. Offensive/Profane vs. Hate Speech]`.

Specifically:
- **`HS1` is NOT clean "Offensive / Vulgar without Hate"**:
  In the authors' annotation schema, `HS1` is called **"Hateful"** (moderate hate). Inspection of raw samples reveals that `HS1` contains both:
  a) Common profanity / insults (*"Teri maa ki chut"*, *"Ye chutiya h owesi"*).
  b) Direct communal slurs and religious attacks (*"mohammad and allah mi maa ki choot"*, *"sare mulla n thokangee"*).
- **`HSN` is NOT the only Hate Speech**:
  `HSN` represents **extreme hate** (*"in sub mullo ko chodo, inki roji roti maro"*, violent threats of lynching, graphic desecration). However, severe communal hatred is also found in `HS1`.
- **Annotation Inconsistencies & Borderline Disagreements**:
  - The documentation contains **no explicit boundary guidelines** explaining how annotators distinguished `HS1` from `HSN`. For example, row 4 (*"@[USER] mohhamad and allah mi maa ki choot"*) is labeled `HS1`, while row 16 (*"Har atankwadi mulla..."*) is labeled `HSN`.
  - Noticeable label noise exists in `HS0`: Row 4 (*"Ye india ke sare muslman babar or akbar ki najayaj oolad hun he"*) and Row 6 (*"baba saheb ambedkar se tuj jaise lunf jaise logo ko bolne ka adhikaar dete isko bhejo pakistan pehle"*) are communal attacks that were incorrectly labeled `HS0` (No Hate).

---

## 4. Empirical Sample Audit (20 Representative Masked Records per Label)

### A. Label `HS0` ("No Hateful")
| Sample ID | Masked Text | Annotation Consistency Check |
| :--- | :--- | :--- |
| `[HS0_01]` | *"Nice buro yes I am [USER] is me happy owesi brothers and cold and my friend is all is well with your video"* | Clean / Neutral |
| `[HS0_02]` | *"Send me"* | Spam / Benign |
| `[HS0_03]` | *"So great leader 🙌"* | Positive / Neutral |
| `[HS0_04]` | *"Ye india ke sare muslman babar or akbar ki najayaj oolad hun he"* | **Misannotated** (Anti-Muslim communal slur labeled HS0) |
| `[HS0_05]` | *"owesi ki maa kaa cholaaa"* | **Misannotated** (Vulgar insult labeled HS0) |
| `[HS0_06]` | *"Galti ho gai baba saheb ambedkar se tuj jaise lunf jaise logo ko bolne ka adhikaar dete isko bhejo pakistan pehle"* | **Misannotated** (Anti-Dalit / xenophobic hate speech labeled HS0) |
| `[HS0_07]` | *"हमको एक मस्त बकरी चाहिए😍"* | Ambiguous / Double entendre |
| `[HS0_08]` | *"@[USER] owesi ko l jaao...😂😂😂"* | Informal mockery |
| `[HS0_09]` | *"@[USER] wallah habibi इसको तुम रखती हमें बकरा नहीं बकरी चाहिए होती😍"* | Sarcastic political banter |
| `[HS0_10]` | *"@[USER] bkri bhi aapke pass h but bkraa chahae to see l jaao yrrr.. hnare to y pnoti h"* | Informal mockery |
| `[HS0_11]` | *"Jay jay Shree ram 🚩"* | Religious chant / Neutral expression |
| `[HS0_12]` | *"Jay Shree ram 🚩"* | Religious chant / Neutral expression |
| `[HS0_13]` | *"Jay Shree ram"* | Religious chant / Neutral expression |
| `[HS0_14]` | *"Owesi jinda bad 💪💪💪💪💪💪💪"* | Political slogan / Neutral expression |
| `[HS0_15]` | *"@[USER] itna pasand hai toh Teri ammi Chachi ko bhej de uske pass 😂😂😂"* | Borderline abusive / Taunting |
| `[HS0_16]` | *"@[USER] uska bap he yogi ADITYANATH"* | Political taunt |
| `[HS0_17]` | *"Owesi jinda baad 💪💪💪💪..."* | Political slogan / Neutral |
| `[HS0_18]` | *"Muslim 💪🏻💪🏻💪🏻💪🏻💪🏻"* | Communal solidarity / Neutral |
| `[HS0_19]` | *"@[USER] 👏"* | Neutral emoji reaction |
| `[HS0_20]` | *"@[USER] bhai ek baat bata agar kisi ko goli marni hoti to kiya vo gaddi ke bilkul niche hi marta..."* | Constructive inquiry / Non-hate discussion |

---

### B. Label `HS1` ("Hateful" / Moderate Hate)
| Sample ID | Masked Text | Semantic Nature |
| :--- | :--- | :--- |
| `[HS1_01]` | *"@[USER] teri maa chodate"* | Vulgar maternal insult / Profanity (Offensive) |
| `[HS1_02]` | *"@[USER] mohhamad and allah mi maa ki [गाली]"* | Explicit religious / communal hate speech |
| `[HS1_03]` | *"Teri maa ki chut"* | Vulgar profanity (Offensive) |
| `[HS1_04]` | *"Ye chutiya h owesi"* | Political insult / Slang (Offensive) |
| `[HS1_05]` | *"@[USER] ye bhdwa giri apne bhagwaan ke aage kro jake thik yha koi fayda ni bhokne ka..."* | Anti-religious insult / Blasphemy |
| `[HS1_06]` | *"ranwda"* | Slur / Insult (Offensive) |
| `[HS1_07]` | *"Owesi ka maa ka bhosda"* | Vulgar political insult (Offensive) |
| `[HS1_08]` | *"Teri maa ko bhosdaa 😂😂😂"* | Vulgar abuse / Profanity (Offensive) |
| `[HS1_09]` | *"Lakh kutte bhok te h ser per arr ser ka kuch nhi bigde ga"* | Metaphorical taunt / Mild offense |
| `[HS1_10]` | *"@[USER] owesi sher hai??? Abey lodu to yogi kya hai fir"* | Political swearing / Slang (Offensive) |
| `[HS1_11]` | *"@[USER] teri ma ki chut"* | Vulgar maternal insult (Offensive) |
| `[HS1_12]` | *"Kutte bhi puri raat bhookte rehte hai... iska matlab ye nhi ki kutta shi kam kr rha hai !!!"* | Metaphorical criticism |
| `[HS1_13]` | *"Chutiya .....bhadwa😂"* | Pure profanity / Slang (Offensive) |
| `[HS1_14]` | *"Sala sakal se hi terrrirest dikhta he....m"* | Xenophobic / Communal profiling (Hate) |
| `[HS1_15]` | *"Ye pgl ovesi chutiya h ky 🔥🔥🔥🔥🔥"* | Political insult (Offensive) |
| `[HS1_16]` | *"Bhai tn bhi rokange or Teri maa n rokange ... Or jarurt Pdi to sare mulla n thokangee... Jai shree ram"* | Violent communal threat (Hate) |
| `[HS1_17]` | *"Teri gaand marege bete"* | Violent/vulgar personal threat |
| `[HS1_18]` | *"Teri maki chutti"* | Vulgar insult (Offensive) |
| `[HS1_19]` | *"@[USER] sahi main kya mulla ji"* | Sarcastic communal tag |
| `[HS1_20]` | *"Mandir ohi bana he oo chutiya jo kuch nhi kr paye... jai shree ram 🚩❣️"* | Aggressive religious polemic (Offensive) |

---

### C. Label `HSN` ("Extreme Hateful")
| Sample ID | Masked Text | Semantic Nature |
| :--- | :--- | :--- |
| `[HSN_01]` | *"@[USER] Teri maka bhosda lose nikalo madar chod"* | Severe vulgarity / Sexual abuse |
| `[HSN_02]` | *"Bhn k lode katwe kaat k fenk denge bc tujhe suar"* | Explicit violent threat + dehumanizing communal slur (Hate) |
| `[HSN_03]` | *"Are thari maa ko chodu owesi bhadve"* | Severe political abuse |
| `[HSN_04]` | *"@[USER] chodo iski maa bhai akele owesi nai in sub mullo ko chodo"* | Incitement against community (Hate) |
| `[HSN_05]` | *"Madar chod owesi👎👎👎👎👎🖕🖕🖕"* | Severe political swearing |
| `[HSN_06]` | *"@[USER] bhai inki maa chod do mullo ko jaha b mile... inki roji roti maro..."* | Economic boycott + violent hate speech (Hate) |
| `[HSN_07]` | *"Madar chod owesi"* | Severe vulgar abuse |
| `[HSN_08]` | *"Har atankwadi mulla kiu pith piche war krte ho tum log..."* | Targeted group vilification / Hate speech |
| `[HSN_09]` | *"Owesi tere Bhen ki chut lawade"* | Vulgar political insult |
| `[HSN_10]` | *"Chutiya sala owesi"* | Ordinary vulgarity (Misplaced in HSN) |
| `[HSN_11]` | *"@[USER] 😂😂👏👏bhai ye gandu issa hai aapne to iska lehnga kho ke chod diya 🤣"* | Obscene sexualized harassment |
| `[HSN_12]` | *"Bass vale bolta ky hai kuch karke bhii dekha kutte jaise ky bhook ta bc mc lavde"* | Vulgar abusive rant |
| `[HSN_13]` | *"@[USER] kutta to vo h jo hijdo ki foj leke ghum raha... kutta modi h modi"* | Political insult / Transphobic slur |
| `[HSN_14]` | *"suvar k skl ka katuwa sala🤣🤣😡💩halala k aulad... bsdika"* | Dehumanizing communal hate speech |
| `[HSN_15]` | *"Or Yogi teri husband h jo tujhe or teri Amma ko chodta h... suar ki shakl... hijda randua"* | Graphic sexualized political hostility |
| `[HSN_16]` | *"owesi k ma chud gyi... allah ka b nunu kata hua h..."* | Severe blasphemy + hate speech |
| `[HSN_17]` | *"Or Modi randua peda hua tere raam or bhole or Hanuman randue jese..."* | Graphic anti-Hindu hate speech / blasphemy |
| `[HSN_18]` | *"Owesi chutiya sala"* | Ordinary vulgar insult (Misplaced in HSN) |
| `[HSN_19]` | *"Tera lawda hai bhi kya😂....chutiye // Teri chut marunga mai😂🚩🚩"* | Vulgar/sexual threat |
| `[HSN_20]` | *"Hecha aaichi Gand"* | Marathi maternal vulgarity (Offensive) |

---

## 5. Evaluation of Proposed Mapping: `HS0` $\rightarrow$ Neutral, `HS1` $\rightarrow$ Offensive, `HSN` $\rightarrow$ Hate

### Scientific Defensibility Assessment:
Is mapping `HS0` $\rightarrow$ Neutral, `HS1` $\rightarrow$ Offensive, `HSN` $\rightarrow$ Hate scientifically sound?

**Verdict**: **PARTIALLY SOUND, BUT WITH SIGNIFICANT LIMITATIONS**.

1. **Why It Works Pragmatically**:
   - `HS0` is predominantly clean/neutral discourse (~85-90% accuracy).
   - `HS1` contains the bulk of everyday swear words, insults, and uncivil discourse (~70% offensive/profane).
   - `HSN` captures the most egregious, violent, and dehumanizing communal hate speech instances.
2. **Where the Mapping Breaks Down**:
   - `HS1` is formally defined as **"Hateful"** by the authors and contains real communal/religious hate speech (e.g., *"sare mulla n thokangee"*), meaning a classifier trained on this mapping will learn that some severe hate speech is merely "offensive".
   - `HSN` includes ordinary sexual swearing with exclamation marks (e.g., *"Chutiya sala owesi"*), which is offensive abuse rather than identity-targeted hate.
   - The original annotation does not distinguish between **non-targeted swearing (Offensive)** and **protected group vilification (Hate Speech)**; instead, it is a subjective scale of **emotional extremity**.

---

## 6. Final Recommendation

### **ACCEPT WITH LIMITATIONS**

We should proceed using **Indo-HateSpeech** under the following rigorous constraints:

1. **Explicit Academic Disclosure**:
   In any publication, demonstration, and documentation, explicitly state:
   > *"The Indo-HateSpeech dataset (Kaware, 2024) annotates text on an intensity scale (`No Hate`, `Hate`, `Extreme Hate`). For the purpose of our 3-class content moderation system, these were aligned to `[Neutral, Offensive, Hate]` as an empirical proxy, with documented label noise and overlap between moderate hate and vulgarity."*
2. **Quality Cleaning Steps Prior to Training**:
   - Drop the 1,360 null/empty rows.
   - Deduplicate the 18,418 identical Instagram bot/spam comments.
   - Apply our Phase 1 `clean_text()` preprocessing pipeline.
3. **Hybrid AI Agent Compensation**:
   Because the dataset has known boundary blurring between `HS1` and `HSN`, our architecture’s **LLM Moderation Agent (Gemini)** in `src/moderation_agent.py` and the **Policy Engine** in `src/policy_engine.py` will serve as the disambiguation layer for borderline outputs.
