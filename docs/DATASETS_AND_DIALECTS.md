# BAI (Bareeed Artificial Intelligence) — Datasets & Multilingual Pipeline

> **Specification Standard:** AM Standard (SAM Category)
> **Document Identifier:** AM-DATA-BAI-1.0.0
> **Initiator:** Khwarzma
> **Target System:** Bareeed Language Architecture
> **Status:** Active Dataset Specification

---

## 1. Zero-User-Data & Synthetic Generation Architecture

In strict adherence to the **AM Standard**, BAI v1 prohibits the collection, scraping, or utilization of private user emails or production mail logs for dataset creation or model fine-tuning.

To construct a robust dataset without privacy compromises:

* **Synthetic Pipeline (`training/generate_data.py`):** Open-weights LLMs (such as Llama 3 and Mistral) operating within free T4 GPU environments (e.g., Google Colab) generate synthetic email payloads across diverse transactional, promotional, spam, and OTP categories.
* **Deterministic Verification (`training/validate_data.py`):** Generated items undergo heuristic rule validation, structural JSON syntax checking, and label sanity checks before inclusion in training sets.

```text
+---------------------------------------------------------------------------------+
|                         SYNTHETIC DATA GENERATION PIPELINE                      |
|                                                                                 |
|  +--------------------+      +--------------------+      +-------------------+  |
|  | Open-Weights LLM   | ---> | Heuristic Rule     | ---> | Schema & Label    |  |
|  | (Llama 3 / Mistral)|      | Filtering Pipeline |      | Validation Check  |  |
|  +--------------------+      +--------------------+      +-------------------+  |
|                                                                    |            |
|                                                                    v            |
|                                                          +-------------------+  |
|                                                          | Balanced Dataset  |  |
|                                                          | (train/val.jsonl) |  |
|                                                          +-------------------+  |
+---------------------------------------------------------------------------------+

```

---

## 2. Dialect Matrix Coverage (29 Regional Variants)

BAI v1 incorporates explicit linguistic patterns for 19 Arabic dialects and 10 English variants, enabling accurate categorization of regional code-switching, informal phrasings, and localized slang.

### 2.1 Arabic Dialect Coverage (19 Variants)

```text
Arabic (ar)
├── Gulf & Peninsula: ar-SA (Saudi), ar-AE (Emirati), ar-KW (Kuwaiti), ar-QA (Qatari), ar-OM (Omani), ar-BH (Bahraini), ar-YE (Yemeni)
├── Levant & Egypt: ar-EG (Egyptian), ar-LEV-SY (Syrian), ar-LEV-LB (Lebanese), ar-LEV-JO (Jordanian), ar-LEV-PS (Palestinian)
├── North Africa & Maghrebi: ar-SD (Sudanese), ar-IQ (Iraqi), ar-LY (Libyan), ar-TN (Tunisian), ar-DZ (Algerian), ar-MA (Moroccan)
└── Modern Standard: ar-MSA (Fus'ha)

```

### 2.2 English Dialect Coverage (10 Variants)

```text
English (en)
├── Primary Dialects: en-US (American), en-GB (British), en-CA (Canadian), en-AU (Australian)
├── Regional & International: en-IN (Indian), en-ZA (South African), en-NZ (New Zealand), en-IE (Irish)
└── Global & Mixed: en-SG (Singaporean), en-INT (International Code-Switched)

```

---

## 3. Dynamic Configuration & Tokenizer Schema (`config/languages.json`)

Dialect handling, keyword weights, and pre-processing normalization rules are declared dynamically in `config/languages.json` to allow linguistic updates without modifying C++ core source files.

```json
{
  "supported_languages": {
    "ar": {
      "code": "ar",
      "name": "Arabic",
      "script": "arabic",
      "dialects": [
        {"code": "ar-EG", "name": "Egyptian", "keywords_weight": 1.2},
        {"code": "ar-SA", "name": "Saudi", "keywords_weight": 1.1},
        {"code": "ar-LEV", "name": "Levantine", "keywords_weight": 1.0},
        {"code": "ar-MAG", "name": "Maghrebi", "keywords_weight": 1.0}
      ],
      "tokenization_config": {
        "strip_tatweel": true,
        "normalize_alef": true,
        "normalize_yaa": true,
        "preserve_numbers_for_otp": true,
        "remove_diacritics": true
      }
    },
    "en": {
      "code": "en",
      "name": "English",
      "script": "latin",
      "dialects": [
        {"code": "en-US", "name": "American"},
        {"code": "en-GB", "name": "British"}
      ],
      "tokenization_config": {
        "lowercase": true,
        "collapse_whitespace": true,
        "strip_control_characters": true
      }
    }
  }
}

```

---

## 4. Dataset Directory Topology & JSONL Schemas

Training and evaluation datasets are organized within the development repository under structured paths to maintain clean separation between raw traces and processed outputs:

```text
data/
├── raw/                      <-- Raw synthetic outputs from generation pipelines
│   ├── ar/                   <-- Category JSON files (otp.json, spam.json, ads.json)
│   └── en/                   <-- Category JSON files (otp.json, spam.json, ads.json)
├── processed/                <-- Validated JSONL datasets for PyTorch training
│   ├── train.jsonl           <-- Balanced training dataset
│   └── val.jsonl             <-- Cross-validation test set
└── dialects_mapping/         <-- Keyword maps for all 29 variants
    ├── ar_dialects.json
    └── en_dialects.json

```

### 4.1 Sample Raw Generation Payload (`data/raw/ar/otp.json`)

```json
[
  {
    "raw_subject": "رمز التحقق الخاص بك",
    "raw_body": "يا هلا بك، كود التفعيل لمرة واحدة للدخول هو 839201. صلاحية الكود 5 دقائق فقط.",
    "language": "ar",
    "dialect": "ar-SA",
    "target_category": "BAIT",
    "extracted_features": {
      "has_digits": true,
      "otp_length": 6,
      "expiry_minutes": 5
    }
  }
]

```

### 4.2 Sample Processed JSONL Entry (`data/processed/train.jsonl`)

```json
{
  "text": "كود التفعيل الخاص بك في بريد هو 739201. انتهى وقت الكود خلال 5 دقائق.",
  "language": "ar",
  "dialect": "ar-EG",
  "label": "BAIT",
  "is_otp": true,
  "otp_metadata": {
    "detected": true,
    "validity_window_minutes": 5
  }
}

```

---

## 5. Dialect Feature Mappings & Keyword Weighting Architecture

To support accurate sub-task extraction across diverse regional phrasings, `data/dialects_mapping/ar_dialects.json` isolates localized temporal markers, OTP indicators, and marketing expressions.

```json
{
  "ar-EG": {
    "otp_indicators": ["رمز", "كود", "تأكيد", "رقم افتراضي", "باسورد"],
    "expiry_indicators": ["خلال", "صلاحية", "ينتهي", "بعد"],
    "time_units": {
      "minutes": ["دقايق", "دقيقة", "دقائق"]
    }
  },
  "ar-SA": {
    "otp_indicators": ["رمز التحقق", "كود الدخول", "رمز التفعيل", "كلمة المرور"],
    "expiry_indicators": ["صالح لمدة", "ينتهي خلال", "ينتهي بعد"],
    "time_units": {
      "minutes": ["دقيقة", "دقائق", "دقايك"]
    }
  },
  "ar-LEV": {
    "otp_indicators": ["رمز التأكيد", "كود التفعيل", "رقم التفعيل"],
    "expiry_indicators": ["صالح لـ", "خلال زمن", "ينتهي بـ"],
    "time_units": {
      "minutes": ["دقايق", "دقيقة"]
    }
  }
}

```

---

## 6. Pre-processing & Normalization Execution

The pre-processing pipeline standardizes variations in spelling, character forms, and decorative marks across dialects to prevent vocabulary inflation while ensuring numeric sequences remain intact.

```text
Raw Text Input
  │
  ├──► Character Normalization (Alef/Yaa Unification, Tatweel Removal)
  │
  ├──► Diacritic Removal (Tashkeel Strip)
  │
  ├──► Numeric Sequence Preservation (OTP Boundary Identification)
  │
  └──► Tokenization Vector Sequence (Ready for Transformer Encoder)

```
