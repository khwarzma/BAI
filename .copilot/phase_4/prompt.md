# Phase 4 Part 2: Python Generation & Validation Scripts Specifications

## 1. Overview & Objective
Implement two highly optimized Python scripts in `training/`:
1. `training/generate_data.py`: Multi-threaded/async dataset generation pipeline interacting with local Ollama API, relying on `data/manifest/generation_manifest.json` and `data/manifest/company_patterns.json`.
2. `training/validate_data.py`: Pre-training quality assurance engine to validate JSON schemas, check dataset balance, perform SHA-256 deduplication, normalize text (NFKC), and verify OTP logic before splitting.

---

## 2. Target File 1: `training/generate_data.py`

### Key Requirements:
- **Manifest Integration**: Reads configuration, system prompts, scenarios, dialect rules, and company patterns from `data/manifest/`.
- **Checkpointing & State Tracking**:
  - Maintains `data/checkpoint/progress_state.json` to log generated counts per dialect and category.
  - Automatically resumes from the last state without duplicating work.
- **Anti-Duplication**:
  - Calculates SHA-256 hashes for normalized text content.
  - Maintains/checks `data/checkpoint/hashes_registry.db` (SQLite or set) to reject duplicates instantly.
- **Ollama Client Execution**:
  - Issues REST API calls to local Ollama endpoints (e.g. `http://localhost:11434/api/generate`).
  - Implements exponential backoff and retries for failed requests.
- **Output Storage**: Writes raw generated samples directly into `data/raw/ar/{dialect}.jsonl`, `data/raw/en/{dialect}.jsonl`, and `data/raw/global/{type}.jsonl`.

---

## 3. Target File 2: `training/validate_data.py`

### Key Requirements:
- **Schema & Quality Verification**:
  - Validates required fields (`text`, `category_label`, `otp_label`, `language`, `dialect`).
  - Ensures NFKC Unicode normalization for Arabic and English text.
  - Verifies that all `BAIT` category emails with OTP codes explicitly set `otp_label = 1.0` and `confidence_multiplier = 1.5`.
- **Statistical Report Generation**:
  - Calculates balance metrics across categories and all 29 dialects.
  - Writes comprehensive summary report to `data/processed/dataset_stats.json`.
- **Train/Val/Test Splitting**:
  - Splits validated raw dataset (80% Train, 10% Val, 10% Test).
  - Saves final processed sets into `data/processed/train.jsonl`, `val.jsonl`, and `test.jsonl`.

---

## 4. Execution Directives
- Write clean, robust, fully typed, production-ready Python 3.10+ code.
- Provide complete script implementations in `output.txt` without omissions or `# TODO` placeholders.