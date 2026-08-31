# TASK SPECIFICATION: Phase 2 — Language-Agnostic Configurations & Dialect Mappings

## Context & Requirements
We are implementing the dynamic configuration and mapping layer for BAI. 
These files must hold all language-specific and dialect-specific definitions, leaving the core C++ and Python engines entirely language-agnostic.

---

## File 1: `config/languages.json`
Define global language settings and high-level routing parameters:
- System default languages (`ar` for Arabic, `en` for English).
- Language detection thresholds and tokenizers metadata references.
- Schema defining categories (`INBOX_PINNED`, `INBOX`, `BAIT`, `BAIS`, `BAIADS`).

---

## File 2: `config/rules.json`
Define decision-making rules and execution thresholds:
- Confidence threshold for autonomous decision execution (e.g., `0.85`).
- Rules for OTP detection (high-priority bypass flag).
- System behavior rules when confidence falls below threshold (e.g., trigger prompt fallback).

---

## File 3: `data/dialects_mapping/ar_dialects.json`
Define canonical structures and region mappings for the 19 Arabic dialects:
- Map ISO region codes to dialect identifiers (e.g., `ar-EG` -> Egyptian, `ar-SA` -> Hijazi/Najdi, `ar-LEV` -> Levantine, etc., covering all 19 target Arabic variations).
- Include subword normalization tokens and common dialectal indicators used by downstream validation scripts.

---

## File 4: `data/dialects_mapping/en_dialects.json`
Define canonical structures for the 10 English dialects:
- Map region codes to dialect identifiers (e.g., `en-US`, `en-GB`, `en-CA`, `en-AU`, `en-IN`, etc., covering 10 variations).
- Include structural features and normalization indicators.

---

## Output Rules:
- DO NOT modify existing python/C++ source code.
- Write ALL generated JSON contents clearly into `.copilot/phase_2/output.txt` with appropriate section markers (`FILE: <path>`).
- Pause execution after outputting.