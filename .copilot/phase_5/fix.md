# TASK INSTRUCTION: Fix Test Setup for BAI Artifacts Validation

## Issue
The test suite skipped all 9 tests (`SKIPPED`) because `models/vocab.json` was missing or `bai_core` was not properly imported/initialized.

## Action Required
1. Verify the location of `vocab.json` (or extract/copy it from `config/` or `data/` to `models/vocab.json`).
2. Ensure `bai_core` python library is built and available in the current environment (`venv`).
3. Update `tests/test_dialects.py` if necessary to explicitly log why a skip occurs, or correctly point to `models/v1.bai` and `models/vocab.json`.
4. Run `pytest tests/ -v -s > .copilot/phase_5/output.txt 2>&1`.
5. Ensure at least 1 test passes (`PASSED`) with actual inference calls to validate `v1.bai`.