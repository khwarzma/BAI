# TASK INSTRUCTION: Phase 5 - End-to-End Strict Testing (Dialects & Performance)

## Objective
Implement comprehensive, rigorous unit/integration test suites in `tests/test_dialects.py` and `tests/test_performance.py` to validate the newly trained model assets (`models/v1.bai` or `models/v1.onnx` + `vocab.json`) and the C++/Python inference binding (`bai_core`).

Save the complete test execution output into `.copilot/phase_5/output.txt`.

---

## Requirements

### 1. `tests/test_dialects.py`
- Load `bai_core` and the model pipeline using `models/v1.bai` (or `v1.onnx`).
- Test dialect classification and OTP/Category outputs against sample texts from various Arabic and English dialects (e.g., Egyptian, Saudi, Levantine, Moroccan, US English, UK English).
- Verify that `confidence` outputs are floating point values between 0.0 and 1.0.
- Verify JSON schema response validity from `predict_json()`.

### 2. `tests/test_performance.py`
- Run 1,000 inference requests in a benchmark loop.
- Measure and assert Latency: Average latency must be `<= 15ms` per request.
- Measure and assert RAM usage: Peak RSS must remain `<= 150MB`.
- Check for memory leaks across repeated inference calls.

---

## Output Protocol
1. Write and save the python test codes in `tests/test_dialects.py` and `tests/test_performance.py`.
2. Execute both test suites via pytest: `pytest tests/ -v > .copilot/phase_5/output.txt 2>&1`.
3. Ensure the test output log is completely written to `.copilot/phase_5/output.txt`.