# TASK SPECIFICATION: Phase 1 — Core AI Model Architecture & Export Pipeline

## Context & Architecture Requirements
We are building BAI (Bareeed Artificial Intelligence), a lightweight, language-agnostic Micro-Transformer classification engine running on Python 3.12 / C++23.
You are tasked with generating 3 Python scripts for the first phase. The implementation MUST be completely language-agnostic (no hardcoded language or dialect logic inside the model architecture).

---

## File 1: `training/model.py`
Implement a PyTorch Micro-Transformer Encoder class `BaiMicroEncoder` with multi-task classification heads.

### Structural Requirements:
- **Architecture**: 6 Encoder layers, 8 Attention heads, Hidden Dimension ($d_{model}$) = 256, Feed-Forward Dimension ($d_{ff}$) = 1024, Max Context Window = 512 subword tokens.
- **Inputs**: Tokenized sequence tensor `input_ids` and `attention_mask`.
- **Outputs (Multi-Task Heads)**:
  1. `logits_category`: Classification logits for 5 categories (`INBOX_PINNED`, `INBOX`, `BAIT`, `BAIS`, `BAIADS`).
  2. `logits_otp`: Binary classification logits for OTP detection (`is_otp`).
  3. `confidence`: Scalar output representing model output certainty score (sigmoid dynamic scalar).
- Must include helper functions for initializing weights cleanly.

---

## File 2: `training/train.py`
Implement an optimized training script designed for efficiency and execution in environments with runtime limits (e.g., Colab T4 / Local CPU/GPU).

### Structural Requirements:
- Loads train and validation datasets from `data/processed/train.jsonl` and `val.jsonl`.
- Implements `AdamW` optimizer, linear warmup with cosine decay scheduler, and `FP16` Mixed Precision training (via `torch.cuda.amp`).
- **Auto-Checkpointing**: Saves checkpoints every N steps to a specified directory to prevent data loss on unexpected session terminations.
- Computes loss as a weighted sum of Category Cross-Entropy and OTP Binary Cross-Entropy.
- Includes step logging for evaluation loss and metric evaluation.

---

## File 3: `training/export.py`
Implement a model exporter that converts the trained PyTorch model into an ONNX binary and applies dynamic INT8 quantization.

### Structural Requirements:
- Exports the PyTorch model to ONNX format with dynamic axes for batch size and sequence length.
- Applies ONNX Runtime Dynamic INT8 Quantization (`onnxruntime.quantization.quantize_dynamic`) to reduce model weight footprint.
- Saves the quantized final artifact directly as `models/v1.bai` (or output path parameter).
- Validates the exported `.bai` model by running a test inference pass using `onnxruntime.InferenceSession`.

---

## Output Rules:
- DO NOT overwrite any files directly in the repository yet.
- Put the entire output code for all 3 files into `.copilot/phase_1/output.txt`.
- Pause process execution upon writing to `output.txt` and await approval.