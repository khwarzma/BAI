# FIX SPECIFICATION: Phase 1 — Core AI Model & Export Pipeline

Please correct the issues in `.copilot/phase_1/output.txt` and regenerate the updated version back into `.copilot/phase_1/output.txt`.

---

## Required Fixes:

### 1. File: `training/model.py`
- Fix `attention_mask` handling inside `forward()`:
  - If `attention_mask` is provided as `(batch_size, seq_len)` where `1` = real token and `0` = pad token, convert it properly to `src_key_padding_mask` boolean matrix (`attention_mask == 0`).
  - Ensure the output signature strictly matches: `Tuple[torch.Tensor, torch.Tensor, torch.Tensor]`.
  - Fix CLS token extraction: Keep `cls_output = encoded[:, 0, :]` but ensure layer normalization is applied safely.

### 2. File: `training/train.py`
- Update `collate_fn`:
  - Support configurable `pad_token_id` (default = 0).
  - Pad both `input_ids` and `attention_mask` cleanly up to `max_len` within the batch.
- Update `BaiTrainer`:
  - Fix device assignment logic for `autocast` (`device_type='cuda'` or `'cpu'`).

### 3. File: `training/export.py`
- Fix `export_to_onnx`:
  - Ensure positional dummy input arguments explicitly map to `(input_ids, attention_mask)`.
  - Explicitly mark `opset_version=17` for seamless compatibility with C++ ONNX Runtime.
  - Verify that INT8 quantized output `.bai` maintains identical input/output node names.

---

## Instructions:
Apply these strictly without altering the underlying architecture (6 layers, 8 heads, 256 d_model). Output the corrected 3 files into `.copilot/phase_1/output.txt`. 