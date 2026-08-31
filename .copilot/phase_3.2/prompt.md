# TASK SPECIFICATION: Phase 3.2 — High-Level Subsystems Integration & Pybind11

## Context & Architecture Alignment
We are completing the missing C++ subsystems in the `core/` directory.
IMPORTANT: The low-level ONNX engine and C API are ALREADY implemented and located in:
- `core/include/bai/engine.hpp` and `core/include/bai/c_api.h`
- `core/src/engine.cpp` and `core/src/c_api.cpp`

DO NOT modify or recreate the files under `core/include/bai/` or their source implementations. 
Instead, read and inspect their structure, then implement the surrounding high-level system components so they integrate perfectly with `bai::BaiEngine`.

---

## Directory Structure to Fulfill:
core/
├── bindings/
│   └── bai_pybind.cpp
├── include/
│   ├── bai/ (DO NOT TOUCH - EXISTING)
│   ├── inference_engine.hpp
│   ├── json_builder.hpp
│   └── tokenizer.hpp
└── src/
├── c_api.cpp (DO NOT TOUCH - EXISTING)
├── engine.cpp (DO NOT TOUCH - EXISTING)
├── inference_engine.cpp
├── json_builder.cpp
└── tokenizer.cpp


---

## Component Specifications:

### 1. `core/include/tokenizer.hpp` & `core/src/tokenizer.cpp`
- Class `BaiTokenizer` in namespace `bai`.
- Load subword vocabulary or JSON mapping rules (`initialize(const std::string& vocab_path)`).
- `encode(const std::string& text, size_t max_seq_len)` returning a pair/struct of `input_ids` and `attention_mask` (vectors of `int64_t`).

### 2. `core/include/json_builder.hpp` & `core/src/json_builder.cpp`
- Class `JsonBuilder` in namespace `bai`.
- Construct fast, low-overhead JSON response strings containing category prediction, OTP status, confidence score, and meta execution info.

### 3. `core/include/inference_engine.hpp` & `core/src/inference_engine.cpp`
- Class `InferencePipeline` in namespace `bai`.
- Integrates `BaiTokenizer`, `BaiEngine` (`#include "bai/engine.hpp"`), and `JsonBuilder`.
- High-level prediction methods:
  - `predict(const std::string& text) -> InferenceResult`
  - `predict_json(const std::string& text) -> std::string`

### 4. `core/bindings/bai_pybind.cpp`
- Export `pybind11` module named `bai_core`.
- Bind `InferencePipeline` for easy Python invocation.
- Bind `BaiEngine` and `EngineConfig` structs for raw direct invocation if needed.

---

## Output Rules:
- Output ALL remaining 7 files (`tokenizer.hpp/cpp`, `json_builder.hpp/cpp`, `inference_engine.hpp/cpp`, `bai_pybind.cpp`) clearly into `.copilot/phase_3.2/output.txt` using section markers (`FILE: <path>`).
- Ensure full C++23 standard compatibility.
- Pause execution after outputting.