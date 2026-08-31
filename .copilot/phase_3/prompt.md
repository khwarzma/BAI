# TASK SPECIFICATION: Phase 3 — C++23 High-Performance Inference Engine with C API

## Context & Requirements
We are building the core inference engine for BAI using modern **C++23** and **ONNX Runtime C++ API**.
The engine must follow strict zero-dynamic-memory allocation patterns in hot loops and provide a zero-cost error handling mechanism using `std::expected`.
Additionally, a pure C API wrapper (`extern "C"`) must be exported for dynamic linking and pybind11 integration.

---

## Architecture Constraints:
- Standard: C++23 (`-std=c++23`).
- Error Handling: Use `std::expected<T, std::string>` or custom error codes; no raw `throw` in performance-critical execution paths.
- Memory: Avoid allocations inside `infer()` hot path. Pre-allocate ONNX input/output tensors and buffers during initialization.
- Headers & Source structure:
  - `include/bai/engine.hpp`: High-level C++23 class `BaiEngine`.
  - `include/bai/c_api.h`: Pure C API interface.
  - `src/core/engine.cpp`: C++23 Engine implementation.
  - `src/core/c_api.cpp`: Implementation of the C interface.

---

## File 1: `include/bai/engine.hpp`
Define the main C++23 `BaiEngine` class:
- Config struct `EngineConfig` (model_path, num_threads, device_id, use_gpu, etc.).
- Output struct `InferenceResult` containing:
  - `std::array<float, 5> category_logits`
  - `float otp_logit`
  - `float confidence`
- Method `initialize(const EngineConfig& config) -> std::expected<void, std::string>`
- Method `infer(std::span<const int64_t> input_ids, std::span<const int64_t> attention_mask) -> std::expected<InferenceResult, std::string>`
- Proper RAII cleanup for ONNX Runtime sessions, environments, and memory info.

---

## File 2: `include/bai/c_api.h`
Define exported C bindings:
- Opaque handle: `typedef struct BaiEngineHandle_t* BaiEngineHandle;`
- Status codes enum: `BAI_SUCCESS`, `BAI_ERROR_INIT_FAILED`, `BAI_ERROR_INFERENCE_FAILED`, `BAI_ERROR_INVALID_PARAM`.
- `bai_engine_create(const char* model_path, int num_threads, BaiEngineHandle* out_handle);`
- `bai_engine_infer(BaiEngineHandle handle, const int64_t* input_ids, const int64_t* attention_mask, size_t seq_len, float* out_category_logits, float* out_otp_logit, float* out_confidence);`
- `bai_engine_destroy(BaiEngineHandle handle);`

---

## File 3: `src/core/engine.cpp`
Implement `BaiEngine`:
- Include `<onnxruntime_cxx_api.h>`.
- Properly set up `Ort::Env`, `Ort::SessionOptions`, and `Ort::Session`.
- Pre-allocate input/output `Ort::Value` structures where applicable to eliminate reallocation overhead during `infer()`.
- Run model forward pass and populate `InferenceResult`.

---

## File 4: `src/core/c_api.cpp`
Implement C bindings translating C calls to `BaiEngine` methods with safe C-style error reporting.

---

## Output Rules:
- Output ALL files clearly into `.copilot/phase_3/output.txt` using section markers (`FILE: <path>`).
- Do NOT generate CMake files yet (Phase 4 will handle build systems and Python bindings).
- Pause execution after outputting.