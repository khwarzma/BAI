# BAI (Bareeed Artificial Intelligence) — Technical Architecture

> **Specification Standard:** AM Standard (SAM Category)
> **Document Identifier:** AM-ARCH-BAI-1.0.0
> **Initiator:** Khwarzma
> **Target System:** Bareeed Infrastructure
> **Status:** Architecture Reference Document

---

## 1. High-Level System Topology

The BAI system relies on a clean separation of concerns between training pipelines, C++ core inference, and the Django host application.

```text
+---------------------------------------------------------------------------------+
|                                 TRAINING ENVIRONMENT                            |
|  +-------------------+      +--------------------+      +--------------------+  |
|  |  Data Generator   | ---> | PyTorch Transformer| ---> |  Quantizer & Export|  |
|  | (Llama / Gemini)  |      |   (Micro-Encoder)  |      |   (export.py)      |  |
|  +-------------------+      +--------------------+      +--------------------+  |
+----------------------------------------------------------- | -------------------+
                                                             | Generates v1.bai
                                                             v
+---------------------------------------------------------------------------------+
|                             SERVER PRODUCTION RUNTIME                           |
|                                                                                 |
|  +---------------------------------------------------------------------------+  |
|  |                           Django Web Application                          |  |
|  +---------------------------------------------------------------------------+  |
|                                      |                                          |
|                         Low-Latency Pybind11 Calls                              |
|                                      v                                          |
|  +---------------------------------------------------------------------------+  |
|  |                     C++23 Native Engine (bai_engine.so)                   |  |
|  |                                                                           |  |
|  |  +--------------------+   +-------------------+   +--------------------+  |  |
|  |  | Zero-Copy Tokenizer|   | ONNX Runtime / C++|   | JSON Output Builder|  |  |
|  |  | (languages.json)   |   | Core (v1.bai)     |   | (Zero Allocation)  |  |  |
|  |  +--------------------+   +-------------------+   +--------------------+  |  |
|  +---------------------------------------------------------------------------+  |
+---------------------------------------------------------------------------------+

```

---

## 2. Micro-Transformer Encoder Architecture

The neural core of BAI v1 is a custom Micro-Transformer Encoder designed for high-speed sequence classification and parameter frugality.

### 2.1 Model Specifications

* **Parameter Count:** ~15M to 25M trainable parameters.
* **Layer Depth:** 6 Encoder Blocks.
* **Attention Heads:** 8 Multi-Head Self-Attention layers optimized with FlashAttention primitives.
* **Hidden Dimensionality ($d_{model}$):** 256.
* **Feed-Forward Dimensionality ($d_{ff}$):** 1024.
* **Maximum Context Window:** 512 subword tokens (truncating non-essential email tails while preserving subject and header context).

---

## 3. C++23 Inference Core Design (`core/`)

To fulfill the strict sub-15ms execution requirement on standard dual-core hardware, the inference engine is implemented in native C++23 using zero-dynamic-memory-allocation principles within execution loops.

### 3.1 Subword Tokenization Pipeline

* Loads language mapping, diacritic normalization rules, and vocabulary mappings directly from `config/languages.json`.
* Operates on std::string_view and std::span buffers to avoid memory allocation during token sequence preparation.

### 3.2 ONNX Runtime Engine Integration

* Wraps the ONNX Runtime C++ API to load quantized `v1.bai` model artifacts.
* Pre-allocates execution tensors at initial load time, preventing memory allocations during single-message inference passes.

### 3.3 Zero-Allocation JSON Builder

* Formats classification categories, confidence scores, and OTP metadata into valid JSON payloads without dynamic heap memory allocation.

---

## 4. Host Integration & Pybind11 Native Interface

Django communicates directly with the compiled C++ shared object via Pybind11 wrappers, eliminating IPC (Inter-Process Communication) and HTTP overhead.

### 4.1 Python Interface Definition

```python
import bai_engine

# Initialize engine globally during Django startup
engine = bai_engine.InferenceEngine(
    model_path="/our/server/root/models/v1.bai",
    config_path="/our/server/root/config/languages.json",
    rules_path="/our/server/root/config/rules.json"
)

# Inference execution call (sub-15ms)
result_json = engine.predict(
    subject="Your login code is 948201",
    body="Use this code to verify your session. Valid for 5 minutes.",
    sender="noreply@service.com"
)

```

---

## 5. Zero-Downtime Model Hot-Swapping

To update weights (e.g., from `v1.bai` to `v1.1.bai`) without restarting Django processes or dropping incoming mail:

1. **Memory-Mapped Loading (`mmap`):** The C++ engine loads model weights into memory-mapped buffer spaces.
2. **Atomic Pointer Swapping:** When a new `.bai` binary is verified, a control signal triggers the C++ engine to load the new weights into a secondary pointer and atomically swap the active pointer (`std::atomic<EngineState*>`).
3. **Zero Interruption:** Older threads finish processing under the previous pointer while new incoming requests immediately utilize the updated weights.
