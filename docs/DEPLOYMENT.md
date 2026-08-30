# BAI (Bareeed Artificial Intelligence) — Production Deployment & Operations Architecture

> **Specification Standard:** AM Standard (SAM Category)
> **Document Identifier:** AM-DEPL-BAI-1.0.0
> **Initiator & Owner:** Khwarzma
> **Target Application Infrastructure:** Bareeed Production Servers
> **Status:** Active Operational & Deployment Specification

---

## 1. System Hardware Boundaries & Environment Isolation

The deployment architecture of BAI v1 is engineered under strict execution boundaries to ensure complete co-existence with web application processes (Django / Gunicorn / Nginx) on standard 2-core CPU hardware configurations.

```text
+---------------------------------------------------------------------------------+
|                               SYSTEM SECURITY BOUNDARY                          |
|                                                                                 |
|  +---------------------------------------------------------------------------+  |
|  |                         Django Web Process Layer                          |  |
|  +---------------------------------------------------------------------------+  |
|                                      |                                          |
|                          Direct Ctypes / Pybind11 API                           |
|                                      v                                          |
|  +---------------------------------------------------------------------------+  |
|  |                 BAI Engine Native C++ Shared Library (.so)                |  |
|  |                                                                           |  |
|  |  +--------------------+   +-------------------+   +--------------------+  |  |
|  |  | Zero-Copy Tokenizer|   | ONNX Micro-Kernel |   | JSON Output Stream |  |  |
|  |  | (In-Memory Buffer) |   | (GGUF / INT8)     |   | (Static Buffer)    |  |  |
|  |  +--------------------+   +-------------------+   +--------------------+  |  |
|  +---------------------------------------------------------------------------+  |
+---------------------------------------------------------------------------------+

```

### 1.1 Infrastructure & Hardware Capping Parameters

| Parameter / Resource | Enforcement Constraint | Verification Mechanism |
| --- | --- | --- |
| **Operating System** | Linux (Debian 12 / Ubuntu 22.04 LTS x86_64) | POSIX Kernel Interface |
| **CPU Architecture** | 2 CPU Cores Maximum (AVX2 Enabled) | System Cgroups Thread Binding |
| **System RAM Limit** | Hard cap $\le 150 \text{ MB}$ total resident set size (RSS) | Static Memory Allocator & Valgrind |
| **Disk Storage I/O** | $0 \text{ bytes}$ dynamic disk writes during inference | In-Memory Payload Processing |
| **Dependency Footprint** | Zero PyTorch, Zero Heavy ML Runtimes | Native Dynamic Shared Library (`.so`) |

---

## 2. Server Directory Layout & Artifact Isolation

The deployment directory contains only pre-compiled shared binaries, quantized model parameters, and runtime configuration mappings.

```text
/our/server/root/
├── config/
│   ├── languages.json            <-- Dynamic dialect maps & normalization rules
│   └── rules.json                <-- System lifecycle & auto-purge rules
├── core/
│   └── bai_engine.so             <-- Native C++23 compiled shared library object
├── bridge/
│   └── bai_pybind.so             <-- Native Python binding module extension
└── models/
    ├── v1.bai                    <-- Active quantized micro-weights file (GGUF / ONNX)
    └── v1.1.bai                  <-- Staged model weights file for atomic hot-swap

```

---

## 3. C++ Shared Library Compilation & Optimization Pipeline

Building the native execution engine requires compilation with maximum optimization flags using C++23 standards.

### 3.1 Advanced CMake Build Script (`core/CMakeLists.txt`)

```cmake
cmake_minimum_required(VERSION 3.22)
project(bai_engine LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# High-performance compiler optimization flags
add_compile_options(
    -O3
    -march=native
    -flto
    -fno-rtti
    -fno-exceptions
    -Wall
    -Wextra
)

# Core Dynamic Shared Object
add_library(bai_engine SHARED
    src/tokenizer.cpp
    src/inference_engine.cpp
    src/json_builder.cpp
)

target_include_directories(bai_engine PUBLIC include/)

```

### 3.2 Compilation & Strip Sequence

```bash
# Navigate to C++ core directory
cd BAI/core

# Configure out-of-source build with Release optimizations
cmake -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_STANDARD=23

# Compile shared library objects
cmake --build build --config Release -j$(nproc)

# Strip debug symbols to reduce dynamic binary footprint
strip --strip-unneeded build/libbai_engine.so
strip --strip-unneeded build/bai_pybind.so

# Deploy compiled artifacts to server location
cp build/libbai_engine.so /our/server/root/core/bai_engine.so
cp build/bai_pybind.so /our/server/root/bridge/bai_pybind.so

```

---

## 4. Integration with Django Host Application

Django communicates with the pre-compiled C++ shared object via Pybind11 bindings initialized at application startup.

### 4.1 Django Application Bootstrapper (`apps.py`)

```python
import sys
from django.apps import AppConfig

# Inject server bridge location into runtime path
sys.path.append("/our/server/root/bridge")
import bai_pybind  # Native C++ module link

class BaiEngineConfig(AppConfig):
    name = 'bai_engine_integration'
    engine_instance = None

    def ready(self):
        """
        Instantiates single global engine context on Django startup.
        Loads model weights once into shared RAM space.
        """
        BaiEngineConfig.engine_instance = bai_pybind.InferenceEngine(
            model_path="/our/server/root/models/v1.bai",
            config_path="/our/server/root/config/languages.json",
            rules_path="/our/server/root/config/rules.json"
        )

```

### 4.2 In-Memory Inference Execution Service (`services.py`)

```python
import json
from django.apps import apps
from typing import Dict, Any

def process_incoming_mail_payload(subject: str, body: str, sender: str) -> Dict[str, Any]:
    """
    Passes raw mail payload buffers to C++ engine.
    Executes sub-15ms classification with zero disk writes.
    """
    config = apps.get_app_config('bai_engine_integration')
    engine = config.engine_instance

    # Native C++ execution pass
    raw_json_response = engine.predict(
        subject=subject,
        body=body,
        sender=sender
    )

    return json.loads(raw_json_response)

```

---

## 5. Zero-Downtime Model Hot-Swapping Procedure

Updating model weights (e.g., from `v1.bai` to `v1.1.bai`) is executed atomically using memory-mapped buffer pointer swapping without restarting Gunicorn or dropping active mail tasks.

```text
                                HOT-SWAPPING PIPELINE

  New Weight File        Integrity Verification         Atomic Memory Swap
 +---------------+       +--------------------+       +--------------------+
 |   v1.1.bai    |  -->  |  SHA-256 Checksum  |  -->  | Signal C++ Engine  |
 | File Staged   |       |  Verification      |       | (Pointer Reload)   |
 +---------------+       +--------------------+       +--------------------+
                                                                |
                                                                v
                                                      Zero Dropped Requests /
                                                      Zero Service Restarts

```

### 5.1 Hot-Swap Execution Script (`deploy_weights.sh`)

```bash
#!/usr/bin/env bash
set -euo pipefail

MODEL_DIR="/our/server/root/models"
NEW_MODEL="v1.1.bai"
TARGET_LINK="v1.bai"
CHECKSUM_FILE="v1.1.bai.sha256"

cd "${MODEL_DIR}"

# Step 1: Verify model file checksum integrity
sha256sum -c "${CHECKSUM_FILE}"

# Step 2: Atomic symlink replacement
ln -sf "${NEW_MODEL}" active_weights.tmp
mv -Tf active_weights.tmp "${TARGET_LINK}"

# Step 3: Send POSIX signal to Gunicorn processes to reload pointer memory
pkill -USR1 -f "gunicorn.*bareeed"

```

---

## 6. Real-Time Resource Auditing & SLA Verification

To guarantee adherence to the **AM Standard** during production operation, system health checks continuously monitor execution boundaries:

* **Memory Footprint Audit:** Verifies that system RAM usage remains capped below $150 \text{ MB}$.
```bash
ps aux | grep bareeed | awk '{sum+=$6} END {print "Total Memory: " sum/1024 " MB"}'

```


* **Latency Verification:** Confirms inference execution latency satisfies the sub-15ms constraint via the telemetry metadata emitted by the C++ core:
```json
"telemetry": {
  "inference_time_ms": 7.34,
  "memory_footprint_mb": 112.4
}

```
