# Khwarazma BAI Deployment

## Deployment boundary

Khwarazma BAI is intended to run inside the Bareeed backend boundary so that mail classification can remain close to the product's privacy and authorization controls. The engine returns classification signals; the Bareeed application owns mailbox writes, user prompts, retention, purge scheduling, access control, observability, and production rollout policy.

The public source surface includes the C++23 engine core, ONNX Runtime bridge, C API, and pybind11 module. Production model weights remain closed and proprietary. A local model artifact in a development checkout does not imply permission to redistribute or deploy it in another service.

## Requirements

- CMake 3.20 or newer.
- A compiler with C++23 support.
- Python 3 development headers and interpreter.
- `pybind11` discoverable from the active Python environment.
- ONNX Runtime headers and a compatible shared library.
- An authorized compatible BAI model artifact and `models/vocab.json`.

Python training/export/test dependencies are listed in `requirements.txt`. Installing that file does not install a C++ compiler, CMake, ONNX Runtime development files, or a packaged `bai_core` wheel.

## Build the native extension

The checked-in build is defined by `core/CMakeLists.txt`:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt

cmake -S core -B core/build \
  -DPython3_EXECUTABLE="$(command -v python)" \
  -DCMAKE_BUILD_TYPE=Release
cmake --build core/build --config Release
```

The project sets C++23 and position-independent code, builds the `bai_static` library, and conditionally generates the `bai_core` Python extension with `pybind11_add_module()` when pybind11 is found through CMake. The extension is defined by `core/bindings/bai_pybind.cpp`; it links the same native sources and ONNX Runtime library as the core target. The module exposes `EngineConfig`, `BaiEngine`, `InferencePipeline`, result types, and `get_version()`.

`core/CMakeLists.txt` compiles `src/engine.cpp`, `src/c_api.cpp`, `src/tokenizer.cpp`, `src/json_builder.cpp`, and `src/inference_engine.cpp`. It requires Python Interpreter/Development components, queries the active interpreter for `pybind11.get_cmake_dir()`, and links a compatible ONNX Runtime shared library. The Python target is optional at configure time; the static core target can still be produced if pybind11 is unavailable.

The current CMake file contains an absolute developer-machine path to `libonnxruntime.so`. A portable deployment must replace this with a toolchain/package-manager path or an explicit deployment configuration. The generated extension filename is platform- and Python-version-specific.

The model artifact is an ONNX-compatible file produced by `training/export.py`: PyTorch is exported with opset 17, then dynamic INT8 quantization writes the configured output path (normally `models/v1.bai`) while preserving the input/output node names. The `.bai` suffix is a repository naming convention; the C++ loader passes the path directly to `Ort::Session` and does not implement a separate custom model format.

## Runtime initialization

```python
import bai_core

config = bai_core.EngineConfig()
config.model_path = "/authorized/models/v1.bai"
config.num_threads = 2
config.max_seq_length = 512

pipeline = bai_core.InferencePipeline()
pipeline.initialize(config, "/authorized/models/vocab.json")
result = pipeline.predict_json("Your verification code is 654321.")
```

Initialization creates an ONNX Runtime environment and session, configures graph optimization and threads, selects the configured provider, loads the model, and allocates input buffers. Model loading is excluded from the repository's request-latency benchmark.

For C++ consumers, include `core/include/inference_engine.hpp` and link the native implementation and ONNX Runtime. For C-compatible consumers, use `core/include/bai/c_api.h`; retain the returned opaque handle until `bai_engine_destroy()`.

## Runtime operations and safety

The engine is designed for one initialized instance to serve text requests, but its mutable input buffers are not protected by an explicit synchronization primitive. Serialize calls on a shared instance or provision independent instances per concurrent execution context. Validate deployment behavior under the host web server's worker/thread model.

`use_gpu` and `enable_fp16` require special care: CUDA provider code is conditional on `ENABLE_CUDA`, and `enable_fp16` is not currently consumed by `engine.cpp`. GPU deployment is not established by the available build or tests.

The implementation does not provide the previously described memory-mapped model loading or atomic pointer swap. Model updates should therefore be treated as an application-level process/session replacement procedure and validated with a controlled rollout; do not claim zero-downtime hot swapping from this repository alone.

## Performance acceptance

The available test command is:

```bash
PYTHONPATH=core/build python -m pytest tests -q -s
```

The performance test checks 1,000 sequential warmed requests, mean latency `<= 15 ms`, RSS `<= 150 MB`, and RSS growth `<= 5 MB`. A recorded environment measured 0.6939 ms mean latency and 14.54 MB RSS. These results are evidence for that environment only; production acceptance should repeat the benchmark on the target host, with target worker count, input-length distribution, cold-start behavior, and concurrency profile.

## BUSL-1.1 and closed weights

This repository is licensed under the **Business Source License 1.1** in `LICENSE`. The current terms permit non-production development, educational use, security auditing, interoperability, and evaluation. Without written authorization from Khwarazma, the Licensed Work may not be deployed in a live production environment, embedded in a commercial service, used to build a competing product, or sold/hosted as a service.

The Change Date is **2030-09-03**, after which the license converts to Apache License 2.0 as specified in `LICENSE`, subject to the license terms. The license does not grant rights to proprietary BAI weights, checkpoints, private Bareeed data, or production credentials. Model files must be obtained and deployed only under explicit Khwarazma authorization.

## Production review requirements

Because BAI is directly integrated with Bareeed's production backend, every deployment change should receive strict pull-request review for:

- privacy and data handling;
- model input/output compatibility;
- ABI and Python compatibility;
- memory ownership and concurrency;
- error propagation and failure recovery;
- latency/RSS regression;
- model provenance and closed-weight access;
- rollback and operational observability.

For licensing, deployment authorization, or contribution questions, contact `Khwarzma@bareeed.com` or `im4@bareeed.com`.
