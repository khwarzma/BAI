# Khwarazma BAI Technical Specification

This specification records the implementation-backed v1 contract for Khwarazma BAI, the Bareeed Artificial Intelligence engine used for compact email triage.

## 1. Operational goals and evidence

The performance targets are:

| Metric | Target | Repository evidence |
| --- | ---: | --- |
| Mean request latency | `< 15 ms` | `tests/test_performance.py` asserts `<= 15.0 ms`; a recorded local run measured **0.69 ms mean** (0.6939 ms) over 1,000 sequential requests. |
| Process RSS | `< 150 MB` | The test asserts `<= 150 MB`; a recorded local run measured 14.54 MB RSS after the request loop. |
| RSS growth during loop | `<= 5 MB` | Asserted by the performance test. |

The benchmark warms the pipeline with ten calls, then measures 1,000 sequential `predict_json()` calls plus Python `json.loads`. It excludes initialization and model loading, and does not establish cold-start, concurrent, GPU, p95/p99, or cross-machine performance. RSS is a process high-water mark obtained through `resource.getrusage`; it is not a model-only memory measurement.

`tests/test_performance.py` records each request with `time.perf_counter()`, computes the arithmetic mean with `statistics.fmean()`, and asserts average latency, final high-water RSS, and RSS growth. In the recorded environment the mean was approximately **0.69 ms** (0.6939 ms) and post-loop RSS was **14.54 MB**. These results demonstrate the thresholds for that environment and test shape; they are not a universal service-level guarantee.

The engine's performance design includes a compact transformer, ONNX Runtime execution, fixed-size input buffers, graph optimization configuration, and optional thread configuration. The native path still performs tokenization, buffer copies, temporary ONNX value construction, inference, postprocessing, JSON serialization, and (in the test) JSON parsing.

## 2. Input contract

The public text path accepts a non-empty UTF-8-oriented `std::string`. `InferencePipeline::predict()` rejects empty text. The tokenizer produces:

- `input_ids`: `int64` token IDs;
- `attention_mask`: `int64` values where 1 represents a real token and 0 represents padding;
- `[CLS]` at the beginning and `[SEP]` at the end;
- padding to the requested maximum length, normally 512.

The lower-level engine requires equal, non-empty spans whose length does not exceed `EngineConfig::max_seq_length`. It does not validate the semantic correctness of individual token IDs or mask values.

## 3. ONNX model contract

The model and runtime must agree on these names and shapes:

| Direction | Name | Meaning |
| --- | --- | --- |
| Input | `input_ids` | Batch token IDs, `int64`. |
| Input | `attention_mask` | Batch token/padding mask, `int64`. |
| Output | `logits_category` | Five category logits: inbox pinned, inbox, bait, bais, baiads. |
| Output | `logits_otp` | One binary OTP classification logit. |
| Output | `confidence` | One confidence score produced by the model. |

`BaiEngine::initialize()` caches these names and `BaiEngine::infer()` requires exactly three output tensors. Category output is copied into `std::array<float, 5>`, while OTP and confidence use their first scalar values.

The contract is positional as well as nominal: category logits are expected to contain five values, and the OTP/confidence outputs are expected to contain one scalar for the single-item inference path. A model with different node names, output order, or dimensions is not compatible without coordinated changes to `training/export.py` and `core/src/engine.cpp`.

The Python model (`training/model.py`) uses a six-layer `nn.TransformerEncoder` with `d_model=256`, eight attention heads, feed-forward width 1024, maximum sequence length 512, and five category outputs by default. `training/export.py` exports with ONNX opset 17 and validates the exported output contract before quantization.

## 4. Postprocessing and confidence

Category probabilities are computed using stable softmax:

```text
max = max(category_logits)
e_i = exp(category_logits_i - max)
p_i = e_i / sum(e_i)
```

The category is the index with the highest `p_i`. The runtime maps indexes to:

```text
0 -> inbox_pinned
1 -> inbox
2 -> bait
3 -> bais
4 -> baiads
```

OTP confidence uses the sigmoid function:

```text
sigmoid(x) = 1 / (1 + exp(-x))
```

`otp_detected` is true when the raw OTP logit is greater than zero. The model's overall `confidence` is already sigmoid-bounded in `BaiMicroEncoder.confidence_head`, and the native runtime returns it without applying a second transform.

The learned confidence objective is implemented in `BaiTrainer._compute_loss()`:

```text
0.7 * CrossEntropy(category)
+ 0.3 * BCEWithLogits(OTP)
+ 0.2 * MSE(confidence, category_correctness)
```

The policy thresholds in `config/rules.json` (including 0.85 autonomous execution, 0.75 high confidence, 0.50 low confidence, 0.30 reject, and 0.90 auto-flag) are metadata for the application policy. The native runtime does not load or enforce them. There is no separate temperature-scaling, isotonic, or Platt calibration stage in the inspected source.

`training/export.py` validates the quantized artifact with `onnxruntime.InferenceSession`, random `int64` inputs of shape `(1, 128)`, three outputs, shapes `(1, 5)`, `(1, 1)`, and `(1, 1)`, and confidence values in `[0, 1]`. The exporter uses a `(1, 256)` dummy input for export and declares dynamic batch and sequence axes for both inputs.

## 5. JSON result contract

`predict_json()` serializes:

```json
{
  "category": "inbox_pinned",
  "category_confidence": 0.0000,
  "otp_detected": false,
  "otp_confidence": 0.5000,
  "overall_confidence": 0.0000,
  "execution_time_ms": 0.0000,
  "model_version": "1.0.0"
}
```

The values are formatted by `JsonBuilder` in `core/src/json_builder.cpp`. Application-level fields such as purge schedules, language labels, dialect labels, sender metadata, and user prompts are not emitted by the current native JSON builder.

Important implementation limitation: `InferencePipeline::predict_json()` currently passes an all-zero category-logit array to `JsonBuilder` after obtaining the real prediction. Equal logits select `inbox_pinned`, so JSON category output can disagree with `predict()`'s actual model-selected category. Consumers should treat this as a known correctness risk until fixed.

## 6. Resource and portability constraints

`EngineConfig` exposes thread count, device ID, GPU selection, FP16 selection, maximum sequence length, and graph optimization level. `enable_fp16` is copied through the API but is not consumed by the current engine implementation. CUDA setup is conditional on `ENABLE_CUDA`; the checked-in CMake configuration does not define that macro.

The repository contains no source-backed guarantee of a two-core cap, zero disk writes, universal sub-15 ms latency, or a literal zero-allocation path. Those are operational goals or design claims requiring deployment-specific verification.
