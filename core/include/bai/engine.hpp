
#pragma once

#include "onnxruntime_cxx_api.h"

#include <array>
#include <span>
#include <string>
#include <memory>
#include <mutex>
#include <expected>
#include <vector>

namespace bai {

inline constexpr size_t kModelSequenceLength = 256;

/**
 * Configuration for BaiEngine initialization.
 */
struct EngineConfig {
    std::string model_path;           // Path to .bai ONNX model file
    int num_threads = 4;               // Number of inference threads
    int device_id = 0;                 // GPU device ID (0 for CPU)
    bool use_gpu = false;              // Enable GPU inference
    bool enable_fp16 = false;           // Enable FP16 computation (GPU only)
    size_t max_seq_length = kModelSequenceLength; // ONNX model sequence length
    int graph_optimization_level = 2;  // ONNX graph optimization (0-3)
};

/**
 * Inference result container.
 */
struct InferenceResult {
    std::array<float, 5> category_logits;  // [INBOX_PINNED, INBOX, BAIT, BAIS, BAIADS]
    float otp_logit;                       // Binary OTP classification logit
    float confidence;                      // Model confidence score [0.0, 1.0]
};

/**
 * BaiMicroEncoder inference engine with zero-allocation hot path.
 * 
 * Provides efficient inference using ONNX Runtime C++ API with:
 * - Pre-allocated input/output buffers
 * - No dynamic allocations in critical inference path
 * - RAII resource management
 * - std::expected error handling
 */
class BaiEngine {
public:
    /**
     * Constructor.
     */
    BaiEngine();

    /**
     * Destructor - cleans up ONNX Runtime resources.
     */
    ~BaiEngine();

    // Disable copy operations
    BaiEngine(const BaiEngine&) = delete;
    BaiEngine& operator=(const BaiEngine&) = delete;

    // Move operations
    BaiEngine(BaiEngine&& other) noexcept;
    BaiEngine& operator=(BaiEngine&& other) noexcept;

    /**
     * Initialize the inference engine with configuration.
     * 
     * @param config Engine configuration
     * @return std::expected with void on success, error message on failure
     */
    std::expected<void, std::string> initialize(const EngineConfig& config);

    /**
     * Run inference on tokenized input.
     * 
     * Zero-allocation inference path - all buffers pre-allocated during init.
     * 
     * @param input_ids Tokenized input sequence (exactly 256 tokens)
     * @param attention_mask Attention mask (1=token, 0=padding)
     * @return std::expected containing InferenceResult on success, error message on failure
     */
    std::expected<InferenceResult, std::string> infer(
        std::span<const int64_t> input_ids,
        std::span<const int64_t> attention_mask
    );

    /**
     * Check if engine is initialized.
     * @return true if ready for inference
     */
    bool is_initialized() const { return session_ != nullptr; }

private:
    // ONNX Runtime components
    std::unique_ptr<Ort::Env> env_;
    std::unique_ptr<Ort::Session> session_;
    std::unique_ptr<Ort::MemoryInfo> memory_info_;

    // Configuration
    EngineConfig config_;

    // Pre-allocated buffers for inference
    std::vector<int64_t> input_ids_buffer_;
    std::vector<int64_t> attention_mask_buffer_;

    // Cached node names
    std::vector<const char*> input_names_;
    std::vector<const char*> output_names_;
    mutable std::mutex inference_mutex_;

    /**
     * Validate input dimensions and constraints.
     * @return error message, empty string if valid
     */
    std::string validate_inputs(
        std::span<const int64_t> input_ids,
        std::span<const int64_t> attention_mask
    ) const;
};

}  // namespace bai
