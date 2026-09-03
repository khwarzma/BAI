#pragma once

#include "bai/engine.hpp"
#include "tokenizer.hpp"
#include "json_builder.hpp"
#include <chrono>
#include <string>
#include <expected>

namespace bai {

/**
 * High-level prediction result container.
 */
struct PredictionResult {
    std::string category;           // Predicted category name
    float category_confidence;      // Confidence for category prediction
    bool otp_detected;              // Whether OTP was detected
    float otp_confidence;           // Confidence for OTP detection
    float overall_confidence;       // Overall model confidence
    double execution_time_ms;       // Inference time
    std::array<float, 5> category_logits; // Raw model category logits
    float otp_logit;                // Raw model OTP logit
};

/**
 * InferencePipeline - High-level end-to-end inference orchestration.
 * 
 * Coordinates:
 * 1. Text input -> tokenization (BaiTokenizer)
 * 2. Tokenized input -> inference (BaiEngine)
 * 3. Raw logits -> formatted output (JsonBuilder)
 * 
 * Provides simplified Python-friendly interface to complex multi-component system.
 * All errors handled via std::expected to ensure no exceptions in critical paths.
 */
class InferencePipeline {
public:
    /**
     * Constructor.
     */
    InferencePipeline();

    /**
     * Destructor.
     */
    ~InferencePipeline() = default;

    // Disable copy, allow move
    InferencePipeline(const InferencePipeline&) = delete;
    InferencePipeline& operator=(const InferencePipeline&) = delete;

    InferencePipeline(InferencePipeline&&) noexcept = default;
    InferencePipeline& operator=(InferencePipeline&&) noexcept = default;

    /**
     * Initialize pipeline with model and tokenizer.
     * 
     * @param engine_config ONNX engine configuration
     * @param vocab_path Path to tokenizer vocabulary JSON
     * @return std::expected with void on success, error message on failure
     */
    std::expected<void, std::string> initialize(
        const EngineConfig& engine_config,
        const std::string& vocab_path
    );

    /**
     * Run inference and return typed result.
     * 
     * Steps:
     * 1. Encode text to tokens
     * 2. Run ONNX inference
     * 3. Extract results
     * 
     * @param text Input text
     * @return std::expected containing PredictionResult on success, error on failure
     */
    std::expected<PredictionResult, std::string> predict(const std::string& text);

    /**
     * Run inference and return formatted JSON.
     * 
     * @param text Input text
     * @return std::expected containing JSON string on success, error on failure
     */
    std::expected<std::string, std::string> predict_json(const std::string& text);

    /**
     * Check if pipeline is initialized.
     * @return true if both tokenizer and engine are ready
     */
    bool is_initialized() const;

private:
    BaiEngine engine_;
    BaiTokenizer tokenizer_;
    JsonBuilder json_builder_;

    /**
     * Helper to measure execution time.
     */
    class ScopedTimer {
    public:
        ScopedTimer(double& out_ms) : out_ms_(out_ms) {
            start_ = std::chrono::high_resolution_clock::now();
        }
        ~ScopedTimer() {
            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start_);
            out_ms_ = duration.count() / 1000.0;
        }
    private:
        double& out_ms_;
        std::chrono::high_resolution_clock::time_point start_;
    };
};

}  // namespace bai
