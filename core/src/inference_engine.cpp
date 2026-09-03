#include "inference_engine.hpp"

#include <array>
#include <chrono>
#include <cmath>
#include <span>

namespace bai {

InferencePipeline::InferencePipeline() = default;

std::expected<void, std::string> InferencePipeline::initialize(
    const EngineConfig& engine_config,
    const std::string& vocab_path
) {
    if (auto engine_result = engine_.initialize(engine_config); !engine_result) {
        return std::unexpected("Engine initialization failed: " + engine_result.error());
    }

    if (auto tokenizer_result = tokenizer_.initialize(vocab_path); !tokenizer_result) {
        return std::unexpected("Tokenizer initialization failed: " + tokenizer_result.error());
    }

    return {};
}

std::expected<PredictionResult, std::string> InferencePipeline::predict(const std::string& text) {
    if (!is_initialized()) {
        return std::unexpected("Pipeline not initialized. Call initialize() first.");
    }
    if (text.empty()) {
        return std::unexpected("Input text is empty.");
    }

    double execution_time_ms = 0.0;
    {
        ScopedTimer timer(execution_time_ms);

        const auto tokenized = tokenizer_.encode(text, kModelSequenceLength);
        if (!tokenized) {
            return std::unexpected("Tokenization failed: " + tokenized.error());
        }

        const auto infer_result = engine_.infer(
            std::span<const int64_t>(tokenized->input_ids),
            std::span<const int64_t>(tokenized->attention_mask)
        );
        if (!infer_result) {
            return std::unexpected("Inference failed: " + infer_result.error());
        }

        const auto& raw_result = infer_result.value();
        float max_logit = raw_result.category_logits[0];
        for (int i = 1; i < 5; ++i) {
            if (raw_result.category_logits[i] > max_logit) {
                max_logit = raw_result.category_logits[i];
            }
        }

        std::array<float, 5> exp_logits{};
        float sum = 0.0f;
        for (int i = 0; i < 5; ++i) {
            exp_logits[i] = std::exp(raw_result.category_logits[i] - max_logit);
            sum += exp_logits[i];
        }

        int best_category = 0;
        float best_prob = exp_logits[0] / sum;
        for (int i = 1; i < 5; ++i) {
            const float prob = exp_logits[i] / sum;
            if (prob > best_prob) {
                best_prob = prob;
                best_category = i;
            }
        }

        const float otp_confidence = 1.0f / (1.0f + std::exp(-raw_result.otp_logit));
        const bool otp_detected = raw_result.otp_logit > 0.0f;
        static constexpr const char* CATEGORIES[] = {"inbox_pinned", "inbox", "bait", "bais", "baiads"};

        return PredictionResult{
            CATEGORIES[best_category],
            best_prob,
            otp_detected,
            otp_confidence,
            raw_result.confidence,
            execution_time_ms,
            raw_result.category_logits,
            raw_result.otp_logit
        };
    }
}

std::expected<std::string, std::string> InferencePipeline::predict_json(const std::string& text) {
    const auto predict_result = predict(text);
    if (!predict_result) {
        return std::unexpected(std::string("Error: ") + predict_result.error());
    }

    return json_builder_.build_response(
        predict_result->category_logits,
        predict_result->otp_logit,
        predict_result->overall_confidence,
        predict_result->execution_time_ms
    );
}

bool InferencePipeline::is_initialized() const {
    return engine_.is_initialized() && tokenizer_.is_initialized();
}

}  // namespace bai
