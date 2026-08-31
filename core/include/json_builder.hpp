#pragma once

#include <array>
#include <cmath>
#include <string>

namespace bai {

class JsonBuilder {
public:
    JsonBuilder() = default;
    ~JsonBuilder() = default;

    JsonBuilder(const JsonBuilder&) = delete;
    JsonBuilder& operator=(const JsonBuilder&) = delete;
    JsonBuilder(JsonBuilder&&) noexcept = default;
    JsonBuilder& operator=(JsonBuilder&&) noexcept = default;

    static constexpr std::array<const char*, 5> CATEGORIES = {
        "inbox_pinned", "inbox", "bait", "bais", "baiads"
    };

    std::string build_response(
        const std::array<float, 5>& category_logits,
        float otp_logit,
        float confidence,
        double execution_time_ms
    ) const;

    std::string build_error(const std::string& error_message) const;

    std::string build_validation_error(
        const std::string& field_name,
        const std::string& reason
    ) const;

private:
    static constexpr const char* MODEL_VERSION = "1.0.0";

    std::array<float, 5> softmax_categories(const std::array<float, 5>& logits) const;

    static inline float sigmoid(float x) {
        return 1.0f / (1.0f + std::exp(-x));
    }

    std::string build_json_manually(
        const char* category,
        float category_conf,
        bool otp_detected,
        float otp_conf,
        float overall_conf,
        double exec_time
    ) const;
};

}  // namespace bai
