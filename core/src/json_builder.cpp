#include "json_builder.hpp"

#include <array>
#include <cmath>
#include <iomanip>
#include <sstream>

namespace bai {

std::string JsonBuilder::build_response(
    const std::array<float, 5>& category_logits,
    float otp_logit,
    float confidence,
    double execution_time_ms
) const {
    auto probs = softmax_categories(category_logits);

    int best_category = 0;
    float best_prob = probs[0];
    for (int i = 1; i < 5; ++i) {
        if (probs[i] > best_prob) {
            best_prob = probs[i];
            best_category = i;
        }
    }

    const bool otp_detected = otp_logit > 0.0f;
    const float otp_conf = sigmoid(otp_logit);

    return build_json_manually(
        CATEGORIES[best_category],
        best_prob,
        otp_detected,
        otp_conf,
        confidence,
        execution_time_ms
    );
}

std::string JsonBuilder::build_error(const std::string& error_message) const {
    std::ostringstream oss;
    oss << "{"
        << "\"error\":\"" << error_message << "\","
        << "\"status\":\"error\","
        << "\"model_version\":\"" << MODEL_VERSION << "\""
        << "}";
    return oss.str();
}

std::string JsonBuilder::build_validation_error(
    const std::string& field_name,
    const std::string& reason
) const {
    std::ostringstream oss;
    oss << "{"
        << "\"error\":\"Validation error in field '" << field_name << "'\","
        << "\"reason\":\"" << reason << "\","
        << "\"status\":\"validation_error\","
        << "\"model_version\":\"" << MODEL_VERSION << "\""
        << "}";
    return oss.str();
}

std::array<float, 5> JsonBuilder::softmax_categories(const std::array<float, 5>& logits) const {
    float max_logit = logits[0];
    for (int i = 1; i < 5; ++i) {
        if (logits[i] > max_logit) {
            max_logit = logits[i];
        }
    }

    std::array<float, 5> exp_logits{};
    float sum = 0.0f;
    for (int i = 0; i < 5; ++i) {
        exp_logits[i] = std::exp(logits[i] - max_logit);
        sum += exp_logits[i];
    }

    std::array<float, 5> probs{};
    for (int i = 0; i < 5; ++i) {
        probs[i] = exp_logits[i] / sum;
    }
    return probs;
}

std::string JsonBuilder::build_json_manually(
    const char* category,
    float category_conf,
    bool otp_detected,
    float otp_conf,
    float overall_conf,
    double exec_time
) const {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(4);
    oss << "{"
        << "\"category\":\"" << category << "\","
        << "\"category_confidence\":" << category_conf << ","
        << "\"otp_detected\":" << (otp_detected ? "true" : "false") << ","
        << "\"otp_confidence\":" << otp_conf << ","
        << "\"overall_confidence\":" << overall_conf << ","
        << "\"execution_time_ms\":" << exec_time << ","
        << "\"model_version\":\"" << MODEL_VERSION << "\""
        << "}";
    return oss.str();
}

}  // namespace bai
