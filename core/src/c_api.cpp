#include "bai/c_api.h"
#include "bai/engine.hpp"
#include <cstring>
#include <memory>
#include <span>

/**
 * C API implementation wrapping BaiEngine C++ class.
 * Converts C-style error codes and memory management to C++ exceptions/expected.
 */

namespace {
    // Convert EngineConfig from C to C++
    bai::EngineConfig c_config_to_cpp(const BaiEngineConfig* c_config) {
        bai::EngineConfig cpp_config;
        if (c_config->model_path) {
            cpp_config.model_path = c_config->model_path;
        }
        cpp_config.num_threads = c_config->num_threads;
        cpp_config.device_id = c_config->device_id;
        cpp_config.use_gpu = c_config->use_gpu != 0;
        cpp_config.enable_fp16 = c_config->enable_fp16 != 0;
        cpp_config.max_seq_length = c_config->max_seq_length;
        cpp_config.graph_optimization_level = c_config->graph_optimization_level;
        return cpp_config;
    }
}

// Global error storage (thread-local for multi-threaded safety would be better)
thread_local static std::string last_error_message;

extern "C" {

BaiStatusCode bai_engine_create(
    const BaiEngineConfig* config,
    BaiEngineHandle* out_handle) {
    
    if (!config || !out_handle) {
        return BAI_ERROR_INVALID_PARAM;
    }

    try {
        // Create engine instance
        auto engine = std::make_unique<bai::BaiEngine>();
        
        // Initialize with config
        bai::EngineConfig cpp_config = c_config_to_cpp(config);
        auto init_result = engine->initialize(cpp_config);
        
        if (!init_result) {
            last_error_message = init_result.error();
            return BAI_ERROR_INIT_FAILED;
        }

        // Store opaque handle as a typed struct wrapper.
        auto* handle = new BaiEngineHandle_t{engine.release()};
        *out_handle = handle;
        return BAI_SUCCESS;

    } catch (const std::bad_alloc&) {
        last_error_message = "Memory allocation failed";
        return BAI_ERROR_MEMORY_ALLOC_FAILED;
    } catch (const std::exception& e) {
        last_error_message = e.what();
        return BAI_ERROR_INIT_FAILED;
    }
}

BaiStatusCode bai_engine_infer(
    BaiEngineHandle handle,
    const int64_t* input_ids,
    const int64_t* attention_mask,
    size_t seq_len,
    BaiInferenceResult* out_result) {
    
    if (!handle || !input_ids || !attention_mask || !out_result) {
        return BAI_ERROR_INVALID_PARAM;
    }

    if (seq_len == 0) {
        return BAI_ERROR_INVALID_PARAM;
    }

    try {
        // Cast opaque handle back to BaiEngine*.
        auto engine = handle->engine;
        
        if (!engine || !engine->is_initialized()) {
            last_error_message = "Engine not initialized";
            return BAI_ERROR_NOT_INITIALIZED;
        }

        // Create spans from C arrays
        auto input_ids_span = std::span<const int64_t>(input_ids, seq_len);
        auto attention_mask_span = std::span<const int64_t>(attention_mask, seq_len);

        // Run inference
        auto result = engine->infer(input_ids_span, attention_mask_span);
        
        if (!result) {
            last_error_message = result.error();
            return BAI_ERROR_INFERENCE_FAILED;
        }

        // Copy result to C struct
        const auto& inference_result = result.value();
        
        // Copy category logits
        for (size_t i = 0; i < 5; ++i) {
            out_result->category_logits[i] = inference_result.category_logits[i];
        }
        
        out_result->otp_logit = inference_result.otp_logit;
        out_result->confidence = inference_result.confidence;

        return BAI_SUCCESS;

    } catch (const std::exception& e) {
        last_error_message = e.what();
        return BAI_ERROR_INFERENCE_FAILED;
    }
}

BaiStatusCode bai_engine_destroy(BaiEngineHandle handle) {
    if (!handle) {
        return BAI_ERROR_INVALID_HANDLE;
    }

    try {
        auto engine = handle->engine;
        delete engine;
        delete handle;
        return BAI_SUCCESS;
    } catch (const std::exception& e) {
        last_error_message = e.what();
        return BAI_ERROR_INFERENCE_FAILED;
    }
}

const char* bai_get_error_message(BaiStatusCode status) {
    switch (status) {
        case BAI_SUCCESS:
            return "Success";
        case BAI_ERROR_INIT_FAILED:
            return "Engine initialization failed";
        case BAI_ERROR_INFERENCE_FAILED:
            return "Inference execution failed";
        case BAI_ERROR_INVALID_PARAM:
            return "Invalid parameter";
        case BAI_ERROR_MODEL_NOT_FOUND:
            return "Model file not found";
        case BAI_ERROR_MEMORY_ALLOC_FAILED:
            return "Memory allocation failed";
        case BAI_ERROR_INVALID_HANDLE:
            return "Invalid engine handle";
        case BAI_ERROR_NOT_INITIALIZED:
            return "Engine not initialized";
        default:
            return "Unknown error";
    }
}

const char* bai_get_version(void) {
    return "1.0.0";
}

}  // extern "C"

