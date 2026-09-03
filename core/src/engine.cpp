
#include "bai/engine.hpp"
#include <onnxruntime_cxx_api.h>
#include <algorithm>
#include <cstring>
#include <iostream>

namespace bai {

BaiEngine::BaiEngine() 
    : env_(nullptr), session_(nullptr), memory_info_(nullptr) {
}

BaiEngine::~BaiEngine() {
    // RAII cleanup - unique_ptr handles deallocation
}

BaiEngine::BaiEngine(BaiEngine&& other) noexcept
    : env_(std::move(other.env_)),
      session_(std::move(other.session_)),
      memory_info_(std::move(other.memory_info_)),
      config_(std::move(other.config_)),
      input_ids_buffer_(std::move(other.input_ids_buffer_)),
      attention_mask_buffer_(std::move(other.attention_mask_buffer_)),
      input_names_(std::move(other.input_names_)),
      output_names_(std::move(other.output_names_)) {
}

BaiEngine& BaiEngine::operator=(BaiEngine&& other) noexcept {
    if (this != &other) {
        env_ = std::move(other.env_);
        session_ = std::move(other.session_);
        memory_info_ = std::move(other.memory_info_);
        config_ = std::move(other.config_);
        input_ids_buffer_ = std::move(other.input_ids_buffer_);
        attention_mask_buffer_ = std::move(other.attention_mask_buffer_);
        input_names_ = std::move(other.input_names_);
        output_names_ = std::move(other.output_names_);
    }
    return *this;
}

std::expected<void, std::string> BaiEngine::initialize(
    const EngineConfig& config) {
    
    try {
        if (config.max_seq_length != kModelSequenceLength) {
            return std::unexpected(
                "max_seq_length must be " + std::to_string(kModelSequenceLength)
            );
        }
        if (config.num_threads <= 0) {
            return std::unexpected("num_threads must be positive");
        }
        config_ = config;

        // Initialize ONNX Runtime environment
        env_ = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "bai_engine");

        // Configure session options
        auto session_options = std::make_unique<Ort::SessionOptions>();
        session_options->SetIntraOpNumThreads(config.num_threads);
        session_options->SetGraphOptimizationLevel(
            static_cast<GraphOptimizationLevel>(config.graph_optimization_level)
        );

        // Enable GPU if requested
        if (config.use_gpu) {
            #ifdef ENABLE_CUDA
            OrtCUDAProviderOptions cuda_options{};
            cuda_options.device_id = config.device_id;
            session_options->AppendExecutionProvider_CUDA(cuda_options);
            #endif
        } else {
            // CPU is the default provider in the ONNX Runtime session.
        }

        // Create ONNX Runtime session
        const wchar_t* model_path_w = nullptr;
        #ifdef _WIN32
        // Convert model path to wide string on Windows
        std::wstring model_path_wide(config.model_path.begin(), config.model_path.end());
        model_path_w = model_path_wide.c_str();
        session_ = std::make_unique<Ort::Session>(
            *env_, model_path_w, *session_options
        );
        #else
        // Use char* directly on Unix-like systems
        session_ = std::make_unique<Ort::Session>(
            *env_, config.model_path.c_str(), *session_options
        );
        #endif

        // Create memory info for CPU
        memory_info_ = std::make_unique<Ort::MemoryInfo>(
            Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)
        );

        // Pre-allocate input/output buffers
        input_ids_buffer_.resize(config.max_seq_length);
        attention_mask_buffer_.resize(config.max_seq_length);

        // Cache input node names
        input_names_.clear();
        input_names_.push_back("input_ids");
        input_names_.push_back("attention_mask");

        // Cache output node names
        output_names_.clear();
        output_names_.push_back("logits_category");
        output_names_.push_back("logits_otp");
        output_names_.push_back("confidence");

        return {};
    } catch (const Ort::Exception& e) {
        return std::unexpected(
            std::string("ONNX Runtime error: ") + e.what()
        );
    } catch (const std::exception& e) {
        return std::unexpected(
            std::string("Initialization error: ") + e.what()
        );
    }
}

std::string BaiEngine::validate_inputs(
    std::span<const int64_t> input_ids,
    std::span<const int64_t> attention_mask) const {
    
    if (input_ids.empty()) {
        return "input_ids cannot be empty";
    }
    
    if (input_ids.size() != attention_mask.size()) {
        return "input_ids and attention_mask must have same length";
    }
    
    if (input_ids.size() > config_.max_seq_length) {
        return "Sequence length exceeds max_seq_length";
    }
    
    return "";  // Empty string = valid
}

std::expected<InferenceResult, std::string> BaiEngine::infer(
    std::span<const int64_t> input_ids,
    std::span<const int64_t> attention_mask) {
    
    if (!is_initialized()) {
        return std::unexpected("Engine not initialized");
    }

    // Validate inputs (non-critical path)
    std::string validation_error = validate_inputs(input_ids, attention_mask);
    if (!validation_error.empty()) {
        return std::unexpected(validation_error);
    }

    std::lock_guard lock(inference_mutex_);
    try {
        // Zero-allocation path: copy into pre-allocated buffers
        const size_t seq_len = input_ids.size();
        
        // Zero-initialize buffers (padding with zeros)
        std::fill(input_ids_buffer_.begin(), input_ids_buffer_.end(), 0);
        std::fill(attention_mask_buffer_.begin(), attention_mask_buffer_.end(), 0);
        
        // Copy actual data
        std::copy(input_ids.begin(), input_ids.end(), input_ids_buffer_.begin());
        std::copy(attention_mask.begin(), attention_mask.end(), 
                  attention_mask_buffer_.begin());

        // Create input tensors from pre-allocated buffers
        std::vector<int64_t> input_shape{1, static_cast<int64_t>(config_.max_seq_length)};
        
        auto input_ids_tensor = Ort::Value::CreateTensor<int64_t>(
            static_cast<const OrtMemoryInfo*>(*memory_info_),
            input_ids_buffer_.data(),
            input_ids_buffer_.size(),
            input_shape.data(),
            input_shape.size()
        );
        
        auto attention_mask_tensor = Ort::Value::CreateTensor<int64_t>(
            static_cast<const OrtMemoryInfo*>(*memory_info_),
            attention_mask_buffer_.data(),
            attention_mask_buffer_.size(),
            input_shape.data(),
            input_shape.size()
        );

        // Prepare input tensors
        std::vector<Ort::Value> input_tensors;
        input_tensors.emplace_back(std::move(input_ids_tensor));
        input_tensors.emplace_back(std::move(attention_mask_tensor));

        // Run inference
        auto output_tensors = session_->Run(
            Ort::RunOptions{nullptr},
            input_names_.data(),
            input_tensors.data(),
            input_tensors.size(),
            output_names_.data(),
            output_names_.size()
        );

        if (output_tensors.size() != 3) {
            return std::unexpected(
                "Expected 3 outputs, got " + std::to_string(output_tensors.size())
            );
        }

        // Extract results
        InferenceResult result{};
        
        // Extract category logits (batch_size=1, num_categories=5)
        float* category_data = output_tensors[0].GetTensorMutableData<float>();
        std::copy(category_data, category_data + 5, result.category_logits.begin());
        
        // Extract OTP logit (batch_size=1, 1)
        float* otp_data = output_tensors[1].GetTensorMutableData<float>();
        result.otp_logit = otp_data[0];
        
        // Extract confidence (batch_size=1, 1)
        float* confidence_data = output_tensors[2].GetTensorMutableData<float>();
        result.confidence = confidence_data[0];

        return result;

    } catch (const Ort::Exception& e) {
        return std::unexpected(
            std::string("ONNX Runtime inference error: ") + e.what()
        );
    } catch (const std::exception& e) {
        return std::unexpected(
            std::string("Inference error: ") + e.what()
        );
    }
}

}  // namespace bai
