#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <span>
#include <stdexcept>
#include <vector>

#include "bai/engine.hpp"
#include "inference_engine.hpp"

namespace py = pybind11;
using namespace bai;

PYBIND11_MODULE(bai_core, m) {
    m.doc() = "BAI Core - Micro-Transformer Inference Engine Python Bindings";

    py::class_<EngineConfig>(m, "EngineConfig")
        .def(py::init<>())
        .def_readwrite("model_path", &EngineConfig::model_path)
        .def_readwrite("num_threads", &EngineConfig::num_threads)
        .def_readwrite("device_id", &EngineConfig::device_id)
        .def_readwrite("use_gpu", &EngineConfig::use_gpu)
        .def_readwrite("enable_fp16", &EngineConfig::enable_fp16)
        .def_readwrite("max_seq_length", &EngineConfig::max_seq_length)
        .def_readwrite("graph_optimization_level", &EngineConfig::graph_optimization_level);

    py::class_<InferenceResult>(m, "InferenceResult")
        .def(py::init<>())
        .def_readwrite("category_logits", &InferenceResult::category_logits)
        .def_readwrite("otp_logit", &InferenceResult::otp_logit)
        .def_readwrite("confidence", &InferenceResult::confidence);

    py::class_<PredictionResult>(m, "PredictionResult")
        .def(py::init<>())
        .def_readwrite("category", &PredictionResult::category)
        .def_readwrite("category_confidence", &PredictionResult::category_confidence)
        .def_readwrite("otp_detected", &PredictionResult::otp_detected)
        .def_readwrite("otp_confidence", &PredictionResult::otp_confidence)
        .def_readwrite("overall_confidence", &PredictionResult::overall_confidence)
        .def_readwrite("execution_time_ms", &PredictionResult::execution_time_ms);

    py::class_<BaiEngine>(m, "BaiEngine")
        .def(py::init<>())
        .def("initialize", [](BaiEngine& self, const EngineConfig& config) {
            auto result = self.initialize(config);
            if (!result) {
                throw std::runtime_error(result.error());
            }
        })
        .def("infer", [](BaiEngine& self,
            const std::vector<int64_t>& input_ids,
            const std::vector<int64_t>& attention_mask) {
            auto result = self.infer(
                std::span<const int64_t>(input_ids),
                std::span<const int64_t>(attention_mask)
            );
            if (!result) {
                throw std::runtime_error(result.error());
            }
            return result.value();
        })
        .def("is_initialized", &BaiEngine::is_initialized);

    py::class_<InferencePipeline>(m, "InferencePipeline")
        .def(py::init<>())
        .def("initialize", [](InferencePipeline& self, const EngineConfig& config, const std::string& vocab_path) {
            auto result = self.initialize(config, vocab_path);
            if (!result) {
                throw std::runtime_error(result.error());
            }
        })
        .def("predict", [](InferencePipeline& self, const std::string& text) {
            auto result = self.predict(text);
            if (!result) {
                throw std::runtime_error(result.error());
            }
            return result.value();
        })
        .def("predict_json", [](InferencePipeline& self, const std::string& text) {
            auto result = self.predict_json(text);
            if (!result) {
                throw std::runtime_error(result.error());
            }
            return result.value();
        })
        .def("is_initialized", &InferencePipeline::is_initialized);

    m.def("get_version", []() { return "1.0.0"; });
}
