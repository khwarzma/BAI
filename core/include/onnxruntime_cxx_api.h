#pragma once

#include <cstddef>
#include <cstdint>
#include <exception>
#include <memory>
#include <string>
#include <vector>

#define ORT_LOGGING_LEVEL_WARNING 0

namespace onnxruntime {

class RunOptions {
public:
    explicit RunOptions(void*) {}
};

enum GraphOptimizationLevel {
    ORT_DISABLE_ALL = 0,
    ORT_ENABLE_BASIC = 1,
    ORT_ENABLE_EXTENDED = 2,
    ORT_ENABLE_ALL = 3
};

enum AllocatorType {
    OrtArenaAllocator = 0
};

enum MemType {
    OrtMemTypeDefault = 0
};

class Exception : public std::exception {
public:
    explicit Exception(const std::string& message) : message_(message) {}
    const char* what() const noexcept override { return message_.c_str(); }
private:
    std::string message_;
};

class Env {
public:
    Env(int, const char*) {}
};

class SessionOptions {
public:
    void SetIntraOpNumThreads(int) {}
    void SetGraphOptimizationLevel(GraphOptimizationLevel) {}
    void AppendExecutionProvider_CPU() {}
};

class MemoryInfo {
public:
    static MemoryInfo CreateCpu(AllocatorType, MemType) { return MemoryInfo{}; }
};

class Value {
public:
    Value() = default;

    template <typename T>
    static Value CreateTensor(
        const MemoryInfo&, T* data, size_t len,
        const int64_t* shape, size_t shape_size
    ) {
        Value v;
        v.storage_ = std::shared_ptr<void>(data, [](void*) {});
        v.size_ = len;
        (void)shape;
        (void)shape_size;
        return v;
    }

    template <typename T>
    static Value CreateTensor(
        const MemoryInfo&, const T* data, size_t len,
        const int64_t* shape, size_t shape_size
    ) {
        Value v;
        v.storage_ = std::shared_ptr<void>(const_cast<T*>(data), [](void*) {});
        v.size_ = len;
        (void)shape;
        (void)shape_size;
        return v;
    }

    template <typename T>
    T* GetTensorMutableData() {
        return reinterpret_cast<T*>(storage_.get());
    }

    template <typename T>
    const T* GetTensorMutableData() const {
        return reinterpret_cast<const T*>(storage_.get());
    }

    bool empty() const { return storage_ == nullptr; }
    size_t size() const { return size_; }

    template <typename T>
    static Value CreateOwnedTensor(size_t count, const T& init = T{}) {
        auto* ptr = new T[count]();
        for (size_t i = 0; i < count; ++i) {
            ptr[i] = init;
        }
        Value v;
        v.storage_ = std::shared_ptr<void>(ptr, [](void* raw) {
            delete[] static_cast<int64_t*>(raw);
        });
        v.size_ = count;
        return v;
    }

private:
    std::shared_ptr<void> storage_;
    size_t size_ = 0;
};

class Session {
public:
    Session(const Env&, const char*, const SessionOptions&) {}
    Session(const Env&, const std::string&, const SessionOptions&) {}

    std::vector<Value> Run(
        const RunOptions&,
        const char* const*,
        const Value*,
        size_t,
        const char* const*,
        size_t
    ) const {
        std::vector<Value> outputs;
        outputs.emplace_back(Value::CreateOwnedTensor<float>(5, 0.0f));
        outputs.emplace_back(Value::CreateOwnedTensor<float>(1, 0.0f));
        outputs.emplace_back(Value::CreateOwnedTensor<float>(1, 1.0f));
        return outputs;
    }
};

}  // namespace onnxruntime

namespace Ort = onnxruntime;

constexpr onnxruntime::AllocatorType OrtArenaAllocator = onnxruntime::OrtArenaAllocator;
constexpr onnxruntime::MemType OrtMemTypeDefault = onnxruntime::OrtMemTypeDefault;
