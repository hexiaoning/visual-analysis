#pragma once

#include <array>
#include <filesystem>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include <onnxruntime_cxx_api.h>
#include <opencv2/core.hpp>

namespace visual {

class ModelSession {
public:
    explicit ModelSession(const std::filesystem::path& path)
        : session_(environment(), path.c_str(), options()) {
        Ort::AllocatorWithDefaultOptions allocator;
        for (size_t index = 0; index < session_.GetInputCount(); ++index) {
            inputNames_.emplace_back(session_.GetInputNameAllocated(index, allocator).get());
        }
        for (size_t index = 0; index < session_.GetOutputCount(); ++index) {
            outputNames_.emplace_back(session_.GetOutputNameAllocated(index, allocator).get());
        }
        for (const auto& name : inputNames_) inputs_.push_back(name.c_str());
        for (const auto& name : outputNames_) outputs_.push_back(name.c_str());
    }

    ModelSession(const ModelSession&) = delete;
    ModelSession& operator=(const ModelSession&) = delete;

    const std::vector<std::string>& inputNames() const { return inputNames_; }
    const std::vector<std::string>& outputNames() const { return outputNames_; }

    std::vector<int64_t> inputShape(size_t index) const {
        return session_.GetInputTypeInfo(index).GetTensorTypeAndShapeInfo().GetShape();
    }

    template<class T>
    static Ort::Value tensor(std::span<T> data, std::span<const int64_t> shape) {
        auto memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        return Ort::Value::CreateTensor<T>(memory, data.data(), data.size(), shape.data(), shape.size());
    }

    std::vector<Ort::Value> run(std::span<const Ort::Value> tensors) {
        if (tensors.size() != inputs_.size()) throw std::runtime_error("Unexpected ONNX input count.");
        return session_.Run(Ort::RunOptions{nullptr}, inputs_.data(), tensors.data(), tensors.size(),
                            outputs_.data(), outputs_.size());
    }

    std::vector<Ort::Value> run(cv::Mat blob) {
        if (blob.dims != 4 || blob.type() != CV_32F || !blob.isContinuous()) {
            throw std::runtime_error("Expected a contiguous NCHW float tensor.");
        }
        const std::array<int64_t, 4> shape{blob.size[0], blob.size[1], blob.size[2], blob.size[3]};
        const auto value = tensor<float>({blob.ptr<float>(), blob.total()}, shape);
        return run(std::span<const Ort::Value>(&value, 1));
    }

private:
    static Ort::Env& environment() {
        static auto env = [] {
            Ort::Env created{ORT_LOGGING_LEVEL_ERROR, "VisualAnalysis"};
            created.DisableTelemetryEvents();
            return created;
        }();
        return env;
    }

    static Ort::SessionOptions options() {
        Ort::SessionOptions options;
        options.SetIntraOpNumThreads(2);
        options.SetInterOpNumThreads(1);
        return options;
    }

    Ort::Session session_;
    std::vector<std::string> inputNames_, outputNames_;
    std::vector<const char*> inputs_, outputs_;
};

}
