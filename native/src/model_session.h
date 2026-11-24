#pragma once

#include "inference_backend.h"
#include "openvino_model.h"

#include <array>
#include <filesystem>
#include <fstream>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include <onnxruntime_cxx_api.h>
#include <opencv2/core.hpp>

namespace visual {

class ModelSession {
public:
    explicit ModelSession(const std::filesystem::path& path, bool preferGpu = true)
        : path_(path), profile_(inferenceProfile(path)), usingGpu_(preferGpu && gpuAvailable()),
          useCpuArena_(preferGpu),
          session_(nullptr) {
        if(usingGpu_ && intelGpuAvailable() && usesAttentionPipeline()) {
            try { openVino_=std::make_unique<OpenVinoModel>(path); }
            catch(const std::exception&) { }
        }
        if(openVino_) {
            inputNames_=openVino_->inputNames();
            outputNames_=openVino_->outputNames();
        } else {
            session_=createSession();
            Ort::AllocatorWithDefaultOptions allocator;
            for (size_t index = 0; index < session_.GetInputCount(); ++index) {
                inputNames_.emplace_back(session_.GetInputNameAllocated(index, allocator).get());
            }
            for (size_t index = 0; index < session_.GetOutputCount(); ++index) {
                outputNames_.emplace_back(session_.GetOutputNameAllocated(index, allocator).get());
            }
        }
        for (const auto& name : inputNames_) inputs_.push_back(name.c_str());
        for (const auto& name : outputNames_) outputs_.push_back(name.c_str());
    }

    ModelSession(const ModelSession&) = delete;
    ModelSession& operator=(const ModelSession&) = delete;

    ~ModelSession() { finishProfile(); }

    const std::vector<std::string>& inputNames() const { return inputNames_; }
    const std::vector<std::string>& outputNames() const { return outputNames_; }

    std::vector<int64_t> inputShape(size_t index) const {
        if(openVino_) return openVino_->inputShape(index);
        return session_.GetInputTypeInfo(index).GetTensorTypeAndShapeInfo().GetShape();
    }

    template<class T>
    static Ort::Value tensor(std::span<T> data, std::span<const int64_t> shape) {
        auto memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        return Ort::Value::CreateTensor<T>(memory, data.data(), data.size(), shape.data(), shape.size());
    }

    std::vector<Ort::Value> run(std::span<const Ort::Value> tensors) {
        if (tensors.size() != inputs_.size()) throw std::runtime_error("Unexpected ONNX input count.");
        try {
            auto output=evaluate(tensors);
            if(usingGpu_) ++gpuRuns_;
            return output;
        } catch (const std::exception& error) {
            if (!usingGpu_) throw;
            finishProfile();
            openVino_.reset();
            session_ = Ort::Session{nullptr};
            usingGpu_ = false;
            reportGpuFallback(path_);
            recordFallback(error);
            session_ = createSession();
            return evaluate(tensors);
        }
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

    Ort::SessionOptions options() {
        Ort::SessionOptions options;
        options.SetIntraOpNumThreads(2);
        options.SetInterOpNumThreads(1);
        if(!useCpuArena_) options.DisableCpuMemArena();
        if(usingGpu_ && usesAttentionPipeline()) options.AddConfigEntry("ep.dml.disable_graph_fusion","1");
        if (usingGpu_) enableGpu(options);
        if (!profile_.empty()) options.EnableProfiling(profile_.c_str());
        return options;
    }

    bool usesAttentionPipeline() const {
        const auto name=path_.stem().wstring();
        return name.starts_with(L"chinese_clip_");
    }

    Ort::Session createSession() {
        try {
            return Ort::Session(environment(), path_.c_str(), options());
        } catch (const Ort::Exception& error) {
            if (!usingGpu_) throw;
            usingGpu_ = false;
            reportGpuFallback(path_);
            recordFallback(error);
            return Ort::Session(environment(), path_.c_str(), options());
        }
    }

    std::vector<Ort::Value> evaluate(std::span<const Ort::Value> tensors) {
        if(openVino_) return openVino_->run(tensors);
        return session_.Run(Ort::RunOptions{nullptr}, inputs_.data(), tensors.data(), tensors.size(),
                            outputs_.data(), outputs_.size());
    }

    void finishProfile() noexcept {
        if (profile_.empty()) return;
        try {
            if(!openVino_) {
                Ort::AllocatorWithDefaultOptions allocator;
                session_.EndProfilingAllocated(allocator);
            }
            if(gpuRuns_>0) {
                const auto suffix=openVino_?L".ovgpu":L".gpu";
                std::ofstream(std::filesystem::path(profile_.native()+suffix)) << gpuRuns_;
            }
        } catch (const std::exception&) {
        }
    }

    void recordFallback(const std::exception& error) const {
        if(!profile_.empty()) std::ofstream(std::filesystem::path(profile_.native()+L".fallback")) << error.what();
    }

    std::filesystem::path path_, profile_;
    bool usingGpu_;
    bool useCpuArena_;
    size_t gpuRuns_{};
    Ort::Session session_;
    std::unique_ptr<OpenVinoModel> openVino_;
    std::vector<std::string> inputNames_, outputNames_;
    std::vector<const char*> inputs_, outputs_;
};

}
