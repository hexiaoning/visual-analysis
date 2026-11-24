#pragma once

#include <filesystem>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include <onnxruntime_cxx_api.h>

namespace visual {

class OpenVinoModel {
public:
    explicit OpenVinoModel(const std::filesystem::path& path);
    ~OpenVinoModel();
    std::vector<std::string> inputNames() const;
    std::vector<std::string> outputNames() const;
    std::vector<int64_t> inputShape(size_t index) const;
    std::vector<Ort::Value> run(std::span<const Ort::Value> inputs);

private:
    struct State;
    std::unique_ptr<State> state_;
};

}
