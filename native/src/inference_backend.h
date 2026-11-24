#pragma once

#include <filesystem>
#include <cstdint>
#include <optional>
#include <string>

#include <onnxruntime_cxx_api.h>

namespace visual {

bool gpuAvailable();
bool intelGpuAvailable();
std::optional<std::uint64_t> gpuMemoryBudget();
std::wstring inferenceBackendLabel();
bool enableGpu(Ort::SessionOptions& options);
void reportGpuFallback(const std::filesystem::path& model);
std::filesystem::path inferenceProfile(const std::filesystem::path& model);

}
