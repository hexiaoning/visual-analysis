#include "inference_backend.h"

#include <atomic>
#include <memory>
#include <mutex>

#include <Windows.h>
#include <d3d12.h>
#include <dxgi1_4.h>
#include <dml_provider_factory.h>

namespace visual {
namespace {

struct ReleaseCom {
    void operator()(IUnknown* object) const { object->Release(); }
};

struct Adapter {
    int index{-1};
    size_t memory{};
    UINT vendor{};
    std::wstring name;
};

std::wstring environmentValue(const wchar_t* key) {
    const auto size = GetEnvironmentVariableW(key, nullptr, 0);
    if (!size) return {};
    std::wstring value(size, L'\0');
    const auto length = GetEnvironmentVariableW(key, value.data(), size);
    if (length >= size) return {};
    value.resize(length);
    return value;
}

const Adapter& selectedAdapter() {
    static const auto selected = [] {
        Adapter best;
        if (environmentValue(L"VISUAL_ANALYSIS_BACKEND") == L"cpu") return best;
        IDXGIFactory1* rawFactory{};
        if (FAILED(CreateDXGIFactory1(IID_PPV_ARGS(&rawFactory)))) return best;
        const std::unique_ptr<IDXGIFactory1, ReleaseCom> factory(rawFactory);
        for (UINT index = 0; ; ++index) {
            IDXGIAdapter1* rawAdapter{};
            if (FAILED(factory->EnumAdapters1(index, &rawAdapter))) break;
            const std::unique_ptr<IDXGIAdapter1, ReleaseCom> adapter(rawAdapter);
            DXGI_ADAPTER_DESC1 description{};
            if (FAILED(adapter->GetDesc1(&description)) || (description.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) ||
                FAILED(D3D12CreateDevice(adapter.get(), D3D_FEATURE_LEVEL_11_0, __uuidof(ID3D12Device), nullptr))) {
                continue;
            }
            if (best.index < 0 || description.DedicatedVideoMemory > best.memory) {
                best = {static_cast<int>(index), description.DedicatedVideoMemory,
                        description.VendorId, description.Description};
            }
        }
        return best;
    }();
    return selected;
}

std::mutex fallbackMutex;
std::wstring lastFallback;

}

bool gpuAvailable() { return selectedAdapter().index >= 0; }
bool intelGpuAvailable() { return gpuAvailable() && selectedAdapter().vendor == 0x8086; }

std::optional<std::uint64_t> gpuMemoryBudget() {
    const auto index=selectedAdapter().index;
    if(index<0) return std::nullopt;
    IDXGIFactory1* rawFactory{};
    if(FAILED(CreateDXGIFactory1(IID_PPV_ARGS(&rawFactory)))) return std::nullopt;
    const std::unique_ptr<IDXGIFactory1,ReleaseCom> factory(rawFactory);
    IDXGIAdapter1* rawAdapter{};
    if(FAILED(factory->EnumAdapters1(static_cast<UINT>(index),&rawAdapter))) return std::nullopt;
    const std::unique_ptr<IDXGIAdapter1,ReleaseCom> adapter(rawAdapter);
    IDXGIAdapter3* rawMemory{};
    if(FAILED(adapter->QueryInterface(IID_PPV_ARGS(&rawMemory)))) return std::nullopt;
    const std::unique_ptr<IDXGIAdapter3,ReleaseCom> memory(rawMemory);
    DXGI_QUERY_VIDEO_MEMORY_INFO usage{};
    if(FAILED(memory->QueryVideoMemoryInfo(0,DXGI_MEMORY_SEGMENT_GROUP_LOCAL,&usage))) return std::nullopt;
    return usage.Budget>usage.CurrentUsage?usage.Budget-usage.CurrentUsage:0;
}

std::wstring inferenceBackendLabel() {
    if (!gpuAvailable()) return L"CPU 兼容模式 · 无已启用的 DirectX 12 显卡";
    std::lock_guard lock(fallbackMutex);
    auto text = L"GPU · " + selectedAdapter().name;
    if (!lastFallback.empty()) text += L" · " + lastFallback + L" 使用 CPU";
    return text;
}

bool enableGpu(Ort::SessionOptions& options) {
    if (!gpuAvailable()) return false;
    const OrtDmlApi* dml{};
    Ort::ThrowOnError(Ort::GetApi().GetExecutionProviderApi("DML", ORT_API_VERSION,
                                                           reinterpret_cast<const void**>(&dml)));
    options.DisableMemPattern().SetExecutionMode(ORT_SEQUENTIAL);
    Ort::ThrowOnError(dml->SessionOptionsAppendExecutionProvider_DML(options, selectedAdapter().index));
    return true;
}

void reportGpuFallback(const std::filesystem::path& model) {
    std::lock_guard lock(fallbackMutex);
    lastFallback = model.stem().wstring();
}

std::filesystem::path inferenceProfile(const std::filesystem::path& model) {
    const auto directory = environmentValue(L"VISUAL_ANALYSIS_GPU_PROFILE");
    if (directory.empty()) return {};
    static std::atomic<unsigned int> sequence{};
    return std::filesystem::path(directory) /
        (model.stem().wstring() + L"-" + std::to_wstring(GetCurrentProcessId()) + L"-" +
         std::to_wstring(sequence.fetch_add(1)));
}

}
