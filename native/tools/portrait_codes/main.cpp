#include <windows.h>
#include <onnxruntime_cxx_api.h>
#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/imgcodecs.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace {
namespace fs = std::filesystem;
constexpr int side = 512;
constexpr size_t plane = side * side;
constexpr std::array<int64_t, 4> rgb_shape{1, 3, side, side};
constexpr std::array<int64_t, 3> code_shape{1, 16, 16};
constexpr std::array<int64_t, 3> logits_shape{1, 256, 1024};

std::vector<char> read_bytes(const fs::path& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file)
        throw std::runtime_error("Cannot open input file");
    std::vector<char> bytes{std::istreambuf_iterator<char>(file), {}};
    if (file.bad())
        throw std::runtime_error("Cannot read input file");
    return bytes;
}

void write_bytes(const fs::path& path, const void* data, size_t length) {
    if (fs::exists(path))
        throw std::runtime_error("Output already exists");
    const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
    auto pending = path;
    pending += L"." + std::to_wstring(GetCurrentProcessId()) + L"." + std::to_wstring(stamp) + L".tmp";
    try {
        std::ofstream file(pending, std::ios::binary);
        file.write(static_cast<const char*>(data), static_cast<std::streamsize>(length));
        file.close();
        if (!file)
            throw std::runtime_error("Cannot save output file");
        if (!MoveFileExW(pending.c_str(), path.c_str(), MOVEFILE_WRITE_THROUGH))
            throw std::runtime_error("Cannot publish output file");
    } catch (...) {
        std::error_code ignored;
        fs::remove(pending, ignored);
        throw;
    }
}

cv::Mat read_image(const fs::path& path) {
    const auto bytes = read_bytes(path);
    if (bytes.empty() || bytes.size() > static_cast<size_t>(std::numeric_limits<int>::max()))
        throw std::invalid_argument("Invalid image file size");
    const cv::Mat data(1, static_cast<int>(bytes.size()), CV_8UC1, const_cast<char*>(bytes.data()));
    const auto img = cv::imdecode(data, cv::IMREAD_COLOR);
    if (img.size() != cv::Size(side, side) || img.type() != CV_8UC3)
        throw std::runtime_error("Expected 512 by 512 BGR input");
    return cv::dnn::blobFromImage(img, 1. / 127.5, {side, side}, cv::Scalar::all(127.5), true, false, CV_32F);
}

template<size_t N>
void require_tensor(const Ort::Value& tensor, ONNXTensorElementDataType element,
                    const std::array<int64_t, N>& expected) {
    const auto type = tensor.GetTensorTypeAndShapeInfo();
    const auto actual = type.GetShape();
    if (!std::ranges::equal(actual, expected) || type.GetElementType() != element)
        throw std::runtime_error("Unexpected ONNX result contract");
}

void save_face(const fs::path& path, const float* generated) {
    cv::Mat restored(side, side, CV_8UC3);
    for (int y = 0; y < side; ++y) {
        auto* row = restored.ptr<cv::Vec3b>(y);
        for (int x = 0; x < side; ++x) {
            for (int c = 0; c < 3; ++c) {
                const float v = generated[c * plane + y * side + x];
                if (!std::isfinite(v))
                    throw std::runtime_error("Nonfinite output pixel");
                row[x][2 - c] = static_cast<uint8_t>(std::clamp((v + 1.f) * .5f * 255.f + .5f, 0.f, 255.f));
            }
        }
    }
    std::vector<uint8_t> bytes;
    if (!cv::imencode(".png", restored, bytes))
        throw std::runtime_error("Cannot encode output PNG");
    write_bytes(path, bytes.data(), bytes.size());
}

void blend_codes(const fs::path& raw_path, const fs::path& seed_path, const fs::path& target) {
    constexpr int classes = 1024;
    constexpr int positions = 256;
    constexpr auto bytes = positions * classes * sizeof(float);
    const auto raw_bytes = read_bytes(raw_path);
    const auto seed_bytes = read_bytes(seed_path);
    if (raw_bytes.size() != bytes || seed_bytes.size() != bytes)
        throw std::invalid_argument("Expected two full grids of FP32 code scores");
    std::vector<float> raw(positions * classes), seed(positions * classes);
    std::memcpy(raw.data(), raw_bytes.data(), bytes);
    std::memcpy(seed.data(), seed_bytes.data(), bytes);
    std::array<int64_t, positions> codes{};
    for (int cell = 0; cell < positions; ++cell) {
        const float dx = ((cell % 16) * 32.f + 15.5f - 254.f) / 151.f;
        const float dy = ((cell / 16) * 32.f + 15.5f - 297.f) / 139.f;
        const float support = std::clamp((1.f - dx * dx - dy * dy) / .75f, 0.f, 1.f);
        const float weight = .03f + .79f * (support * support) * (3.f - 2.f * support);
        float best = -std::numeric_limits<float>::infinity();
        for (int c = 0; c < classes; ++c) {
            const auto offset = cell * classes + c;
            const float score = raw[offset] + (seed[offset] - raw[offset]) * weight;
            if (!std::isfinite(score))
                throw std::invalid_argument("Nonfinite blended code score");
            if (score > best) {
                best = score;
                codes[cell] = c;
            }
        }
    }
    write_bytes(target, codes.data(), sizeof(codes));
}

void run(int argc, wchar_t** argv) {
    const std::wstring_view mode(argv[1]);
    const std::wstring_view backend(argv[2]);
    const bool encode = mode == L"encode";
    const bool restore = mode == L"restore";
    const int stride = encode || restore ? 2 : 3;
    if ((!encode && !restore && mode != L"decode") || (backend != L"cpu" && backend != L"dml") ||
        argc < 4 + stride || (argc - 4) % stride != 0)
        throw std::invalid_argument("Expected encode/decode/restore cpu/dml model.onnx and complete image [code] output sets");

    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "portrait-exposure-codes");
    Ort::SessionOptions options;
    options.SetIntraOpNumThreads(1);
    options.SetInterOpNumThreads(1);
    options.SetExecutionMode(ORT_SEQUENTIAL);
    options.DisableMemPattern();
    options.DisableCpuMemArena();
    options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    options.EnableProfiling(backend == L"dml" ? L"codes-dml" : L"codes-cpu");
    if (backend == L"dml") {
        using Append = OrtStatus*(ORT_API_CALL*)(OrtSessionOptions*, int);
        const auto dll = GetModuleHandleW(L"onnxruntime.dll");
        const auto append = reinterpret_cast<Append>(GetProcAddress(dll, "OrtSessionOptionsAppendExecutionProvider_DML"));
        if (!append)
            throw std::runtime_error("DirectML provider missing");
        Ort::ThrowOnError(append(options, 0));
    }
    Ort::Session net(env, argv[3], options);
    const size_t input_count = encode || restore ? 1 : 2;
    if (net.GetInputCount() != input_count || net.GetOutputCount() != 1)
        throw std::invalid_argument("Wrong model for the requested route");
    Ort::AllocatorWithDefaultOptions allocator;
    const auto output = net.GetOutputNameAllocated(0, allocator);
    const auto output_ptr = output.get();
    const auto image_name = net.GetInputNameAllocated(0, allocator);
    const std::array<const char*, 2> names{restore ? image_name.get() : "image", "codes"};
    const auto memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    for (int i = 4; i < argc; i += stride) {
        auto blob = read_image(argv[i]);
        std::array<Ort::Value, 2> values{Ort::Value::CreateTensor<float>(memory, blob.ptr<float>(), blob.total(),
                                          rgb_shape.data(), rgb_shape.size()), Ort::Value{nullptr}};
        std::array<int64_t, 256> codes{};
        if (!encode && !restore) {
            const auto data = read_bytes(argv[i + 1]);
            if (data.size() != sizeof(codes))
                throw std::invalid_argument("Expected exactly 256 code indices in binary int64 form");
            std::memcpy(codes.data(), data.data(), data.size());
            if (!std::ranges::all_of(codes, [](int64_t x) { return x >= 0 && x < 1024; }))
                throw std::invalid_argument("Code index is outside [0, 1024)");
            values[1] = Ort::Value::CreateTensor<int64_t>(memory, codes.data(), codes.size(), code_shape.data(), code_shape.size());
        }
        const auto start = std::chrono::steady_clock::now();
        const auto out = net.Run(Ort::RunOptions{nullptr}, names.data(), values.data(), input_count, &output_ptr, 1);
        const auto seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        const auto* pixels = out[0].GetTensorData<float>();
        if (encode) {
            require_tensor(out[0], ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, logits_shape);
            if (!std::all_of(pixels, pixels + 256 * 1024, [](float x) { return std::isfinite(x); }))
                throw std::runtime_error("Nonfinite encoder output");
            write_bytes(argv[i + 1], pixels, 256 * 1024 * sizeof(float));
        } else {
            require_tensor(out[0], ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, rgb_shape);
            save_face(argv[i + (restore ? 1 : 2)], pixels);
        }
        std::cout << "seconds=" << seconds << '\n';
    }
    std::cout << "profile=" << net.EndProfilingAllocated(allocator).get() << '\n';
}
}

int wmain(int argc, wchar_t** argv) {
    try {
        cv::setNumThreads(1);
        if (argc > 1 && std::wstring_view(argv[1]) == L"validate") {
            if (argc != 3)
                throw std::invalid_argument("Expected validate image.png");
            read_image(argv[2]);
            return 0;
        }
        if (argc > 1 && std::wstring_view(argv[1]) == L"blend") {
            if (argc < 5 || (argc - 2) % 3 != 0)
                throw std::invalid_argument("Expected blend raw.f32 seed.f32 output.bin [...]");
            for (int i = 2; i < argc; i += 3)
                blend_codes(argv[i], argv[i + 1], argv[i + 2]);
            return 0;
        }
        if (argc < 6)
            throw std::invalid_argument("Expected mode, device, local model, input and output paths");
        run(argc, argv);
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
