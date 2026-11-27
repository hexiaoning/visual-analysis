#include <openvino/openvino.hpp>
#include <openvino/frontend/onnx/extension/conversion.hpp>
#include <openvino/op/add.hpp>
#include <openvino/op/constant.hpp>
#include <openvino/op/deformable_convolution.hpp>
#include <openvino/op/reshape.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace fs = std::filesystem;
namespace {

// The exporter supplies the ONNX DeformConv ordering and zero bias when
// the author checkpoint has no bias. Match padded bilinear sampling at edges.
ov::OutputVector convertDeformableConvolution(const ov::frontend::NodeContext& node) {
    if (node.get_input_size() != 5 || node.get_attribute<int64_t>("group", 1) != 1 ||
        node.get_attribute<int64_t>("offset_group", 1) != 1 ||
        node.get_attribute<std::vector<int64_t>>("strides") != std::vector<int64_t>{1, 1} ||
        node.get_attribute<std::vector<int64_t>>("dilations") != std::vector<int64_t>{1, 1} ||
        node.get_attribute<std::vector<int64_t>>("pads") != std::vector<int64_t>{1, 1, 1, 1}) {
        throw std::invalid_argument("Unexpected deformable convolution contract.");
    }
    auto features = std::make_shared<ov::op::v8::DeformableConvolution>(
        node.get_input(0), node.get_input(2), node.get_input(1), node.get_input(4),
        ov::Strides{1, 1}, ov::CoordinateDiff{1, 1}, ov::CoordinateDiff{1, 1},
        ov::Strides{1, 1}, ov::op::PadType::EXPLICIT, 1, 1, true);
    auto shape = ov::op::v0::Constant::create(ov::element::i64, {4}, {1, -1, 1, 1});
    auto bias = std::make_shared<ov::op::v1::Reshape>(node.get_input(3), shape, false);
    return {std::make_shared<ov::op::v1::Add>(features, bias)};
}

cv::Mat readPortrait(const fs::path& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file)
        throw std::runtime_error("Cannot read the input image.");
    const std::vector<unsigned char> data{std::istreambuf_iterator<char>(file), {}};
    cv::Mat image = cv::imdecode(data, cv::IMREAD_COLOR);
    if (image.size() != cv::Size(512, 512))
        throw std::invalid_argument("An aligned 512x512 portrait is required.");
    return image;
}

void writeImage(const fs::path& path, const cv::Mat& image) {
    std::vector<unsigned char> png;
    if (!cv::imencode(".png", image, png))
        throw std::runtime_error("Cannot encode the candidate image.");
    std::ofstream file(path, std::ios::binary);
    file.write(reinterpret_cast<const char*>(png.data()), static_cast<std::streamsize>(png.size()));
    file.close();
    if (!file)
        throw std::runtime_error("Cannot save the candidate image.");
}

std::vector<float> moments(int argc, wchar_t** argv) {
    if (argc == 5)
        return {0.f, .5f, 1.f};
    std::vector<float> results;
    for (int i = 5; i < argc; ++i) {
        const std::wstring text(argv[i]);
        std::size_t used{};
        const float value = std::stof(text, &used);
        if (used != text.size() || !std::isfinite(value) || value < 0.f || value > 1.f)
            throw std::invalid_argument("Each moment must be in [0, 1].");
        if (std::find(results.begin(), results.end(), value) == results.end())
            results.push_back(value);
    }
    return results;
}

void produce(const std::string& device, const fs::path& graph, const cv::Mat& portrait,
             const fs::path& output, const std::vector<float>& requested) {
    constexpr int side = 256;
    constexpr std::size_t plane = side * side;
    cv::setNumThreads(1);
    cv::Mat reduced;
    cv::resize(portrait, reduced, {side, side}, 0, 0, cv::INTER_AREA);
    const cv::Mat tensor = cv::dnn::blobFromImage(reduced, 1. / 255.0, {side, side}, {}, true);

    ov::Core core;
    core.add_extension(std::make_shared<ov::frontend::onnx::ConversionExtension>("DeformConv", convertDeformableConvolution));
    auto parameters = ov::AnyMap{ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY),
                                 ov::hint::inference_precision(ov::element::f32)};
    if (device == "CPU") parameters.emplace(ov::inference_num_threads.name(), 3);
    auto compiled = core.compile_model(graph.wstring(), device, parameters);
    const auto devices = compiled.get_property(ov::execution_devices);
    if (devices.empty() || !std::ranges::all_of(devices, [&](const std::string& d) { return d.starts_with(device); }) ||
        compiled.input().get_shape() != ov::Shape{1, 4, side, side} ||
        compiled.output().get_shape() != ov::Shape{1, 3, side, side} ||
        compiled.input().get_element_type() != ov::element::f32 || compiled.output().get_element_type() != ov::element::f32) {
        throw std::runtime_error("The compiled portrait graph or execution device differs from the requested contract.");
    }

    const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
    fs::path pending = output;
    pending += L".pending-" + std::to_wstring(stamp);
    if (fs::exists(output) || !fs::create_directory(pending))
        throw std::runtime_error("Use a new destination inside an existing directory.");
    try {
        auto request = compiled.create_infer_request();
        auto input = request.get_input_tensor();
        std::copy_n(tensor.ptr<float>(), 3 * plane, input.data<float>());
        for (std::size_t index = 0; index < requested.size(); ++index) {
            const auto time = std::chrono::steady_clock::now();
            std::fill_n(input.data<float>() + 3 * plane, plane, requested[index]);
            request.infer();
            auto result = request.get_output_tensor();
            auto* values = result.data<float>();
            if (!std::all_of(values, values + 3 * plane, [](float x) { return std::isfinite(x); }))
                throw std::runtime_error("The model returned non-finite pixels.");
            const std::vector<cv::Mat> planes{
                {side, side, CV_32F, values + 2 * plane}, {side, side, CV_32F, values + plane}, {side, side, CV_32F, values}};
            cv::Mat small, expanded, image;
            cv::merge(planes, small);
            cv::resize(small, expanded, portrait.size(), 0, 0, cv::INTER_CUBIC);
            expanded.convertTo(image, CV_8UC3, 255.0);
            const std::string name = "candidate-" + std::to_string(index + 1) + ".png";
            writeImage(pending / name, image);
            std::cout << name << " moment=" << requested[index] << " device=" << devices.front()
                      << " seconds=" << std::chrono::duration<double>(std::chrono::steady_clock::now() - time).count()
                      << '\n';
        }
        fs::rename(pending, output);
    } catch (...) {
        std::error_code ignored;
        for (std::size_t i = 0; i < requested.size(); ++i)
            fs::remove(pending / ("candidate-" + std::to_string(i + 1) + ".png"), ignored);
        fs::remove(pending, ignored);
        throw;
    }
}
}

int wmain(int argc, wchar_t** argv) {
    try {
        if (argc < 5)
            throw std::invalid_argument("Use PortraitMoments.exe CPU|GPU graph.onnx portrait.png output [moment ...]");
        const std::wstring_view device(argv[1]);
        if (device != L"CPU" && device != L"GPU")
            throw std::invalid_argument("Expected CPU or GPU.");
        produce(device == L"CPU" ? "CPU" : "GPU", argv[2], readPortrait(argv[3]), argv[4], moments(argc, argv));
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
