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
#include <array>
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

ov::CompiledModel compileFloatGraph(ov::Core& core, const std::string& device, const fs::path& graph) {
    auto parameters = ov::AnyMap{ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY),
                                 ov::hint::inference_precision(ov::element::f32)};
    if (device == "CPU") parameters.emplace(ov::inference_num_threads.name(), 3);
    auto compiled = core.compile_model(graph.wstring(), device, parameters);
    const auto devices = compiled.get_property(ov::execution_devices);
    if (devices.empty() || !std::ranges::all_of(devices, [&](const std::string& d) { return d.starts_with(device); }) ||
        compiled.inputs().size() != 1 || compiled.outputs().size() != 1 ||
        compiled.input().get_element_type() != ov::element::f32 || compiled.output().get_element_type() != ov::element::f32) {
        throw std::runtime_error("The compiled portrait graph or execution device differs from the requested contract.");
    }
    return compiled;
}

ov::CompiledModel compilePortraitGraph(ov::Core& core, const std::string& device, const fs::path& graph,
                                       const ov::Shape& input, const ov::Shape& output) {
    auto compiled = compileFloatGraph(core, device, graph);
    if (compiled.input().get_shape() != input || compiled.output().get_shape() != output)
        throw std::runtime_error("Unexpected portrait graph dimensions.");
    return compiled;
}

template <class Write>
void publishPortraits(const fs::path& output, std::size_t count, Write write) {
    const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
    fs::path pending = output;
    pending += L".pending-" + std::to_wstring(stamp);
    if (fs::exists(output) || !fs::create_directory(pending))
        throw std::runtime_error("Use a new destination inside an existing directory.");
    const auto name = [](std::size_t index) { return "candidate-" + std::to_string(index + 1) + ".png"; };
    try {
        for (std::size_t index = 0; index < count; ++index)
            write(pending / name(index), index);
        fs::rename(pending, output);
    } catch (...) {
        std::error_code ignored;
        for (std::size_t i = 0; i < count; ++i)
            fs::remove(pending / name(i), ignored);
        fs::remove(pending, ignored);
        throw;
    }
}

cv::Mat rgbBlob(const cv::Mat& portrait, int side) {
    cv::Mat reduced;
    cv::resize(portrait, reduced, {side, side}, 0, 0, cv::INTER_AREA);
    return cv::dnn::blobFromImage(reduced, 1. / 255.0, {side, side}, {}, true);
}

cv::Mat bgr(const float* rgb, int side) {
    const auto plane = static_cast<std::size_t>(side) * side;
    // OpenCV's plane headers borrow this read-only input for the merge.
    auto* pixels = const_cast<float*>(rgb);
    const std::array<cv::Mat, 3> planes{{{side, side, CV_32F, pixels + 2 * plane},
                                        {side, side, CV_32F, pixels + plane}, {side, side, CV_32F, pixels}}};
    cv::Mat out;
    cv::merge(planes.data(), planes.size(), out);
    return out;
}

void renderRefinedSequence(ov::InferRequest& seed_session, ov::InferRequest& request, int side,
                           const std::string& device, const cv::Mat& portrait, const fs::path& output) {
    constexpr int seed_side = 240;
    const auto plane = static_cast<std::size_t>(side) * side;
    constexpr std::size_t seed_plane = seed_side * seed_side;
    constexpr std::array<std::size_t, 4> positions{0, 2, 4, 6};
    const bool original_resolution = side == portrait.rows;
    const cv::Mat tensor = rgbBlob(portrait, seed_side);
    std::copy_n(tensor.ptr<float>(), 3 * seed_plane, seed_session.get_input_tensor().data<float>());
    seed_session.infer();
    const auto raw_seeds = seed_session.get_output_tensor();
    const auto* raw_values = raw_seeds.data<const float>();
    if (!std::all_of(raw_values, raw_values + raw_seeds.get_size(), [](float v) { return std::isfinite(v); }))
        throw std::runtime_error("The sequence graph returned non-finite pixels.");
    std::vector<float> seed_values(raw_values, raw_values + raw_seeds.get_size());
    for (auto& pixel : seed_values) pixel = std::clamp(pixel, 0.f, 1.f);

    const auto resize_planes = [&](const float* source, float* target) {
        for (std::size_t channel = 0; channel < 3; ++channel) {
            const cv::Mat image(seed_side, seed_side, CV_32F, const_cast<float*>(source + channel * seed_plane));
            cv::Mat reduced(side, side, CV_32F, target + channel * plane);
            cv::resize(image, reduced, {side, side}, 0, 0, original_resolution ? cv::INTER_CUBIC : cv::INTER_AREA);
            if (original_resolution) {
                cv::Mat bytes;
                reduced.convertTo(bytes, CV_8U, 255.0);
                bytes.convertTo(reduced, CV_32F, 1. / 255.);
            }
        }
    };
    std::vector<float> reduced(21 * plane);
    for (std::size_t phase = 0; phase < 7; ++phase)
        resize_planes(seed_values.data() + phase * 3 * seed_plane, reduced.data() + phase * 3 * plane);
    auto* packed = request.get_input_tensor().data<float>();
    if (original_resolution) {
        const cv::Mat full_observation = rgbBlob(portrait, side);
        std::copy_n(full_observation.ptr<float>(), 3 * plane, packed);
    } else {
        resize_planes(tensor.ptr<float>(), packed);
    }
    for (std::size_t element = 0; element < 3 * plane; ++element) {
        float mean = 0.f, low = 1.f, high = 0.f;
        for (std::size_t phase = 0; phase < 7; ++phase) {
            const float value = reduced[element + phase * 3 * plane];
            mean += value;
            low = std::min(low, value);
            high = std::max(high, value);
        }
        packed[element + 6 * plane] = mean / 7.f;
        packed[element + 9 * plane] = high - low;
    }
    publishPortraits(output, positions.size(), [&](const fs::path& path, std::size_t index) {
        const auto started = std::chrono::steady_clock::now();
        const auto phase = positions[index];
        auto* conditioning = reduced.data() + phase * 3 * plane;
        std::copy_n(conditioning, 3 * plane, packed + 3 * plane);
        std::fill_n(packed + 12 * plane, plane, index == 0 || index == 3 ? 1.f : 1.f / 3.f);
        request.infer();
        const auto prediction = request.get_output_tensor();
        auto* values = prediction.data<float>();
        if (!std::all_of(values, values + 3 * plane, [](float v) { return std::isfinite(v); }))
            throw std::runtime_error("The refinement graph returned non-finite pixels.");
        cv::Mat seed, patch, original;
        if (original_resolution) {
            seed = bgr(conditioning, side);
            patch = bgr(values, side) - seed;
        } else {
            const cv::Mat delta = bgr(values, side) - bgr(conditioning, side);
            cv::resize(bgr(seed_values.data() + phase * 3 * seed_plane, seed_side), seed, portrait.size(), 0, 0,
                       cv::INTER_CUBIC);
            seed.convertTo(original, CV_8UC3, 255.0);
            original.convertTo(seed, CV_32FC3, 1. / 255.);
            cv::resize(delta, patch, portrait.size(), 0, 0, cv::INTER_CUBIC);
        }
        for (int y = 0; y < seed.rows; ++y) {
            auto* pixels = seed.ptr<cv::Vec3f>(y);
            const auto* edits = patch.ptr<cv::Vec3f>(y);
            const double dy = (y - 280.) / 207.;
            for (int x = 0; x < seed.cols; ++x) {
                const double dx = (x - 256.) / 189.;
                const float weight = static_cast<float>(std::clamp((1. - dx * dx - dy * dy) / .52, 0., 1.));
                pixels[x] += edits[x] * (weight * weight * (3.f - 2.f * weight));
            }
        }
        seed.convertTo(original, CV_8UC3, 255.0);
        writeImage(path, original);
        std::cout << path.filename().string() << " position=" << phase + 1 << " device=" << device
                  << " seconds=" << std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count() << '\n';
    });
}

void refineSequence(const std::string& device, wchar_t** argv, int argc) {
    constexpr int seed_side = 240;
    cv::setNumThreads(1);
    ov::Core core;
    auto seed_model = compilePortraitGraph(core, device, argv[3], {1, 3, seed_side, seed_side}, {1, 21, seed_side, seed_side});
    auto refine_model = compileFloatGraph(core, device, argv[4]);
    const auto input = refine_model.input().get_shape();
    if (input != ov::Shape{1, 13, 192, 192} && input != ov::Shape{1, 13, 512, 512})
        throw std::invalid_argument("Expected a 192 or 512 pixel refinement graph with 13 input planes.");
    const auto side = input[2];
    if (refine_model.output().get_shape() != ov::Shape{1, 3, side, side})
        throw std::invalid_argument("Expected an RGB refinement at the input resolution.");
    auto seed_request = seed_model.create_infer_request();
    auto refine_request = refine_model.create_infer_request();
    const auto executed = refine_model.get_property(ov::execution_devices).front();
    for (int arg = 5; arg < argc; arg += 2)
        renderRefinedSequence(seed_request, refine_request, static_cast<int>(side), executed,
                              readPortrait(argv[arg]), argv[arg + 1]);
}

void produce(const std::string& device, const fs::path& graph, const cv::Mat& portrait,
             const fs::path& output, const std::vector<float>& requested) {
    constexpr int side = 256;
    constexpr std::size_t plane = side * side;
    cv::setNumThreads(1);
    const cv::Mat tensor = rgbBlob(portrait, side);

    ov::Core core;
    core.add_extension(std::make_shared<ov::frontend::onnx::ConversionExtension>("DeformConv", convertDeformableConvolution));
    auto compiled = compilePortraitGraph(core, device, graph, {1, 4, side, side}, {1, 3, side, side});
    const auto devices = compiled.get_property(ov::execution_devices);
    auto request = compiled.create_infer_request();
    auto input = request.get_input_tensor();
    std::copy_n(tensor.ptr<float>(), 3 * plane, input.data<float>());
    publishPortraits(output, requested.size(), [&](const fs::path& path, std::size_t index) {
        const auto time = std::chrono::steady_clock::now();
        std::fill_n(input.data<float>() + 3 * plane, plane, requested[index]);
        request.infer();
        auto result = request.get_output_tensor();
        auto* values = result.data<float>();
        if (!std::all_of(values, values + 3 * plane, [](float x) { return std::isfinite(x); }))
            throw std::runtime_error("The model returned non-finite pixels.");
        cv::Mat expanded, image;
        cv::resize(bgr(values, side), expanded, portrait.size(), 0, 0, cv::INTER_CUBIC);
        expanded.convertTo(image, CV_8UC3, 255.0);
        writeImage(path, image);
        std::cout << path.filename().string() << " moment=" << requested[index] << " device=" << devices.front()
                  << " seconds=" << std::chrono::duration<double>(std::chrono::steady_clock::now() - time).count() << '\n';
    });
}
}

int wmain(int argc, wchar_t** argv) {
    try {
        const bool refine = argc > 1 && std::wstring_view(argv[1]) == L"--refine-sequence";
        if ((refine && (argc < 7 || argc % 2 == 0)) || (!refine && argc < 5))
            throw std::invalid_argument("Use PortraitMoments.exe CPU|GPU graph.onnx portrait.png output [moment ...]\n"
                                        "or PortraitMoments.exe --refine-sequence CPU|GPU sequence.onnx refinement.onnx "
                                        "portrait.png output [portrait2.png output2 ...]");
        const std::wstring_view device(argv[refine ? 2 : 1]);
        if (device != L"CPU" && device != L"GPU")
            throw std::invalid_argument("Expected CPU or GPU.");
        const std::string backend = device == L"CPU" ? "CPU" : "GPU";
        if (refine)
            refineSequence(backend, argv, argc);
        else
            produce(backend, argv[2], readPortrait(argv[3]), argv[4], moments(argc, argv));
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
