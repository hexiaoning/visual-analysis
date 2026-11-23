#include "portrait_selection.h"

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <nlohmann/json.hpp>
#include <opencv2/imgcodecs.hpp>
#include <stdexcept>
#include <string_view>
#include <windows.h>

namespace fs = std::filesystem;
using portrait_selection::Result;
using portrait_selection::Route;

namespace {
cv::Mat read_image(const fs::path &input) {
    std::ifstream stream(input, std::ios::binary);
    if (!stream)
        throw std::runtime_error("Cannot open input image");
    std::vector<unsigned char> bytes((std::istreambuf_iterator<char>(stream)), {});
    const auto image = cv::imdecode(bytes, cv::IMREAD_COLOR);
    if (image.empty())
        throw std::runtime_error("Failed to decode input image");
    return image;
}

void write_image(const fs::path &output, const cv::Mat &image) {
    std::vector<unsigned char> bytes;
    if (!cv::imencode(".png", image, bytes))
        throw std::runtime_error("PNG encoding failed");
    std::ofstream stream(output, std::ios::binary);
    stream.write(reinterpret_cast<const char *>(bytes.data()), bytes.size());
    stream.close();
    if (!stream)
        throw std::runtime_error("Failed to save candidate PNG");
}

const char *route_name(Route name) { return name == Route::codeformer ? "codeformer" : "amd"; }

using Json = nlohmann::ordered_json;

Json routes(const std::vector<Route> &selection) {
    auto names = Json::array();
    for (Route route : selection) {
        names.push_back(route_name(route));
    }
    return names;
}

Json nullable(const std::optional<double> &value) { return value ? Json(*value) : Json(nullptr); }

Json evidence(const portrait_selection::Evidence &values) {
    return {{"clarity", values.clarity},
            {"clarity_vs_source", values.clarity_relative_to_input},
            {"replay_error", nullable(values.replay_error)}};
}

void write_result(const fs::path &file, const Result &r, bool fit_duplicate, const Json &outputs) {
    constexpr std::array<const char *, 5> regions{"jaw", "nose", "left_eye", "right_eye", "mouth"};
    Json shifts = Json::object();
    for (size_t i = 0; i < regions.size(); ++i) {
        shifts[regions[i]] = r.regional_shift[i];
    }
    Json report{{"selected", routes(r.selected(fit_duplicate))},
                {"previous_rule_selected", routes(r.original_display)},
                {"fitted_duplicate_rule_selected", routes(r.fitted_display)},
                {"fit_duplicate_enabled", fit_duplicate},
                {"input_clarity", r.input_clarity},
                {"relative_clarity", r.relative_clarity},
                {"codeformer", evidence(r.codeformer)},
                {"amd", evidence(r.amd)},
                {"regional_shift_px", shifts},
                {"replay_supported", r.replay_supported},
                {"duplicate_eligible", r.duplicate_eligible},
                {"amd_fraction", nullable(r.amd_fraction)},
                {"observed_luminance_png", "observed-luminance.png"},
                {"outputs", outputs}};
    std::ofstream stream(file, std::ios::binary);
    stream << report.dump(2) << '\n';
    stream.close();
    if (!stream)
        throw std::runtime_error("Failed to save selection.json");
}

fs::path publish(const fs::path &directory, const cv::Mat &cf, const cv::Mat &amd, const Result &r,
                 bool fit_duplicate) {
    fs::create_directories(directory);
    fs::path pending, complete;
    do {
        auto id = std::chrono::steady_clock::now().time_since_epoch().count();
        std::string name = "run-" + std::to_string(GetCurrentProcessId()) + "-" + std::to_string(id);
        complete = directory / name;
        pending = directory / (name + ".tmp");
    } while (fs::exists(complete) || !fs::create_directory(pending));
    try {
        write_image(pending / "observed-luminance.png", r.observed_image);
        Json outputs = Json::array();
        const auto &selection = r.selected(fit_duplicate);
        for (Route route : selection) {
            const std::string stem = "candidate-" + std::to_string(outputs.size() + 1);
            const auto &values = route == Route::codeformer ? r.codeformer : r.amd;
            write_image(pending / (stem + ".png"), route == Route::codeformer ? cf : amd);
            Json entry{
                {"route", route_name(route)}, {"png", stem + ".png"}, {"replay_luminance_png", nullptr}};
            if (!values.replay_image.empty()) {
                const auto replay = stem + "-replay.png";
                write_image(pending / replay, values.replay_image);
                entry["replay_luminance_png"] = replay;
            }
            outputs.push_back(std::move(entry));
        }
        write_result(pending / "selection.json", r, fit_duplicate, outputs);
        fs::rename(pending, complete);
    } catch (...) {
        // Only remove the private, uniquely created staging folder for this call.
        std::error_code ec;
        fs::remove_all(pending, ec);
        throw;
    }
    return complete;
}
} // namespace

int wmain(int argc, wchar_t **argv) {
    try {
        constexpr auto usage =
            "PortraitCandidates.exe [--fit-duplicate] landmarks.onnx source.png "
            "codeformer.png amd.png output-directory [source2.png codeformer2.png amd2.png output2 ...]";
        if (argc == 2 && std::wstring_view(argv[1]) == L"--help") {
            std::cout << usage << '\n';
            return 0;
        }
        const bool fit_duplicate = argc > 1 && std::wstring_view(argv[1]) == L"--fit-duplicate";
        const int model_arg = fit_duplicate ? 2 : 1;
        const int first_sample = model_arg + 1;
        if (argc - first_sample < 4 || (argc - first_sample) % 4 != 0)
            throw std::invalid_argument(usage);
        cv::setNumThreads(1);
        portrait_selection::Selector select(argv[model_arg]);
        for (int i = first_sample; i < argc; i += 4) {
            const auto source = read_image(argv[i]);
            const auto cf = read_image(argv[i + 1]);
            const auto amd = read_image(argv[i + 2]);
            const Result result = select.select(source, cf, amd);
            const fs::path path = fs::absolute(publish(argv[i + 3], cf, amd, result, fit_duplicate));
            const auto utf8 = path.u8string();
            std::cout.write(reinterpret_cast<const char *>(utf8.data()), utf8.size()) << std::endl;
        }
        return 0;
    } catch (std::exception const &e) {
        std::cerr << e.what() << std::endl;
        return 1;
    }
}
