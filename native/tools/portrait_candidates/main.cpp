#include "portrait_selection.h"
#include "exposure_normalizer.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <nlohmann/json.hpp>
#include <opencv2/imgcodecs.hpp>
#include <stdexcept>
#include <string_view>
#include <utility>
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

void publish_exposure(const fs::path &output, const cv::Mat &image) {
    const auto timestamp = std::chrono::steady_clock::now().time_since_epoch().count();
    fs::path pending = output;
    pending += ".tmp-" + std::to_string(GetCurrentProcessId()) + "-" + std::to_string(timestamp);
    try {
        write_image(pending, image);
        if (!MoveFileExW(pending.c_str(), output.c_str(), 0))
            throw std::runtime_error("Could not publish exposure output as a new file");
    } catch (...) {
        std::error_code ignored;
        fs::remove(pending, ignored);
        throw;
    }
}

const char *route_name(Route name) { return name == Route::codeformer ? "codeformer" : "amd"; }

using Json = nlohmann::ordered_json;

portrait_selection::TextureGuard read_guard(const fs::path &path) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream)
        throw std::runtime_error("Cannot open texture guard model");
    const auto json = Json::parse(stream);
    portrait_selection::TextureGuard guard{};
    const auto &coefficients = json.at("coefficients");
    if (json.at("schema") != "portrait-texture-linear-v1" ||
        json.at("feature_count") != guard.feature_count || !coefficients.is_array() ||
        coefficients.size() != guard.feature_count)
        throw std::invalid_argument("Expected a 96-feature portrait texture model, version 1");
    const auto number = [](const Json &v) {
        if (!v.is_number() || !std::isfinite(v.get<double>()))
            throw std::invalid_argument("Invalid texture model coefficient");
        return v.get<double>();
    };
    guard.bias = number(json.at("bias"));
    guard.cutoff = number(json.at("cutoff"));
    std::transform(coefficients.begin(), coefficients.end(), guard.coefficients.begin(), number);
    return guard;
}

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

void write_result(const fs::path &file, const Result &r, bool fit_duplicate, const std::vector<Route> &selection,
                  const Json &outputs, std::optional<size_t> compared_alternate) {
    constexpr std::array<const char *, 5> regions{"jaw", "nose", "left_eye", "right_eye", "mouth"};
    Json shifts = Json::object();
    for (size_t i = 0; i < regions.size(); ++i) {
        shifts[regions[i]] = r.regional_shift[i];
    }
    Json report{
        {"selected", routes(selection)},
        {"previous_rule_selected", routes(r.original_display)},
        {"fitted_duplicate_rule_selected", routes(r.fitted_display)},
        {"fit_duplicate_enabled", fit_duplicate},
        {"mode", r.guarded_display ? "texture-guard" : (fit_duplicate ? "fit-duplicate" : "default")},
        {"texture_guard_selected", r.guarded_display ? routes(*r.guarded_display) : Json(nullptr)},
        {"replacement_texture_score", nullable(r.replacement_texture_score)},
        {"texture_cutoff", nullable(r.texture_cutoff)},
        {"input_clarity", r.input_clarity},
        {"relative_clarity", r.relative_clarity},
        {"codeformer", evidence(r.codeformer)},
        {"amd", evidence(r.amd)},
        {"regional_shift_px", shifts},
        {"visible_shape_difference", r.visible_shape_difference},
        {"replay_supported", r.replay_supported},
        {"duplicate_eligible", r.duplicate_eligible},
        {"amd_fraction", nullable(r.amd_fraction)},
        {"observed_luminance_png", "observed-luminance.png"},
        {"outputs", outputs}};
    if (compared_alternate) {
        report["mode"] = "select-by-replay";
        report["compared_alternate_index"] = *compared_alternate;
    }
    std::ofstream stream(file, std::ios::binary);
    stream << report.dump(2) << '\n';
    stream.close();
    if (!stream)
        throw std::runtime_error("Failed to save selection.json");
}

fs::path publish(const fs::path &directory, const cv::Mat &cf, const cv::Mat &amd, const Result &r,
                 bool fit_duplicate, const std::vector<Route> &selection,
                 std::optional<size_t> compared_alternate = std::nullopt) {
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
        write_result(pending / "selection.json", r, fit_duplicate, selection, outputs, compared_alternate);
        fs::rename(pending, complete);
    } catch (...) {
        // Only remove the private, uniquely created staging folder for this call.
        std::error_code ec;
        fs::remove_all(pending, ec);
        throw;
    }
    return complete;
}

enum class ReplayChoice { primary, distinct_pair, replacement };

ReplayChoice choose_by_replay(const Result &result) {
    if (result.amd.clarity_relative_to_input < 1.20)
        return ReplayChoice::primary;
    if (result.duplicate_eligible && result.amd_fraction.value_or(0.) > .50 &&
        result.amd.replay_error && result.codeformer.replay_error &&
        *result.amd.replay_error <= *result.codeformer.replay_error)
        return ReplayChoice::replacement;
    return result.original_display.size() == 2 ? ReplayChoice::distinct_pair : ReplayChoice::primary;
}

fs::path select_by_replay(wchar_t **argv, int argc) {
    cv::setNumThreads(1);
    portrait_selection::Selector selector(argv[2]);
    const auto source = read_image(argv[3]);
    const auto primary = read_image(argv[4]);
    struct Alternate {
        size_t index;
        cv::Mat image;
        Result result;
        ReplayChoice choice;
    };
    std::optional<Alternate> best;
    for (int arg = 6; arg < argc; ++arg) {
        auto image = read_image(argv[arg]);
        auto result = selector.select(source, primary, image);
        const auto choice = choose_by_replay(result);
        const bool better_replay = best && choice == best->choice && choice != ReplayChoice::primary &&
                                   result.amd.replay_error < best->result.amd.replay_error;
        if (!best || choice > best->choice || better_replay)
            best = Alternate{static_cast<size_t>(arg - 5), std::move(image), std::move(result), choice};
    }
    const auto &alternate = best.value();
    std::vector<Route> selection{Route::codeformer};
    if (alternate.choice == ReplayChoice::replacement)
        selection.front() = Route::amd;
    else if (alternate.choice == ReplayChoice::distinct_pair)
        selection.push_back(Route::amd);
    return publish(argv[5], primary, alternate.image, alternate.result, true, selection, alternate.index);
}

void print_output_path(const fs::path &path) {
    const auto utf8 = fs::absolute(path).u8string();
    std::cout.write(reinterpret_cast<const char *>(utf8.data()), utf8.size()) << std::endl;
}
} // namespace

int wmain(int argc, wchar_t **argv) {
    try {
        constexpr auto usage =
            "PortraitCandidates.exe [--fit-duplicate | --texture-guard model.json] landmarks.onnx "
            "source.png "
            "codeformer.png amd.png output-directory [source2.png codeformer2.png amd2.png output2 ...]\n"
            "PortraitCandidates.exe --normalize-exposure model.json source.png output.png\n"
            "PortraitCandidates.exe --match-candidate-tone candidate.png guide.png output.png\n"
            "PortraitCandidates.exe --select-by-replay landmarks.onnx source.png primary.png output-directory "
            "alternate.png [alternate2.png ...]";
        if (argc == 2 && std::wstring_view(argv[1]) == L"--help") {
            std::cout << usage << '\n';
            return 0;
        }
        const std::wstring_view option = argc > 1 ? argv[1] : L"";
        if (option == L"--select-by-replay") {
            if (argc < 7)
                throw std::invalid_argument(usage);
            print_output_path(select_by_replay(argv, argc));
            return 0;
        }
        if (option == L"--match-candidate-tone") {
            if (argc != 5)
                throw std::invalid_argument(usage);
            const fs::path output(argv[4]);
            if (fs::exists(output))
                throw std::invalid_argument("Tone-matched output must be a new path");
            cv::setNumThreads(1);
            const auto candidate = read_image(argv[2]);
            const auto guide = read_image(argv[3]);
            publish_exposure(output, portrait_selection::match_candidate_tone(candidate, guide));
            return 0;
        }
        if (option == L"--normalize-exposure") {
            if (argc != 5)
                throw std::invalid_argument(usage);
            const fs::path output(argv[4]);
            if (fs::exists(output))
                throw std::invalid_argument("Exposure output must be a new path");
            cv::setNumThreads(1);
            const auto result = portrait_selection::normalize_exposure(read_image(argv[3]), argv[2]);
            publish_exposure(output, result.image);
            std::cout << Json{{"applied", result.applied}, {"gain_bgr", result.gain_bgr},
                              {"black_level", result.black_level}, {"evidence", result.evidence},
                              {"recovered_grid", result.recovered_grid}}.dump() << '\n';
            return 0;
        }
        const bool protect = option == L"--texture-guard";
        const bool fit_duplicate = protect || option == L"--fit-duplicate";
        const int model_arg = protect ? 3 : (fit_duplicate ? 2 : 1);
        const int first_sample = model_arg + 1;
        if (argc - first_sample < 4 || (argc - first_sample) % 4 != 0)
            throw std::invalid_argument(usage);
        cv::setNumThreads(1);
        const auto guard = protect ? std::optional(read_guard(argv[2])) : std::nullopt;
        portrait_selection::Selector select(argv[model_arg]);
        for (int i = first_sample; i < argc; i += 4) {
            const auto source = read_image(argv[i]);
            const auto cf = read_image(argv[i + 1]);
            const auto amd = read_image(argv[i + 2]);
            const Result result = select.select(source, cf, amd, guard ? &*guard : nullptr);
            print_output_path(publish(argv[i + 3], cf, amd, result, fit_duplicate, result.selected(fit_duplicate)));
        }
        return 0;
    } catch (std::exception const &e) {
        std::cerr << e.what() << std::endl;
        return 1;
    }
}
