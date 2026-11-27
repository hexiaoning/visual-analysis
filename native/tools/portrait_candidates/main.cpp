#include "portrait_selection.h"
#include "appearance_cover.h"
#include "exposure_normalizer.h"
#include "exposure_recombination.h"
#include "fine_band_limit.h"
#include "geometry_calibration.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <numeric>
#include <nlohmann/json.hpp>
#include <opencv2/imgcodecs.hpp>
#include <optional>
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

void publish_new_image(const fs::path &output, const cv::Mat &image) {
    const auto timestamp = std::chrono::steady_clock::now().time_since_epoch().count();
    fs::path pending = output;
    pending += ".tmp-" + std::to_string(GetCurrentProcessId()) + "-" + std::to_string(timestamp);
    try {
        write_image(pending, image);
        if (!MoveFileExW(pending.c_str(), output.c_str(), 0))
            throw std::runtime_error("Could not publish output as a new file");
    } catch (...) {
        std::error_code ignored;
        fs::remove(pending, ignored);
        throw;
    }
}

cv::Mat combine_detail_consensus(const std::array<cv::Mat, 3> &candidates) {
    for (const auto &image : candidates)
        if (image.size() != cv::Size(512, 512) || image.type() != CV_8UC3)
            throw std::invalid_argument("Expected three aligned 512x512 BGR portraits");
    // Clamp the third value to the range of the first two to obtain the median.
    cv::Mat lower, upper, result;
    cv::min(candidates[0], candidates[1], lower);
    cv::max(candidates[0], candidates[1], upper);
    cv::max(candidates[2], lower, result);
    cv::min(result, upper, result);
    return result;
}

const char *route_name(Route name) { return name == Route::codeformer ? "codeformer" : "amd"; }

using Json = nlohmann::ordered_json;

using DonorWeights = std::array<std::array<float, 3>, 4>;

DonorWeights read_donor_weights(const fs::path &path) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream)
        throw std::runtime_error("Cannot open local donor weights");
    const auto model = Json::parse(stream);
    const auto &weights = model.at("weights");
    DonorWeights result{};
    if (model.at("schema") != "portrait-local-donors-v1" || !weights.is_array() || weights.size() != result.size())
        throw std::invalid_argument("Expected four facial regions in a version 1 local donor profile");
    for (size_t region = 0; region < result.size(); ++region) {
        const auto &row = weights.at(region);
        if (!row.is_array() || row.size() != result[region].size())
            throw std::invalid_argument("Expected three donor weights per facial region");
        double total = 0.0;
        for (size_t donor = 0; donor < result[region].size(); ++donor) {
            if (!row[donor].is_number())
                throw std::invalid_argument("Donor weights must be finite numbers");
            const double weight = row[donor].get<double>();
            if (!std::isfinite(weight) || weight < 0.0 || weight > 1.0)
                throw std::invalid_argument("Donor weights must be between zero and one");
            result[region][donor] = static_cast<float>(weight);
            total += weight;
        }
        if (total > 1.0 + 1e-7)
            throw std::invalid_argument("Donor weights cannot sum above one within a facial region");
    }
    return result;
}

cv::Mat compose_local_donors(const cv::Mat &base, const std::array<cv::Mat, 3> &donors, const DonorWeights &weights) {
    const auto check = [](const cv::Mat &image) {
        if (image.size() != cv::Size(512, 512) || image.type() != CV_8UC3)
            throw std::invalid_argument("Local donor mixing requires aligned 512x512 BGR images");
    };
    check(base);
    for (const auto &donor : donors)
        check(donor);
    struct FacialWindow { float x, y, width, height; };
    constexpr std::array windows{
        FacialWindow{190.0f, 240.0f, 49.0f, 40.0f}, FacialWindow{315.0f, 240.0f, 49.0f, 40.0f},
        FacialWindow{255.0f, 302.0f, 43.0f, 39.0f}, FacialWindow{255.0f, 373.0f, 79.0f, 41.0f},
    };
    auto output = base.clone();
    for (int y = 0; y < base.rows; ++y) {
        const auto *reference = base.ptr<cv::Vec3b>(y);
        auto *result = output.ptr<cv::Vec3b>(y);
        const std::array pixels{donors[0].ptr<cv::Vec3b>(y), donors[1].ptr<cv::Vec3b>(y), donors[2].ptr<cv::Vec3b>(y)};
        for (int x = 0; x < base.cols; ++x) {
            std::array<float, windows.size()> masks{};
            float total = 0.0f;
            for (size_t i = 0; i < windows.size(); ++i) {
                const auto &region = windows[i];
                const float dx = (static_cast<float>(x) - region.x) / region.width;
                const float dy = (static_cast<float>(y) - region.y) / region.height;
                const float fraction = std::clamp((1.18f - dx * dx - dy * dy) / 0.57f, 0.0f, 1.0f);
                masks[i] = fraction * fraction * (3.0f - 2.0f * fraction);
                total += masks[i];
            }
            if (total == 0.0f)
                continue;
            std::array<float, 3> amount{};
            for (size_t i = 0; i < windows.size(); ++i)
                for (size_t donor = 0; donor < amount.size(); ++donor)
                    amount[donor] += weights[i][donor] * (masks[i] / std::max(total, 1.0f));
            for (int channel = 0; channel < 3; ++channel) {
                const float value = reference[x][channel];
                float adjustment = 0.0f;
                for (size_t donor = 0; donor < amount.size(); ++donor)
                    adjustment += amount[donor] * (static_cast<float>(pixels[donor][x][channel]) - value);
                result[x][channel] = cv::saturate_cast<std::uint8_t>(value + adjustment);
            }
        }
    }
    return output;
}

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

template <typename WriteRun>
fs::path publish_run(const fs::path &directory, WriteRun write_run) {
    fs::create_directories(directory);
    fs::path pending, complete;
    do {
        auto id = std::chrono::steady_clock::now().time_since_epoch().count();
        std::string name = "run-" + std::to_string(GetCurrentProcessId()) + "-" + std::to_string(id);
        complete = directory / name;
        pending = directory / (name + ".tmp");
    } while (fs::exists(complete) || !fs::create_directory(pending));
    try {
        write_run(pending);
        fs::rename(pending, complete);
    } catch (...) {
        // Only remove the private, uniquely created staging folder for this call.
        std::error_code ec;
        fs::remove_all(pending, ec);
        throw;
    }
    return complete;
}

fs::path publish(const fs::path &directory, const cv::Mat &cf, const cv::Mat &amd, const Result &r,
                 bool fit_duplicate, const std::vector<Route> &selection,
                 std::optional<size_t> compared_alternate = std::nullopt) {
    return publish_run(directory, [&](const fs::path &pending) {
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
    });
}

fs::path mouth_variants(wchar_t **argv, int argc) {
    cv::setNumThreads(1);
    std::vector<double> openings;
    for (int arg = 5; arg < argc; ++arg) {
        const std::wstring value(argv[arg]);
        std::size_t used = 0;
        const double opening = std::stod(value, &used);
        if (used != value.size() || !std::isfinite(opening) || opening <= 0. || opening > 1.)
            throw std::invalid_argument("Expected mouth openings in (0, 1].");
        if (std::find(openings.begin(), openings.end(), opening) == openings.end())
            openings.push_back(opening);
    }
    const auto photo = read_image(argv[3]);
    portrait_selection::Selector model(argv[2]);
    const auto variants = model.mouth_variants(photo, openings);
    return publish_run(argv[4], [&](const fs::path &pending) {
        Json outputs = Json::array();
        for (const auto &variant : variants) {
            const std::string filename = "candidate-" + std::to_string(outputs.size() + 1) + ".png";
            write_image(pending / filename, variant.image);
            outputs.push_back({{"requested_opening", variant.requested_opening},
                               {"output_opening", variant.output_opening}, {"png", filename}});
        }
        std::ofstream receipt(pending / "selection.json", std::ios::binary);
        receipt << Json{{"mode", "mouth-variants"}, {"outputs", std::move(outputs)}}.dump(2) << '\n';
        receipt.close();
        if (!receipt)
            throw std::runtime_error("Could not save mouth variant report");
    });
}

fs::path recombine_exposure_pair(const fs::path &directory, const portrait_selection::ShapeDescriptor &first,
                                 const portrait_selection::ShapeDescriptor &second) {
    using portrait_selection::ExposureRegion;
    return publish_run(directory, [&](const fs::path &pending) {
        Json variants = Json::object();
        for (const auto &[region, name] : std::array{std::pair{ExposureRegion::eyes, "eyes"},
                                                     std::pair{ExposureRegion::mouth, "mouth"}}) {
            const auto pair = portrait_selection::recombine_exposure(first, second, region);
            Json outputs = Json::array();
            if (std::string_view(pair.status) == "generated") {
                const std::string a = std::string(name) + "-first.png";
                const std::string b = std::string(name) + "-second.png";
                write_image(pending / a, pair.first);
                write_image(pending / b, pair.second);
                outputs = {a, b};
            }
            variants[name] = {{"status", pair.status}, {"pose_shift", pair.pose_shift}, {"context_error", pair.context_error},
                              {"ideal_mean_error", pair.ideal_mean_error}, {"saved_mean_error", pair.saved_mean_error},
                              {"saved_mean_rms", pair.saved_mean_rms}, {"changed_pixels", {pair.changed_first, pair.changed_second}},
                              {"outputs", std::move(outputs)}};
        }
        std::ofstream receipt(pending / "selection.json", std::ios::binary);
        receipt << Json{{"mode", "exposure-recombination"}, {"regions", std::move(variants)}}.dump(2) << '\n';
        receipt.close();
        if (!receipt)
            throw std::runtime_error("Could not save exposure recombination report");
    });
}

Json write_indexed_candidates(const fs::path &directory, const std::vector<cv::Mat> &portraits,
                              const std::vector<std::size_t> &indices) {
    Json outputs = Json::array();
    for (const auto index : indices) {
        const std::string filename = "candidate-" + std::to_string(outputs.size() + 1) + ".png";
        write_image(directory / filename, portraits[index]);
        outputs.push_back({{"input_index", index}, {"png", filename}});
    }
    return outputs;
}

class ShapeComparisons {
public:
    ShapeComparisons(const fs::path &model, const std::vector<cv::Mat> &portraits, bool record_checks = true)
        : model_(model), portraits_(portraits), descriptions_(portraits.size()), record_checks_(record_checks) {}

    bool different(std::size_t previous, std::size_t proposed) {
        if (!selector_)
            selector_.emplace(model_);
        for (const auto index : {previous, proposed}) {
            if (!descriptions_[index])
                descriptions_[index] = selector_->describe_shape(portraits_[index]);
        }
        const bool visible = portrait_selection::visibly_different(*descriptions_[previous], *descriptions_[proposed]);
        if (record_checks_)
            checks.push_back({{"previous", previous}, {"proposed", proposed}, {"visible", visible}});
        return visible;
    }

    Json checks = Json::array();

private:
    fs::path model_;
    const std::vector<cv::Mat> &portraits_;
    std::optional<portrait_selection::Selector> selector_;
    std::vector<std::optional<portrait_selection::ShapeDescriptor>> descriptions_;
    bool record_checks_;
};

fs::path inspect_appearance_bank(wchar_t **argv, int argc) {
    cv::setNumThreads(1);
    const std::wstring radius_text(argv[2]);
    std::size_t used = 0;
    const double radius = std::stod(radius_text, &used);
    if (used != radius_text.size() || !std::isfinite(radius) || radius <= 0. || radius > 2.)
        throw std::invalid_argument("Expected a radius in (0, 2]");
    const auto source = read_image(argv[4]);
    std::vector<cv::Mat> portraits;
    for (int i = 6; i < argc; ++i)
        portraits.push_back(read_image(argv[i]));
    const auto evidence = portrait_selection::analyse_appearance_bank(source, portraits);
    const bool compare_landmarks = std::wstring_view(argv[3]) != L"--distance-only";
    ShapeComparisons landmarks(argv[3], portraits, false);
    std::vector<std::vector<bool>> compatible(portraits.size(), std::vector<bool>(portraits.size()));
    for (std::size_t i = 0; i < portraits.size(); ++i) {
        for (std::size_t j = i + 1; j < portraits.size(); ++j) {
            const bool visible = evidence.distances[i][j] > radius && (!compare_landmarks || landmarks.different(i, j));
            compatible[i][j] = compatible[j][i] = visible;
        }
    }
    return publish_run(argv[5], [&](const fs::path &directory) {
        const Json report = {
            {"radius", radius}, {"source_detail", evidence.observed_detail}, {"detail", evidence.candidate_detail},
            {"replay_errors", evidence.replay_errors}, {"distance", evidence.distances}, {"compatible", compatible},
            {"landmarks_enabled", compare_landmarks},
        };
        std::ofstream receipt(directory / "selection.json", std::ios::binary);
        receipt << report.dump() << '\n';
        receipt.close();
        if (!receipt)
            throw std::runtime_error("Could not save appearance bank evidence");
    });
}

fs::path extend_shape_cover(wchar_t **argv, int argc) {
    cv::setNumThreads(1);
    const auto parse = [](const wchar_t *argument) {
        const std::wstring text(argument);
        std::size_t used = 0;
        const double number = std::stod(text, &used);
        if (used != text.size())
            throw std::invalid_argument("Invalid shape extension parameter");
        return number;
    };
    const double radius = parse(argv[2]);
    const double tolerance = parse(argv[3]);
    const std::wstring text(argv[4]);
    std::size_t used = 0;
    const auto kept = std::stoull(text, &used);
    int source_index = 5;
    const bool expand = std::wstring_view(argv[source_index]) == L"--max-additions";
    std::size_t max_additions = 1;
    if (expand) {
        const std::wstring count(argv[source_index + 1]);
        if (count.empty() || !std::ranges::all_of(count, [](wchar_t c) { return c >= L'0' && c <= L'9'; }))
            throw std::invalid_argument("Expected a nonnegative maximum number of additions");
        max_additions = static_cast<std::size_t>(std::stoull(count));
        source_index += 2;
    }
    const bool compare_landmarks = std::wstring_view(argv[source_index]) == L"--landmarks";
    const int model_index = source_index + 1;
    if (compare_landmarks)
        source_index += 2;
    const int first_candidate = source_index + 2;
    if (argc <= first_candidate)
        throw std::invalid_argument("Expected the source, output directory and at least one candidate");
    if (used != text.size() || kept == 0 || kept > static_cast<unsigned long long>(argc - first_candidate))
        throw std::invalid_argument("Expected a positive protected count within the candidate bank");
    const auto source = read_image(argv[source_index]);
    std::vector<cv::Mat> portraits;
    portraits.reserve(static_cast<std::size_t>(argc - first_candidate));
    for (int arg = first_candidate; arg < argc; ++arg)
        portraits.push_back(read_image(argv[arg]));
    const auto protected_count = static_cast<std::size_t>(kept);
    std::optional<ShapeComparisons> comparisons;
    std::function<bool(std::size_t, std::size_t)> compatible;
    if (expand && compare_landmarks) {
        comparisons.emplace(argv[model_index], portraits);
        compatible = [&](std::size_t previous, std::size_t proposed) {
            return comparisons->different(previous, proposed);
        };
    }
    auto chosen = portrait_selection::extend_appearances(source, portraits, protected_count, radius, tolerance,
                                                        max_additions, compatible);
    Json landmark_checks = Json::array();
    if (!expand && compare_landmarks && chosen.representatives.size() > protected_count) {
        portrait_selection::Selector selector(argv[model_index]);
        const auto &extra = portraits[chosen.representatives.back()];
        bool distinct = true;
        for (std::size_t i = 0; i < protected_count; ++i) {
            const bool visible = selector.select(source, portraits[i], extra).visible_shape_difference;
            landmark_checks.push_back(visible);
            distinct = distinct && visible;
        }
        if (!distinct)
            chosen.representatives.pop_back();
    }
    return publish_run(argv[source_index + 1], [&](const fs::path &pending) {
        Json report = {{"mode", "evidence-extension"}, {"radius", radius}, {"replay_tolerance", tolerance},
                             {"kept_count", kept}, {"replay_errors", chosen.replay_errors},
                             {"distance_to_protected", chosen.distances_to_kept},
                             {"landmarks_enabled", compare_landmarks}, {"landmark_checks", landmark_checks},
                             {"outputs", write_indexed_candidates(pending, portraits, chosen.representatives)}};
        if (expand) {
            report["max_additions"] = max_additions;
            report["pair_checks"] = comparisons ? comparisons->checks : Json::array();
        }
        std::ofstream receipt(pending / "selection.json", std::ios::binary);
        receipt << report.dump(2) << '\n';
        receipt.close();
        if (!receipt)
            throw std::runtime_error("Could not save evidence extension report");
    });
}

fs::path refine_representatives(wchar_t **argv, int argc) {
    cv::setNumThreads(1);
    const std::wstring count(argv[2]);
    if (count.empty() || !std::ranges::all_of(count, [](wchar_t c) { return c >= L'0' && c <= L'9'; }))
        throw std::invalid_argument("Expected a positive number of displayed candidates");
    const auto kept = static_cast<std::size_t>(std::stoull(count));
    const auto source = read_image(argv[4]);
    std::vector<cv::Mat> portraits;
    for (int arg = 6; arg < argc; ++arg)
        portraits.push_back(read_image(argv[arg]));
    ShapeComparisons comparisons(argv[3], portraits);
    const auto replacement = portrait_selection::refine_representative(source, portraits, kept,
        [&](auto previous, auto proposed) { return comparisons.different(previous, proposed); });
    std::vector<std::size_t> selected(kept);
    std::iota(selected.begin(), selected.end(), 0);
    Json improvement = nullptr;
    if (replacement) {
        selected[replacement->previous] = replacement->proposed;
        improvement = {{"previous", replacement->previous}, {"proposed", replacement->proposed},
                       {"distance", replacement->distance}, {"minimum_relative_gain", replacement->score},
                       {"replay_improvements", replacement->replay_improvements}};
    }
    return publish_run(argv[5], [&](const fs::path &pending) {
        const Json report = {{"mode", "representative-refinement"}, {"kept_count", kept},
                             {"replacement", improvement}, {"pair_checks", comparisons.checks},
                             {"outputs", write_indexed_candidates(pending, portraits, selected)}};
        std::ofstream receipt(pending / "selection.json", std::ios::binary);
        receipt << report.dump(2) << '\n';
        receipt.close();
        if (!receipt)
            throw std::runtime_error("Could not write representative refinement receipt");
    });
}

fs::path refine_clear_representatives(wchar_t **argv, int argc) {
    cv::setNumThreads(1);
    const auto read_count = [](const wchar_t* argument) {
        const std::wstring_view text(argument);
        if (text.empty() || !std::ranges::all_of(text, [](wchar_t c) { return c >= L'0' && c <= L'9'; }))
            throw std::invalid_argument("Expected unsigned representative and addition counts");
        return std::stoull(argument);
    };
    const auto kept = read_count(argv[2]), added = read_count(argv[3]);
    const auto count = static_cast<std::size_t>(argc - 7);
    if (kept == 0 || kept > count || added > count - kept)
        throw std::invalid_argument("Displayed counts exceed the candidate bank or omit its first representative");
    const auto source = read_image(argv[5]);
    std::vector<cv::Mat> portraits;
    portraits.reserve(count);
    for (int i = 7; i < argc; ++i)
        portraits.push_back(read_image(argv[i]));
    ShapeComparisons comparisons(argv[4], portraits);
    const auto replacement = portrait_selection::refine_clear_representative(
        source, portraits, static_cast<std::size_t>(kept), static_cast<std::size_t>(added),
        [&](auto previous, auto proposed) { return comparisons.different(previous, proposed); });
    std::vector<std::size_t> selected(static_cast<std::size_t>(kept + added));
    std::iota(selected.begin(), selected.end(), 0);
    Json outcome = nullptr;
    if (replacement) {
        selected[replacement->previous] = replacement->proposed;
        Json regional = Json::array();
        for (const auto gain : replacement->regional_gain)
            regional.push_back(gain ? Json(*gain) : Json(nullptr));
        outcome = {{"previous", replacement->previous}, {"proposed", replacement->proposed},
                   {"distance", replacement->distance}, {"clarity_gain", replacement->clarity_gain},
                   {"regional_gain", regional}, {"replay_deltas", replacement->replay_deltas},
                   {"replay_limits", replacement->replay_limits},
                   {"maximum_offset_change", replacement->maximum_offset_change}};
    }
    return publish_run(argv[6], [&](const fs::path &pending) {
        const Json report = {{"mode", "clarity-refinement"}, {"kept_count", kept}, {"added_count", added},
                             {"replacement", outcome}, {"pair_checks", comparisons.checks},
                             {"outputs", write_indexed_candidates(pending, portraits, selected)}};
        std::ofstream receipt(pending / "selection.json", std::ios::binary);
        receipt << report.dump(2) << '\n';
        receipt.close();
        if (!receipt)
            throw std::runtime_error("Could not write clarity refinement receipt");
    });
}

fs::path cover_shapes(wchar_t **argv, int argc) {
    cv::setNumThreads(1);
    const std::wstring value(argv[2]);
    size_t used = 0;
    const double radius = std::stod(value, &used);
    if (used != value.size())
        throw std::invalid_argument("Invalid appearance cover radius");
    std::vector<cv::Mat> portraits;
    portraits.reserve(static_cast<size_t>(argc - 4));
    for (int arg = 4; arg < argc; ++arg)
        portraits.push_back(read_image(argv[arg]));
    const auto cover = portrait_selection::cover_appearances(portraits, radius);
    return publish_run(argv[3], [&](const fs::path &pending) {
        Json report = {
            {"mode", "shape-cover"}, {"radius", radius}, {"nearest_input_index", cover.nearest},
            {"measured_distance", cover.distances}, {"outputs", write_indexed_candidates(pending, portraits, cover.representatives)},
        };
        std::ofstream receipt(pending / "selection.json", std::ios::binary);
        receipt << report.dump(2) << '\n';
        receipt.close();
        if (!receipt)
            throw std::runtime_error("Could not save shape cover report");
    });
}

enum class ReplayChoice { primary, distinct_pair, replacement };

ReplayChoice choose_by_replay(const Result &result) {
    if (result.amd.clarity_relative_to_input < 1.20)
        return ReplayChoice::primary;
    if (result.duplicate_eligible && result.amd_fraction.value_or(0.) > .50 &&
        result.amd.replay_error && result.codeformer.replay_error &&
        *result.amd.replay_error <= *result.codeformer.replay_error) {
        // Extra band-pass energy can come from screen texture returning in the alternate.
        return result.relative_clarity <= 1.40 ? ReplayChoice::replacement : ReplayChoice::primary;
    }
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
            "PortraitCandidates.exe --condition-candidate-tone scale candidate.png guide.png output.png\n"
            "PortraitCandidates.exe --calibrate-capture-color model.json source.png candidate.png output.png\n"
            "PortraitCandidates.exe --calibrate-capture-geometry model.json source.png first-guide.png "
            "second-guide.png candidate.png output.png\n"
            "PortraitCandidates.exe --cap-fine-band aligned.png max-rms output.png\n"
            "PortraitCandidates.exe --compose-local-donors weights.json baseline.png first.png second.png "
            "third.png output.png\n"
            "PortraitCandidates.exe --combine-detail-consensus first.png second.png third.png output.png\n"
            "PortraitCandidates.exe --mouth-variants landmarks.onnx aligned.png output-directory opening [opening2 ...]\n"
            "PortraitCandidates.exe --recombine-exposure landmarks.onnx first.png second.png output-directory [first2.png second2.png output2 ...]\n"
            "PortraitCandidates.exe --extend-shape-cover radius replay-tolerance kept-count [--max-additions count] [--landmarks model.onnx] source.png output-directory candidate.png [more.png ...]\n"
            "PortraitCandidates.exe --refine-representatives kept-count landmarks.onnx source.png output-directory shown.png [more.png ...]\n"
            "PortraitCandidates.exe --refine-clear-representatives kept-count added-count landmarks.onnx source.png output-directory shown.png [more.png ...]\n"
            "PortraitCandidates.exe --inspect-appearance-bank radius landmarks.onnx|--distance-only source.png output-directory candidate.png [more.png ...]\n"
            "PortraitCandidates.exe --cover-shapes radius output-directory primary.png [candidate.png ...]\n"
            "PortraitCandidates.exe --select-by-replay landmarks.onnx source.png primary.png output-directory "
            "alternate.png [alternate2.png ...]";
        if (argc == 2 && std::wstring_view(argv[1]) == L"--help") {
            std::cout << usage << '\n';
            return 0;
        }
        const std::wstring_view option = argc > 1 ? argv[1] : L"";
        if (option == L"--extend-shape-cover") {
            if (argc < 8)
                throw std::invalid_argument(usage);
            print_output_path(extend_shape_cover(argv, argc));
            return 0;
        }
        if (option == L"--refine-representatives") {
            if (argc < 7)
                throw std::invalid_argument(usage);
            print_output_path(refine_representatives(argv, argc));
            return 0;
        }
        if (option == L"--refine-clear-representatives") {
            if (argc < 8)
                throw std::invalid_argument(usage);
            print_output_path(refine_clear_representatives(argv, argc));
            return 0;
        }
        if (option == L"--inspect-appearance-bank") {
            if (argc < 7)
                throw std::invalid_argument(usage);
            print_output_path(inspect_appearance_bank(argv, argc));
            return 0;
        }
        if (option == L"--mouth-variants") {
            if (argc < 6)
                throw std::invalid_argument(usage);
            print_output_path(mouth_variants(argv, argc));
            return 0;
        }
        if (option == L"--recombine-exposure") {
            if (argc < 6 || (argc - 3) % 3 != 0)
                throw std::invalid_argument(usage);
            cv::setNumThreads(1);
            portrait_selection::Selector model(argv[2]);
            for (int i = 3; i < argc; i += 3) {
                const auto first = model.describe_shape(read_image(argv[i]));
                const auto second = model.describe_shape(read_image(argv[i + 1]));
                print_output_path(recombine_exposure_pair(argv[i + 2], first, second));
            }
            return 0;
        }
        if (option == L"--cover-shapes") {
            if (argc < 5)
                throw std::invalid_argument(usage);
            print_output_path(cover_shapes(argv, argc));
            return 0;
        }
        if (option == L"--select-by-replay") {
            if (argc < 7)
                throw std::invalid_argument(usage);
            print_output_path(select_by_replay(argv, argc));
            return 0;
        }
        if (option == L"--cap-fine-band") {
            if (argc != 5)
                throw std::invalid_argument(usage);
            const fs::path output(argv[4]);
            if (fs::exists(output))
                throw std::invalid_argument("Fine band output must be a new path");
            const std::wstring value(argv[3]);
            size_t used = 0;
            const double maximum = std::stod(value, &used);
            if (used != value.size())
                throw std::invalid_argument("Invalid fine band limit");
            cv::setNumThreads(1);
            const auto limited = portrait_selection::cap_fine_band(read_image(argv[2]), maximum);
            publish_new_image(output, limited.image);
            std::cout << Json{{"target_rms", maximum}, {"output_rms", limited.rms}, {"reached", limited.reached}}.dump()
                      << '\n';
            return 0;
        }
        if (option == L"--compose-local-donors") {
            if (argc != 8)
                throw std::invalid_argument(usage);
            const fs::path output(argv[7]);
            if (fs::exists(output))
                throw std::invalid_argument("Local donor output must be a new path");
            cv::setNumThreads(1);
            const auto weights = read_donor_weights(argv[2]);
            const auto baseline = read_image(argv[3]);
            const std::array donors{read_image(argv[4]), read_image(argv[5]), read_image(argv[6])};
            publish_new_image(output, compose_local_donors(baseline, donors, weights));
            return 0;
        }
        if (option == L"--combine-detail-consensus") {
            if (argc != 6)
                throw std::invalid_argument(usage);
            const fs::path output(argv[5]);
            if (fs::exists(output))
                throw std::invalid_argument("Consensus output must be a new path");
            cv::setNumThreads(1);
            const std::array candidates{read_image(argv[2]), read_image(argv[3]), read_image(argv[4])};
            publish_new_image(output, combine_detail_consensus(candidates));
            return 0;
        }
        if (option == L"--condition-candidate-tone") {
            if (argc != 6)
                throw std::invalid_argument(usage);
            const std::wstring parameter(argv[2]);
            std::size_t used = 0;
            const double scale = std::stod(parameter, &used);
            if (used != parameter.size())
                throw std::invalid_argument("Invalid tone conditioning scale");
            const fs::path output(argv[5]);
            if (fs::exists(output))
                throw std::invalid_argument("Conditioned output must be a new path");
            cv::setNumThreads(1);
            const auto candidate = read_image(argv[3]);
            const auto guide = read_image(argv[4]);
            publish_new_image(output, portrait_selection::condition_candidate_tone(candidate, guide, scale));
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
            publish_new_image(output, portrait_selection::match_candidate_tone(candidate, guide));
            return 0;
        }
        if (option == L"--calibrate-capture-color") {
            if (argc != 6)
                throw std::invalid_argument(usage);
            const fs::path output(argv[5]);
            if (fs::exists(output))
                throw std::invalid_argument("Color calibration output must be a new path");
            cv::setNumThreads(1);
            const auto source = read_image(argv[3]);
            const auto candidate = read_image(argv[4]);
            const auto result = portrait_selection::calibrate_capture_color(source, candidate, argv[2]);
            publish_new_image(output, result.image);
            std::cout << Json{{"log_gain_blue_red", result.log_gain_blue_red}}.dump() << '\n';
            return 0;
        }
        if (option == L"--calibrate-capture-geometry") {
            if (argc != 8)
                throw std::invalid_argument(usage);
            const fs::path output(argv[7]);
            if (fs::exists(output))
                throw std::invalid_argument("Geometry calibration output must be a new path");
            cv::setNumThreads(1);
            const std::array observations{read_image(argv[3]), read_image(argv[4]), read_image(argv[5])};
            const auto candidate = read_image(argv[6]);
            publish_new_image(output, portrait_selection::calibrate_capture_geometry(observations, candidate, argv[2]));
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
            publish_new_image(output, result.image);
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
