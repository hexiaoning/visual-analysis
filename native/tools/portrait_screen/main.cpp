#include "screen_restore.h"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

namespace {

namespace fs = std::filesystem;
using json = nlohmann::json;
using Clock = std::chrono::steady_clock;

fs::path path(const json& value) {
    const auto& name = value.get_ref<const json::string_t&>();
    return fs::path(std::u8string(name.begin(), name.end()));
}

cv::Mat load(const json& name) {
    std::ifstream file(path(name), std::ios::binary);
    if (!file) {
        throw std::runtime_error("Could not open a source image");
    }
    const std::vector<unsigned char> bytes((std::istreambuf_iterator<char>(file)), {});
    cv::Mat data = cv::imdecode(bytes, cv::IMREAD_COLOR);
    if (data.empty()) {
        throw std::runtime_error("Could not decode a source image");
    }
    return data;
}

void writeImage(const fs::path& location, const cv::Mat& pixels) {
    std::vector<unsigned char> encoded;
    if (!cv::imencode(".png", pixels, encoded)) {
        throw std::runtime_error("Could not encode a portrait");
    }
    std::ofstream stream(location, std::ios::binary);
    stream.write(reinterpret_cast<const char*>(encoded.data()), encoded.size());
    if (!stream) {
        throw std::runtime_error("Could not save a portrait");
    }
}

cv::Matx23d transform(const json& input) {
    cv::Matx23d affine;
    for (int y = 0; y < 2; ++y) {
        for (int x = 0; x < 3; ++x) {
            affine(y, x) = input.at(y).at(x).get<double>();
        }
    }
    if (!cv::checkRange(cv::Mat(affine))) {
        throw std::invalid_argument("Invalid source-to-crop transform");
    }
    return affine;
}

cv::Size size(const json& input) {
    const auto w = input.at(0).get<int>();
    const auto h = input.at(1).get<int>();
    if (w < 1 || h < 1) {
        throw std::invalid_argument("Invalid source raster size");
    }
    return {w, h};
}

cv::Mat featheredDifference(const cv::Mat& source, const cv::Mat& cleaned) {
    cv::Mat raw, processed;
    source.convertTo(raw, CV_32F);
    cleaned.convertTo(processed, CV_32F);
    cv::Mat delta = processed - raw;
    const float width = std::min(source.rows, source.cols) * (17.f / 256.f);
    for (int y = 0; y < delta.rows; ++y) {
        auto* row = delta.ptr<cv::Vec3f>(y);
        for (int x = 0; x < delta.cols; ++x) {
            const int edge = std::min({x, y, delta.cols - 1 - x, delta.rows - 1 - y});
            row[x] *= std::clamp(edge / width, 0.f, 1.f);
        }
    }
    return delta;
}

cv::Mat alignedPhase(const cv::Mat& native, const cv::Mat& cleaned, const cv::Mat& raw,
                     const json& transport, const cv::Matx23d& affine) {
    cv::Mat phase;
    const auto mode = transport.at("mode").get<std::string>();
    if (mode == "direct") {
        cv::warpAffine(cleaned, phase, affine, raw.size(), cv::INTER_LINEAR, cv::BORDER_REFLECT_101);
        return phase;
    }
    const cv::Mat sourceDelta = featheredDifference(native, cleaned);
    cv::Mat delta;
    if (mode == "delta") {
        cv::warpAffine(sourceDelta, delta, transform(transport.at("matrix")), raw.size(), cv::INTER_CUBIC);
    } else if (mode == "reduced") {
        cv::Mat full, reduced, sized;
        cv::warpAffine(sourceDelta, full, transform(transport.at("withinSource")), size(transport.at("full")),
                       cv::INTER_NEAREST);
        cv::resize(full, reduced, size(transport.at("reduced")), 0., 0., cv::INTER_AREA);
        cv::resize(reduced, sized, size(transport.at("resized")), 0., 0., cv::INTER_AREA);
        cv::warpAffine(sized, delta, transform(transport.at("matrix")), raw.size(),
                       cv::INTER_LINEAR, cv::BORDER_REFLECT_101);
    } else {
        throw std::invalid_argument("Invalid phase transport method");
    }
    cv::Mat aligned;
    raw.convertTo(aligned, CV_32F);
    aligned += delta;
    aligned.convertTo(phase, CV_8UC3);
    return phase;
}

json frequencies(const std::vector<cv::Point2f>& values) {
    json result = json::array();
    for (auto p : values) {
        result.push_back({p.x, p.y});
    }
    return result;
}

void validateSample(const std::string& key) {
    if (key.empty() || key.find_first_not_of("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-") !=
        std::string::npos) {
        throw std::invalid_argument("Invalid screen sample identifier");
    }
}

json run(const json& rows, const fs::path& output, bool writeNarrow) {
    json audit = json::object();
    for (auto entry = rows.begin(); entry != rows.end(); ++entry) {
        const auto& key = entry.key();
        validateSample(key);
        const auto& data = entry.value();
        const auto native = load(data.at("native"));
        const auto raw = load(data.at("alignedRaw"));
        const auto base = load(data.at("previous"));
        const auto affine = transform(data.at("nativeToAligned"));
        const auto start = Clock::now();
        const auto peaks = visual::screen::detect(native);
        const auto detected = Clock::now();
        const auto cleaned = visual::screen::averagePeriods(native, peaks);
        const auto phase = peaks.periods.empty() ? raw.clone()
            : alignedPhase(native, cleaned, raw, data.at("transport"), affine);
        const auto sampled = Clock::now();
        const auto result = visual::screen::selectBase(raw, base, phase, peaks, affine);
        const auto stop = Clock::now();
        const auto ms = [](auto t) { return std::chrono::duration<double, std::milli>(t).count(); };
        const auto out = output / key;
        fs::create_directories(out);
        writeImage(out / L"phase.png", phase);
        writeImage(out / L"selected.png", result.usePhase ? phase : base);
        audit[key] = {{"chosen", result.usePhase ? "phase" : "previous"}, {"frequencies", frequencies(peaks.frequencies)},
                      {"periods", frequencies(peaks.periods)}, {"carrierRatio", result.carrierRatio},
                      {"midbandRatio", result.midbandRatio}, {"baseCoarseError", result.baseCoarseError},
                      {"phaseCoarseError", result.phaseCoarseError}, {"detectMs", ms(detected - start)},
                      {"sampleMs", ms(sampled - detected)}, {"selectMs", ms(stop - sampled)}};
        if (writeNarrow) {
            const auto begin = Clock::now();
            const auto narrow = peaks.periods.empty() ? raw.clone() : alignedPhase(
                native, visual::screen::subtractCoherentCarrier(native, peaks), raw, data.at("transport"), affine);
            audit[key]["subtractMs"] = ms(Clock::now() - begin);
            writeImage(out / L"narrow.png", narrow);
        }
        std::cout << key << ": " << (result.usePhase ? "phase" : "previous") << '\n';
    }
    return audit;
}

json runCaptureReplay(const json& rows, const fs::path& output) {
    json audit = json::object();
    for (const auto& [key, data] : rows.items()) {
        validateSample(key);
        const auto& candidates = data.at("candidates");
        const auto prior = data.at("previous").get<std::vector<std::string>>();
        if (!candidates.is_object() || prior.empty() || prior.size() > 2 ||
            (prior.size() == 2 && prior[0] == prior[1])) {
            throw std::invalid_argument("Expected named crops and one or two distinct previously displayed candidates");
        }
        auto names = prior;
        for (const auto& [name, image] : candidates.items()) {
            if (std::find(prior.begin(), prior.end(), name) == prior.end()) {
                names.push_back(name);
            }
        }
        std::vector<cv::Mat> crops;
        crops.reserve(names.size());
        for (const auto& name : names) {
            crops.push_back(load(candidates.at(name)));
        }
        const auto native = load(data.at("native"));
        const auto affine = transform(data.at("nativeToAligned"));
        const auto result = visual::screen::rankAtCaptureResolution(native, affine, crops, prior.size() == 2);
        const auto out = output / key;
        fs::create_directories(out);
        writeImage(out / L"selected.png", crops[result.selected]);
        if (prior.size() == 2) {
            writeImage(out / L"selected-2.png", crops[1]);
        }
        json scores = json::object();
        for (std::size_t i = 0; i < names.size(); ++i) {
            scores[names[i]] = {{"error", result.scores[i].error}, {"detailRatio", result.scores[i].relativeDetail}};
        }
        const auto selected = prior.size() == 2 ? prior : std::vector<std::string>{names[result.selected]};
        audit[key] = {{"selected", selected}, {"blur", result.blur}, {"scores", scores}};
    }
    return audit;
}

}

int wmain(int argc, wchar_t** argv) {
    try {
        if (argc != 3) {
            throw std::invalid_argument("Usage: PortraitScreen.exe manifest.json output-directory");
        }
        std::ifstream manifest(fs::path(argv[1]), std::ios::binary);
        if (!manifest) {
            throw std::runtime_error("Cannot open the manifest");
        }
        const auto config = json::parse(manifest);
        const auto& rows = config.at("rows");
        if (!rows.is_object() || rows.empty()) {
            throw std::invalid_argument("The manifest must contain at least one named sample");
        }
        const bool writeNarrow = config.value("writeNarrow", false);
        const fs::path output(argv[2]);
        if (!output.parent_path().empty()) {
            fs::create_directories(output.parent_path());
        }
        if (!fs::create_directory(output)) {
            throw std::runtime_error("Choose a new output directory for this run");
        }
        cv::setNumThreads(1);
        const auto results = config.value("rankAtCaptureResolution", false)
            ? runCaptureReplay(rows, output) : run(rows, output, writeNarrow);
        const auto pending = output / L"results.pending.json";
        std::ofstream report(pending, std::ios::binary);
        report << results.dump(2) << '\n';
        report.close();
        if (!report) {
            throw std::runtime_error("Could not save the report");
        }
        fs::rename(pending, output / L"results.json");
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
