#include "grid_fusion.h"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace visual::screen {
namespace {
namespace fs = std::filesystem;
using Json = nlohmann::json;

constexpr int Side = 512;
constexpr std::size_t Views = 8;
using Planes = std::array<cv::Mat, Views>;
using Bands = std::array<cv::Mat, 3>;
using ViewBands = std::array<Bands, Views>;

fs::path path(const Json& value) {
    const auto text = value.get<std::string>();
    return fs::path(std::u8string(text.begin(), text.end()));
}

cv::Mat luminance(const cv::Mat& rgb) {
    cv::Mat gray;
    cv::cvtColor(rgb, gray, cv::COLOR_BGR2GRAY);
    return gray;
}

cv::Mat blur(const cv::Mat& pixels, double sigma) {
    cv::Mat out;
    cv::GaussianBlur(pixels, out, {}, sigma, 0., cv::BORDER_REFLECT_101);
    return out;
}

Bands bands(const cv::Mat& pixels) {
    auto y = luminance(pixels);
    const auto a = blur(y, .6);
    const auto b = blur(y, 1.5);
    const auto c = blur(y, 4.);
    const auto d = blur(y, 12.);
    return {a - b, b - c, c - d};
}

cv::Mat median(const Planes& planes) {
    cv::Mat out(planes[0].size(), planes[0].type());
    for (int y = 0; y < out.rows; ++y) {
        auto* target = out.ptr<float>(y);
        std::array<const float*, Views> rows{};
        for (std::size_t view = 0; view < Views; ++view) {
            rows[view] = planes[view].ptr<float>(y);
        }
        for (int x = 0; x < out.cols * out.channels(); ++x) {
            std::array<float, Views> vals{};
            for (std::size_t view = 0; view < Views; ++view) {
                vals[view] = rows[view][x];
            }
            auto middle = vals.begin() + Views / 2;
            std::nth_element(vals.begin(), middle, vals.end());
            const auto low = *std::max_element(vals.begin(), middle);
            target[x] = .5f * (low + *middle);
        }
    }
    return out;
}

struct Detail {
    Bands median;
    Bands signal;
};

Detail stability(const Planes& rgb) {
    ViewBands measurements;
    for (std::size_t view = 0; view < Views; ++view) {
        measurements[view] = bands(rgb[view]);
    }
    Detail result;
    for (std::size_t scale = 0; scale < result.median.size(); ++scale) {
        Planes scales, differences;
        for (std::size_t view = 0; view < Views; ++view) {
            scales[view] = measurements[view][scale];
        }
        const auto center = median(scales);
        for (std::size_t view = 0; view < Views; ++view) {
            cv::absdiff(scales[view], center, differences[view]);
        }
        const auto deviation = median(differences);
        result.median[scale] = center;
        const auto strength = blur(cv::abs(center), 3.);
        const cv::Mat noise = blur(deviation, 3.) + .0025;
        cv::divide(strength, noise, result.signal[scale]);
    }
    return result;
}

std::pair<cv::Mat, cv::Mat> combine(const Planes& fhdmi, const Planes& uhdm, const cv::Mat& observed) {
    const auto f = stability(fhdmi);
    const auto u = stability(uhdm);
    const cv::Mat base = .5f * (median(fhdmi) + median(uhdm));
    const auto baseBands = bands(base);
    const auto raw = bands(observed);
    const std::array<float, 3> gain{.48f, .37f, .15f};
    cv::Mat fusion = cv::Mat::zeros(base.size(), CV_32F);
    cv::Mat extra = cv::Mat::zeros(base.size(), CV_32F);
    for (std::size_t i = 0; i < gain.size(); ++i) {
        const cv::Mat reliability = cv::max(f.signal[i] + u.signal[i], .00001);
        const cv::Mat weighted = f.signal[i].mul(f.median[i]) + u.signal[i].mul(u.median[i]);
        cv::Mat estimate;
        cv::divide(weighted, reliability, estimate);
        fusion += estimate - baseBands[i];
        const auto cross = blur(estimate.mul(raw[i]), 2.);
        cv::Mat squared = blur(estimate.mul(estimate), 2.).mul(blur(raw[i].mul(raw[i]), 2.));
        cv::sqrt(squared, squared);
        cv::Mat localCorrelation;
        cv::divide(cross, 2. * squared + 1e-6, localCorrelation);
        const cv::Mat confidence = cv::min(cv::max(localCorrelation + .5, 0.), 1.);
        const cv::Mat floor = cv::min(f.signal[i], u.signal[i]);
        const cv::Mat consensus = cv::min(cv::max((floor - .8) / 1.2, 0.), 1.);
        extra += gain[i] * estimate.mul(confidence).mul(consensus);
    }
    auto assemble = [&](const cv::Mat& detail) {
        cv::Mat rgb, result;
        cv::merge(std::vector<cv::Mat>(3, detail), rgb);
        cv::add(base, rgb, result);
        result.convertTo(result, CV_8UC3, 255.);
        return result;
    };
    return {assemble(fusion), assemble(fusion + extra)};
}

cv::Mat replaceColor(const cv::Mat& detail, const cv::Mat& color, bool useTone) {
    cv::Mat d, c;
    detail.convertTo(d, CV_32FC3, 1. / 255.);
    color.convertTo(c, CV_32FC3, 1. / 255.);
    auto y = luminance(d);
    const auto yc = luminance(c);
    if (useTone) {
        y += cv::min(cv::max(blur(yc - y, 7.5), -.16), .16);
    }
    cv::Mat out(detail.size(), detail.type());
    for (int j = 0; j < Side; ++j) {
        for (int i = 0; i < Side; ++i) {
            const auto colorY = yc.at<float>(j, i);
            const auto value = y.at<float>(j, i);
            const auto centered = c.at<cv::Vec3f>(j, i) - cv::Vec3f::all(colorY);
            const float upper = std::max({centered[0], centered[1], centered[2], 1e-7f});
            const float lower = std::max({-centered[0], -centered[1], -centered[2], 1e-7f});
            const float factor = std::min({1.f, (1.f - value) / upper, value / lower});
            auto& result = out.at<cv::Vec3b>(j, i);
            for (int k = 0; k < 3; ++k) {
                result[k] = cv::saturate_cast<unsigned char>((value + factor * centered[k]) * 255.f);
            }
        }
    }
    return out;
}

cv::Mat readColor(const fs::path& file) {
    std::ifstream stream(file, std::ios::binary);
    const std::vector<unsigned char> bytes((std::istreambuf_iterator<char>(stream)), {});
    if (bytes.empty()) {
        throw std::runtime_error("Could not open a source image");
    }
    auto img = cv::imdecode(bytes, cv::IMREAD_COLOR);
    if (img.size() != cv::Size(Side, Side)) {
        throw std::runtime_error("Expected aligned 512 x 512 color crop");
    }
    return img;
}

Planes align(const Json& frames, const std::string& model) {
    if (!frames.is_array() || frames.size() != Views) {
        throw std::runtime_error("Expected eight inference outputs");
    }
    Planes aligned;
    for (std::size_t i = 0; i < Views; ++i) {
        const auto& data = frames.at(i);
        const auto tensor = path(data.at("tensor").at("path"));
        const auto output = tensor.parent_path() / (tensor.stem().string() + "-" + model + ".f32");
        std::ifstream stream(output, std::ios::binary);
        std::vector<float> values(3 * Side * Side);
        if (!stream.read(reinterpret_cast<char*>(values.data()), static_cast<std::streamsize>(values.size() * sizeof(float)))) {
            throw std::runtime_error("Unable to read inference tensor");
        }
        std::vector<cv::Mat> channels;
        for (int k : {2, 1, 0}) {
            channels.emplace_back(Side, Side, CV_32F, values.data() + k * Side * Side);
        }
        cv::Mat pixels;
        cv::merge(channels, pixels);
        if (!cv::checkRange(pixels)) {
            throw std::invalid_argument("Invalid color tensor");
        }
        pixels = cv::min(cv::max(pixels, 0.), 1.);
        const int turn = data.at("pose").at(0).get<int>();
        if (turn != 0) {
            const std::array<int, 3> inverse{cv::ROTATE_90_CLOCKWISE, cv::ROTATE_180, cv::ROTATE_90_COUNTERCLOCKWISE};
            cv::rotate(pixels, pixels, inverse.at(static_cast<std::size_t>(turn - 1)));
        }
        if (data.at("pose").at(1).get<bool>()) {
            cv::flip(pixels, pixels, 1);
        }
        cv::Matx23d matrix;
        for (int y = 0; y < 2; ++y) {
            for (int x = 0; x < 3; ++x) {
                matrix(y, x) = data.at("to_grid").at(y).at(x).get<double>();
            }
        }
        cv::warpAffine(pixels, aligned[i], matrix, cv::Size(Side, Side), cv::INTER_LINEAR, cv::BORDER_REFLECT_101);
    }
    return aligned;
}

void save(const fs::path& file, const cv::Mat& pixels) {
    std::vector<unsigned char> bytes;
    if (!cv::imencode(".png", pixels, bytes)) {
        throw std::runtime_error("Unable to encode image");
    }
    std::ofstream stream(file, std::ios::binary);
    stream.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    if (!stream) {
        throw std::runtime_error("Unable to write image");
    }
}
}

cv::Mat mixGeneratedDetail(const cv::Mat& anchor, const cv::Mat& generated) {
    const cv::Size expected(Side, Side);
    for (const auto* image : {&anchor, &generated}) {
        if (image->size() != expected || image->type() != CV_8UC3) {
            throw std::invalid_argument("Expected aligned 512x512 color portraits");
        }
    }

    cv::Mat anchorFloat, generatedFloat;
    anchor.convertTo(anchorFloat, CV_32FC3, 1. / 255.);
    generated.convertTo(generatedFloat, CV_32FC3, 1. / 255.);
    const auto yAnchor = luminance(anchorFloat);
    const auto yGenerated = luminance(generatedFloat);
    const cv::Mat delta = (yGenerated - yAnchor) - (blur(yGenerated, 4.) - blur(yAnchor, 4.));

    cv::Mat output = anchor.clone();
    for (int y = 0; y < Side; ++y) {
        const auto* correction = delta.ptr<float>(y);
        const auto* original = anchor.ptr<cv::Vec3b>(y);
        auto* target = output.ptr<cv::Vec3b>(y);
        const float dy = (y - 292.f) / 190.f;
        for (int x = 0; x < Side; ++x) {
            const float dx = (x - 256.f) / 184.f;
            const float fade = std::clamp((1.f - dx * dx - dy * dy) / .33f, 0.f, 1.f);
            const float detail = 255.f * std::clamp(.4f * fade * correction[x], -.075f, .075f);
            for (int channel = 0; channel < 3; ++channel) {
                target[x][channel] = cv::saturate_cast<unsigned char>(original[x][channel] + detail);
            }
        }
    }
    return output;
}

std::size_t runGridFusion(std::istream& manifest, const fs::path& output) {
    const auto plan = Json::parse(manifest);
    const auto& rows = plan.at("rows");
    if (!rows.is_object() || rows.empty()) {
        throw std::invalid_argument("The manifest must contain at least one named sample");
    }
    if (!output.parent_path().empty()) {
        fs::create_directories(output.parent_path());
    }
    if (!fs::create_directory(output)) {
        throw std::invalid_argument("Choose a new output directory");
    }
    for (const auto& [key, row] : rows.items()) {
        if (key.empty() ||
            key.find_first_not_of("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_") != std::string::npos) {
            throw std::invalid_argument("Invalid sample name");
        }
        cv::Mat raw;
        readColor(path(row.at("observed"))).convertTo(raw, CV_32FC3, 1. / 255.);
        const auto color = readColor(path(row.at("color")));
        const auto fhdmi = align(row.at("frames"), "fhdmi");
        const auto uhdm = align(row.at("frames"), "uhdm");
        const auto [blend, detail] = combine(fhdmi, uhdm, raw);
        const auto folder = output / ("case-" + key);
        if (!fs::create_directory(folder)) {
            throw std::invalid_argument("Duplicate sample name");
        }
        save(folder / "bandmix.png", blend);
        save(folder / "bandmix-detail.png", detail);
        save(folder / "color-mix.png", replaceColor(detail, color, false));
        save(folder / "color-tone-mix.png", replaceColor(detail, color, true));
    }
    return rows.size();
}

}
