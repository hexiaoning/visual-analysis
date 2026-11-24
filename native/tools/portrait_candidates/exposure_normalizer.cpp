#include "exposure_normalizer.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <nlohmann/json.hpp>
#include <numeric>
#include <opencv2/imgproc.hpp>
#include <stdexcept>
#include <vector>

namespace portrait_selection {
namespace {
constexpr int side = 512;
constexpr float linear_floor = .003f;
constexpr size_t feature_count = 99;
constexpr size_t output_count = 4;
using Values = std::vector<double>;

cv::Mat log_linear_bgr(const cv::Mat &image) {
    std::array<float, 256> lookup{};
    for (size_t i = 0; i < lookup.size(); ++i) {
        const float value = static_cast<float>(i) / 255.f;
        const float linear = value <= .04045f ? value / 12.92f : std::pow((value + .055f) / 1.055f, 2.4f);
        lookup[i] = std::log(linear + linear_floor);
    }
    cv::Mat result(image.size(), CV_32FC3);
    for (int y = 0; y < image.rows; ++y) {
        const auto *input = image.ptr<unsigned char>(y);
        auto *output = result.ptr<float>(y);
        for (int x = 0; x < 3 * image.cols; ++x)
            output[x] = lookup[input[x]];
    }
    return result;
}

template <size_t count>
using Row = std::array<double, count>;

struct ExposureModel {
    Row<feature_count> center;
    Row<feature_count> scale;
    std::vector<Row<feature_count>> basis;
    std::vector<Row<output_count>> weights;
    Values classifier;
    double cutoff;
};

ExposureModel read_exposure_model(const std::filesystem::path &path) {
    using Json = nlohmann::json;
    std::ifstream stream(path);
    if (!stream)
        throw std::runtime_error("Cannot open exposure model");
    const auto json = Json::parse(stream);
    ExposureModel model{json.at("center").get<Row<feature_count>>(),
                        json.at("scale").get<Row<feature_count>>(),
                        json.at("centers").get<std::vector<Row<feature_count>>>(),
                        json.at("weights").get<std::vector<Row<output_count>>>(),
                        json.at("classifier").get<Values>(), json.at("gate_limit").get<double>()};
    const size_t dimensions = 1 + feature_count + model.basis.size();
    const auto finite = [](const auto &row) {
        return std::all_of(row.begin(), row.end(), [](double value) { return std::isfinite(value); });
    };
    if (model.basis.empty() || model.basis.size() > 1024 || model.weights.size() != dimensions ||
        model.classifier.size() != dimensions || !std::isfinite(model.cutoff) ||
        !finite(model.center) || !finite(model.scale) || !finite(model.classifier) ||
        std::any_of(model.scale.begin(), model.scale.end(), [](double value) { return value <= 0.; }) ||
        !std::all_of(model.basis.begin(), model.basis.end(), finite) ||
        !std::all_of(model.weights.begin(), model.weights.end(), finite))
        throw std::invalid_argument("Invalid exposure model");
    return model;
}

double quantile(const Values &sorted, double probability) {
    const double idx = probability * static_cast<double>(sorted.size() - 1);
    const size_t left = static_cast<size_t>(idx);
    const double fraction = idx - static_cast<double>(left);
    return sorted[left] + fraction * (sorted[left + 1] - sorted[left]);
}

Row<feature_count> describe(const cv::Mat &pixels) {
    constexpr std::array quantiles{.03, .10, .25, .50, .75, .90, .97, .995};
    constexpr std::array color_quantiles{.15, .50, .85};
    const std::array<cv::Rect, 3> regions{{{0, 0, 96, 96}, {22, 30, 52, 51}, {24, 41, 48, 28}}};
    cv::Mat reduced;
    cv::resize(pixels, reduced, {96, 96}, 0., 0., cv::INTER_AREA);
    Row<feature_count> result{};
    size_t index = 0;
    for (const auto &region : regions) {
        std::array<Values, 3> channels;
        std::array<Values, 2> chroma;
        for (auto &values : channels)
            values.reserve(region.area());
        for (auto &values : chroma)
            values.reserve(region.area());
        const cv::Mat patch = reduced(region);
        cv::Scalar mean, deviation;
        cv::meanStdDev(patch, mean, deviation);
        for (int y = 0; y < patch.rows; ++y) {
            const auto *row = patch.ptr<cv::Vec3f>(y);
            for (int x = 0; x < patch.cols; ++x) {
                const auto point = row[x];
                const float total = std::max(.075f, point[0] + point[1] + point[2]);
                for (size_t ch = 0; ch < channels.size(); ++ch)
                    channels[ch].push_back(point[static_cast<int>(ch)]);
                chroma[0].push_back(point[0] / total);
                chroma[1].push_back(point[2] / total);
            }
        }
        for (auto &values : channels)
            std::sort(values.begin(), values.end());
        for (auto &values : chroma)
            std::sort(values.begin(), values.end());
        for (const auto percentile : quantiles)
            for (const auto &values : channels)
                result[index++] = quantile(values, percentile);
        for (const auto percentile : color_quantiles)
            for (const auto &values : chroma)
                result[index++] = quantile(values, percentile);
        for (int ch = 0; ch < 3; ++ch)
            result[index++] = deviation[ch];
    }
    return result;
}

cv::Mat interpolation(int n) {
    cv::Mat identity = cv::Mat::eye(n, n, CV_32F), up;
    cv::resize(identity, up, {side, n}, 0., 0., cv::INTER_LINEAR);
    return up.t();
}

cv::Mat normalized(const cv::Mat &line) {
    cv::Mat centered = line - cv::mean(line)[0];
    return centered / std::max(cv::norm(centered, cv::NORM_L2), 1.e-8);
}

int estimate_grid(const cv::Mat &pixels) {
    cv::Mat gray;
    cv::cvtColor(pixels, gray, cv::COLOR_BGR2GRAY);
    const auto projection = [](const cv::Mat &img) {
        cv::Mat difference = img.rowRange(2, side) - img.rowRange(1, side - 1) * 2.f + img.rowRange(0, side - 2);
        cv::Mat clipped, averaged, smooth;
        cv::min(cv::abs(difference), .11, clipped);
        cv::reduce(clipped, averaged, 1, cv::REDUCE_AVG);
        cv::GaussianBlur(averaged.t(), smooth, {0, 1}, 3.3);
        cv::Mat band = averaged - smooth.t();
        return normalized(band.rowRange(15, side - 2 - 15));
    };
    const cv::Mat rows = projection(gray), columns = projection(gray.t());
    double best = -2., next = -2.;
    int found = 0;
    for (int n = 32; n < 104; ++n) {
        const cv::Mat up = interpolation(n);
        const cv::Mat edges = up.rowRange(2, side) - up.rowRange(1, side - 1) * 2.f + up.rowRange(0, side - 2);
        cv::Mat magnitude;
        cv::reduce(cv::abs(edges), magnitude, 1, cv::REDUCE_SUM);
        const cv::Mat shape = normalized(magnitude.rowRange(15, side - 2 - 15));
        const double score = std::min(rows.dot(shape), columns.dot(shape));
        if (score >= best) {
            next = best;
            best = score;
            found = n;
        } else if (score > next) {
            next = score;
        }
    }
    return best >= .67 && best - next >= .11 ? found : 0;
}

cv::Mat correct(const cv::Mat &pixels, int grid, const std::array<double, 3> &gain, double black) {
    std::array<cv::Mat, 3> channels;
    cv::split(pixels, channels.data());
    cv::Mat inverse;
    if (grid) {
        const cv::Mat up = interpolation(grid);
        if (!cv::solve(up.t() * up, up.t(), inverse, cv::DECOMP_CHOLESKY))
            throw std::runtime_error("Failed to reconstruct the sampling grid");
    }
    for (size_t channel = 0; channel < channels.size(); ++channel) {
        cv::Mat &plane = channels[channel];
        if (grid) {
            plane = inverse * plane * inverse.t();
            cv::max(plane, 0., plane);
            cv::min(plane, 1., plane);
        }
        for (int y = 0; y < plane.rows; ++y) {
            float *values = plane.ptr<float>(y);
            for (int x = 0; x < plane.cols; ++x) {
                const float normalized = (values[x] - static_cast<float>(black)) / static_cast<float>(gain[channel]);
                values[x] = std::clamp(normalized, 0.f, 1.f);
            }
        }
    }
    cv::Mat corrected;
    cv::merge(channels.data(), channels.size(), corrected);
    if (grid)
        cv::resize(corrected, corrected, {side, side}, 0., 0., cv::INTER_CUBIC);
    cv::Mat result(side, side, CV_8UC3);
    for (int y = 0; y < side; ++y) {
        const auto *input = corrected.ptr<float>(y);
        auto *output = result.ptr<unsigned char>(y);
        for (int x = 0; x < 3 * side; ++x)
            output[x] = static_cast<unsigned char>(std::clamp(input[x] * 255.f + .5f, 0.f, 255.f));
    }
    return result;
}
}

ExposureResult normalize_exposure(const cv::Mat &image, const std::filesystem::path &model_path) {
    if (image.type() != CV_8UC3 || image.size() != cv::Size(side, side))
        throw std::invalid_argument("Exposure normalization requires an aligned 512x512 BGR image");
    const ExposureModel model = read_exposure_model(model_path);
    cv::Mat pixels(image.size(), CV_32FC3);
    // Match the training domain without an intermediate 8-bit reduction.
    for (int y = 0; y < side; ++y) {
        const auto *input = image.ptr<unsigned char>(y);
        auto *output = pixels.ptr<float>(y);
        for (int x = 0; x < 3 * side; ++x)
            output[x] = input[x] / 255.f;
    }
    auto features = describe(pixels);
    for (size_t i = 0; i < features.size(); ++i)
        features[i] = std::clamp((features[i] - model.center[i]) / model.scale[i], -6., 6.);

    Values design{1.};
    design.insert(design.end(), features.begin(), features.end());
    for (const auto &point : model.basis) {
        double distance = 0.;
        for (size_t i = 0; i < features.size(); ++i) {
            const double difference = point[i] - features[i];
            distance += difference * difference;
        }
        design.push_back(std::exp(-distance / (1.45 * feature_count)));
    }
    const double evidence = std::inner_product(design.begin(), design.end(), model.classifier.begin(), 0.);
    const int grid = estimate_grid(pixels);
    if (evidence < model.cutoff)
        return {image, {1., 1., 1.}, 0., evidence, grid, false};
    Row<output_count> values{};
    for (size_t i = 0; i < design.size(); ++i)
        for (size_t j = 0; j < values.size(); ++j)
            values[j] += design[i] * model.weights[i][j];
    std::array<double, 3> gain{std::clamp(values[0], .24, .83), std::clamp(values[1], .24, .83),
                               std::clamp(values[2], .24, .83)};
    const double black = std::clamp(values[3], 0., .06);
    return {correct(pixels, grid, gain, black), gain, black, evidence, grid, true};
}

cv::Mat match_candidate_tone(const cv::Mat &candidate, const cv::Mat &guide) {
    for (const auto *image : {&candidate, &guide}) {
        if (image->type() != CV_8UC3 || image->size() != cv::Size(side, side))
            throw std::invalid_argument("Tone matching requires two aligned 512x512 BGR images");
    }
    const auto pixels = log_linear_bgr(candidate);
    const cv::Mat delta = log_linear_bgr(guide) - pixels;
    cv::Mat coarse, correction;
    cv::resize(delta, coarse, {64, 64}, 0., 0., cv::INTER_AREA);
    cv::GaussianBlur(coarse, coarse, {0, 0}, 6.);
    cv::resize(coarse, correction, candidate.size(), 0., 0., cv::INTER_LINEAR);

    cv::Mat result(candidate.size(), candidate.type());
    for (int y = 0; y < side; ++y) {
        const auto *input = pixels.ptr<cv::Vec3f>(y);
        const auto *shift = correction.ptr<cv::Vec3f>(y);
        auto *output = result.ptr<cv::Vec3b>(y);
        const float dy = (static_cast<float>(y) - 297.f) / 207.f;
        for (int x = 0; x < side; ++x) {
            const float dx = (static_cast<float>(x) - 256.f) / 215.f;
            const float radius = std::sqrt(dx * dx + dy * dy);
            const float fade = std::clamp((radius - .67f) / .33f, 0.f, 1.f);
            const float mix = .75f * (.5f + .5f * std::cos(static_cast<float>(CV_PI) * fade));
            cv::Vec3f linear;
            for (int channel = 0; channel < 3; ++channel) {
                const float adjustment = mix * std::clamp(shift[x][channel], -.77f, .77f);
                linear[channel] = std::max(std::exp(input[x][channel] + adjustment) - linear_floor, 0.f);
            }
            // Use one highlight scale for all channels to retain the corrected color.
            const float highlight = std::max({1.f, linear[0], linear[1], linear[2]});
            for (int channel = 0; channel < 3; ++channel) {
                const float value = linear[channel] / highlight;
                const float srgb = value <= .0031308f ? 12.92f * value : 1.055f * std::pow(value, 1.f / 2.4f) - .055f;
                output[x][channel] = cv::saturate_cast<unsigned char>(srgb * 255.f);
            }
        }
    }
    return result;
}
}
