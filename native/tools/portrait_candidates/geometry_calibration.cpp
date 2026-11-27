#include "geometry_calibration.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <nlohmann/json.hpp>
#include <opencv2/imgproc.hpp>
#include <stdexcept>
#include <vector>

namespace portrait_selection {
namespace {

constexpr int side = 512;
constexpr size_t feature_count = 31;
using Features = std::array<double, feature_count>;

struct GeometryModel {
    Features center;
    Features scale;
    std::array<double, 2> bias;
    std::array<std::array<double, 2>, feature_count> weights;
};

GeometryModel read_model(const std::filesystem::path &path) {
    std::ifstream stream(path);
    if (!stream)
        throw std::runtime_error("Cannot open capture geometry model");
    const auto json = nlohmann::json::parse(stream);
    const auto count = [](const auto &row, size_t n) { return row.is_array() && row.size() == n; };
    const auto &weights = json.at("weights");
    if (json.at("schema") != "portrait-capture-geometry-linear-v1" || !count(json.at("center"), feature_count) ||
        !count(json.at("scale"), feature_count) || !count(json.at("bias"), 2) || !count(weights, feature_count) ||
        !std::all_of(weights.begin(), weights.end(), [&](const auto &row) { return count(row, 2); }))
        throw std::invalid_argument("Invalid capture geometry model shape");
    const GeometryModel model{json.at("center").get<Features>(), json.at("scale").get<Features>(),
                              json.at("bias").get<std::array<double, 2>>(),
                              weights.get<decltype(GeometryModel::weights)>()};
    const auto finite = [](const auto &row) {
        return std::all_of(row.begin(), row.end(), [](double value) { return std::isfinite(value); });
    };
    if (!finite(model.center) || !finite(model.scale) || !finite(model.bias) ||
        !std::all_of(model.weights.begin(), model.weights.end(), finite) ||
        std::any_of(model.scale.begin(), model.scale.end(), [](double value) { return value <= 0.; }))
        throw std::invalid_argument("Invalid capture geometry model coefficients");
    return model;
}

cv::Mat gaussian(const cv::Mat &image, double sigma) {
    cv::Mat result;
    cv::GaussianBlur(image, result, {0, 0}, sigma);
    return result;
}

cv::Mat normalized_bands(const cv::Mat &image) {
    cv::Mat unit, gray;
    image.convertTo(unit, CV_32FC3, 1. / 255.);
    cv::cvtColor(unit, gray, cv::COLOR_BGR2GRAY);
    std::array<cv::Mat, 2> bands{gaussian(gray, 1.1) - gaussian(gray, 5.5),
                                 gaussian(gray, 2.1) - gaussian(gray, 9.2)};
    for (auto &band : bands) {
        cv::Mat energy = gaussian(band.mul(band), 6.);
        cv::sqrt(energy + .00017, energy);
        cv::divide(band, energy, band);
    }
    cv::Mat result;
    cv::merge(bands.data(), bands.size(), result);
    return result;
}

struct Sample {
    cv::Vec2d gradient;
    double error;
    double weight;
    int band;
    int fold;
};

cv::Vec2d displacement(const std::vector<Sample> &rows, int band, int held_out) {
    const auto used = [&](const auto &row) { return (band < 0 || row.band == band) && row.fold != held_out; };
    cv::Vec2d ridge{};
    for (const auto &row : rows)
        if (used(row))
            ridge += row.weight * row.gradient.mul(row.gradient);
    ridge = {std::max(ridge[0], 1.e-8) * .036, std::max(ridge[1], 1.e-8) * .036};
    cv::Vec2d result{};
    for (int iteration = 0; iteration < 2; ++iteration) {
        double xx = ridge[0], xy = 0., yy = ridge[1];
        cv::Vec2d target{};
        for (const auto &row : rows) {
            if (!used(row))
                continue;
            const double residual = row.error - row.gradient.dot(result);
            const double w = row.weight * std::min(1., .48 / std::max(std::abs(residual), 1.e-7));
            const auto d = row.gradient;
            xx += w * d[0] * d[0];
            xy += w * d[0] * d[1];
            yy += w * d[1] * d[1];
            target += (w * row.error) * d;
        }
        const double determinant = xx * yy - xy * xy;
        result = {(yy * target[0] - xy * target[1]) / determinant, (xx * target[1] - xy * target[0]) / determinant};
    }
    return result;
}

double held_gain(const std::vector<Sample> &samples, const cv::Vec2d &shift, int fold) {
    double before = 0., after = 0., weights = 0.;
    for (const auto &row : samples) {
        if (row.fold != fold)
            continue;
        const double a = std::min(std::abs(row.error), .9);
        const double b = std::min(std::abs(row.error - row.gradient.dot(shift)), .9);
        before += row.weight * a * a;
        after += row.weight * b * b;
        weights += row.weight;
    }
    return (before - after) / std::max(before, .00001 * weights);
}

struct Pixel {
    cv::Point position;
    cv::Vec2f base;
    std::array<cv::Vec2d, 2> gradients;
    double weight;
    int fold;
};

std::vector<Pixel> neighborhood(const cv::Mat &base, const cv::Mat &gx, const cv::Mat &gy, int x, int y) {
    // Match the training descriptor's alternating spatial blocks, including negative offsets.
    const auto tile = [](int p) { return p >= 0 ? p / 10 : (p - 9) / 10; };
    std::vector<Pixel> pixels;
    pixels.reserve(35 * 35);
    for (int dy = -34; dy <= 34; dy += 2) {
        for (int dx = -34; dx <= 34; dx += 2) {
            const cv::Point p{x + dx, y + dy};
            const auto u = gx.at<cv::Vec2f>(p);
            const auto v = gy.at<cv::Vec2f>(p);
            const double contrast = std::sqrt(.5 * (u.dot(u) + v.dot(v)));
            const double window = std::exp(-(dx * dx + dy * dy) / 800.);
            const int fold = (tile(dx) + tile(dy)) % 2 == 0 ? 1 : 0;
            pixels.push_back({p, base.at<cv::Vec2f>(p), {{{u[0], v[0]}, {u[1], v[1]}}},
                              window * std::clamp(contrast / .17, .08, 1.), fold});
        }
    }
    return pixels;
}

void describe_observation(const std::vector<Pixel> &pixels, const cv::Mat &other, double *values) {
    std::vector<Sample> rows;
    rows.reserve(pixels.size() * 2);
    cv::Vec2d bias{};
    double weight = 0., ab = 0., aa = 0., bb = 0.;
    for (const auto &pixel : pixels) {
        const auto observed = other.at<cv::Vec2f>(pixel.position);
        const auto base = pixel.base;
        for (int band = 0; band < 2; ++band) {
            const double error = static_cast<double>(observed[band]) - base[band];
            bias[band] += pixel.weight * error;
            rows.push_back({pixel.gradients[band], error, pixel.weight, band, pixel.fold});
        }
        weight += pixel.weight;
        ab += observed.dot(base);
        aa += observed.dot(observed);
        bb += base.dot(base);
    }
    bias *= 1. / weight;
    for (auto &row : rows)
        row.error -= bias[row.band];

    const auto all = displacement(rows, -1, -1);
    const auto fine = displacement(rows, 0, -1);
    const auto coarse = displacement(rows, 1, -1);
    const auto first = displacement(rows, -1, 0);
    const auto second = displacement(rows, -1, 1);
    const std::array offsets{all[0], all[1], fine[0], fine[1], coarse[0], coarse[1]};
    std::transform(offsets.begin(), offsets.end(), values, [](double v) { return std::clamp(v, -4., 4.); });
    values[6] = std::clamp(std::min(held_gain(rows, first, 0), held_gain(rows, second, 1)), -.5, .75);
    values[7] = cv::norm(first - second);
    const double count = static_cast<double>(pixels.size()) * 2.;
    values[8] = ab / std::max(std::sqrt(aa * bb), count * 1.e-6);
}

cv::Vec2f predict(const std::vector<Pixel> &pixels, const std::array<cv::Mat, 3> &observations, int x, int y,
                  const GeometryModel &model) {
    Features features{};
    for (size_t i = 0; i < observations.size(); ++i)
        describe_observation(pixels, observations[i], features.data() + 9 * i);
    double xx = 0., xy = 0., yy = 0.;
    for (const auto &p : pixels) {
        for (const auto &g : p.gradients) {
            xx += g[0] * g[0];
            xy += g[0] * g[1];
            yy += g[1] * g[1];
        }
    }
    const double trace = xx + yy;
    const double root = std::hypot(xx - yy, 2. * xy);
    features[27] = std::log(std::max(trace, .001));
    features[28] = .5 * (trace - root) / std::max(.5 * (trace + root), .0001);
    features[29] = (x - 257.) / 147.;
    features[30] = (y - 280.) / 128.;
    auto shift = model.bias;
    for (size_t i = 0; i < features.size(); ++i) {
        const double v = std::clamp((features[i] - model.center[i]) / model.scale[i], -3.8, 3.8);
        for (size_t j = 0; j < shift.size(); ++j)
            shift[j] += model.weights[i][j] * v;
    }
    if (!std::isfinite(shift[0]) || !std::isfinite(shift[1]))
        throw std::invalid_argument("Capture geometry model produced a nonfinite displacement");
    return {.82f * static_cast<float>(std::clamp(shift[0], -1.35, 1.35)),
            .82f * static_cast<float>(std::clamp(shift[1], -1.35, 1.35))};
}

float face_weight(int x, int y) {
    const float dx = (static_cast<float>(x) - 255.f) / 162.f;
    const float dy = (static_cast<float>(y) - 290.f) / 141.f;
    const float a = std::clamp((1.11f - (dx * dx + dy * dy)) / .33f, 0.f, 1.f);
    return a * a * (3.f - 2.f * a);
}

} // namespace

cv::Mat calibrate_capture_geometry(const std::array<cv::Mat, 3> &observations, const cv::Mat &candidate,
                                   const std::filesystem::path &model_path) {
    const auto valid = [](const cv::Mat &image) { return image.size() == cv::Size(side, side) && image.type() == CV_8UC3; };
    if (!valid(candidate) || !std::all_of(observations.begin(), observations.end(), valid))
        throw std::invalid_argument("Capture geometry calibration requires four aligned 512x512 BGR images");
    const auto model = read_model(model_path);
    const auto base = normalized_bands(candidate);
    const std::array guides{normalized_bands(observations[0]), normalized_bands(observations[1]),
                            normalized_bands(observations[2])};
    cv::Mat gx, gy;
    cv::Sobel(base, gx, CV_32F, 1, 0, 3, .125);
    cv::Sobel(base, gy, CV_32F, 0, 1, 3, .125);

    cv::Mat field = cv::Mat::zeros(candidate.size(), CV_32FC2);
    cv::Mat coverage = cv::Mat::zeros(candidate.size(), CV_32F);
    for (int y = 172; y <= 412; y += 48) {
        for (int x = 136; x <= 376; x += 48) {
            const auto shift = predict(neighborhood(base, gx, gy, x, y), guides, x, y, model);
            for (int dy = -44; dy <= 44; ++dy) {
                const float u = std::cos(static_cast<float>(CV_PI) * .5f * dy / 45.f);
                for (int dx = -44; dx <= 44; ++dx) {
                    const float v = std::cos(static_cast<float>(CV_PI) * .5f * dx / 45.f);
                    const float w = u * u * v * v;
                    coverage.at<float>(y + dy, x + dx) += w;
                    field.at<cv::Vec2f>(y + dy, x + dx) += shift * w;
                }
            }
        }
    }

    cv::Mat mask(candidate.size(), CV_8U);
    for (int y = 0; y < side; ++y) {
        auto *sample = field.ptr<cv::Vec2f>(y);
        auto *insides = mask.ptr<uchar>(y);
        const auto *support = coverage.ptr<float>(y);
        for (int x = 0; x < side; ++x) {
            const float weight = face_weight(x, y);
            sample[x] *= weight / std::max(support[x], .17f);
            sample[x] += cv::Vec2f(static_cast<float>(x), static_cast<float>(y));
            insides[x] = weight == 0.f ? 255 : 0;
        }
    }
    cv::Mat result;
    cv::remap(candidate, result, field, {}, cv::INTER_CUBIC, cv::BORDER_REFLECT_101);
    candidate.copyTo(result, mask);
    return result;
}

} // namespace portrait_selection
