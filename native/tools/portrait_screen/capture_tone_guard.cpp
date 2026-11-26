#include "screen_restore.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>
#include <tuple>

#include <opencv2/imgproc.hpp>

namespace visual::screen {
namespace {

using Pixels = std::array<std::vector<cv::Point>, 2>;

cv::Mat luma(const cv::Mat& pixels) {
    cv::Mat f, y;
    pixels.convertTo(f, CV_32FC3, 1. / 255.);
    cv::cvtColor(f, y, cv::COLOR_BGR2GRAY);
    return y;
}

cv::Mat blur(const cv::Mat& image, double sigma) {
    cv::Mat output;
    cv::GaussianBlur(image, output, {}, sigma, 0., cv::BORDER_REFLECT_101);
    return output;
}

cv::Mat wave(const cv::Mat& image) {
    return blur(image, 1.1) - blur(image, 3.4);
}

std::array<cv::Mat, 4> windows() {
    std::array<cv::Mat, 4> result;
    const std::array<cv::Point, 4> positions{{{193, 240}, {319, 240}, {257, 314}, {257, 371}}};
    const std::array<cv::Point, 4> scales{{{45, 31}, {45, 31}, {37, 47}, {61, 35}}};
    for (std::size_t c = 0; c < result.size(); ++c) {
        result[c].create(512, 512, CV_32F);
        for (int y = 0; y < 512; ++y) {
            for (int x = 0; x < 512; ++x) {
                const auto dx = static_cast<double>(x - positions[c].x) / scales[c].x;
                const auto dy = static_cast<double>(y - positions[c].y) / scales[c].y;
                result[c].at<float>(y, x) = static_cast<float>(std::exp(-.5 * (dx * dx + dy * dy)));
            }
        }
    }
    return result;
}

double contour(const cv::Mat& a, const cv::Mat& b, const cv::Mat& w) {
    return cv::sum(a.mul(b).mul(w))[0] /
        std::max(std::sqrt(cv::sum(a.mul(a).mul(w))[0] * cv::sum(b.mul(b).mul(w))[0]), 1.e-8);
}

Pixels folds(const cv::Mat& image) {
    cv::Mat fractional;
    image.convertTo(fractional, CV_32FC3, 1. / 255.);
    const auto c = blur(fractional, 5.);
    Pixels p;
    for (int y = 0; y < c.rows; ++y) {
        for (int x = 0; x < c.cols; ++x) {
            const auto dx = (x - 256.) / 137.;
            const auto dy = (y - 302.) / 153.;
            const auto v = c.at<cv::Vec3f>(y, x);
            if (dx * dx + dy * dy < 1. && *std::min_element(v.val, v.val + 3) > .05f
                && *std::max_element(v.val, v.val + 3) < .93f) {
                p[static_cast<std::size_t>((x / 32 + y / 32) % 2 == 0)].emplace_back(x, y);
            }
        }
    }
    return p;
}

double quantile(const std::vector<float>& sorted, double value) {
    const auto index = value * static_cast<double>(sorted.size() - 1);
    const auto lo = static_cast<std::size_t>(index);
    return sorted[lo] + (sorted[std::min(lo + 1, sorted.size() - 1)] - sorted[lo])
        * (index - static_cast<double>(lo));
}

std::vector<float> sampled(const cv::Mat& m, const std::vector<cv::Point>& points) {
    std::vector<float> result;
    result.reserve(points.size());
    for (const auto& point : points) {
        result.push_back(m.at<float>(point));
    }
    std::sort(result.begin(), result.end());
    return result;
}

struct Evidence {
    double center;
    double absolute;
    double low;
};

std::array<Evidence, 2> evidence(const cv::Mat& source, const cv::Mat& v, const Pixels& p) {
    std::array<Evidence, 2> result{};
    const cv::Mat residual = source - v;
    for (std::size_t i = 0; i < p.size(); ++i) {
        const auto z = sampled(residual, p[i]);
        double total = 0.;
        for (const auto value : z) {
            total += std::abs(value);
        }
        result[i] = {quantile(z, .5), total / static_cast<double>(z.size()), quantile(z, .25)};
    }
    return result;
}

cv::Mat tone(const cv::Mat& source, const cv::Mat& first, const cv::Mat& image, const Pixels& p) {
    std::array<double, 2> gain{}, bias{};
    for (std::size_t c = 0; c < p.size(); ++c) {
        const auto x = sampled(first, p[c]);
        const auto z = sampled(source, p[c]);
        gain[c] = std::clamp((quantile(z, .8) - quantile(z, .2)) /
            std::max(quantile(x, .8) - quantile(x, .2), .025), .8, 1.4);
        bias[c] = std::clamp(quantile(z, .5) - gain[c] * quantile(x, .5), -.115, .115);
    }
    if (std::abs(gain[0] - gain[1]) > .12 || std::abs(bias[0] - bias[1]) > .030) {
        return image;
    }
    const auto g = .5 * (gain[0] + gain[1]);
    const auto b = .5 * (bias[0] + bias[1]);
    cv::Mat out(image.size(), image.type());
    for (int y = 0; y < out.rows; ++y) {
        for (int x = 0; x < out.cols; ++x) {
            const auto dx = (x - 256.) / 135.;
            const auto dy = (y - 302.) / 140.;
            const auto mask = std::clamp((std::exp(-.5 * (dx * dx + dy * dy)) - .018) / .76, 0., 1.);
            const auto d = std::clamp(.85 * ((g - 1.) * first.at<float>(y, x) + b), -.10, .15) * mask;
            const auto origin = image.at<cv::Vec3b>(y, x);
            auto& target = out.at<cv::Vec3b>(y, x);
            for (int c = 0; c < 3; ++c) {
                target[c] = cv::saturate_cast<uchar>(origin[c] + 255. * d);
            }
        }
    }
    return out;
}

void requireAlignedFace(const cv::Mat& image) {
    if (image.size() != cv::Size(512, 512) || image.type() != CV_8UC3) {
        throw std::invalid_argument("Expected an aligned 512x512 color crop");
    }
}

}

CaptureToneChoice chooseCaptureTone(const cv::Mat& observed, const cv::Mat& before,
                                    const std::vector<cv::Mat>& alternatives, double minimumDeficit) {
    requireAlignedFace(observed);
    requireAlignedFace(before);
    for (const auto& alternative : alternatives) {
        requireAlignedFace(alternative);
    }
    if (!std::isfinite(minimumDeficit) || minimumDeficit < 0. || minimumDeficit > 255.) {
        throw std::invalid_argument("The capture tone gap must be between 0 and 255");
    }

    CaptureToneChoice selected{before, std::nullopt, false};
    const auto points = folds(observed);
    if (std::any_of(points.begin(), points.end(), [](const auto& fold) { return fold.size() < 1200; })) {
        return selected;
    }
    const auto observedY = luma(observed);
    const auto low = blur(observedY, 7.5);
    const auto firstY = luma(before);
    const auto first = blur(firstY, 7.5);
    const auto profile = evidence(low, first, points);
    if (std::any_of(profile.begin(), profile.end(), [&](const auto& fold) {
        return fold.center < minimumDeficit / 255. || fold.low < .010;
    })) {
        return selected;
    }

    const auto current = wave(firstY);
    const auto anchor = wave(observedY);
    static const auto weights = windows();
    std::array<double, 4> currentAgreement{};
    for (std::size_t i = 0; i < weights.size(); ++i) {
        currentAgreement[i] = contour(current, anchor, weights[i]);
    }
    std::optional<std::tuple<double, double, double>> highest;
    auto consider = [&](const cv::Mat& candidate, std::optional<std::size_t> alternative, bool adjustedTone) {
        const auto candidateY = luma(candidate);
        const auto signal = evidence(low, blur(candidateY, 7.5), points);
        double benefit = 1.;
        for (std::size_t i = 0; i < profile.size(); ++i) {
            const auto baseline = profile[i];
            const auto proposed = signal[i];
            if (baseline.absolute - proposed.absolute < .017 || proposed.absolute > .68 * baseline.absolute ||
                proposed.center < -.03 || proposed.center > .72 * baseline.center) {
                return;
            }
            benefit = std::min(benefit, baseline.absolute - proposed.absolute);
        }
        const auto candidateBand = wave(candidateY);
        double shape = 1.;
        double agreement = 1.;
        for (std::size_t i = 0; i < weights.size(); ++i) {
            shape = std::min(shape, contour(candidateBand, current, weights[i]));
            agreement = std::min(agreement, contour(candidateBand, anchor, weights[i]) - currentAgreement[i]);
        }
        if (shape < .88 || agreement < -.045) {
            return;
        }
        const auto rank = std::tuple{benefit, shape, agreement};
        if (!highest || rank > *highest) {
            highest = rank;
            selected = {candidate, alternative, adjustedTone};
        }
    };
    for (std::size_t i = 0; i < alternatives.size(); ++i) {
        consider(alternatives[i], i, false);
    }
    consider(tone(low, first, before, points), std::nullopt, true);
    return selected;
}

}
