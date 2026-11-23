#include "texture_guard.h"
#include <algorithm>
#include <cmath>
#include <opencv2/imgproc.hpp>
#include <stdexcept>
#include <vector>

namespace portrait_selection {
namespace {
cv::Mat smooth(const cv::Mat &channel, double sigma) {
    cv::Mat filtered;
    cv::GaussianBlur(channel, filtered, {}, sigma);
    return filtered;
}

// Each sample is a 16px tile whose top left falls inside the frozen face ellipse.
std::vector<double> sorted_tile_means(const cv::Mat &plane) {
    std::vector<double> samples;
    for (int y = 64; y <= 196; y += 16) {
        for (int x = 48; x <= 192; x += 16) {
            double dx = (x - 128) / 80., dy = (y - 136) / 82.;
            if (dx * dx + dy * dy < 1.) {
                samples.push_back(cv::mean(plane(cv::Rect(x, y, 16, 16)))[0]);
            }
        }
    }
    std::sort(samples.begin(), samples.end());
    return samples;
}

double log_quantile(const std::vector<double> &sorted, double probability) {
    double index = probability * static_cast<double>(sorted.size() - 1);
    size_t lower = static_cast<size_t>(index);
    double fraction = index - static_cast<double>(lower);
    return std::log(.0001 + (1. - fraction) * sorted[lower] + fraction * sorted[lower + 1]);
}
} // namespace

double TextureGuard::score(const cv::Mat &candidate) const {
    if (candidate.type() != CV_8UC3 || candidate.size() != cv::Size(512, 512))
        throw std::invalid_argument("Texture score requires an aligned 512x512 BGR image");
    cv::Mat resized;
    cv::resize(candidate, resized, {256, 256}, 0., 0., cv::INTER_AREA);
    cv::Mat values(resized.size(), CV_32FC3);
    for (int y = 0; y < resized.rows; ++y) {
        const auto *input = resized.ptr<unsigned char>(y);
        auto *output = values.ptr<float>(y);
        for (int i = 0; i < 3 * resized.cols; ++i)
            output[i] = input[i] / 255.f;
    }
    cv::Mat bgr[3];
    cv::split(values, bgr);
    const cv::Mat blue_luma = bgr[0] * .114f, green_luma = bgr[1] * .587f, red_luma = bgr[2] * .299f;
    const std::array<cv::Mat, 3> channels{blue_luma + green_luma + red_luma, bgr[2] - bgr[1],
                                          bgr[0] - bgr[1]};
    size_t feature = 0;
    double result = bias;
    for (const auto &channel : channels) {
        const std::array<cv::Mat, 5> bands{channel, smooth(channel, .6), smooth(channel, 1.3),
                                           smooth(channel, 3.), smooth(channel, 6.)};
        const cv::Mat detail = channel - bands.back();
        cv::Mat contrast = smooth(detail.mul(detail), 7.);
        cv::max(contrast, 0., contrast);
        cv::sqrt(contrast, contrast);
        contrast += .025f;
        for (size_t i = 1; i < bands.size(); ++i) {
            const cv::Mat magnitude = cv::abs(bands[i - 1] - bands[i]);
            const std::array<cv::Mat, 2> planes{magnitude, magnitude / contrast};
            for (const auto &plane : planes) {
                const auto samples = sorted_tile_means(plane);
                for (double probability : {.1, .35, .65, .9})
                    result += coefficients[feature++] * log_quantile(samples, probability);
            }
        }
    }
    return result;
}
} // namespace portrait_selection
