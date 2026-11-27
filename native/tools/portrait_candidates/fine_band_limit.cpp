#include "fine_band_limit.h"

#include <algorithm>
#include <cmath>
#include <numbers>
#include <opencv2/imgproc.hpp>
#include <stdexcept>

namespace portrait_selection {
namespace {
constexpr int side = 512;
constexpr int pad = 32;
const cv::Rect measurement(107, 160, 298, 270);

cv::Mat fine_band(const cv::Mat &luminance) {
    cv::Mat fine, broad;
    cv::GaussianBlur(luminance, fine, {}, 0.65);
    cv::GaussianBlur(luminance, broad, {}, 2.20);
    return fine - broad;
}

cv::Mat luminance(const cv::Mat &source) {
    cv::Mat rgb, gray;
    source.convertTo(rgb, CV_32FC3, 1.0 / 255.0);
    cv::cvtColor(rgb, gray, cv::COLOR_BGR2GRAY);
    return gray;
}

cv::Mat correction_direction(const cv::Mat &band) {
    cv::Mat expanded, coefficients, correction;
    cv::copyMakeBorder(band, expanded, pad, pad, pad, pad, cv::BORDER_REFLECT_101);
    expanded.convertTo(expanded, CV_64F);
    cv::dft(expanded, coefficients, cv::DFT_COMPLEX_OUTPUT);
    for (int y = 0; y < coefficients.rows; ++y) {
        auto *row = coefficients.ptr<cv::Vec2d>(y);
        const double fy = double(std::min(y, coefficients.rows - y)) / coefficients.rows;
        for (int x = 0; x < coefficients.cols; ++x) {
            const double fx = double(std::min(x, coefficients.cols - x)) / coefficients.cols;
            const double frequency = 2.0 * std::numbers::pi * std::numbers::pi * (fx * fx + fy * fy);
            const double band_response = std::exp(-0.65 * 0.65 * frequency) - std::exp(-2.20 * 2.20 * frequency);
            const double cutoff = 0.40 * 0.40 * frequency;
            const double filter = -band_response / (band_response * band_response + 0.006) * std::exp(-cutoff * cutoff);
            row[x] *= filter;
        }
    }
    cv::idft(coefficients, correction, cv::DFT_SCALE | cv::DFT_REAL_OUTPUT);
    cv::Mat cropped;
    correction(cv::Rect(pad, pad, side, side)).convertTo(cropped, CV_32F);
    return cropped;
}

cv::Mat face_weights() {
    cv::Mat weights(side, side, CV_32F);
    for (int y = 0; y < side; ++y) {
        auto *row = weights.ptr<float>(y);
        const float dy = (static_cast<float>(y) - 286.0f) / 148.0f;
        for (int x = 0; x < side; ++x) {
            const float dx = (static_cast<float>(x) - 253.0f) / 156.0f;
            const float h = std::clamp((1.20f - dx * dx - dy * dy) / 0.36f, 0.0f, 1.0f);
            row[x] = h * h * (3.0f - 2.0f * h);
        }
    }
    return weights;
}

cv::Mat adjust(const cv::Mat &source, const cv::Mat &change, const cv::Mat &weights, float strength) {
    cv::Mat output(source.size(), source.type());
    for (int y = 0; y < source.rows; ++y) {
        const auto *original = source.ptr<cv::Vec3b>(y);
        auto *final = output.ptr<cv::Vec3b>(y);
        const auto *correction = change.ptr<float>(y);
        const auto *weight = weights.ptr<float>(y);
        for (int x = 0; x < source.cols; ++x) {
            const auto pixel = original[x];
            const float offset = std::clamp(correction[x] * strength, -0.091f, 0.091f) * 255.0f * weight[x];
            const float delta = std::clamp(offset, -float(std::min({pixel[0], pixel[1], pixel[2]})),
                                           255.0f - std::max({pixel[0], pixel[1], pixel[2]}));
            const int gray_offset = cvRound(delta);
            for (int channel = 0; channel < 3; ++channel)
                final[x][channel] = static_cast<uchar>(pixel[channel] + gray_offset);
        }
    }
    return output;
}
} // namespace

FineBandResult cap_fine_band(const cv::Mat &source, double maximum_rms) {
    if (source.size() != cv::Size(side, side) || source.type() != CV_8UC3)
        throw std::invalid_argument("Fine band limiting requires an aligned 512x512 BGR image");
    if (!std::isfinite(maximum_rms) || maximum_rms <= 0.0 || maximum_rms > 1.0)
        throw std::invalid_argument("The fine band limit must be greater than zero and at most one");
    const cv::Mat band = fine_band(luminance(source));
    const double measured = cv::norm(band(measurement), cv::NORM_L2);
    const double normalization = std::sqrt(static_cast<double>(measurement.area()));
    const double initial_rms = measured / normalization;
    if (initial_rms <= 1.02 * maximum_rms)
        return {source.clone(), initial_rms, true};
    const double target = maximum_rms * normalization;
    const cv::Mat change = correction_direction(band);
    const cv::Mat mask = face_weights();
    double lower = 0.0, upper = 0.95;
    cv::Mat result;
    double achieved = measured;
    for (int step = 0; step < 19; ++step) {
        const double strength = (lower + upper) / 2.0;
        result = adjust(source, change, mask, static_cast<float>(strength));
        const cv::Mat changed = fine_band(luminance(result));
        achieved = cv::norm(changed(measurement), cv::NORM_L2);
        if (achieved > target)
            lower = strength;
        else
            upper = strength;
    }
    const double rms = achieved / normalization;
    return {result, rms, rms <= 1.02 * maximum_rms};
}
} // namespace portrait_selection
