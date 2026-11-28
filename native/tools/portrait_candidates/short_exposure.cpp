#include "short_exposure.h"

#include <algorithm>
#include <stdexcept>

namespace portrait_selection {
namespace {

int ceil_divide(int numerator, int denominator) {
    return numerator / denominator + (numerator % denominator > 0);
}

int floor_divide(int numerator, int denominator) {
    return numerator / denominator - (numerator % denominator < 0);
}

void check_image(const cv::Mat& image) {
    if (image.empty() || image.type() != CV_8UC3)
        throw std::invalid_argument("Expected a nonempty three-channel RGB8 image");
}

void check_pair(const cv::Mat& first, const cv::Mat& second) {
    check_image(first);
    check_image(second);
    if (first.size() != second.size())
        throw std::invalid_argument("Exposure image dimensions differ");
}

} // namespace

ShortExposureCamera::ShortExposureCamera(int short_weight, int long_weight, int companion_tolerance)
    : short_weight_(short_weight), long_weight_(long_weight) {
    if (short_weight < 1 || short_weight > max_weight || long_weight < 1 || long_weight > max_weight ||
        companion_tolerance < 0 || companion_tolerance > 255)
        throw std::invalid_argument("Exposure weights or companion tolerance are out of range");
    total_weight_ = short_weight + long_weight;
    for (int level = 0; level <= 255; ++level) {
        const Interval sum{
            std::max(0, ceil_divide((2 * level - 1) * total_weight_, 2)),
            std::min(255 * total_weight_, ceil_divide((2 * level + 1) * total_weight_, 2) - 1)};
        const Interval allowed_long{std::max(0, level - companion_tolerance),
                                    std::min(255, level + companion_tolerance)};
        // The unclipped quantization bin has at least long_weight integer sums.
        // At 0 or 255 the clipped end is already a physical bound on both frames.
        sums_[level] = sum;
        long_values_[level] = allowed_long;
        short_values_[level] = {
            std::max(0, ceil_divide(sum.low - long_weight_ * allowed_long.high, short_weight_)),
            std::min(255, floor_divide(sum.high - long_weight_ * allowed_long.low, short_weight_))};
    }
}

std::pair<cv::Mat, cv::Mat> ShortExposureCamera::bounds(const cv::Mat& observed) const {
    check_image(observed);
    cv::Mat low(observed.size(), CV_8UC3), high(observed.size(), CV_8UC3);
    for (int y = 0; y < observed.rows; ++y) {
        const auto* source = observed.ptr<unsigned char>(y);
        auto* lower = low.ptr<unsigned char>(y);
        auto* upper = high.ptr<unsigned char>(y);
        for (size_t i = 0; i < size_t(observed.cols) * 3; ++i) {
            const auto limits = short_values_[source[i]];
            lower[i] = static_cast<unsigned char>(limits.low);
            upper[i] = static_cast<unsigned char>(limits.high);
        }
    }
    return {std::move(low), std::move(high)};
}

cv::Mat ShortExposureCamera::companion(const cv::Mat& observed, const cv::Mat& short_frame) const {
    check_pair(observed, short_frame);
    cv::Mat long_frame(observed.size(), CV_8UC3);
    for (int y = 0; y < observed.rows; ++y) {
        const auto* source = observed.ptr<unsigned char>(y);
        const auto* known = short_frame.ptr<unsigned char>(y);
        auto* output = long_frame.ptr<unsigned char>(y);
        for (size_t i = 0; i < size_t(observed.cols) * 3; ++i) {
            const auto sum = sums_[source[i]];
            const auto allowed = long_values_[source[i]];
            const auto contribution = short_weight_ * known[i];
            const int low = std::max(allowed.low, ceil_divide(sum.low - contribution, long_weight_));
            const int high = std::min(allowed.high, floor_divide(sum.high - contribution, long_weight_));
            if (low > high)
                throw std::invalid_argument("Short frame cannot produce the observation with this exposure model");
            output[i] = static_cast<unsigned char>(std::clamp(int(source[i]), low, high));
        }
    }
    return long_frame;
}

cv::Mat ShortExposureCamera::replay(const cv::Mat& short_frame, const cv::Mat& long_frame) const {
    check_pair(short_frame, long_frame);
    cv::Mat result(short_frame.size(), CV_8UC3);
    for (int y = 0; y < short_frame.rows; ++y) {
        const auto* short_pixels = short_frame.ptr<unsigned char>(y);
        const auto* long_pixels = long_frame.ptr<unsigned char>(y);
        auto* output = result.ptr<unsigned char>(y);
        for (size_t i = 0; i < size_t(short_frame.cols) * 3; ++i) {
            const int sum = short_weight_ * short_pixels[i] + long_weight_ * long_pixels[i];
            output[i] = static_cast<unsigned char>((2 * sum + total_weight_) / (2 * total_weight_));
        }
    }
    return result;
}

} // namespace portrait_selection
