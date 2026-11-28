#include "short_exposure.h"

#include <algorithm>
#include <cstdlib>
#include <stdexcept>
#include <utility>

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
        sums_[level] = sum;
        long_values_[level] = allowed_long;
    }
}

int ShortExposureCamera::render_level(int short_pixel, int long_pixel) const {
    const int sum = short_weight_ * short_pixel + long_weight_ * long_pixel;
    return (2 * sum + total_weight_) / (2 * total_weight_);
}

ShortExposureBounds ShortExposureCamera::bounds(const cv::Mat& observed, int capture_tolerance) const {
    check_image(observed);
    if (capture_tolerance < 0 || capture_tolerance > 255)
        throw std::invalid_argument("Capture tolerance must be between 0 and 255");
    std::array<Interval, 256> short_values{};
    for (int level = 0; level <= 255; ++level) {
        // Adjacent quantization bins share a boundary. Integer sums spanning any
        // interior bin contain at least long_weight_ consecutive values.
        const int first = sums_[std::max(0, level - capture_tolerance)].low;
        const int last = sums_[std::min(255, level + capture_tolerance)].high;
        const auto companion = long_values_[level];
        short_values[level] = {
            std::max(0, ceil_divide(first - long_weight_ * companion.high, short_weight_)),
            std::min(255, floor_divide(last - long_weight_ * companion.low, short_weight_))};
    }
    cv::Mat low(observed.size(), CV_8UC3), high(observed.size(), CV_8UC3);
    for (int y = 0; y < observed.rows; ++y) {
        const auto* source = observed.ptr<unsigned char>(y);
        auto* lower = low.ptr<unsigned char>(y);
        auto* upper = high.ptr<unsigned char>(y);
        for (size_t i = 0; i < size_t(observed.cols) * 3; ++i) {
            const auto limits = short_values[source[i]];
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
            const auto allowed = long_values_[source[i]];
            // Incrementing the companion can move the rounded observation by
            // at most one level. Every level between the endpoints is attainable.
            const int closest = std::clamp(int(source[i]), render_level(known[i], allowed.low),
                                           render_level(known[i], allowed.high));
            const auto sum = sums_[closest];
            const auto contribution = short_weight_ * known[i];
            const int low = std::max(allowed.low, ceil_divide(sum.low - contribution, long_weight_));
            const int high = std::min(allowed.high, floor_divide(sum.high - contribution, long_weight_));
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
            output[i] = static_cast<unsigned char>(render_level(short_pixels[i], long_pixels[i]));
        }
    }
    return result;
}

ShortExposureCover::ShortExposureCover(const cv::Mat& observed, const ShortExposureCamera& camera,
                                       int capture_tolerance, std::size_t channel_budget,
                                       cv::Point short_offset)
    : observed_(observed.clone()), channel_budget_(channel_budget) {
    check_image(observed_);
    if (short_offset.x <= -observed.cols || short_offset.x >= observed.cols ||
        short_offset.y <= -observed.rows || short_offset.y >= observed.rows)
        throw std::invalid_argument("Short exposure translation must overlap the observation");
    sensor_overlap_ = {std::max(0, short_offset.x), std::max(0, short_offset.y),
                       observed.cols - std::abs(short_offset.x), observed.rows - std::abs(short_offset.y)};
    short_overlap_ = sensor_overlap_ - short_offset;
    const auto bounds_in_short_frame = [&](int tolerance) {
        auto overlap_bounds = camera.bounds(observed_(sensor_overlap_), tolerance);
        if (short_offset == cv::Point{})
            return overlap_bounds;
        // The observation places no constraint on short-frame pixels outside the sensor crop.
        ShortExposureBounds limits{cv::Mat(observed_.size(), CV_8UC3, cv::Scalar::all(0)),
                                   cv::Mat(observed_.size(), CV_8UC3, cv::Scalar::all(255))};
        overlap_bounds.lower.copyTo(limits.lower(short_overlap_));
        overlap_bounds.upper.copyTo(limits.upper(short_overlap_));
        return limits;
    };
    exact_ = bounds_in_short_frame(0);
    allowed_ = channel_budget_ ? bounds_in_short_frame(capture_tolerance) : exact_;
}

cv::Mat ShortExposureCover::short_frame_on_sensor(const cv::Mat& short_frame) const {
    check_pair(observed_, short_frame);
    if (sensor_overlap_ == short_overlap_)
        return short_frame;
    // Choose one feasible outside-crop short exposure for the replay witness.
    cv::Mat on_sensor = observed_.clone();
    short_frame(short_overlap_).copyTo(on_sensor(sensor_overlap_));
    return on_sensor;
}

cv::Mat ShortExposureCover::project(const cv::Mat& proposal) const {
    check_pair(observed_, proposal);
    cv::Mat result;
    cv::max(proposal, exact_.lower, result);
    cv::min(result, exact_.upper, result);
    if (!channel_budget_)
        return result;
    // A relaxed choice only consumes one allowed channel. Spend those choices
    // where their independently attainable drop in squared error is largest.
    struct Improvement { size_t offset; int value; unsigned char pixel; };
    std::vector<Improvement> improvements;
    const auto contiguous = proposal.isContinuous() ? proposal : proposal.clone();
    for (size_t i = 0; i < result.total() * result.elemSize(); ++i) {
        const int original = contiguous.data[i];
        const int extremum = std::clamp(original, int(allowed_.lower.data[i]), int(allowed_.upper.data[i]));
        const int before = original - result.data[i];
        const int after = original - extremum;
        const int gain = before * before - after * after;
        if (gain > 0)
            improvements.push_back({i, gain, static_cast<unsigned char>(extremum)});
    }
    if (channel_budget_ < improvements.size()) {
        const auto middle = improvements.begin() + static_cast<std::ptrdiff_t>(channel_budget_);
        std::nth_element(improvements.begin(), middle, improvements.end(), [](const auto& a, const auto& b) {
            return a.value == b.value ? a.offset < b.offset : a.value > b.value;
        });
        improvements.resize(channel_budget_);
    }
    for (const auto& improvement : improvements)
        result.data[improvement.offset] = improvement.pixel;
    return result;
}

ShortExposureCover::CellRange ShortExposureCover::cell_range(size_t i, int width) const {
    return {allowed_.lower.data[i] / width, allowed_.upper.data[i] / width,
            exact_.lower.data[i] / width, exact_.upper.data[i] / width};
}

bool ShortExposureCover::requires_error(const CellRange& range, int digit) {
    return digit < range.exact_first || digit > range.exact_last;
}

void ShortExposureCover::reset_suffix(cv::Mat& cursor, size_t start, int width, size_t allowance) const {
    for (size_t i = start; i < cursor.total() * cursor.elemSize(); ++i) {
        const auto range = cell_range(i, width);
        const int least = allowance ? range.first : range.exact_first;
        cursor.data[i] = static_cast<unsigned char>(least);
        allowance -= requires_error(range, least);
    }
}

cv::Mat ShortExposureCover::first_cell(int width) const {
    cv::Mat cursor(observed_.size(), observed_.type());
    reset_suffix(cursor, 0, width, channel_budget_);
    return cursor;
}

void ShortExposureCover::validate_cell(const cv::Mat& cursor, int width) const {
    check_pair(observed_, cursor);
    if (!cursor.isContinuous())
        throw std::invalid_argument("Expected a contiguous exposure page cursor");
    size_t used = 0;
    for (size_t i = 0; i < cursor.total() * cursor.elemSize(); ++i) {
        const auto range = cell_range(i, width);
        const int digit = cursor.data[i];
        if (digit < range.first || digit > range.last)
            throw std::invalid_argument("Exposure page cursor falls outside the feasible cells");
        used += requires_error(range, digit);
    }
    if (used > channel_budget_)
        throw std::invalid_argument("Exposure page cursor exceeds the capture error budget");
}

bool ShortExposureCover::advance_cell(cv::Mat& cursor, int width) const {
    size_t used = 0;
    const size_t channels = cursor.total() * cursor.elemSize();
    for (size_t i = 0; i < channels; ++i)
        used += requires_error(cell_range(i, width), cursor.data[i]);
    for (size_t i = channels; i > 0; --i) {
        const auto range = cell_range(i - 1, width);
        const int digit = cursor.data[i - 1];
        used -= requires_error(range, digit);
        const bool can_relax = used < channel_budget_;
        const int least = can_relax ? range.first : range.exact_first;
        const int last = can_relax ? range.last : range.exact_last;
        const int next = std::max(digit + 1, least);
        if (next <= last) {
            cursor.data[i - 1] = static_cast<unsigned char>(next);
            reset_suffix(cursor, i, width, channel_budget_ - used - requires_error(range, next));
            return true;
        }
    }
    return false;
}

cv::Mat ShortExposureCover::representative(const cv::Mat& cursor, int width) const {
    cv::Mat candidate(observed_.size(), observed_.type());
    for (size_t i = 0; i < candidate.total() * candidate.elemSize(); ++i) {
        const auto digit = cursor.data[i];
        const auto& bounds = requires_error(cell_range(i, width), digit) ? allowed_ : exact_;
        const int bin = digit * width;
        const int lower = std::max(int(bounds.lower.data[i]), bin);
        const int upper = std::min(int(bounds.upper.data[i]), bin + width - 1);
        candidate.data[i] = static_cast<unsigned char>(std::clamp(int(observed_.data[i]), lower, upper));
    }
    return candidate;
}

} // namespace portrait_selection
