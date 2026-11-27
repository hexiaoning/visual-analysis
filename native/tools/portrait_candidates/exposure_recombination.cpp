#include "exposure_recombination.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>

#include <opencv2/imgproc.hpp>

namespace portrait_selection {
namespace {

void validate(const ShapeDescriptor& frame) {
    if (frame.image.type() != CV_8UC3 || frame.image.size() != cv::Size(512, 512) || frame.points.size() != 98 ||
        !std::ranges::all_of(frame.points, [](const auto& p) { return std::isfinite(p.x) && std::isfinite(p.y); }))
        throw std::invalid_argument("Expected an aligned 512x512 BGR frame and finite WFLW landmarks.");
}

cv::Rect2f component(const ShapeDescriptor& first, const ShapeDescriptor& second, int begin, int end) {
    cv::Rect2f bounds;
    bool initialized = false;
    for (const auto* frame : {&first, &second}) {
        for (int i = begin; i < end; ++i) {
            const auto p = frame->points[i];
            const cv::Rect2f pixel(p.x, p.y, 1.f, 1.f);
            bounds = initialized ? bounds | pixel : pixel;
            initialized = true;
        }
    }
    return bounds;
}

bool add_mask(cv::Mat& mask, cv::Rect2f bounds, float margin) {
    if (bounds.width < 8.f || bounds.width > 150.f || bounds.height > 100.f || bounds.x < 8.f || bounds.y < 8.f ||
        bounds.br().x > mask.cols - 8.f || bounds.br().y > mask.rows - 8.f)
        return false;
    constexpr float feather = 18.f;
    const float left = bounds.x - margin, right = bounds.br().x + margin;
    const float top = bounds.y - margin, bottom = bounds.br().y + margin;
    const int x_min = std::max(0, static_cast<int>(std::floor(left - feather)));
    const int y_min = std::max(0, static_cast<int>(std::floor(top - feather)));
    const int x_max = std::min(mask.cols, static_cast<int>(std::ceil(right + feather)));
    const int y_max = std::min(mask.rows, static_cast<int>(std::ceil(bottom + feather)));
    for (int y = y_min; y < y_max; ++y) {
        auto* values = mask.ptr<float>(y);
        for (int x = x_min; x < x_max; ++x) {
            const float outside_x = std::max({left - x, x - right, 0.f});
            const float outside_y = std::max({top - y, y - bottom, 0.f});
            const float t = std::clamp(1.f - std::hypot(outside_x, outside_y) / feather, 0.f, 1.f);
            values[x] = std::max(values[x], t * t * (3.f - 2.f * t));
        }
    }
    return true;
}

double pose_shift(const ShapeDescriptor& first, const ShapeDescriptor& second) {
    std::array<double, 9> difference{};
    for (std::size_t i = 0; i < difference.size(); ++i)
        difference[i] = cv::norm(first.points[51 + i] - second.points[51 + i]);
    const auto middle = difference.begin() + difference.size() / 2;
    std::nth_element(difference.begin(), middle, difference.end());
    return *middle;
}

double context_error(const cv::Mat& first, const cv::Mat& second, const cv::Mat& mask) {
    cv::Mat a, b;
    cv::GaussianBlur(first, a, {}, 3.0, 0., cv::BORDER_REFLECT_101);
    cv::GaussianBlur(second, b, {}, 3.0, 0., cv::BORDER_REFLECT_101);
    double sum = 0.;
    std::size_t count = 0;
    for (int y = 0; y < mask.rows; ++y) {
        const auto* m = mask.ptr<float>(y);
        const auto* p = a.ptr<cv::Vec3b>(y);
        const auto* q = b.ptr<cv::Vec3b>(y);
        for (int x = 0; x < mask.cols; ++x) {
            if (m[x] <= .06f || m[x] >= .40f)
                continue;
            for (int c = 0; c < 3; ++c) sum += std::abs(static_cast<int>(p[x][c]) - q[x][c]);
            count += 3;
        }
    }
    return count ? sum / (255. * count) : 1.;
}

const std::array<double, 256>& linear_transfer() {
    static const auto table = [] {
        std::array<double, 256> values{};
        for (std::size_t i = 0; i < values.size(); ++i) {
            const double s = i / 255.;
            values[i] = s <= .04045 ? s / 12.92 : std::pow((s + .055) / 1.055, 2.4);
        }
        return values;
    }();
    return table;
}

unsigned char encode(double linear) {
    const auto& values = linear_transfer();
    auto after = std::lower_bound(values.begin(), values.end(), linear);
    if (after == values.end()) return 255;
    if (after != values.begin() && linear - *(after - 1) < *after - linear) --after;
    return static_cast<unsigned char>(after - values.begin());
}

void exchange(const cv::Mat& first, const cv::Mat& second, const cv::Mat& mask, ExposureRecombination& result) {
    result.first = first.clone();
    result.second = second.clone();
    const auto& values = linear_transfer();
    double squared = 0.;
    for (int y = 0; y < mask.rows; ++y) {
        const auto* m = mask.ptr<float>(y);
        auto* a = result.first.ptr<cv::Vec3b>(y);
        auto* b = result.second.ptr<cv::Vec3b>(y);
        for (int x = 0; x < mask.cols; ++x) {
            if (m[x] == 0.f) continue;
            const auto before_a = a[x], before_b = b[x];
            for (int c = 0; c < 3; ++c) {
                const double p = values[before_a[c]], q = values[before_b[c]];
                const double delta = m[x] * (q - p);
                const double new_a = p + delta, new_b = q - delta;
                result.ideal_mean_error = std::max(result.ideal_mean_error, std::abs((new_a + new_b) - (p + q)) / 2.);
                a[x][c] = encode(new_a);
                b[x][c] = encode(new_b);
                const double error = (values[a[x][c]] + values[b[x][c]] - p - q) / 2.;
                result.saved_mean_error = std::max(result.saved_mean_error, std::abs(error));
                squared += error * error;
            }
            result.changed_first += a[x] != before_a;
            result.changed_second += b[x] != before_b;
        }
    }
    result.saved_mean_rms = std::sqrt(squared / (3. * mask.total()));
}

}

ExposureRecombination recombine_exposure(const ShapeDescriptor& first, const ShapeDescriptor& second, ExposureRegion region) {
    validate(first);
    validate(second);
    ExposureRecombination result;
    result.pose_shift = pose_shift(first, second);
    if (result.pose_shift > 12.) {
        result.status = "pose-difference";
        return result;
    }

    cv::Mat mask(first.image.size(), CV_32F, cv::Scalar(0));
    const bool valid_mask = region == ExposureRegion::eyes
        ? add_mask(mask, component(first, second, 60, 68), 7.f) && add_mask(mask, component(first, second, 68, 76), 7.f)
        : add_mask(mask, component(first, second, 76, 96), 8.f);
    if (!valid_mask) {
        result.status = "unsupported-region";
        return result;
    }
    result.context_error = context_error(first.image, second.image, mask);
    if (result.context_error > .085) {
        result.status = "context-difference";
        return result;
    }

    exchange(first.image, second.image, mask, result);
    result.status = std::min(result.changed_first, result.changed_second) >= 16 ? "generated" : "unchanged";
    return result;
}

}
