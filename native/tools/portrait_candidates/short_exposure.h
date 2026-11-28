#pragma once

#include <array>
#include <cstddef>
#include <vector>
#include <opencv2/core.hpp>

namespace portrait_selection {

struct ShortExposureBounds {
    cv::Mat lower;
    cv::Mat upper;
};

class ShortExposureCamera {
public:
    static constexpr int max_weight = 1'000'000;

    ShortExposureCamera(int short_weight, int long_weight, int companion_tolerance);

    ShortExposureBounds bounds(const cv::Mat& observed, int capture_tolerance = 0) const;
    cv::Mat replay(const cv::Mat& short_frame, const cv::Mat& long_frame) const;
    cv::Mat companion(const cv::Mat& observed, const cv::Mat& short_frame) const;

private:
    struct Interval { int low; int high; };

    int short_weight_;
    int long_weight_;
    int total_weight_;
    std::array<Interval, 256> sums_{};
    std::array<Interval, 256> long_values_{};

    int render_level(int short_pixel, int long_pixel) const;
};

class ShortExposureCover {
public:
    ShortExposureCover(const cv::Mat& observed, const ShortExposureCamera& camera,
                       int capture_tolerance, std::size_t channel_budget,
                       cv::Point short_offset = {});

    const ShortExposureBounds& exact_bounds() const { return exact_; }
    const ShortExposureBounds& allowed_bounds() const { return allowed_; }
    cv::Mat project(const cv::Mat& proposal) const;
    cv::Mat short_frame_on_sensor(const cv::Mat& short_frame) const;

    cv::Mat first_cell(int width) const;
    void validate_cell(const cv::Mat& cursor, int width) const;
    bool advance_cell(cv::Mat& cursor, int width) const;
    cv::Mat representative(const cv::Mat& cursor, int width) const;

private:
    struct CellRange { int first; int last; int exact_first; int exact_last; };

    CellRange cell_range(std::size_t channel, int width) const;
    static bool requires_error(const CellRange& range, int digit);
    void reset_suffix(cv::Mat& cursor, std::size_t start, int width, std::size_t allowance) const;

    cv::Mat observed_;
    const std::size_t channel_budget_;
    cv::Rect sensor_overlap_;
    cv::Rect short_overlap_;
    ShortExposureBounds exact_;
    ShortExposureBounds allowed_;
};

} // namespace portrait_selection
