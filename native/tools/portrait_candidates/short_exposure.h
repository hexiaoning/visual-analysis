#pragma once

#include <array>
#include <utility>
#include <opencv2/core.hpp>

namespace portrait_selection {

class ShortExposureCamera {
public:
    static constexpr int max_weight = 1'000'000;

    ShortExposureCamera(int short_weight, int long_weight, int companion_tolerance);

    std::pair<cv::Mat, cv::Mat> bounds(const cv::Mat& observed) const;
    cv::Mat replay(const cv::Mat& short_frame, const cv::Mat& long_frame) const;
    cv::Mat companion(const cv::Mat& observed, const cv::Mat& short_frame) const;

private:
    struct Interval { int low; int high; };

    int short_weight_;
    int long_weight_;
    int total_weight_;
    std::array<Interval, 256> sums_{};
    std::array<Interval, 256> long_values_{};
    std::array<Interval, 256> short_values_{};
};

} // namespace portrait_selection
