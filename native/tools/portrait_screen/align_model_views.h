#pragma once

#include <array>
#include <optional>

#include <opencv2/core.hpp>

namespace visual::screen {

struct AlignedModelViews {
    cv::Mat original;
    cv::Mat whole;
    cv::Mat aligned;
    std::optional<cv::Point2f> wholeOffset;
    std::array<std::optional<cv::Point2f>, 3> viewOffsets;
};

AlignedModelViews alignModelViews(const cv::Mat& observed, const cv::Mat& anchor,
                                  const std::array<cv::Mat, 3>& views);

}
