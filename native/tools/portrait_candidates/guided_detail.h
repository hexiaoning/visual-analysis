#pragma once

#include <opencv2/core/mat.hpp>

namespace portrait_selection {

struct GuidedDetail {
    cv::Mat warped;
    cv::Mat adjusted;
    double cycle_support;
    double contrast_support;
    double peak_displacement;
};

GuidedDetail transfer_guided_detail(const cv::Mat& donor, const cv::Mat& guide);

}
