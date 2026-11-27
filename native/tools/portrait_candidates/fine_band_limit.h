#pragma once

#include <opencv2/core.hpp>

namespace portrait_selection {
struct FineBandResult {
    cv::Mat image;
    double rms;
    bool reached;
};

FineBandResult cap_fine_band(const cv::Mat &source, double maximum_rms);
}
