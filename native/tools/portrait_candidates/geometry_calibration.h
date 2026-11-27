#pragma once

#include <array>
#include <filesystem>
#include <opencv2/core/mat.hpp>

namespace portrait_selection {

cv::Mat calibrate_capture_geometry(const std::array<cv::Mat, 3> &observations, const cv::Mat &candidate,
                                   const std::filesystem::path &model_path);

} // namespace portrait_selection
