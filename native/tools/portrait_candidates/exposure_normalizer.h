#pragma once

#include <array>
#include <filesystem>
#include <opencv2/core/mat.hpp>

namespace portrait_selection {
struct ExposureResult {
    cv::Mat image;
    std::array<double, 3> gain_bgr;
    double black_level;
    double evidence;
    int recovered_grid;
    bool applied;
};

ExposureResult normalize_exposure(const cv::Mat &image, const std::filesystem::path &model_path);
cv::Mat match_candidate_tone(const cv::Mat &candidate, const cv::Mat &guide);
}
