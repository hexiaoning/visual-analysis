#pragma once

#include <array>
#include <filesystem>
#include <opencv2/core.hpp>
#include <vector>

namespace portrait_selection {

using ExposureFrames = std::array<cv::Mat, 4>;

struct BoundedDirection {
    cv::Mat image;
    int region;
    int axis;
    int sign;
    int phase;
    double peak_light_error;
};

struct DirectionGroup {
    std::array<int, 4> weight_units{};
    int seed_family{};
    double initial_replay{};
    double fitted_replay{};
    double seed_spread{};
    bool eligible{};
    ExposureFrames fitted_states;
    std::vector<BoundedDirection> candidates;
};

std::vector<DirectionGroup> learned_exposure_directions(
    const std::filesystem::path& dictionary, const cv::Mat& input,
    const std::array<cv::Mat, 2>& anchors, const std::array<ExposureFrames, 2>& families);

}
