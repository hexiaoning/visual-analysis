#pragma once

#include <opencv2/core/mat.hpp>

namespace portrait_consensus {
// All three inputs must be aligned CV_8UC3 BGR 512x512 images of the same observation.
// For an eligible duplicate, choose AMD if the result exceeds 0.5, else CodeFormer.
double amd_fraction(const cv::Mat &observed, const cv::Mat &codeformer, const cv::Mat &amd);
} // namespace portrait_consensus
