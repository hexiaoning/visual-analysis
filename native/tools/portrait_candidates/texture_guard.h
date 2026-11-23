#pragma once
#include <array>
#include <cstddef>
#include <opencv2/core/mat.hpp>

namespace portrait_selection {
struct TextureGuard {
    static constexpr std::size_t feature_count = 96;
    std::array<double, feature_count> coefficients;
    double bias;
    double cutoff;
    double score(const cv::Mat &candidate) const;
};
} // namespace portrait_selection
