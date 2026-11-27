#pragma once

#include <cstddef>
#include <vector>

#include <opencv2/core.hpp>

namespace portrait_selection {

struct AppearanceCover {
    std::vector<std::size_t> representatives;
    std::vector<std::size_t> nearest;
    std::vector<double> distances;
};

// The first aligned candidate is the starting representative. The radius
// applies to measured geometry in the supplied bank of candidates.
AppearanceCover cover_appearances(const std::vector<cv::Mat>& candidates, double radius);

}
