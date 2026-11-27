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

struct EvidenceExtension {
    std::vector<std::size_t> representatives;
    std::vector<double> replay_errors;
    std::vector<double> distances_to_kept;
};

// Preserve kept_count initial portraits. Add at most one candidate that
// explains the observation and covers a distinct part of the finite bank.
EvidenceExtension extend_appearances(const cv::Mat& source, const std::vector<cv::Mat>& candidates,
                                     std::size_t kept_count, double radius, double replay_tolerance);

}
