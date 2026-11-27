#pragma once

#include <cstddef>
#include <array>
#include <functional>
#include <optional>
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

// Preserve kept_count initial portraits and extend the measured cover. The
// predicate can veto a proposed pair before its candidate is added.
EvidenceExtension extend_appearances(const cv::Mat& source, const std::vector<cv::Mat>& candidates,
                                     std::size_t kept_count, double radius, double replay_tolerance,
                                     std::size_t max_additions = 1,
                                     const std::function<bool(std::size_t, std::size_t)>& compatible = {});

struct RepresentativeReplacement {
    std::size_t previous;
    std::size_t proposed;
    double distance;
    double score;
    std::array<double, 4> replay_improvements;
};

// Find a single replacement for the displayed prefix. Check that it represents
// the same appearance as its predecessor and differs from the rest.
std::optional<RepresentativeReplacement> refine_representative(
    const cv::Mat& source, const std::vector<cv::Mat>& candidates, std::size_t kept_count,
    const std::function<bool(std::size_t, std::size_t)>& visibly_different);

}
