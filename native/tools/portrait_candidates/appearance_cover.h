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

struct AppearanceEvidence {
    double observed_detail;
    std::vector<double> replay_errors;
    std::vector<double> candidate_detail;
    std::vector<std::vector<double>> distances;
};

AppearanceEvidence analyse_appearance_bank(const cv::Mat& source, const std::vector<cv::Mat>& candidates);

struct EvidenceExtension {
    std::vector<std::size_t> representatives;
    std::vector<double> replay_errors;
    std::vector<double> distances_to_kept;
    std::size_t standard_count{};
};

struct CloseAppearanceCheck {
    double radius;
    std::function<bool(std::size_t, std::size_t)> compatible;
};

// Preserve kept_count initial portraits and extend the measured cover. The
// predicate can veto a proposed pair before its candidate is added.
EvidenceExtension extend_appearances(const cv::Mat& source, const std::vector<cv::Mat>& candidates,
                                     std::size_t kept_count, double radius, double replay_tolerance,
                                     std::size_t max_additions = 1,
                                     const std::function<bool(std::size_t, std::size_t)>& compatible = {},
                                     const std::optional<CloseAppearanceCheck>& close = std::nullopt);

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

struct ClearRepresentativeReplacement {
    std::size_t previous;
    std::size_t proposed;
    double distance;
    double clarity_gain;
    std::array<std::optional<double>, 4> regional_gain;
    std::array<double, 4> replay_deltas;
    std::array<double, 4> replay_limits;
    int maximum_offset_change;
};

// The bank starts with the original representatives and then the previous
// additions. A replacement for an original cannot collapse an added mode.
std::optional<ClearRepresentativeReplacement> refine_clear_representative(
    const cv::Mat& source, const std::vector<cv::Mat>& candidates,
    std::size_t kept_count, std::size_t added_count,
    const std::function<bool(std::size_t, std::size_t)>& visibly_different);

struct ExposureRepresentativeReplacement {
    ClearRepresentativeReplacement choice;
    std::size_t family;
    std::array<double, 4> baseline_replay;
};

// Evaluate each complete revision against the same four baseline instants.
// Every revision member indexes a candidate outside the displayed prefix.
std::optional<ExposureRepresentativeReplacement> refine_exposure_representative(
    const cv::Mat& source, const std::vector<cv::Mat>& candidates, std::size_t displayed_count,
    const std::array<cv::Mat, 4>& baseline, const std::array<double, 4>& shutter,
    const std::vector<std::array<std::size_t, 4>>& families,
    const std::function<bool(std::size_t, std::size_t)>& visibly_different);

}
