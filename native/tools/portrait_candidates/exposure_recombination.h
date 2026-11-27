#pragma once

#include "portrait_selection.h"

#include <cstddef>

namespace portrait_selection {

enum class ExposureRegion { eyes, mouth };

struct ExposureRecombination {
    cv::Mat first;
    cv::Mat second;
    double pose_shift{};
    double context_error{};
    double ideal_mean_error{};
    double saved_mean_error{};
    double saved_mean_rms{};
    std::size_t changed_first{};
    std::size_t changed_second{};
    const char* status{};
};

// Exchange the same anatomical region in both frames, conserving their linear-light mean.
ExposureRecombination recombine_exposure(const ShapeDescriptor& first, const ShapeDescriptor& second, ExposureRegion region);

}
