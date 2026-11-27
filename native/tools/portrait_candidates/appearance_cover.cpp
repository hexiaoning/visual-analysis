#include "appearance_cover.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

#include <opencv2/imgproc.hpp>

namespace portrait_selection {
namespace {

constexpr double signal_floor = .004;
const std::array<cv::Rect, 4> regions{
    cv::Rect(40, 50, 22, 17), cv::Rect(68, 50, 22, 17),
    cv::Rect(50, 65, 29, 23), cv::Rect(45, 83, 39, 21),
};

struct Patch {
    cv::Mat direction;
    bool visible{};
};

using Geometry = std::array<Patch, regions.size()>;

Geometry geometry(const cv::Mat& portrait) {
    if (portrait.type() != CV_8UC3 || portrait.size() != cv::Size(512, 512))
        throw std::invalid_argument("Expected an aligned 512x512 BGR portrait");
    cv::Mat reduced, gray, fine, coarse;
    cv::resize(portrait, reduced, {128, 128}, 0, 0, cv::INTER_AREA);
    cv::cvtColor(reduced, gray, cv::COLOR_BGR2GRAY);
    gray.convertTo(gray, CV_32F, 1. / 255.);
    cv::GaussianBlur(gray, fine, {}, .8);
    cv::GaussianBlur(gray, coarse, {}, 3.1);
    const cv::Mat band = fine - coarse;
    Geometry result;
    for (std::size_t i = 0; i < regions.size(); ++i) {
        cv::Mat patch = band(regions[i]).clone();
        const auto values = patch.reshape(1, 1);
        std::vector<float> ordered(values.begin<float>(), values.end<float>());
        const auto middle = ordered.begin() + ordered.size() / 2;
        std::nth_element(ordered.begin(), middle, ordered.end());
        const double median = ordered.size() % 2 ? *middle :
            (static_cast<double>(*middle) + *std::max_element(ordered.begin(), middle)) * .5;
        const double raw_rms = cv::norm(patch) / std::sqrt(static_cast<double>(patch.total()));
        patch -= median;
        const double norm = cv::norm(patch);
        if (raw_rms >= signal_floor && norm > 0.) {
            patch /= norm;
            result[i] = {std::move(patch), true};
        }
    }
    return result;
}

double distance(const Geometry& a, const Geometry& b) {
    double limit = 0.;
    for (std::size_t i = 0; i < a.size(); ++i) {
        if (a[i].visible && b[i].visible) {
            const double squared = cv::norm(a[i].direction, b[i].direction, cv::NORM_L2SQR);
            limit = std::max(limit, std::min(2., .5 * squared));
        } else if (a[i].visible != b[i].visible) {
            // A clear edge on one face can disappear entirely on another.
            limit = std::max(limit, 1.);
        }
    }
    return limit;
}

}

AppearanceCover cover_appearances(const std::vector<cv::Mat>& candidates, double radius) {
    if (candidates.empty() || !std::isfinite(radius) || radius < 0. || radius > 2.)
        throw std::invalid_argument("Expected a portrait bank and a finite radius in [0, 2]");
    std::vector<Geometry> features;
    features.reserve(candidates.size());
    for (const auto& candidate : candidates)
        features.push_back(geometry(candidate));

    AppearanceCover result{{}, std::vector<std::size_t>(features.size()),
                           std::vector<double>(features.size(), std::numeric_limits<double>::infinity())};
    std::size_t representative = 0;
    for (;;) {
        result.representatives.push_back(representative);
        for (std::size_t i = 0; i < features.size(); ++i) {
            const double d = i == representative ? 0. : distance(features[i], features[representative]);
            if (d < result.distances[i]) {
                result.nearest[i] = representative;
                result.distances[i] = d;
            }
        }
        const auto farthest = std::max_element(result.distances.begin(), result.distances.end());
        if (*farthest <= radius)
            return result;
        representative = static_cast<std::size_t>(farthest - result.distances.begin());
    }
}

}
