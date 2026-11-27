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

cv::Mat thumbnail(const cv::Mat& portrait) {
    if (portrait.type() != CV_8UC3 || portrait.size() != cv::Size(512, 512))
        throw std::invalid_argument("Expected an aligned 512x512 BGR portrait");
    cv::Mat reduced, gray;
    cv::resize(portrait, reduced, {128, 128}, 0, 0, cv::INTER_AREA);
    cv::cvtColor(reduced, gray, cv::COLOR_BGR2GRAY);
    gray.convertTo(gray, CV_32F, 1. / 255.);
    return gray;
}

Geometry geometry(const cv::Mat& portrait) {
    const auto gray = thumbnail(portrait);
    cv::Mat fine, coarse;
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

const cv::Rect evidence_region(35, 43, 63, 60);
const std::array<cv::Rect, 4> evidence_parts{
    cv::Rect(36, 49, 28, 19), cv::Rect(67, 49, 28, 19),
    cv::Rect(52, 63, 30, 27), cv::Rect(44, 82, 46, 21),
};

cv::Mat smooth(const cv::Mat& image, double sigma) {
    cv::Mat filtered;
    cv::GaussianBlur(image, filtered, {}, sigma, 0., cv::BORDER_REFLECT_101);
    return filtered;
}

double rms(const cv::Mat& patch) {
    return cv::norm(patch) / std::sqrt(static_cast<double>(patch.total()));
}

struct EvidenceShape {
    cv::Mat middle_band;
    double detail_strength;
};

EvidenceShape evidence_shape(const cv::Mat& gray) {
    const auto middle = smooth(gray, 1.3);
    const cv::Mat fine_band = smooth(gray, .52) - middle;
    return {middle - smooth(gray, 3.), rms(fine_band(evidence_region))};
}

double evidence_distance(const EvidenceShape& a, const EvidenceShape& b) {
    double farthest = 0.;
    for (const auto& area : evidence_parts) {
        const auto p = a.middle_band(area), q = b.middle_band(area);
        const double norm_p = rms(p), norm_q = rms(q);
        if (std::min(norm_p, norm_q) >= signal_floor) {
            const double correlation = p.dot(q) / (static_cast<double>(p.total()) * norm_p * norm_q);
            farthest = std::max(farthest, std::clamp(1. - correlation, 0., 2.));
        }
    }
    return farthest;
}

double registered_error(const cv::Mat& gray, const cv::Mat& observed) {
    const auto target = observed(evidence_region);
    const cv::Rect search(32, 40, 69, 66);
    double best = std::numeric_limits<double>::infinity();
    for (double sigma : {.65, 1.25, 2.1}) {
        auto softened = smooth(gray, sigma);
        cv::Mat correlations;
        cv::matchTemplate(softened(search), target, correlations, cv::TM_CCOEFF_NORMED);
        cv::Point position;
        cv::minMaxLoc(correlations, nullptr, nullptr, nullptr, &position);
        const cv::Mat sample = softened(cv::Rect(search.tl() + position, target.size()));
        double sum_input = 0., sum_target = 0., sum_squared = 0., sum_cross = 0.;
        int train_count = 0;
        for (int y = 0; y < target.rows; ++y) {
            for (int x = 0; x < target.cols; ++x) {
                if ((y / 6 + x / 7) % 2 != 0)
                    continue;
                const double pixel = sample.at<float>(y, x), capture = target.at<float>(y, x);
                sum_input += pixel;
                sum_target += capture;
                sum_squared += pixel * pixel;
                sum_cross += pixel * capture;
                ++train_count;
            }
        }
        const double covariance = sum_cross - sum_input * sum_target / train_count;
        const double variance = sum_squared - sum_input * sum_input / train_count;
        const double gain = std::clamp(covariance / std::max(variance, .00001), .72, 1.38);
        const double offset = std::clamp((sum_target - gain * sum_input) / train_count, -.12, .12);
        double cost = 0.;
        int test_count = 0;
        for (int y = 0; y < target.rows; ++y) {
            for (int x = 0; x < target.cols; ++x) {
                if ((y / 6 + x / 7) % 2 == 0)
                    continue;
                cost += std::abs(gain * sample.at<float>(y, x) + offset - target.at<float>(y, x));
                ++test_count;
            }
        }
        best = std::min(best, cost / test_count);
    }
    return best;
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

EvidenceExtension extend_appearances(const cv::Mat& source, const std::vector<cv::Mat>& candidates,
                                     std::size_t kept_count, double radius, double replay_tolerance) {
    if (kept_count == 0 || kept_count > candidates.size() || !std::isfinite(radius) || radius <= 0. || radius > 2. ||
        !std::isfinite(replay_tolerance) || replay_tolerance < 0. || replay_tolerance > 1.)
        throw std::invalid_argument("Expected a nonempty protected prefix, radius in (0, 2], and replay tolerance in [0, 1].");
    const auto observed = thumbnail(source);
    const double observed_detail = evidence_shape(observed).detail_strength;
    const auto count = candidates.size();
    EvidenceExtension result;
    std::vector<EvidenceShape> shapes;
    shapes.reserve(count);
    double anchor_error = std::numeric_limits<double>::infinity();
    for (std::size_t i = 0; i < count; ++i) {
        const auto gray = thumbnail(candidates[i]);
        shapes.push_back(evidence_shape(gray));
        result.replay_errors.push_back(registered_error(gray, observed));
        if (i < kept_count) {
            result.representatives.push_back(i);
            anchor_error = std::min(anchor_error, result.replay_errors.back());
        }
    }
    std::vector<std::vector<double>> distances(count, std::vector<double>(count));
    for (std::size_t i = 0; i < count; ++i) {
        for (std::size_t j = i + 1; j < count; ++j)
            distances[i][j] = distances[j][i] = evidence_distance(shapes[i], shapes[j]);
    }
    result.distances_to_kept.resize(count);
    std::vector<double> weights(count);
    for (std::size_t i = 0; i < count; ++i) {
        result.distances_to_kept[i] = *std::min_element(distances[i].begin(), distances[i].begin() + kept_count);
        const double error = result.replay_errors[i];
        if (error > anchor_error + replay_tolerance || shapes[i].detail_strength < observed_detail)
            continue;
        double density = 0.;
        for (double distance : distances[i])
            density += std::exp(-distance / radius);
        weights[i] = std::exp(-std::max(0., error - anchor_error) / .014) / density;
    }
    double best_gain = 0.;
    std::size_t best = count;
    for (std::size_t i = kept_count; i < count; ++i) {
        if (weights[i] <= 0. || result.distances_to_kept[i] <= radius)
            continue;
        double gain = 0.;
        for (std::size_t j = 0; j < count; ++j)
            gain += weights[j] * std::max(0., result.distances_to_kept[j] - distances[i][j]);
        if (gain > best_gain) {
            best_gain = gain;
            best = i;
        }
    }
    if (best < count)
        result.representatives.push_back(best);
    return result;
}

}
