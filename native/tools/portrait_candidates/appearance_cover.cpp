#include "appearance_cover.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <tuple>
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

cv::Mat thumbnail(const cv::Mat& portrait, int side = 128) {
    if (portrait.type() != CV_8UC3 || portrait.size() != cv::Size(512, 512))
        throw std::invalid_argument("Expected an aligned 512x512 BGR portrait");
    cv::Mat reduced, gray;
    cv::resize(portrait, reduced, {side, side}, 0, 0, cv::INTER_AREA);
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

struct ReplayFold {
    std::vector<cv::Point> training;
    std::vector<cv::Point> checking;
};

struct ReplayFit {
    std::size_t blur;
    cv::Point offset;
    double gain;
    double bias;
    double error;
};

using ReplayImages = std::array<cv::Mat, 3>;
using ReplayFolds = std::array<ReplayFold, 4>;
using ReplayFits = std::array<ReplayFit, 4>;

ReplayImages replay_images(const cv::Mat& image) {
    return {smooth(image, .65), smooth(image, 1.25), smooth(image, 2.1)};
}

ReplayFolds replay_folds(const cv::Mat& observed) {
    ReplayFolds result;
    for (int y = 0; y < evidence_region.height; ++y) {
        for (int x = 0; x < evidence_region.width; ++x) {
            const auto point = evidence_region.tl() + cv::Point(x, y);
            const auto value = observed.at<float>(point);
            if (value <= .032 || value >= .969)
                continue;
            const auto checking = static_cast<std::size_t>((y / 7 + x / 9) % 4);
            for (std::size_t i = 0; i < result.size(); ++i) {
                auto& pixels = i == checking ? result[i].checking : result[i].training;
                pixels.push_back(point);
            }
        }
    }
    return result;
}

double replay_error(const cv::Mat& image, const cv::Mat& observed,
                    const std::vector<cv::Point>& pixels, const ReplayFit& fit) {
    double error = 0.;
    for (const auto& point : pixels) {
        const auto sample = fit.gain * image.at<float>(point + fit.offset) + fit.bias;
        error += std::min(.1, std::abs(sample - observed.at<float>(point)));
    }
    return error / pixels.size();
}

template<std::size_t count_images>
ReplayFit fit_replay(const std::array<cv::Mat, count_images>& images, const cv::Mat& observed,
                     const ReplayFold& fold, int max_offset = 3, double variance_floor = .00001) {
    ReplayFit best{0, {}, 1., 0., std::numeric_limits<double>::infinity()};
    const double count = static_cast<double>(fold.training.size());
    double observed_sum = 0.;
    for (const auto& point : fold.training)
        observed_sum += observed.at<float>(point);
    for (std::size_t i = 0; i < images.size(); ++i) {
        for (int y = -max_offset; y <= max_offset; ++y) {
            for (int x = -max_offset; x <= max_offset; ++x) {
                const cv::Point offset(x, y);
                double image_sum = 0., image_squared = 0., cross = 0.;
                for (const auto& point : fold.training) {
                    const double sample = images[i].at<float>(point + offset);
                    image_sum += sample;
                    image_squared += sample * sample;
                    cross += sample * observed.at<float>(point);
                }
                const double covariance = cross - image_sum * observed_sum / count;
                const double variance = image_squared - image_sum * image_sum / count;
                const double gain = std::clamp(covariance / std::max(variance, variance_floor), .72, 1.38);
                const double bias = std::clamp((observed_sum - gain * image_sum) / count, -.12, .12);
                ReplayFit fit{i, offset, gain, bias, 0.};
                fit.error = replay_error(images[i], observed, fold.training, fit);
                if (fit.error < best.error)
                    best = fit;
            }
        }
    }
    best.error = replay_error(images[best.blur], observed, fold.checking, best);
    return best;
}

using RegionalClarity = std::array<std::optional<double>, 4>;

std::array<cv::Mat, 2> gradients(const cv::Mat& gray, double sigma) {
    std::array<cv::Mat, 2> result;
    const auto image = smooth(gray, sigma);
    cv::Sobel(image, result[0], CV_32F, 1, 0, 3, .125);
    cv::Sobel(image, result[1], CV_32F, 0, 1, 3, .125);
    return result;
}

RegionalClarity oriented_clarity(const cv::Mat& portrait) {
    const auto gray = thumbnail(portrait, 256);
    const auto fine = gradients(gray, .6);
    const auto coarse = gradients(gray, 1.6);
    const cv::Mat gx2 = coarse[0].mul(coarse[0]), gy2 = coarse[1].mul(coarse[1]);
    const cv::Mat xx = smooth(gx2, 2.), yy = smooth(gy2, 2.);
    const cv::Mat xy = smooth(coarse[0].mul(coarse[1]), 2.);
    cv::Mat strength = gx2 + gy2, projected = fine[0].mul(coarse[0]) + fine[1].mul(coarse[1]);
    cv::Mat orientation;
    cv::magnitude(xx - yy, xy * 2., orientation);
    cv::divide(orientation, xx + yy + 1.e-6, orientation);
    RegionalClarity result;
    for (std::size_t i = 0; i < evidence_parts.size(); ++i) {
        const auto part = evidence_parts[i];
        const cv::Rect area(part.x * 2, part.y * 2, part.width * 2, part.height * 2);
        double weighted_gain = 0., support = 0.;
        for (int y = area.y; y < area.br().y; ++y) {
            const auto* directions = orientation.ptr<float>(y);
            const auto* coarse_energy = strength.ptr<float>(y);
            const auto* fine_aligned = projected.ptr<float>(y);
            for (int x = area.x; x < area.br().x; ++x) {
                const double weight = std::max(0., (directions[x] - .52) / .48) * coarse_energy[x];
                const double gain = std::clamp(fine_aligned[x] / std::max(coarse_energy[x], 1.e-6f), 0.f, 3.f);
                support += weight;
                weighted_gain += weight * gain;
            }
        }
        if (support >= .015)
            result[i] = weighted_gain / support;
    }
    return result;
}

bool clarity_improvement(const RegionalClarity& previous, const RegionalClarity& candidate,
                         ClearRepresentativeReplacement& proposal) {
    std::vector<double> gains;
    for (std::size_t i = 0; i < previous.size(); ++i) {
        if (previous[i].value_or(0.) <= 0. || !candidate[i])
            continue;
        const double gain = *candidate[i] / *previous[i] - 1.;
        if (gain < -.018)
            return false;
        proposal.regional_gain[i] = gain;
        gains.push_back(gain);
        proposal.clarity_gain += gain;
    }
    if (gains.size() < 3)
        return false;
    std::sort(gains.begin(), gains.end());
    const auto middle = gains.size() / 2;
    const double median = gains.size() % 2 ? gains[middle] : (gains[middle - 1] + gains[middle]) * .5;
    proposal.clarity_gain /= gains.size();
    return median >= .033 && gains[gains.size() - 2] >= .021;
}

ReplayFits fit_clear_replays(const cv::Mat& portrait, const cv::Mat& observed, const ReplayFolds& folds) {
    const auto gray = thumbnail(portrait);
    const std::array images{gray, smooth(gray, .65), smooth(gray, 1.25), smooth(gray, 2.1)};
    ReplayFits fits;
    for (std::size_t i = 0; i < fits.size(); ++i)
        fits[i] = fit_replay(images, observed, folds[i], 2, 1.e-6);
    return fits;
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

AppearanceEvidence analyse_appearance_bank(const cv::Mat& source, const std::vector<cv::Mat>& candidates) {
    const auto observed = thumbnail(source);
    const auto count = candidates.size();
    AppearanceEvidence result{evidence_shape(observed).detail_strength, {}, {},
                              std::vector<std::vector<double>>(count, std::vector<double>(count))};
    std::vector<EvidenceShape> shapes;
    shapes.reserve(count);
    for (const auto& candidate : candidates) {
        const auto gray = thumbnail(candidate);
        shapes.push_back(evidence_shape(gray));
        result.candidate_detail.push_back(shapes.back().detail_strength);
        result.replay_errors.push_back(registered_error(gray, observed));
    }
    for (std::size_t i = 0; i < count; ++i) {
        for (std::size_t j = i + 1; j < count; ++j)
            result.distances[i][j] = result.distances[j][i] = evidence_distance(shapes[i], shapes[j]);
    }
    return result;
}

EvidenceExtension extend_appearances(const cv::Mat& source, const std::vector<cv::Mat>& candidates,
                                     std::size_t kept_count, double radius, double replay_tolerance,
                                     std::size_t max_additions,
                                     const std::function<bool(std::size_t, std::size_t)>& compatible) {
    if (kept_count == 0 || kept_count > candidates.size() || !std::isfinite(radius) || radius <= 0. || radius > 2. ||
        !std::isfinite(replay_tolerance) || replay_tolerance < 0. || replay_tolerance > 1.)
        throw std::invalid_argument("Expected a nonempty protected prefix, radius in (0, 2], and replay tolerance in [0, 1].");
    auto evidence = analyse_appearance_bank(source, candidates);
    const auto& distances = evidence.distances;
    const auto count = candidates.size();
    EvidenceExtension result;
    result.replay_errors = std::move(evidence.replay_errors);
    const double anchor_error = *std::min_element(result.replay_errors.begin(),
                                                  result.replay_errors.begin() + kept_count);
    for (std::size_t i = 0; i < kept_count; ++i)
        result.representatives.push_back(i);
    result.distances_to_kept.resize(count);
    std::vector<double> weights(count);
    for (std::size_t i = 0; i < count; ++i) {
        result.distances_to_kept[i] = *std::min_element(distances[i].begin(), distances[i].begin() + kept_count);
        const double error = result.replay_errors[i];
        if (error > anchor_error + replay_tolerance || evidence.candidate_detail[i] < evidence.observed_detail)
            continue;
        double density = 0.;
        for (double distance : distances[i])
            density += std::exp(-distance / radius);
        weights[i] = std::exp(-std::max(0., error - anchor_error) / .014) / density;
    }
    auto nearest = result.distances_to_kept;
    std::vector<bool> tried(count);
    while (result.representatives.size() - kept_count < max_additions) {
        double best_gain = 0.;
        std::size_t best = count;
        for (std::size_t i = kept_count; i < count; ++i) {
            if (tried[i] || weights[i] <= 0. || nearest[i] <= radius)
                continue;
            double gain = 0.;
            for (std::size_t j = 0; j < count; ++j)
                gain += weights[j] * std::max(0., nearest[j] - distances[i][j]);
            if (gain > best_gain) {
                best_gain = gain;
                best = i;
            }
        }
        if (best == count)
            break;
        tried[best] = true;
        if (compatible && !std::all_of(result.representatives.begin(), result.representatives.end(),
                                       [&](auto previous) { return compatible(previous, best); }))
            continue;
        result.representatives.push_back(best);
        for (std::size_t i = 0; i < count; ++i)
            nearest[i] = std::min(nearest[i], distances[best][i]);
    }
    return result;
}

std::optional<RepresentativeReplacement> refine_representative(
    const cv::Mat& source, const std::vector<cv::Mat>& candidates, std::size_t kept_count,
    const std::function<bool(std::size_t, std::size_t)>& visibly_different) {
    if (kept_count == 0 || kept_count > candidates.size() || !visibly_different)
        throw std::invalid_argument("Expected a nonempty displayed prefix and a landmark comparison");
    const auto observed = thumbnail(source);
    const auto folds = replay_folds(observed);
    if (std::any_of(folds.begin(), folds.end(), [](const auto& fold) { return fold.checking.size() < 80; }))
        return std::nullopt;

    const auto observed_detail = evidence_shape(observed).detail_strength;
    std::vector<EvidenceShape> shapes;
    std::vector<ReplayFits> fits(kept_count);
    for (std::size_t i = 0; i < kept_count; ++i) {
        const auto gray = thumbnail(candidates[i]);
        shapes.push_back(evidence_shape(gray));
        const auto images = replay_images(gray);
        for (std::size_t j = 0; j < folds.size(); ++j)
            fits[i][j] = fit_replay(images, observed, folds[j]);
    }

    std::vector<RepresentativeReplacement> replacements;
    for (std::size_t i = kept_count; i < candidates.size(); ++i) {
        const auto gray = thumbnail(candidates[i]);
        const auto shape = evidence_shape(gray);
        RepresentativeReplacement proposal{0, i, std::numeric_limits<double>::infinity(),
                                            std::numeric_limits<double>::infinity(), {}};
        for (std::size_t j = 0; j < kept_count; ++j) {
            const double distance = evidence_distance(shape, shapes[j]);
            if (distance < proposal.distance) {
                proposal.distance = distance;
                proposal.previous = j;
            }
        }
        if (proposal.distance > .15 || shape.detail_strength <
            std::max(.98 * observed_detail, .84 * shapes[proposal.previous].detail_strength))
            continue;
        const auto images = replay_images(gray);
        bool supported = true;
        for (std::size_t j = 0; j < folds.size(); ++j) {
            const auto& fit = fits[proposal.previous][j];
            const double gain = fit.error - replay_error(images[fit.blur], observed, folds[j].checking, fit);
            if (gain < std::max(.00015, .03 * fit.error)) {
                supported = false;
                break;
            }
            proposal.replay_improvements[j] = gain;
            proposal.score = std::min(proposal.score, gain / std::max(.001, fit.error));
        }
        if (supported)
            replacements.push_back(proposal);
    }
    std::stable_sort(replacements.begin(), replacements.end(), [](const auto& a, const auto& b) {
        return a.score > b.score;
    });
    for (const auto& proposal : replacements) {
        if (visibly_different(proposal.previous, proposal.proposed))
            continue;
        bool distinct_from_others = true;
        for (std::size_t i = 0; i < kept_count && distinct_from_others; ++i) {
            if (i != proposal.previous)
                distinct_from_others = visibly_different(i, proposal.proposed);
        }
        if (distinct_from_others)
            return proposal;
    }
    return std::nullopt;
}

std::optional<ClearRepresentativeReplacement> refine_clear_representative(
    const cv::Mat& source, const std::vector<cv::Mat>& candidates,
    std::size_t kept_count, std::size_t added_count,
    const std::function<bool(std::size_t, std::size_t)>& visibly_different) {
    if (kept_count == 0 || kept_count > candidates.size() ||
        added_count > candidates.size() - kept_count || !visibly_different)
        throw std::invalid_argument("Expected a nonempty representative prefix and a valid addition count");
    const auto displayed_count = kept_count + added_count;
    const auto observed = thumbnail(source);
    if (displayed_count == candidates.size())
        return std::nullopt;
    const auto folds = replay_folds(observed);
    if (std::any_of(folds.begin(), folds.end(), [](const auto& fold) { return fold.checking.size() < 80; }))
        return std::nullopt;

    constexpr double radius = .055;
    constexpr double minimum_pair_distance = 1.e-7;
    const auto evidence = analyse_appearance_bank(source, candidates);
    std::vector<std::optional<RegionalClarity>> original_clarity(kept_count);
    std::vector<ClearRepresentativeReplacement> proposals;
    for (std::size_t i = kept_count; i < candidates.size(); ++i) {
        const auto& distances = evidence.distances[i];
        const auto nearest = std::min_element(distances.begin(), distances.begin() + kept_count);
        const auto previous = static_cast<std::size_t>(nearest - distances.begin());
        if (*nearest > radius || evidence.candidate_detail[i] < .98 * evidence.observed_detail ||
            evidence.replay_errors[i] > evidence.replay_errors[previous] + .004)
            continue;
        bool loses_separation = false;
        for (std::size_t j = 0; j < kept_count; ++j) {
            if (j != previous && evidence.distances[previous][j] > radius && distances[j] <= radius) {
                loses_separation = true;
                break;
            }
        }
        if (loses_separation)
            continue;
        if (!original_clarity[previous])
            original_clarity[previous] = oriented_clarity(candidates[previous]);
        ClearRepresentativeReplacement proposal{};
        proposal.previous = previous;
        proposal.proposed = i;
        proposal.distance = *nearest;
        if (clarity_improvement(*original_clarity[previous], oriented_clarity(candidates[i]), proposal))
            proposals.push_back(proposal);
    }
    std::sort(proposals.begin(), proposals.end(), [](const auto& left, const auto& right) {
        return std::tie(left.clarity_gain, left.previous, left.proposed) >
               std::tie(right.clarity_gain, right.previous, right.proposed);
    });

    std::vector<std::optional<ReplayFits>> original_replays(kept_count);
    const auto different = [&](std::size_t first, std::size_t second) {
        return evidence.distances[first][second] > minimum_pair_distance && visibly_different(first, second);
    };
    for (auto proposal : proposals) {
        const auto previous = proposal.previous, next = proposal.proposed;
        if (!original_replays[previous])
            original_replays[previous] = fit_clear_replays(candidates[previous], observed, folds);
        const auto& prior = *original_replays[previous];
        const auto trial = fit_clear_replays(candidates[next], observed, folds);
        bool supported = true;
        for (std::size_t i = 0; i < folds.size(); ++i) {
            proposal.replay_deltas[i] = trial[i].error - prior[i].error;
            proposal.replay_limits[i] = std::clamp(.06 * prior[i].error, .0006, .0013);
            const auto nudge = trial[i].offset - prior[i].offset;
            proposal.maximum_offset_change = std::max({proposal.maximum_offset_change,
                                                        std::abs(nudge.x), std::abs(nudge.y)});
            if (proposal.replay_deltas[i] > proposal.replay_limits[i] || proposal.maximum_offset_change > 1) {
                supported = false;
                break;
            }
        }
        if (!supported || different(previous, next) || different(next, previous))
            continue;
        for (std::size_t j = 0; j < kept_count && supported; ++j) {
            if (j != previous && different(previous, j) && !different(next, j))
                supported = false;
        }
        if (!supported)
            continue;
        // Use the same addition guard after a replacement has been chosen.
        // The existing display remains the fallback for a conflicting proposal.
        for (std::size_t j = kept_count; j < displayed_count; ++j) {
            if (evidence.distances[next][j] <= radius || !visibly_different(next, j))
                return std::nullopt;
        }
        return proposal;
    }
    return std::nullopt;
}

}
