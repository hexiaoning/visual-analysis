#include "screen_restore.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

#include <opencv2/imgproc.hpp>

namespace visual::screen {
namespace {

constexpr std::array<double, 4> blurScales{0., .55, 1.15, 1.8};
constexpr std::array<std::array<float, 4>, 4> featureRegions{{
    {192.f, 240.f, 38.f, 24.f}, {320.f, 240.f, 38.f, 24.f},
    {257.f, 310.f, 34.f, 35.f}, {256.f, 373.f, 51.f, 26.f},
}};
const cv::Rect detailRegion(107, 160, 298, 270);

cv::Mat blur(const cv::Mat& src, double scale) {
    if (scale == 0.) {
        return src;
    }
    cv::Mat result;
    cv::GaussianBlur(src, result, {}, scale, scale, cv::BORDER_REFLECT_101);
    return result;
}

cv::Mat grayscale(const cv::Mat& pixels) {
    cv::Mat gray, signal;
    cv::cvtColor(pixels, gray, cv::COLOR_BGR2GRAY);
    gray.convertTo(signal, CV_32F, 1. / 255.);
    return signal;
}

cv::Mat floatLuma(const cv::Mat& bgr) {
    cv::Mat signal, gray;
    bgr.convertTo(signal, CV_32FC3, 1. / 255.);
    cv::cvtColor(signal, gray, cv::COLOR_BGR2GRAY);
    return gray;
}

double detailEnergy(const cv::Mat& bgr) {
    const auto gray = floatLuma(bgr);
    const cv::Mat detail = blur(gray, .65) - blur(gray, 2.2);
    const cv::Mat face = detail(detailRegion);
    return face.dot(face) / static_cast<double>(face.total());
}

cv::Rect captureBounds(cv::Size raster, const cv::Matx23d& nativeToAligned) {
    const double determinant = nativeToAligned(0, 0) * nativeToAligned(1, 1)
        - nativeToAligned(0, 1) * nativeToAligned(1, 0);
    if (!std::isfinite(determinant) || std::abs(determinant) <= 1e-10) {
        throw std::invalid_argument("The native-to-aligned transform must be invertible");
    }
    cv::Matx23d inverse;
    cv::invertAffineTransform(nativeToAligned, inverse);
    cv::Point2d low(std::numeric_limits<double>::infinity(), std::numeric_limits<double>::infinity());
    cv::Point2d high(-low.x, -low.y);
    for (double y : {89., 459.}) {
        for (double x : {58., 454.}) {
            const auto p = inverse * cv::Vec3d(x, y, 1.);
            low.x = std::min(low.x, p[0]);
            low.y = std::min(low.y, p[1]);
            high.x = std::max(high.x, p[0]);
            high.y = std::max(high.y, p[1]);
        }
    }
    const int x = static_cast<int>(std::clamp(std::floor(low.x - 14.), 0., static_cast<double>(raster.width)));
    const int y = static_cast<int>(std::clamp(std::floor(low.y - 14.), 0., static_cast<double>(raster.height)));
    const int right = static_cast<int>(std::clamp(std::ceil(high.x + 15.), 0., static_cast<double>(raster.width)));
    const int bottom = static_cast<int>(std::clamp(std::ceil(high.y + 15.), 0., static_cast<double>(raster.height)));
    if (right - x < 42 || bottom - y < 42) {
        throw std::invalid_argument("The source does not contain enough native face pixels");
    }
    return {x, y, right - x, bottom - y};
}

struct CaptureGrid {
    cv::Mat xCoordinates;
    cv::Mat yCoordinates;
    cv::Mat surround;
    std::array<cv::Mat, featureRegions.size()> features;
};

CaptureGrid makeGrid(const cv::Rect& rect, const cv::Matx23d& affine) {
    CaptureGrid grid{cv::Mat(rect.size(), CV_32F), cv::Mat(rect.size(), CV_32F),
                     cv::Mat::zeros(rect.size(), CV_8U), {}};
    for (auto& weights : grid.features) {
        weights.create(rect.size(), CV_32F);
    }
    for (int y = 0; y < rect.height; ++y) {
        for (int x = 0; x < rect.width; ++x) {
            const auto pos = affine * cv::Vec3d(rect.x + x, rect.y + y, 1.);
            const float u = static_cast<float>(pos[0]), v = static_cast<float>(pos[1]);
            grid.xCoordinates.at<float>(y, x) = u;
            grid.yCoordinates.at<float>(y, x) = v;
            float strongest = 0.f;
            for (std::size_t i = 0; i < featureRegions.size(); ++i) {
                const auto region = featureRegions[i];
                const float dx = (u - region[0]) / region[2];
                const float dy = (v - region[1]) / region[3];
                const float weight = std::max(0.f, 1.f - (dx * dx + dy * dy));
                grid.features[i].at<float>(y, x) = weight;
                strongest = std::max(strongest, weight);
            }
            const float dx = (u - 256.f) / 159.f, dy = (v - 293.f) / 159.f;
            if (dx * dx + dy * dy < 1.f && strongest < .07f) {
                grid.surround.at<unsigned char>(y, x) = 255;
            }
        }
    }
    if (cv::countNonZero(grid.surround) < 100) {
        throw std::invalid_argument("The source does not contain enough pixels around the facial features");
    }
    return grid;
}

cv::Mat matchExposure(const cv::Mat& src, const cv::Mat& outerSource, const cv::Mat& mask) {
    const cv::Mat outer = blur(src, 3.5);
    const double targetMean = cv::mean(outerSource, mask)[0];
    const double sourceMean = cv::mean(outer, mask)[0];
    const cv::Mat a = outerSource - targetMean;
    const cv::Mat b = outer - sourceMean;
    const double gain = std::clamp(cv::mean(a.mul(b), mask)[0] /
                                      std::max(cv::mean(b.mul(b), mask)[0], 1e-8),
                                  .5, 1.8);
    const cv::Mat matched = src * gain + (targetMean - sourceMean * gain);
    return matched;
}

double outerError(const cv::Mat& residual, const cv::Mat& mask) {
    std::vector<float> squared;
    squared.reserve(static_cast<std::size_t>(cv::countNonZero(mask)));
    for (int y = 0; y < mask.rows; ++y) {
        const auto* m = mask.ptr<unsigned char>(y);
        const auto* d = residual.ptr<float>(y);
        for (int x = 0; x < mask.cols; ++x) {
            if (m[x]) {
                squared.push_back(d[x] * d[x]);
            }
        }
    }
    const auto mid = squared.begin() + squared.size() / 2;
    std::nth_element(squared.begin(), mid, squared.end());
    return squared.size() % 2 ? *mid : (*std::max_element(squared.begin(), mid) + *mid) * .5;
}

double featureError(const cv::Mat& raw, const cv::Mat& rawBand, const cv::Mat& rendered,
                    const std::array<cv::Mat, featureRegions.size()>& weights) {
    const cv::Mat renderedBand = blur(rendered, .58) - blur(rendered, 2.4);
    const cv::Mat bandResidual = renderedBand - rawBand;
    const cv::Mat wideResidual = blur(rendered - raw, 2.6);
    const cv::Mat bandSquared = bandResidual.mul(bandResidual);
    const cv::Mat wideSquared = wideResidual.mul(wideResidual);
    const cv::Mat rawSquared = rawBand.mul(rawBand);
    double score = 0.;
    std::size_t count = 0;
    for (const auto& weight : weights) {
        const double area = cv::sum(weight)[0];
        if (area >= 5.) {
            score += cv::sum(weight.mul(bandSquared))[0] / std::max(cv::sum(weight.mul(rawSquared))[0], 1e-6)
                + 3. * cv::sum(weight.mul(wideSquared))[0] / area;
            ++count;
        }
    }
    if (count < 3) {
        throw std::invalid_argument("The source does not contain enough visible facial features");
    }
    return score / static_cast<double>(count);
}

std::optional<double> measureChromaBand(const cv::Mat& bgr, const CaptureGrid& grid) {
    cv::Mat color;
    bgr.convertTo(color, CV_32FC3, 1. / 255.);
    cv::cvtColor(color, color, cv::COLOR_BGR2Lab);
    const cv::Mat detail = blur(color, .55) - blur(color, 2.4);
    const cv::Mat x = (grid.xCoordinates - 256.f) / 118.f;
    const cv::Mat y = (grid.yCoordinates - 298.f) / 125.f;
    cv::Mat skin = x.mul(x) + y.mul(y) < 1.f;
    cv::bitwise_and(skin, grid.surround, skin);
    if (cv::countNonZero(skin) < 100) {
        return std::nullopt;
    }
    const cv::Scalar energy = cv::mean(detail.mul(detail), skin);
    return std::sqrt(energy[1] + energy[2]);
}

bool hasSquareCarrier(const cv::Mat& bgr) {
    const auto periods = detect(bgr).periods;
    if (periods.size() != 2) {
        return false;
    }
    const double first = cv::norm(periods[0]), second = cv::norm(periods[1]);
    return std::max(first, second) <= 1.08 * std::min(first, second) &&
        std::abs(periods[0].dot(periods[1])) <= .12 * first * second;
}

double weightedCorrelation(const cv::Mat& a, const cv::Mat& b, const cv::Mat& weights, double area) {
    const cv::Mat left = a - a.dot(weights) / area;
    const cv::Mat right = b - b.dot(weights) / area;
    const cv::Mat weightedLeft = left.mul(weights);
    const cv::Mat weightedRight = right.mul(weights);
    const double magnitude = std::sqrt(left.dot(weightedLeft) * right.dot(weightedRight));
    return std::clamp(right.dot(weightedLeft) / std::max(magnitude, 1e-8), -1., 1.);
}

double highBandRms(const cv::Mat& signal, const cv::Mat& mask) {
    const cv::Mat band = blur(signal, .47) - blur(signal, 1.25);
    return std::sqrt(cv::mean(band.mul(band), mask)[0]);
}

bool preferContourAlternative(const cv::Mat& observed, const cv::Mat& current, const cv::Mat& alternative,
                              const CaptureGrid& grid, const CaptureReplaySelection& result,
                              std::size_t candidateIndex) {
    const cv::Mat sourceBand = blur(observed, 1.1) - blur(observed, 3.2);
    const cv::Mat currentBand = blur(current, 1.1) - blur(current, 3.2);
    const cv::Mat nextBand = blur(alternative, 1.1) - blur(alternative, 3.2);
    double difference = 0.;
    std::size_t count = 0;
    for (const auto& weights : grid.features) {
        const double area = cv::sum(weights)[0];
        if (area < 5.) {
            continue;
        }
        const double currentAgreement = weightedCorrelation(sourceBand, currentBand, weights, area);
        const double nextAgreement = weightedCorrelation(sourceBand, nextBand, weights, area);
        if (nextAgreement < std::max(.86, currentAgreement - .03)) {
            return false;
        }
        difference += nextAgreement - currentAgreement;
        ++count;
    }
    if (count < 3) {
        return false;
    }
    const double averageDifference = difference / static_cast<double>(count);
    const auto& score = result.scores[candidateIndex];
    const double referenceError = result.scores[result.selected].error;
    const double observedEnergy = std::max(highBandRms(observed, grid.surround), 1e-6);
    const double candidateEnergy = highBandRms(alternative, grid.surround);
    const bool strongerReplay = score.error <= .60 * referenceError && averageDifference >= .012 &&
        candidateEnergy <= .96 * observedEnergy && score.relativeDetail >= .60;
    if (strongerReplay) {
        return true;
    }
    const double currentEnergy = highBandRms(current, grid.surround);
    const double retained = candidateEnergy / std::max(currentEnergy, 1e-7 * observedEnergy);
    const double allowedContourLoss = std::min(.018, .006 + .12 * (1. - retained));
    return result.chromaBandRms && *result.chromaBandRms < 2. && score.error <= .28 &&
        retained <= .98 && averageDifference >= -allowedContourLoss;
}

std::size_t choosePrimary(const std::vector<CaptureReplayScore>& scores, bool allowMoreDetail) {
    const double base = scores.front().error;
    std::size_t selected = 0;
    for (std::size_t i = 1; i < scores.size(); ++i) {
        const auto& score = scores[i];
        const double improvement = (base - score.error) / std::max(base, 1e-7);
        const bool plausible = score.relativeDetail <= 1.4 || (allowMoreDetail && improvement > .20);
        if (plausible && score.relativeDetail >= .60 && improvement > .02 && score.error < scores[selected].error) {
            selected = i;
        }
    }
    if (!allowMoreDetail || selected == 0) {
        return selected;
    }
    const auto& first = scores[selected];
    std::size_t sharper = selected;
    for (std::size_t i = 1; i < scores.size(); ++i) {
        const auto& score = scores[i];
        if (score.relativeDetail >= 1.20 * first.relativeDetail && score.error <= 1.10 * first.error &&
            score.error < .80 * base && (sharper == selected || score.error < scores[sharper].error)) {
            sharper = i;
        }
    }
    return sharper;
}

struct Carrier {
    cv::Mat cosine;
    cv::Mat sine;
    cv::Mat observedCos;
    cv::Mat observedSin;
    cv::Mat support;
    double energy{};
};

using Phase = std::pair<cv::Mat, cv::Mat>;

Phase demodulate(const cv::Mat& high, const Carrier& carrier) {
    return {2. * blur(high.mul(carrier.cosine), 8.),
            2. * blur(high.mul(carrier.sine), 8.)};
}

cv::Mat magnitude(const Phase& phase) {
    cv::Mat strength;
    cv::magnitude(phase.first, phase.second, strength);
    return strength;
}

std::vector<Carrier> measureCarriers(const cv::Mat& aligned, const cv::Mat& projection,
                                     const cv::Matx23d& affine, const Peaks& peaks) {
    const cv::Matx22d basis(affine(0, 0), affine(0, 1), affine(1, 0), affine(1, 1));
    const cv::Mat high = aligned - blur(aligned, 2.);
    const cv::Mat projectionHigh = projection - blur(projection, 2.);
    std::vector<Carrier> carriers;
    for (const auto& period : peaks.periods) {
        cv::Vec2d f = basis.t().inv() * cv::Vec2d(period.x, period.y);
        for (int i = 0; i < 2; ++i) {
            f[i] -= std::floor(f[i] + .5);
        }
        if (cv::norm(f) < .1) {
            continue;
        }
        Carrier carrier;
        carrier.cosine.create(aligned.size(), CV_32F);
        carrier.sine.create(aligned.size(), CV_32F);
        for (int y = 0; y < aligned.rows; ++y) {
            for (int x = 0; x < aligned.cols; ++x) {
                const float angle = static_cast<float>(2. * CV_PI * (x * f[0] + y * f[1]));
                carrier.cosine.at<float>(y, x) = std::cos(angle);
                carrier.sine.at<float>(y, x) = std::sin(angle);
            }
        }
        const auto phase = demodulate(high, carrier);
        const auto amplitude = magnitude(phase);
        const auto coherentAmplitude = magnitude({blur(phase.first, 12.), blur(phase.second, 12.)});
        const cv::Mat coherenceBound = .63 * cv::max(blur(amplitude, 12.), 1e-6);
        cv::bitwise_and(amplitude(detailRegion) > .010,
                        coherentAmplitude(detailRegion) > coherenceBound(detailRegion), carrier.support);
        if (cv::countNonZero(carrier.support) >= 200) {
            carrier.energy = cv::norm(amplitude(detailRegion), cv::NORM_L2SQR, carrier.support);
        }
        auto observed = demodulate(projectionHigh, carrier);
        carrier.observedCos = std::move(observed.first);
        carrier.observedSin = std::move(observed.second);
        carriers.push_back(std::move(carrier));
    }
    return carriers;
}

double carrierRatio(const cv::Mat& high, const std::vector<Carrier>& carriers, double observedEnergy) {
    double energy = 0.;
    for (const auto& carrier : carriers) {
        if (carrier.energy > 0.) {
            const auto amplitude = magnitude(demodulate(high, carrier));
            energy += cv::norm(amplitude(detailRegion), cv::NORM_L2SQR, carrier.support);
        }
    }
    return std::sqrt(energy / observedEnergy);
}

cv::Mat subtractObservedPhase(const cv::Mat& candidate, const cv::Mat& high,
                             const std::vector<Carrier>& carriers) {
    cv::Mat correction = cv::Mat::zeros(high.size(), CV_32F);
    for (const auto& carrier : carriers) {
        const auto a = carrier.observedCos;
        const auto b = carrier.observedSin;
        const auto amplitude = magnitude({a, b});
        const auto steady = magnitude({blur(a, 14.), blur(b, 14.)});
        cv::Mat coherence;
        cv::divide(steady, cv::max(blur(amplitude, 14.), 1e-6), coherence);
        const cv::Mat confidence = cv::min(cv::max((coherence - .45) / .35, 0.), 1.);
        const cv::Mat strength = cv::min(cv::max((amplitude - .004) / .012, 0.), 1.);
        const auto q = demodulate(high, carrier);
        cv::Mat matching;
        cv::divide(q.first.mul(a) + q.second.mul(b), cv::max(amplitude.mul(amplitude), 1e-7), matching);
        const cv::Mat common = cv::min(cv::max(matching, 0.), 1.);
        const cv::Mat wave = a.mul(carrier.cosine) + b.mul(carrier.sine);
        correction += confidence.mul(strength).mul(common).mul(wave);
    }
    correction = cv::min(cv::max(correction, -.12), .12);
    cv::Mat rgb, value;
    cv::merge(std::vector<cv::Mat>(3, correction), rgb);
    candidate.convertTo(value, CV_32F, 1. / 255.);
    value -= rgb;
    value.convertTo(value, CV_8UC3, 255.);
    return value;
}

cv::Mat detailProtection() {
    cv::Mat mask(512, 512, CV_32F);
    for (int y = 0; y < mask.rows; ++y) {
        for (int x = 0; x < mask.cols; ++x) {
            float protect = 0.f;
            for (const auto& region : featureRegions) {
                const auto dx = (static_cast<float>(x) - region[0]) / region[2];
                const auto dy = (static_cast<float>(y) - region[1]) / region[3];
                protect = std::max(protect, std::exp(-.75f * (dx * dx + dy * dy)));
            }
            const float dx = (x - 256.f) / 200.f;
            const float dy = (y - 295.f) / 204.f;
            const float taper = std::clamp((1.f - dx * dx - dy * dy) / .25f, 0.f, 1.f);
            mask.at<float>(y, x) = taper * (1.f - .83f * protect);
        }
    }
    return mask;
}

cv::Mat limitDetail(const cv::Mat& candidate, double baseEnergy) {
    constexpr double maximum = 1.385;
    const double allowedEnergy = maximum * maximum * baseEnergy;
    if (detailEnergy(candidate) <= allowedEnergy) {
        return candidate;
    }
    const auto y = floatLuma(candidate);
    static const auto protection = detailProtection();
    const cv::Mat difference = protection.mul(y - blur(y, 1.7));
    cv::Mat rgbDifference, normalized;
    cv::merge(std::vector<cv::Mat>(3, difference), rgbDifference);
    candidate.convertTo(normalized, CV_32F, 1. / 255.);
    const auto render = [&](double strength) {
        cv::Mat out;
        const cv::Mat next = normalized - strength * rgbDifference;
        next.convertTo(out, CV_8UC3, 255.);
        return out;
    };
    auto kept = render(1.);
    if (detailEnergy(kept) > allowedEnergy) {
        return {};
    }
    double low = 0., high = 1.;
    for (int i = 0; i < 14; ++i) {
        const double mid = (low + high) / 2.;
        auto next = render(mid);
        if (detailEnergy(next) > allowedEnergy) {
            low = mid;
        } else {
            high = mid;
            kept = std::move(next);
        }
    }
    return kept;
}

}

CaptureReplaySelection rankAtCaptureResolution(const cv::Mat& nativeBgr, const cv::Matx23d& nativeToAligned,
                                               const std::vector<cv::Mat>& alignedCandidates, bool hasCurrentPair,
                                               bool guardCaptureDetail,
                                               std::optional<std::size_t> contourAlternative) {
    if (nativeBgr.empty() || nativeBgr.type() != CV_8UC3 || alignedCandidates.empty() ||
        !cv::checkRange(cv::Mat(nativeToAligned)) ||
        std::any_of(alignedCandidates.begin(), alignedCandidates.end(), [](const cv::Mat& image) {
            return image.type() != CV_8UC3 || image.size() != cv::Size(512, 512);
        })) {
        throw std::invalid_argument("Expected an eight-bit color source and aligned 512x512 color candidates");
    }
    if (contourAlternative && (!guardCaptureDetail || *contourAlternative >= alignedCandidates.size())) {
        throw std::invalid_argument("Contour comparison requires an existing candidate and capture detail protection");
    }
    const auto bounds = captureBounds(nativeBgr.size(), nativeToAligned);
    const auto grid = makeGrid(bounds, nativeToAligned);
    const cv::Mat crop = nativeBgr(bounds);
    const cv::Mat raw = grayscale(crop);
    const auto outer = blur(raw, 3.5);
    const auto baseLow = blur(raw, 1.8);
    const cv::Mat band = blur(raw, .58) - blur(raw, 2.4);
    const double scaleSquared = nativeToAligned(0, 0) * nativeToAligned(0, 0)
        + nativeToAligned(1, 0) * nativeToAligned(1, 0);
    const double antialias = .43 * std::sqrt(std::max(0., scaleSquared - 1.));
    const auto resample = [&](const cv::Mat& bgr) {
        cv::Mat image;
        cv::remap(blur(grayscale(bgr), antialias), image, grid.xCoordinates, grid.yCoordinates,
                  cv::INTER_LINEAR, cv::BORDER_REFLECT_101);
        return image;
    };
    const auto primary = resample(alignedCandidates.front());
    CaptureReplaySelection result;
    if (guardCaptureDetail) {
        result.chromaBandRms = measureChromaBand(crop, grid);
    }
    double best = std::numeric_limits<double>::infinity();
    for (double scale : blurScales) {
        const auto matched = matchExposure(blur(primary, scale), outer, grid.surround);
        const double error = outerError(blur(matched, 1.8) - baseLow, grid.surround);
        if (error < best) {
            result.blur = scale;
            best = error;
        }
    }
    const double baseEnergy = std::max(detailEnergy(alignedCandidates.front()), 1e-8);
    const auto replay = [&](const cv::Mat& candidate) {
        return matchExposure(blur(resample(candidate), result.blur), outer, grid.surround);
    };
    result.scores.reserve(alignedCandidates.size());
    for (const auto& candidate : alignedCandidates) {
        const auto corrected = replay(candidate);
        result.scores.push_back({featureError(raw, band, corrected, grid.features),
                                 std::sqrt(detailEnergy(candidate) / baseEnergy)});
    }
    if (!hasCurrentPair) {
        const bool quietChroma = result.chromaBandRms.has_value() && *result.chromaBandRms < 2.;
        result.squareCarrier = quietChroma && hasSquareCarrier(crop);
        result.selected = choosePrimary(result.scores, quietChroma && !result.squareCarrier);
        if (contourAlternative && *contourAlternative != result.selected && !result.squareCarrier &&
            preferContourAlternative(raw, replay(alignedCandidates[result.selected]),
                                     replay(alignedCandidates[*contourAlternative]), grid, result, *contourAlternative)) {
            result.selected = *contourAlternative;
            result.contourAlternativeUsed = true;
        }
    }
    return result;
}

std::vector<CaptureDetailCandidate> balanceCaptureDetail(const cv::Mat& nativeBgr, const cv::Matx23d& affine,
                                                        const std::vector<cv::Mat>& crops,
                                                        const CaptureReplaySelection& replay) {
    std::vector<CaptureDetailCandidate> result;
    if (!replay.squareCarrier) {
        return result;
    }
    if (crops.empty() || crops.size() != replay.scores.size()) {
        throw std::invalid_argument("Detail balancing requires the original capture replay");
    }
    const double baseError = replay.scores.front().error;
    const auto eligible = [&](const CaptureReplayScore& score) {
        return score.relativeDetail > 1.4 && score.relativeDetail <= 1.85 && score.error <= .92 * baseError;
    };
    if (std::none_of(replay.scores.begin(), replay.scores.end(), eligible)) {
        return result;
    }
    const auto region = captureBounds(nativeBgr.size(), affine);
    const auto periods = detect(nativeBgr(region));
    cv::Mat observed, projection;
    cv::warpAffine(nativeBgr, observed, affine, {512, 512}, cv::INTER_LINEAR, cv::BORDER_REFLECT_101);
    cv::warpAffine(floatLuma(nativeBgr), projection, affine, {512, 512}, cv::INTER_LINEAR,
                   cv::BORDER_REFLECT_101);
    const auto carriers = measureCarriers(floatLuma(observed), projection, affine, periods);
    double observedEnergy = 0.;
    for (const auto& carrier : carriers) {
        observedEnergy += carrier.energy;
    }
    if (observedEnergy <= 0.) {
        return result;
    }
    const double baseEnergy = std::max(detailEnergy(crops.front()), 1e-8);
    for (std::size_t i = 1; i < crops.size(); ++i) {
        if (!eligible(replay.scores[i])) {
            continue;
        }
        const auto gray = floatLuma(crops[i]);
        const cv::Mat high = gray - blur(gray, 2.);
        if (carrierRatio(high, carriers, observedEnergy) > .30) {
            continue;
        }
        const auto corrected = subtractObservedPhase(crops[i], high, carriers);
        const auto correctedY = floatLuma(corrected);
        if (carrierRatio(correctedY - blur(correctedY, 2.), carriers, observedEnergy) > .30) {
            continue;
        }
        auto bounded = limitDetail(corrected, baseEnergy);
        if (!bounded.empty()) {
            result.push_back({i, std::move(bounded)});
        }
    }
    return result;
}

}
