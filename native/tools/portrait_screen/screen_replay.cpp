#include "screen_restore.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>

#include <opencv2/imgproc.hpp>

namespace visual::screen {
namespace {

constexpr std::array<double, 4> blurScales{0., .55, 1.15, 1.8};
constexpr std::array<std::array<float, 4>, 4> featureRegions{{
    {192.f, 240.f, 38.f, 24.f}, {320.f, 240.f, 38.f, 24.f},
    {257.f, 310.f, 34.f, 35.f}, {256.f, 373.f, 51.f, 26.f},
}};

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

double detailEnergy(const cv::Mat& bgr) {
    cv::Mat signal, gray;
    bgr.convertTo(signal, CV_32FC3, 1. / 255.);
    cv::cvtColor(signal, gray, cv::COLOR_BGR2GRAY);
    const cv::Mat detail = blur(gray, .65) - blur(gray, 2.2);
    const cv::Mat face = detail(cv::Rect(107, 160, 298, 270));
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

}

CaptureReplaySelection rankAtCaptureResolution(const cv::Mat& nativeBgr, const cv::Matx23d& nativeToAligned,
                                               const std::vector<cv::Mat>& alignedCandidates, bool hasCurrentPair) {
    if (nativeBgr.empty() || nativeBgr.type() != CV_8UC3 || alignedCandidates.empty() ||
        !cv::checkRange(cv::Mat(nativeToAligned)) ||
        std::any_of(alignedCandidates.begin(), alignedCandidates.end(), [](const cv::Mat& image) {
            return image.type() != CV_8UC3 || image.size() != cv::Size(512, 512);
        })) {
        throw std::invalid_argument("Expected an eight-bit color source and aligned 512x512 color candidates");
    }
    const auto bounds = captureBounds(nativeBgr.size(), nativeToAligned);
    const auto grid = makeGrid(bounds, nativeToAligned);
    const cv::Mat raw = grayscale(nativeBgr(bounds));
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
    result.scores.reserve(alignedCandidates.size());
    for (const auto& candidate : alignedCandidates) {
        const auto corrected = matchExposure(blur(resample(candidate), result.blur), outer, grid.surround);
        result.scores.push_back({featureError(raw, band, corrected, grid.features),
                                 std::sqrt(detailEnergy(candidate) / baseEnergy)});
    }
    if (!hasCurrentPair) {
        for (std::size_t index = 1; index < result.scores.size(); ++index) {
            const auto& score = result.scores[index];
            const double base = result.scores.front().error;
            const double improvement = (base - score.error) / std::max(base, 1e-7);
            if (score.relativeDetail >= .60 && score.relativeDetail <= 1.4 && improvement > .02 &&
                score.error < result.scores[result.selected].error) {
                result.selected = index;
            }
        }
    }
    return result;
}

}
