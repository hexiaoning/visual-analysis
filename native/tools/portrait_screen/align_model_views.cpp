#include "align_model_views.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>
#include <utility>
#include <vector>

#include <opencv2/imgproc.hpp>

namespace visual::screen {
namespace {

constexpr int faceSize = 512;
constexpr int gridSize = faceSize / 2;
constexpr int shiftCount = 13;
constexpr int shiftTotal = shiftCount * shiftCount;
constexpr int centerIndex = shiftTotal / 2;
constexpr std::array<std::array<float, 4>, 4> features{{
    {192.f, 240.f, 40.f, 27.f}, {320.f, 240.f, 40.f, 27.f},
    {257.f, 310.f, 36.f, 38.f}, {256.f, 373.f, 53.f, 28.f},
}};

using Bands = std::array<cv::Mat, 2>;

void requireFace(const cv::Mat& image) {
    if (image.size() != cv::Size(faceSize, faceSize) || image.type() != CV_8UC3) {
        throw std::invalid_argument("Expected aligned 512x512 BGR model crops");
    }
}

Bands scales(const cv::Mat& input) {
    cv::Mat rgb, luma, grid;
    input.convertTo(rgb, CV_32FC3, 1. / 255.);
    cv::cvtColor(rgb, luma, cv::COLOR_BGR2GRAY);
    cv::resize(luma, grid, {gridSize, gridSize}, 0., 0., cv::INTER_AREA);
    const auto blur = [&](double sigma) {
        cv::Mat result;
        cv::GaussianBlur(grid, result, {}, sigma, 0., cv::BORDER_REFLECT_101);
        return result;
    };
    return {blur(.62) - blur(2.3), blur(1.2) - blur(3.9)};
}

using Windows = std::array<cv::Mat, 4>;

Windows surroundWindows() {
    Windows result;
    for (auto& w : result) {
        w = cv::Mat::zeros(gridSize, gridSize, CV_32F);
    }
    for (int y = 0; y < gridSize; ++y) {
        for (int x = 0; x < gridSize; ++x) {
            const float dx = (static_cast<float>(x) - 128.f) / 104.f;
            const float dy = (static_cast<float>(y) - 146.f) / 111.f;
            float taper = std::clamp((1.f - (dx * dx + dy * dy)) / .28f, 0.f, 1.f);
            for (const auto& region : features) {
                const float rx = (static_cast<float>(x) - .5f * region[0]) / (.60f * region[2]);
                const float ry = (static_cast<float>(y) - .5f * region[1]) / (.62f * region[3]);
                taper *= 1.f - std::exp(-(rx * rx + ry * ry));
            }
            const std::size_t quadrant = static_cast<std::size_t>((y >= 148 ? 2 : 0) + (x >= 128 ? 1 : 0));
            result[quadrant].at<float>(y, x) = taper;
        }
    }
    return result;
}

struct Region {
    cv::Mat weight;
    std::array<cv::Mat, 2> raw;
    std::array<cv::Mat, 2> anchor;
    std::array<double, 2> rawEnergy;
    std::array<double, 2> anchorEnergy;
};

std::vector<Region> regions(const Bands& raw, const Bands& anchor) {
    static const auto windows = surroundWindows();
    std::vector<Region> accepted;
    for (const auto& w : windows) {
        if (cv::countNonZero(w) < 220) {
            continue;
        }
        const double floor = 1e-5 * cv::sum(w)[0];
        Region region{};
        region.weight = w;
        bool informative = true;
        for (std::size_t i = 0; i < raw.size(); ++i) {
            region.raw[i] = raw[i].mul(w);
            region.anchor[i] = anchor[i].mul(w);
            region.rawEnergy[i] = raw[i].dot(region.raw[i]);
            region.anchorEnergy[i] = anchor[i].dot(region.anchor[i]);
            informative &= region.rawEnergy[i] >= floor && region.anchorEnergy[i] >= floor;
        }
        if (informative) {
            accepted.push_back(std::move(region));
        }
    }
    return accepted;
}

cv::Point2f displacement(int index) {
    return {-3.f + .5f * static_cast<float>(index / shiftCount),
            -3.f + .5f * static_cast<float>(index % shiftCount)};
}

using Correlation = std::array<double, 2>;

Correlation alignment(const Region& region, const Bands& proposed) {
    Correlation result{};
    for (std::size_t i = 0; i < proposed.size(); ++i) {
        const auto& face = proposed[i];
        const double energy = face.dot(face.mul(region.weight));
        result[0] += .5 * face.dot(region.raw[i]) /
            std::sqrt(std::max(energy * region.rawEnergy[i], 1e-10));
        result[1] += .5 * face.dot(region.anchor[i]) /
            std::sqrt(std::max(energy * region.anchorEnergy[i], 1e-10));
    }
    return result;
}

std::optional<cv::Point2f> estimate(const cv::Mat& picture, const std::vector<Region>& allRegions) {
    const auto candidate = scales(picture);
    std::vector<const Region*> strong;
    std::vector<Correlation> baseline;
    for (const auto& r : allRegions) {
        const auto base = alignment(r, candidate);
        if (base[0] >= .68 && base[1] >= .68) {
            strong.push_back(&r);
            baseline.push_back(base);
        }
    }
    if (strong.size() < 3) {
        return std::nullopt;
    }
    cv::Mat x(gridSize, gridSize, CV_32F), y(gridSize, gridSize, CV_32F);
    for (int j = 0; j < gridSize; ++j) {
        for (int i = 0; i < gridSize; ++i) {
            x.at<float>(j, i) = static_cast<float>(i);
            y.at<float>(j, i) = static_cast<float>(j);
        }
    }
    double maximum = -1.;
    std::vector<Correlation> best;
    int bestShift = centerIndex;
    for (int s = 0; s < shiftTotal; ++s) {
        const auto point = displacement(s);
        const cv::Mat u = x + .5 * point.x;
        const cv::Mat v = y + .5 * point.y;
        Bands shifted;
        for (std::size_t i = 0; i < candidate.size(); ++i) {
            cv::remap(candidate[i], shifted[i], u, v, cv::INTER_LINEAR, cv::BORDER_REFLECT_101);
        }
        double sum = 0.;
        std::vector<Correlation> next;
        for (const auto* region : strong) {
            const auto q = alignment(*region, shifted);
            sum += .65 * q[0] + .35 * q[1];
            next.push_back(q);
        }
        if (sum > maximum) {
            maximum = sum;
            best = std::move(next);
            bestShift = s;
        }
    }
    double rawGain = 0., stableGain = 0.;
    int rawVotes = 0, stableVotes = 0;
    for (std::size_t i = 0; i < baseline.size(); ++i) {
        const auto one = best[i][0] - baseline[i][0];
        const auto two = best[i][1] - baseline[i][1];
        if (std::min(one, two) < -.013) {
            return std::nullopt;
        }
        rawGain += one;
        stableGain += two;
        rawVotes += one >= .005;
        stableVotes += two >= .003;
    }
    const double count = static_cast<double>(baseline.size());
    if (bestShift == centerIndex || rawGain < .010 * count || stableGain < .007 * count ||
        rawVotes < 2 || stableVotes < 2) {
        return std::nullopt;
    }
    return displacement(bestShift);
}

cv::Mat applyShift(const cv::Mat& image, std::optional<cv::Point2f> shift) {
    if (!shift) {
        return image;
    }
    cv::Mat x(faceSize, faceSize, CV_32F), y(faceSize, faceSize, CV_32F);
    for (int j = 0; j < faceSize; ++j) {
        for (int i = 0; i < faceSize; ++i) {
            const auto xp = static_cast<float>(i);
            const auto yp = static_cast<float>(j);
            const float dx = (xp - 256.f) / 224.f;
            const float dy = (yp - 274.f) / 235.f;
            const float taper = std::clamp((1.f - (dx * dx + dy * dy)) / .29f, 0.f, 1.f);
            x.at<float>(j, i) = xp + taper * shift->x;
            y.at<float>(j, i) = yp + taper * shift->y;
        }
    }
    cv::Mat result;
    cv::remap(image, result, x, y, cv::INTER_LINEAR, cv::BORDER_REFLECT_101);
    return result;
}

cv::Mat median(const std::array<cv::Mat, 3>& img) {
    const cv::Mat low = cv::min(img[0], img[1]);
    const cv::Mat high = cv::max(img[0], img[1]);
    return cv::min(cv::max(low, img[2]), high);
}

}

AlignedModelViews alignModelViews(const cv::Mat& observed, const cv::Mat& anchor,
                                  const std::array<cv::Mat, 3>& views) {
    requireFace(observed);
    requireFace(anchor);
    for (const auto& view : views) {
        requireFace(view);
    }
    const auto support = regions(scales(observed), scales(anchor));
    AlignedModelViews result;
    result.original = median(views);
    result.wholeOffset = estimate(result.original, support);
    result.whole = applyShift(result.original, result.wholeOffset);
    std::array<cv::Mat, 3> adjusted;
    for (std::size_t i = 0; i < views.size(); ++i) {
        result.viewOffsets[i] = estimate(views[i], support);
        adjusted[i] = applyShift(views[i], result.viewOffsets[i]);
    }
    result.aligned = median(adjusted);
    return result;
}

}
