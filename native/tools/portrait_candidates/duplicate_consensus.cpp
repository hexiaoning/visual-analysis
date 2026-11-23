#include "duplicate_consensus.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <stdexcept>
#include <vector>

namespace portrait_consensus {
namespace {

struct Samples {
    std::vector<cv::Point> calibration;
    std::array<std::vector<cv::Point>, 5> validation;
};

Samples sample_regions() {
    const std::array<cv::Rect, 4> traits{cv::Rect(39, 53, 18, 16), cv::Rect(70, 53, 20, 16),
                                         cv::Rect(53, 69, 22, 16), cv::Rect(45, 85, 37, 18)};
    Samples out;
    for (int y = 0; y < 128; ++y)
        for (int x = 0; x < 128; ++x) {
            cv::Point pt(x, y);
            const auto it =
                std::find_if(traits.begin(), traits.end(), [&](cv::Rect r) { return r.contains(pt); });
            if (it != traits.end()) {
                out.validation[std::distance(traits.begin(), it)].push_back(pt);
                continue;
            }
            double dx = (x - 64) / 36., dy = (y - 76) / 42.;
            if (dx * dx + dy * dy >= 1.)
                continue;
            if ((x / 6 + y / 6) % 2 == 0)
                out.calibration.push_back(pt);
            else
                out.validation[4].push_back(pt);
        }
    return out;
}

cv::Mat reduce(const cv::Mat &image) {
    cv::Mat unit, small;
    image.convertTo(unit, CV_32F, 1. / 255);
    cv::resize(unit, small, cv::Size(128, 128), 0, 0, cv::INTER_AREA);
    return small;
}

struct Projection {
    cv::Mat first, second;
    double sigma = 0, loss = std::numeric_limits<double>::infinity();
};

Projection project(const cv::Mat &y, const cv::Mat &a, const cv::Mat &b,
                   const std::vector<cv::Point> &fit, double sigma) {
    Projection out;
    out.sigma = sigma;
    cv::GaussianBlur(a, out.first, {}, sigma);
    cv::GaussianBlur(b, out.second, {}, sigma);
    auto mid = [&](cv::Point p) {
        return (out.first.at<cv::Vec3f>(p) + out.second.at<cv::Vec3f>(p)) * .5f;
    };
    cv::Vec3d mp{}, my{}, xx{}, xy{};
    for (auto p : fit) {
        mp += cv::Vec3d(mid(p));
        my += cv::Vec3d(y.at<cv::Vec3f>(p));
    }
    double n = static_cast<double>(fit.size());
    mp /= n;
    my /= n;
    for (auto p : fit) {
        cv::Vec3d dx = cv::Vec3d(mid(p)) - mp, dy = cv::Vec3d(y.at<cv::Vec3f>(p)) - my;
        xx += dx.mul(dx);
        xy += dx.mul(dy);
    }
    cv::Vec3d gain{}, bias{};
    for (int c = 0; c < 3; ++c) {
        gain[c] = std::clamp((xy[c] / n + .0009) / (xx[c] / n + .0009), .28, 2.1);
        bias[c] = std::clamp(my[c] - gain[c] * mp[c], -.42, .34);
    }
    for (cv::Mat *option : {&out.first, &out.second}) {
        for (auto it = option->begin<cv::Vec3f>(); it != option->end<cv::Vec3f>(); ++it)
            for (int c = 0; c < 3; ++c)
                (*it)[c] = static_cast<float>(std::clamp((*it)[c] * gain[c] + bias[c], 0., 1.));
    }
    double loss = 0;
    for (auto p : fit) {
        cv::Vec3f error = mid(p) - y.at<cv::Vec3f>(p);
        for (int c = 0; c < 3; ++c)
            loss += std::min(std::abs(error[c]), .14f);
    }
    out.loss = loss / (n * 3);
    return out;
}

} // namespace

double amd_fraction(const cv::Mat &observed, const cv::Mat &first, const cv::Mat &second) {
    for (const cv::Mat *im : {&observed, &first, &second})
        if (im->type() != CV_8UC3 || im->size() != cv::Size(512, 512))
            throw std::invalid_argument("Expected aligned 512x512 BGR input");
    const auto samples = sample_regions();
    auto y = reduce(observed), a = reduce(first), b = reduce(second);
    Projection best;
    for (double sigma : {.8, 1.6, 2.4, 3.3, 4.5}) {
        Projection p = project(y, a, b, samples.calibration, sigma);
        if (p.loss < best.loss)
            best = std::move(p);
    }
    double numer = 0, denom = 0;
    for (auto const &region : samples.validation) {
        cv::Vec3d mean{}, var{};
        for (auto p : region)
            mean += cv::Vec3d(y.at<cv::Vec3f>(p));
        mean /= static_cast<double>(region.size());
        for (auto p : region) {
            auto diff = cv::Vec3d(y.at<cv::Vec3f>(p)) - mean;
            var += diff.mul(diff);
        }
        cv::Vec3d spread;
        for (int c = 0; c < 3; ++c)
            spread[c] = std::max(.035, std::sqrt(var[c] / region.size()));
        double region_num = 0, region_den = 0;
        for (auto p : region)
            for (int c = 0; c < 3; ++c) {
                double meas = y.at<cv::Vec3f>(p)[c], f = best.first.at<cv::Vec3f>(p)[c],
                       s = best.second.at<cv::Vec3f>(p)[c];
                if (meas <= .015 || meas >= .985)
                    continue;
                double z = std::abs(meas - (f + s) * .5) / spread[c];
                double weight = std::min(1., 2. / std::max(1e-5, z)) / (spread[c] * spread[c]);
                double d = s - f;
                region_num += weight * d * (meas - f);
                region_den += weight * d * d;
            }
        numer += region_num / (region.size() * 3);
        denom += region_den / (region.size() * 3);
    }
    return denom > 1.e-8 ? std::clamp(numer / denom, 0., 1.) : .5;
}
} // namespace portrait_consensus
