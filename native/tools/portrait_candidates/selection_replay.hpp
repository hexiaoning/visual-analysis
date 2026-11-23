#pragma once
// Fit capture blur and exposure outside facial features. Score on separate feature regions.

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <vector>

namespace selection {
const cv::Rect face{107, 160, 298, 270};
const std::array<cv::Rect, 4> regions{
    {{39, 53, 18, 16}, {70, 53, 20, 16}, {53, 69, 22, 16}, {45, 85, 37, 18}}};

inline cv::Mat filtered(const cv::Mat &image, double sigma) {
    cv::Mat out;
    cv::GaussianBlur(image, out, {0, 0}, sigma);
    return out;
}

inline cv::Mat grayscale(const cv::Mat &bgr) {
    cv::Mat unit(bgr.size(), CV_32FC3), gray;
    for (int y = 0; y < bgr.rows; ++y) {
        auto in = bgr.ptr<uint8_t>(y);
        auto out = unit.ptr<float>(y);
        for (int j = 0; j < bgr.cols * 3; ++j)
            out[j] = float(in[j]) / 255.f;
    }
    cv::cvtColor(unit, gray, cv::COLOR_BGR2GRAY);
    return gray;
}

inline double clarity(const cv::Mat &gray) {
    cv::Mat band = filtered(gray, .65) - filtered(gray, 2.2);
    return cv::norm(band(face), cv::NORM_L2) / std::sqrt(double(face.area()));
}

inline cv::Mat compact(const cv::Mat &gray) {
    cv::Mat small;
    cv::resize(gray, small, {128, 128}, 0, 0, cv::INTER_AREA);
    return filtered(small, .8);
}

class Observation {
  public:
    explicit Observation(const cv::Mat &input) : observed_(compact(input)) {
        for (int y = 35; y < 109; ++y) {
            for (int x = 30; x < 98; ++x) {
                auto value = observed_.at<float>(y, x);
                if (value <= .045f || value >= .97f)
                    continue;
                bool feature = false;
                for (size_t i = 0; i < regions.size(); ++i) {
                    if (regions[i].contains(cv::Point(x, y))) {
                        feature = true;
                        features_[i].emplace_back(x, y);
                    }
                }
                if (!feature)
                    outside_.emplace_back(x, y);
            }
        }
    }

    const cv::Mat &observed() const { return observed_; }

    double replay(const cv::Mat &gray, cv::Mat &projection) const {
        if (outside_.size() < 230)
            return std::numeric_limits<double>::quiet_NaN();
        const auto candidate = compact(gray);
        const double count = double(outside_.size());
        double measured = 0;
        for (auto p : outside_)
            measured += observed_.at<float>(p);
        measured /= count;
        double bestLoss = std::numeric_limits<double>::infinity(),
               bestError = std::numeric_limits<double>::quiet_NaN();
        cv::Mat bestBlur;
        double bestGain = 1., bestBias = 0.;
        for (double sigma : {1., 1.6, 2.4, 3.3, 4.5}) {
            const auto blur = filtered(candidate, sigma);
            double average = 0;
            for (auto p : outside_)
                average += blur.at<float>(p);
            average /= count;
            double covariance = 0, variance = 0;
            for (auto p : outside_) {
                double x = blur.at<float>(p) - average;
                covariance += x * (observed_.at<float>(p) - measured);
                variance += x * x;
            }
            double gain =
                std::clamp((covariance / count + .0009) / (variance / count + .0009), .28, 2.1);
            double bias = std::clamp(measured - gain * average, -.42, .34);
            auto error = [&](cv::Point p) {
                double predicted = std::clamp(gain * blur.at<float>(p) + bias, 0., 1.);
                return std::min(.14, std::abs(predicted - observed_.at<float>(p)));
            };
            double fitLoss = 0;
            for (auto p : outside_)
                fitLoss += error(p);
            if (fitLoss >= bestLoss)
                continue;
            bestLoss = fitLoss;
            bestBlur = blur;
            bestGain = gain;
            bestBias = bias;
            double total = 0;
            size_t counted = 0;
            for (const auto &region : features_) {
                if (region.size() < 45)
                    continue;
                double sum = 0;
                for (auto p : region)
                    sum += error(p);
                total += sum / region.size();
                ++counted;
            }
            bestError = counted ? total / counted : std::numeric_limits<double>::quiet_NaN();
        }
        if (std::isfinite(bestError)) {
            cv::Mat unit = bestBlur * bestGain + bestBias;
            unit.convertTo(projection, CV_8U, 255.);
        }
        return bestError;
    }

  private:
    cv::Mat observed_;
    std::vector<cv::Point> outside_;
    std::array<std::vector<cv::Point>, 4> features_;
};

} // namespace selection
