#include "portrait.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <numbers>
#include <stdexcept>

#include <opencv2/imgproc.hpp>

namespace visual {
namespace {

cv::Mat motionKernel(cv::Size imageSize, int length, int angle) {
    const int center = length / 2;
    const double radians = angle * std::numbers::pi / 180;
    const cv::Point offset(cvRound(center * std::cos(radians)), cvRound(center * std::sin(radians)));
    const cv::Point origin(center, center);
    cv::Mat support = cv::Mat::zeros(length, length, CV_32F);
    cv::line(support, origin - offset, origin + offset, cv::Scalar(1));
    const double mass = cv::sum(support)[0];
    cv::Mat kernel = cv::Mat::zeros(imageSize, CV_32F);
    for (int row = 0; row < length; ++row) {
        for (int col = 0; col < length; ++col) {
            kernel.at<float>((row + imageSize.height - center) % imageSize.height,
                             (col + imageSize.width - center) % imageSize.width)
                = float(support.at<float>(row, col) / mass);
        }
    }
    return kernel;
}

cv::Mat deblur(const cv::Mat& input, int length, int angle) {
    const int padding = length + 4;
    cv::Mat padded;
    cv::copyMakeBorder(input, padded, padding, padding, padding, padding, cv::BORDER_REFLECT_101);
    cv::Mat filter;
    cv::dft(motionKernel(padded.size(), length, angle), filter, cv::DFT_COMPLEX_OUTPUT);
    for (int row = 0; row < filter.rows; ++row) {
        for (int col = 0; col < filter.cols; ++col) {
            auto& value = filter.at<cv::Vec2f>(row, col);
            const float weight = 1.04f / (value.dot(value) + .04f);
            value = {value[0] * weight, -value[1] * weight};
        }
    }
    std::array<cv::Mat, 3> channels;
    cv::split(padded, channels.data());
    auto result = input.clone();
    for (int channel = 0; channel < 3; ++channel) {
        cv::Mat transformed, restored;
        channels[channel].convertTo(channels[channel], CV_32F);
        cv::dft(channels[channel], transformed, cv::DFT_COMPLEX_OUTPUT);
        cv::mulSpectrums(transformed, filter, transformed, 0);
        cv::idft(transformed, restored, cv::DFT_REAL_OUTPUT | cv::DFT_SCALE);
        for (int row = 0; row < result.rows; ++row) {
            for (int col = 0; col < result.cols; ++col) {
                auto& sample = result.at<cv::Vec3b>(row, col)[channel];
                const auto candidate = restored.at<float>(row + padding, col + padding);
                sample = cv::saturate_cast<uchar>(sample + std::clamp((candidate - sample) * .65f, -30.f, 30.f));
            }
        }
    }
    return result;
}

}

Candidates restorePlate(const cv::Mat& image, cv::Rect region) {
    if (region.width < 16 || region.width > 1024 || region.height < 8 || region.height > 512 ||
        (region & cv::Rect(0, 0, image.cols, image.rows)) != region) {
        throw std::runtime_error("车牌选区宽度须为 16～1024 像素，高度须为 8～512 像素。");
    }
    const auto source = image(region).clone();
    Candidates candidates;
    for (const auto angle : {0, 45, 90, 135}) {
        for (const auto length : {3, 5, 9}) {
            auto prediction = deblur(source, length, angle);
            const auto name = "去拖影 " + std::to_string(angle) + "° / " + std::to_string(length) + "px";
            auto result = scoreRestoration(name, prediction, source);
            result.pixels = std::move(prediction);
            candidates.push_back(std::move(result));
        }
    }
    rankRestorations(candidates);
    Candidates output;
    for (auto& candidate : candidates) {
        if (std::any_of(output.begin(), output.end(), [&](const auto& existing) {
                return cv::norm(candidate.pixels, existing.pixels, cv::NORM_L1) < candidate.pixels.total() * 4.5;
            })) {
            continue;
        }
        output.push_back(std::move(candidate));
        if (output.size() == 3) {
            break;
        }
    }
    return output;
}

}
