#include "guided_detail.h"

#include <algorithm>
#include <cmath>
#include <numbers>
#include <stdexcept>

#include <opencv2/imgproc.hpp>
#include <opencv2/video/tracking.hpp>

namespace portrait_selection {
namespace {

constexpr int working_side = 160;
constexpr int portrait_side = 512;

cv::Mat blur(const cv::Mat& input, double sigma) {
    cv::Mat result;
    cv::GaussianBlur(input, result, {}, sigma, 0., cv::BORDER_REFLECT_101);
    return result;
}

cv::Mat geometry_gray(const cv::Mat& image) {
    cv::Mat reduced, gray, normalized, bytes;
    cv::resize(image, reduced, {working_side, working_side}, 0., 0., cv::INTER_AREA);
    cv::cvtColor(reduced, gray, cv::COLOR_BGR2GRAY);
    gray.convertTo(normalized, CV_32F, 1. / 255.);
    const auto soft = blur(normalized, .8);
    const cv::Mat band = soft - blur(normalized, 3.5);
    const cv::Mat signal = .44 * soft + .56 * (.5 + 3. * band);
    signal.convertTo(bytes, CV_8U, 255.);
    return bytes;
}

cv::Mat grid(int side) {
    cv::Mat coordinates(side, side, CV_32FC2);
    for (int y = 0; y < side; ++y)
        for (int x = 0; x < side; ++x)
            coordinates.at<cv::Vec2f>(y, x) = {static_cast<float>(x), static_cast<float>(y)};
    return coordinates;
}

cv::Mat remapped(const cv::Mat& image, const cv::Mat& coordinates) {
    cv::Mat result;
    cv::remap(image, result, coordinates, {}, cv::INTER_LINEAR, cv::BORDER_REFLECT_101);
    return result;
}

cv::Mat supported_flow(const cv::Mat& donor, const cv::Mat& guide, double& support, double& peak) {
    const auto start = geometry_gray(donor);
    const auto target = geometry_gray(guide);
    auto flow = cv::DISOpticalFlow::create(cv::DISOpticalFlow::PRESET_MEDIUM);
    flow->setFinestScale(0);
    cv::Mat backward, forward;
    flow->calc(target, start, backward);
    flow->calc(start, target, forward);
    const auto reverse = remapped(forward, grid(working_side) + backward);
    double trust = 0.;
    int samples = 0;
    for (int y = 0; y < working_side; ++y) {
        for (int x = 0; x < working_side; ++x) {
            auto& delta = backward.at<cv::Vec2f>(y, x);
            const auto magnitude = std::sqrt(delta.dot(delta));
            const auto restored = delta + reverse.at<cv::Vec2f>(y, x);
            const auto cycle = std::sqrt(restored.dot(restored));
            const auto agreement = std::max(0.f, 1.f - cycle / (1.25f + .18f * magnitude));
            const double radius = std::hypot((x - 80.) / 57., (y - 88.) / 57.);
            const auto weight = .5 + .5 * std::cos(std::numbers::pi * std::clamp((radius - .64) / .36, 0., 1.));
            const auto scale = std::min(1.f, 3.5f / std::max(magnitude, 1.e-6f));
            delta *= static_cast<float>(weight) * agreement * scale;
            if (weight > .8) {
                trust += agreement;
                ++samples;
            }
        }
    }
    support = trust / samples;
    cv::Mat compact, full;
    cv::resize(backward, compact, {32, 32}, 0., 0., cv::INTER_AREA);
    cv::resize(compact, full, {portrait_side, portrait_side}, 0., 0., cv::INTER_LINEAR);
    full *= static_cast<float>(portrait_side) / working_side;
    peak = 0.;
    for (int y = 0; y < full.rows; ++y) {
        for (int x = 0; x < full.cols; ++x) {
            auto& motion = full.at<cv::Vec2f>(y, x);
            const auto length = std::sqrt(motion.dot(motion));
            motion *= std::min(1.f, 11.f / std::max(length, 1.e-6f));
            peak = std::max(peak, static_cast<double>(std::min(11.f, length)));
        }
    }
    return full;
}

cv::Mat middle_band(const cv::Mat& image) {
    cv::Mat floating, gray;
    image.convertTo(floating, CV_32F);
    cv::cvtColor(floating, gray, cv::COLOR_BGR2GRAY);
    return blur(gray, 1.6) - blur(gray, 8.);
}

cv::Mat guide_contrast(const cv::Mat& warped, const cv::Mat& guide, double& support) {
    const auto donor = middle_band(warped);
    const auto target = middle_band(guide);
    const auto overlap = blur(donor.mul(target), 6.);
    const auto donor_power = blur(donor.mul(donor), 6.);
    const auto target_power = blur(target.mul(target), 6.);
    cv::Mat result = warped.clone();
    support = 0.;
    int samples = 0;
    for (int y = 0; y < portrait_side; ++y) {
        for (int x = 0; x < portrait_side; ++x) {
            const auto dx = (x - 256.) / 184., dy = (y - 282.) / 197.;
            const auto window = static_cast<float>(std::clamp((1. - dx * dx - dy * dy) / .72, 0., 1.));
            if (window == 0.f)
                continue;
            const auto power = std::sqrt(donor_power.at<float>(y, x) * target_power.at<float>(y, x));
            const auto affinity = std::clamp(overlap.at<float>(y, x) / (power + 2.f), 0.f, 1.f);
            const auto confidence = std::min(1.f, power / 22.f) * std::max(0.f, (affinity - .20f) / .80f);
            const auto contrast = std::clamp(target.at<float>(y, x) - donor.at<float>(y, x), -22.f, 22.f);
            const auto change = contrast * window * confidence * .78f;
            auto& pixel = result.at<cv::Vec3b>(y, x);
            for (int c = 0; c < 3; ++c)
                pixel[c] = cv::saturate_cast<unsigned char>(pixel[c] + change);
            if (window > .99f) {
                support += confidence;
                ++samples;
            }
        }
    }
    support /= samples;
    return result;
}

} // namespace

GuidedDetail transfer_guided_detail(const cv::Mat& donor, const cv::Mat& guide) {
    const cv::Size size(portrait_side, portrait_side);
    if (donor.size() != size || guide.size() != size || donor.type() != CV_8UC3 || guide.type() != CV_8UC3)
        throw std::invalid_argument("Expected two aligned 512x512 BGR portraits");
    GuidedDetail output{};
    const auto displacement = supported_flow(donor, guide, output.cycle_support, output.peak_displacement);
    output.warped = remapped(donor, grid(portrait_side) + displacement);
    output.adjusted = guide_contrast(output.warped, guide, output.contrast_support);
    return output;
}

} // namespace portrait_selection
