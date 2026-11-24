#pragma once

#include <vector>

#include <opencv2/core.hpp>

namespace visual::screen {

struct Peaks {
    std::vector<cv::Point2f> frequencies;
    std::vector<cv::Point2f> periods;
};

Peaks detect(const cv::Mat& nativeBgr);
cv::Mat averagePeriods(const cv::Mat& nativeBgr, const Peaks& peaks);
cv::Mat subtractCoherentCarrier(const cv::Mat& nativeBgr, const Peaks& peaks);

struct Selection {
    bool usePhase{};
    double carrierRatio{};
    double midbandRatio{};
    double baseCoarseError{};
    double phaseCoarseError{};
};

Selection selectBase(const cv::Mat& alignedRaw, const cv::Mat& previous, const cv::Mat& phase,
                     const Peaks& peaks, const cv::Matx23d& nativeToAligned);

}
