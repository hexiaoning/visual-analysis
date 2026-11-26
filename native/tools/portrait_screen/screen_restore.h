#pragma once

#include <cstddef>
#include <optional>
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

struct CaptureReplayScore {
    double error{};
    double relativeDetail{};
};

struct CaptureReplaySelection {
    double blur{};
    std::size_t selected{};
    std::vector<CaptureReplayScore> scores;
    std::optional<double> chromaBandRms;
};

// The first aligned crop is the current primary. A current pair is preserved.
CaptureReplaySelection rankAtCaptureResolution(const cv::Mat& nativeBgr, const cv::Matx23d& nativeToAligned,
                                               const std::vector<cv::Mat>& alignedCandidates, bool hasCurrentPair,
                                               bool guardCaptureDetail = false);

}
