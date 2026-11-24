#pragma once

#include <atomic>
#include <filesystem>
#include <functional>
#include <opencv2/core.hpp>

namespace visual {

struct VideoProgress {
    std::atomic<bool> cancel{};
    std::atomic<int> frames{};
    std::atomic<int> faces{};
    const std::atomic<bool>* externalCancel{};
    bool cancelled() const { return cancel || (externalCancel && *externalCancel); }
};

void restorePortraitVideo(const std::filesystem::path& source, const std::filesystem::path& target,
                          const std::filesystem::path& models, VideoProgress& progress);

using FrameVisitor = std::function<bool(const cv::Mat&, int, double)>;
using FrameSelector = std::function<bool(int)>;
int visitVideo(const std::filesystem::path& source, const FrameVisitor& visitor, int interval,
               const std::atomic<bool>& cancel, int firstFrame=0, const FrameSelector& select={});
using FrameProcessor = std::function<std::pair<cv::Mat,int>(const cv::Mat&)>;
void renderObjectVideo(const std::filesystem::path& source,const std::filesystem::path& target,
                       const FrameProcessor& transform,VideoProgress& progress);
void renderReviewClip(const std::filesystem::path& source,const std::filesystem::path& target,
                      double from,double until,VideoProgress& progress);

}
