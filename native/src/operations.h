#pragma once

#include "storage.h"

#include <atomic>
#include <chrono>
#include <mutex>
#include <opencv2/core.hpp>

namespace visual {

struct DetectionPreview {
    struct Frame {
        cv::Mat image;
        int index{}, detections{};
    };
    std::atomic<bool> paused{};
    std::mutex mutex;
    Frame latest;

    bool wait(const std::atomic<bool>& cancel) const;
    bool refreshDue() const;
    void update(const cv::Mat& image, int index, int detections);
    Frame current();

private:
    std::chrono::steady_clock::time_point nextRefresh{};
};

Json runOperation(const std::string& action,const Json& request,const std::filesystem::path& models,
                  const std::filesystem::path& worker,const std::atomic<bool>& cancel,
                  DetectionPreview* preview = nullptr);

}
