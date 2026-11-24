#pragma once

#include "storage.h"

#include <atomic>
#include <chrono>
#include <deque>
#include <memory>
#include <mutex>
#include <optional>
#include <opencv2/core.hpp>

namespace visual {

class FaceEngine;
struct FaceSettings;

class FaceScanSession {
public:
    void beginBatch() {
        std::lock_guard lock(mutex_);
        paths_.clear();
        closed_=false;
    }

    std::shared_ptr<FaceEngine> engine(const std::filesystem::path& models,const Json& settings,
                                      const FaceSettings& config);

    bool append(const std::vector<std::filesystem::path>& paths) {
        std::lock_guard lock(mutex_);
        if(closed_) return false;
        paths_.insert(paths_.end(),paths.begin(),paths.end());
        return true;
    }

    std::optional<std::filesystem::path> next() {
        std::lock_guard lock(mutex_);
        if(paths_.empty()) {
            closed_=true;
            return {};
        }
        auto path=std::move(paths_.front());
        paths_.pop_front();
        return path;
    }

private:
    std::mutex mutex_;
    std::deque<std::filesystem::path> paths_;
    bool closed_{};
    std::filesystem::path modelDirectory_;
    Json settings_;
    std::shared_ptr<FaceEngine> engine_;
};

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
                  DetectionPreview* preview = nullptr,FaceScanSession* faceQueue = nullptr);

}
