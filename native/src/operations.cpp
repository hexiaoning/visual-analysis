#include "operations.h"
#include "people.h"
#include "vehicle.h"
#include "media_operations.h"
#include "plate_video.h"

#include <algorithm>
#include <thread>
#include <opencv2/imgproc.hpp>

namespace visual {

bool DetectionPreview::wait(const std::atomic<bool>& cancel) const {
    using namespace std::chrono_literals;
    while(paused && !cancel) std::this_thread::sleep_for(25ms);
    return !cancel;
}

bool DetectionPreview::refreshDue() const {
    return std::chrono::steady_clock::now()>=nextRefresh;
}

void DetectionPreview::update(const cv::Mat& image, int index, int detections) {
    nextRefresh=std::chrono::steady_clock::now()+std::chrono::milliseconds(67);
    cv::Mat resized, bgra;
    const auto ratio=std::min(1.,800./std::max(image.cols,image.rows));
    if(ratio<1.) cv::resize(image,resized,{std::max(1,cvRound(image.cols*ratio)),std::max(1,cvRound(image.rows*ratio))},
                           0,0,cv::INTER_AREA);
    else resized=image;
    cv::cvtColor(resized,bgra,cv::COLOR_BGR2BGRA);
    std::lock_guard lock(mutex);
    latest={std::move(bgra),index,detections};
}

DetectionPreview::Frame DetectionPreview::current() {
    std::lock_guard lock(mutex);
    return latest;
}

Json runOperation(const std::string& action,const Json& request,const std::filesystem::path& models,
                  const std::filesystem::path& worker,const std::atomic<bool>& cancel,DetectionPreview* preview) {
    if(action=="vehicle.analyze") return analyzeVehicle(request,models,worker,cancel,preview);
    if(action=="plate.video") return plateVideo(request,cancel,preview);
    if(action=="semantic.search" || action=="visual.search" || action=="objects.scan" || action=="face.scan" ||
       action=="hits.export") {
        return mediaOperation(action,request,models,cancel,preview);
    }
    return peopleOperation(action,request,models,cancel);
}

}
