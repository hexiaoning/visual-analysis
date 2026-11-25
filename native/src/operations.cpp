#include "operations.h"
#include "people.h"
#include "vehicle.h"
#include "media_operations.h"
#include "plate_video.h"
#include "local_images.h"

#include <algorithm>
#include <array>
#include <stdexcept>
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

namespace {

Json portraitMotion(const Json& request, const std::filesystem::path& models, const std::atomic<bool>& cancel) {
    const auto source = readImage(textPath(request.at("input").get<std::string>()));
    auto output = localPath(textPath(request.at("output").get<std::string>()));
    if (output.extension() != L".png") {
        throw std::invalid_argument("Portrait motion output must be a .png file.");
    }
    if (cancel) throw std::runtime_error("Portrait motion restoration canceled.");
    localPath(models / L"restormer-rsblur-256.onnx");
    std::optional<cv::Matx23d> affine;
    if (const auto supplied = request.find("source_to_aligned"); supplied != request.end()) {
        const auto coefficients = supplied->get<std::array<std::array<double, 3>, 2>>();
        affine = cv::Matx23d(coefficients[0][0], coefficients[0][1], coefficients[0][2],
                             coefficients[1][0], coefficients[1][1], coefficients[1][2]);
    }
    const auto begin = std::chrono::steady_clock::now();
    const auto result = restorePortraitMotion(source, models, affine, request.value("prefer_gpu", false));
    if (cancel) throw std::runtime_error("Portrait motion restoration canceled.");
    writePng(output, result.pixels);
    const auto& matrix = result.sourceToAligned;
    return {{"output", pathText(output)}, {"width", result.pixels.cols}, {"height", result.pixels.rows},
            {"context_pixels", result.contextPixels},
            {"source_to_aligned", {{matrix(0,0), matrix(0,1), matrix(0,2)},
                                    {matrix(1,0), matrix(1,1), matrix(1,2)}}},
            {"elapsed_ms", std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - begin).count()}};
}

}

Json runOperation(const std::string& action,const Json& request,const std::filesystem::path& models,
                  const std::filesystem::path& worker,const std::atomic<bool>& cancel,DetectionPreview* preview,
                  FaceScanSession* faceQueue) {
    if(action=="vehicle.analyze") return analyzeVehicle(request,models,worker,cancel,preview);
    if(action=="plate.video") return plateVideo(request,cancel,preview);
    if(action=="portrait.motion") return portraitMotion(request,models,cancel);
    if(action=="semantic.search" || action=="visual.search" || action=="objects.scan" || action=="face.scan" ||
       action=="hits.export") {
        return mediaOperation(action,request,models,cancel,preview,faceQueue);
    }
    return peopleOperation(action,request,models,cancel);
}

}
