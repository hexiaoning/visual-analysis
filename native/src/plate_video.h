#pragma once

#include "operations.h"

namespace visual {
cv::Rect trackPlateRegion(const cv::Mat& image,const cv::Mat& reference,cv::Rect previous);
cv::Mat fusePlateFrames(const cv::Mat& reference,const std::vector<cv::Mat>& donors);

Json plateVideo(const Json& request,const std::atomic<bool>& cancel,DetectionPreview* processingPreview);
}
