#pragma once

#include <optional>
#include <filesystem>
#include <Windows.h>
#include <opencv2/core.hpp>

namespace visual {
std::optional<cv::Rect> previewNative(HWND owner,const cv::Mat& image,const wchar_t* caption,
                                     bool select,std::optional<cv::Rect> overlay={});
void playNativeVideo(HWND owner,const std::filesystem::path& source,int first,
                     const cv::Mat& initial,cv::Rect initialTarget);
}
