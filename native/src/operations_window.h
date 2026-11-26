#pragma once

#include <filesystem>
#include <Windows.h>
#include <opencv2/core.hpp>

namespace visual {

inline constexpr UINT loadSearchRestoration=WM_APP+15;

struct SearchRestorationInput {
    cv::Mat image;
    cv::Rect region;
};

HWND createWorkflowPage(HWND parent,const std::filesystem::path& models,const std::filesystem::path& worker,size_t workflow);
bool requestWorkflowClose(HWND page);

}
