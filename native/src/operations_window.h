#pragma once

#include <filesystem>
#include <span>
#include <Windows.h>
#include <opencv2/core.hpp>

namespace visual {

inline constexpr UINT loadSearchRestoration=WM_APP+15;

struct SearchRestorationInput {
    cv::Mat image;
    cv::Rect region;
};

std::span<const wchar_t* const> workflowNames();
void openWorkflow(HWND parent,const std::filesystem::path& models,const std::filesystem::path& worker,size_t workflow);

}
