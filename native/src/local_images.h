#pragma once

#include "portrait.h"

namespace visual {

std::filesystem::path localPath(const std::filesystem::path& path);
cv::Mat readImage(const std::filesystem::path& path);
void writePng(const std::filesystem::path& path, const cv::Mat& pixels);
int runCases(const std::filesystem::path& inputs, const std::filesystem::path& outputs,
             const std::filesystem::path& report, const std::filesystem::path& models, bool preferRecall);
std::wstring wide(const std::string& utf8);

}
