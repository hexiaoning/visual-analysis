#pragma once

#include <cstddef>
#include <filesystem>
#include <iosfwd>

#include <opencv2/core/mat.hpp>

namespace visual::screen {

std::size_t runGridFusion(std::istream& manifest, const std::filesystem::path& output);
cv::Mat mixGeneratedDetail(const cv::Mat& anchor, const cv::Mat& generated);

}
