#pragma once

#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

namespace visual {

struct Candidate {
    std::string model;
    cv::Mat pixels;
    double score{};
    double agreement{};
    double detail{};
    bool moderateConfidence{};
};

using Candidates = std::vector<Candidate>;

Candidate scoreRestoration(const std::string& model, const cv::Mat& restored, const cv::Mat& source);
void rankRestorations(Candidates& candidates);

std::vector<Candidates> restorePortraits(
    const std::vector<cv::Mat>& images,
    const std::filesystem::path& modelDirectory,
    std::optional<cv::Rect> region = std::nullopt,
    bool preferRecall = false);

Candidates restorePlate(const cv::Mat& image, cv::Rect region);

class VideoPortraitRestorer {
public:
    explicit VideoPortraitRestorer(const std::filesystem::path& modelDirectory);
    ~VideoPortraitRestorer();
    std::pair<cv::Mat, int> restore(const cv::Mat& pixels);

private:
    struct State;
    std::unique_ptr<State> state_;
};

}
