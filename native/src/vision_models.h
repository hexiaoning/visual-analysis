#pragma once

#include "storage.h"

#include <memory>
#include <opencv2/core.hpp>

namespace visual {

Feature normalized(Feature value);
double similarity(std::span<const float> first, std::span<const float> second);

struct FaceObservation {
    cv::Rect box;
    cv::Mat aligned;
    Feature feature;
    float quality{};
};

struct FaceSettings {
    std::string profile = "auraface";
    bool mirror = true;
    float threshold = .4f;
    int minimumSize = 32;
    float minimumQuality = .22f;
    float enrollmentQuality = .30f;
    double matchThreshold = .45, margin = .06;
    static FaceSettings from(const Json& config, const std::filesystem::path& models);
    std::string modelId() const;
};

class FaceEngine {
public:
    FaceEngine(const std::filesystem::path& models, FaceSettings settings);
    ~FaceEngine();
    std::vector<FaceObservation> faces(const cv::Mat& image);
    FaceObservation reference(const cv::Mat& image);
    const FaceSettings& settings() const;

private:
    struct State;
    std::unique_ptr<State> state_;
};

struct ObjectObservation {
    int category{};
    double score{};
    cv::Rect box;
};

std::string objectName(int category);
cv::Mat markObjects(const cv::Mat& frame, std::span<const ObjectObservation> objects);

class ObjectDetector {
public:
    explicit ObjectDetector(const std::filesystem::path& models);
    ~ObjectDetector();
    std::vector<ObjectObservation> detect(const cv::Mat& image, double threshold, bool people, bool vehicles,
                                          bool otherObjects=false);
private:
    struct State;
    std::unique_ptr<State> state_;
};

class SemanticEncoder {
public:
    explicit SemanticEncoder(const std::filesystem::path& models);
    ~SemanticEncoder();
    Feature image(const cv::Mat& image);
    Feature text(const std::string& text);
private:
    struct State;
    std::unique_ptr<State> state_;
};

}
