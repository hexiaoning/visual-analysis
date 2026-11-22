#include "portrait.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>
#include <utility>

#include <onnxruntime_cxx_api.h>
#include <opencv2/calib3d.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/video/tracking.hpp>

namespace visual {
namespace {

constexpr int detectorSize = 640;
constexpr int portraitSize = 512;
constexpr float detectionThreshold = .4f;
constexpr float overlapThreshold = .4f;
const std::array<cv::Point2f, 5> ffhq{{
    {192.98138f, 239.94708f}, {318.90277f, 240.1936f}, {256.63416f, 314.01935f},
    {201.26117f, 371.41043f}, {313.08905f, 371.15118f},
}};

class Inference {
public:
    explicit Inference(const std::filesystem::path& path)
        : session_(environment(), path.c_str(), options()) {
        if (session_.GetInputCount() != 1) {
            throw std::runtime_error("An unexpected number of ONNX input tensors.");
        }
        inputName_ = session_.GetInputNameAllocated(0, allocator_).get();
        for (size_t index = 0; index < session_.GetOutputCount(); ++index) {
            outputNames_.emplace_back(session_.GetOutputNameAllocated(index, allocator_).get());
        }
        for (const auto& name : outputNames_) {
            outputs_.push_back(name.c_str());
        }
    }

    std::vector<Ort::Value> run(cv::Mat blob) {
        const auto memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        const std::array<int64_t, 4> shape{1, 3, blob.size[2], blob.size[3]};
        auto tensor = Ort::Value::CreateTensor<float>(
            memory, blob.ptr<float>(), blob.total(), shape.data(), shape.size());
        const char* inputName = inputName_.c_str();
        return session_.Run(Ort::RunOptions{nullptr}, &inputName, &tensor, 1, outputs_.data(), outputs_.size());
    }

private:
    static Ort::Env& environment() {
        static Ort::Env environment{ORT_LOGGING_LEVEL_ERROR, "VisualAnalysis"};
        static const bool telemetryDisabled = (environment.DisableTelemetryEvents(), true);
        (void)telemetryDisabled;
        return environment;
    }

    static Ort::SessionOptions options() {
        Ort::SessionOptions options;
        options.SetIntraOpNumThreads(2);
        options.SetInterOpNumThreads(1);
        return options;
    }

    Ort::Session session_;
    Ort::AllocatorWithDefaultOptions allocator_;
    std::string inputName_;
    std::vector<std::string> outputNames_;
    std::vector<const char*> outputs_;
};

struct Detection {
    cv::Rect2f box;
    std::array<cv::Point2f, 5> landmarks;
    float score;
};

float intersectionOverUnion(const cv::Rect2f& first, const cv::Rect2f& second) {
    const float width = std::min(first.br().x, second.br().x) - std::max(first.x, second.x) + 1;
    const float height = std::min(first.br().y, second.br().y) - std::max(first.y, second.y) + 1;
    const float overlap = std::max(width, 0.f) * std::max(height, 0.f);
    const float total = (first.width + 1) * (first.height + 1)
                      + (second.width + 1) * (second.height + 1) - overlap;
    return overlap / std::max(total, 1e-9f);
}

std::vector<Detection> detect(Inference& detector, const cv::Mat& image, float threshold) {
    const double ratio = std::min(double(detectorSize) / image.cols, double(detectorSize) / image.rows);
    const int width = std::max(1, cvRound(image.cols * ratio));
    const int height = std::max(1, cvRound(image.rows * ratio));
    cv::Mat canvas = cv::Mat::zeros(detectorSize, detectorSize, CV_8UC3);
    cv::resize(image, canvas(cv::Rect(0, 0, width, height)), {width, height});
    const auto outputs = detector.run(cv::dnn::blobFromImage(canvas, 1. / 128, {}, {127.5, 127.5, 127.5}, true));
    if (outputs.size() != 9) {
        throw std::runtime_error("SCRFD did not return nine output tensors.");
    }
    const auto clampX = [&](float x) { return float(std::clamp(x / ratio, 0., image.cols - 1.)); };
    const auto clampY = [&](float y) { return float(std::clamp(y / ratio, 0., image.rows - 1.)); };
    std::vector<Detection> detections;
    for (int level = 0; level < 3; ++level) {
        const int stride = 8 << level;
        const size_t grid = detectorSize / stride;
        const auto& scores = outputs[level];
        const auto& boxes = outputs[level + 3];
        const auto& points = outputs[level + 6];
        const size_t count = grid * grid * 2;
        if (scores.GetTensorTypeAndShapeInfo().GetElementCount() != count ||
            boxes.GetTensorTypeAndShapeInfo().GetElementCount() != count * 4 ||
            points.GetTensorTypeAndShapeInfo().GetElementCount() != count * 10) {
            throw std::runtime_error("SCRFD output does not match the 640-pixel anchor grid.");
        }
        for (size_t index = 0; index < count; ++index) {
            const float score = scores.GetTensorData<float>()[index];
            if (score < threshold) {
                continue;
            }
            const float x = float(index / 2 % grid * stride);
            const float y = float(index / 2 / grid * stride);
            const auto* box = boxes.GetTensorData<float>() + index * 4;
            const auto* point = points.GetTensorData<float>() + index * 10;
            Detection found{
                cv::Rect2f(cv::Point2f(clampX(x - box[0] * stride), clampY(y - box[1] * stride)),
                           cv::Point2f(clampX(x + box[2] * stride), clampY(y + box[3] * stride))),
                {}, score
            };
            for (size_t landmark = 0; landmark < found.landmarks.size(); ++landmark) {
                found.landmarks[landmark] = {
                    float((x + point[2 * landmark] * stride) / ratio),
                    float((y + point[2 * landmark + 1] * stride) / ratio),
                };
            }
            detections.push_back(found);
        }
    }
    std::stable_sort(detections.begin(), detections.end(),
                     [](const auto& a, const auto& b) { return a.score > b.score; });
    std::vector<Detection> kept;
    for (const auto& candidate : detections) {
        if (std::none_of(kept.begin(), kept.end(), [&](const auto& existing) {
                return intersectionOverUnion(candidate.box, existing.box) > overlapThreshold;
            })) {
            kept.push_back(candidate);
            if (kept.size() == 100) {
                break;
            }
        }
    }
    return kept;
}

struct Face {
    cv::Mat crop;
    cv::Mat transform;
    float detectionScore;
    cv::Rect2f bounds;
};

std::vector<Face> align(Inference& detector, const cv::Mat& pixels, std::optional<cv::Rect> region,
                       float threshold) {
    std::vector<Face> faces;
    const std::vector<cv::Point2f> destination(ffhq.begin(), ffhq.end());
    for (const auto& detection : detect(detector, pixels, threshold)) {
        const auto center = (detection.box.tl() + detection.box.br()) * .5f;
        if (region && !region->contains(cv::Point(cvRound(center.x), cvRound(center.y)))) {
            continue;
        }
        if (std::min(detection.box.width, detection.box.height) < 24 ||
            cv::norm(detection.landmarks[0] - detection.landmarks[1]) < 6) {
            continue;
        }
        const std::vector<cv::Point2f> source(detection.landmarks.begin(), detection.landmarks.end());
        auto matrix = cv::estimateAffinePartial2D(source, destination, cv::noArray(), cv::LMEDS);
        if (matrix.empty() || !cv::checkRange(matrix)) {
            continue;
        }
        matrix.convertTo(matrix, CV_32F);
        std::vector<cv::Point2f> aligned;
        cv::transform(source, aligned, matrix);
        bool reliable = true;
        for (size_t index = 0; index < aligned.size(); ++index) {
            reliable &= cv::norm(aligned[index] - ffhq[index]) <= portraitSize * .14;
        }
        if (!reliable) {
            continue;
        }
        cv::Mat crop;
        cv::warpAffine(pixels, crop, matrix, {portraitSize, portraitSize},
                       cv::INTER_LINEAR, cv::BORDER_REFLECT_101);
        faces.push_back({std::move(crop), std::move(matrix), detection.score, detection.box});
    }
    return faces;
}

std::vector<Face> findFaces(Inference& detector, const cv::Mat& image, std::optional<cv::Rect> region,
                           bool preferRecall = false) {
    auto faces = align(detector, image, region, detectionThreshold);
    if (!faces.empty()) {
        return faces;
    }
    cv::Mat reducedPattern;
    cv::GaussianBlur(image, reducedPattern, {}, .9);
    faces = align(detector, reducedPattern, region, detectionThreshold);
    if (faces.empty() && preferRecall) {
        faces = align(detector, reducedPattern, region, .32f);
    }
    return faces;
}

cv::Mat inferPortrait(Inference& model, const cv::Mat& crop) {
    const auto output = model.run(cv::dnn::blobFromImage(crop, 1. / 127.5, {}, {127.5, 127.5, 127.5}, true));
    const auto shape = output.at(0).GetTensorTypeAndShapeInfo().GetShape();
    if (shape != std::vector<int64_t>{1, 3, portraitSize, portraitSize}) {
        throw std::runtime_error("Invalid portrait output tensor dimensions.");
    }
    const float* values = output[0].GetTensorData<float>();
    cv::Mat result(portraitSize, portraitSize, CV_8UC3);
    constexpr size_t plane = portraitSize * portraitSize;
    for (size_t index = 0; index < plane; ++index) {
        auto& pixel = result.ptr<cv::Vec3b>()[index];
        for (size_t channel = 0; channel < 3; ++channel) {
            const auto sample = values[index + (2 - channel) * plane];
            if (!std::isfinite(sample)) {
                throw std::runtime_error("Nonfinite portrait model output.");
            }
            pixel[static_cast<int>(channel)] = cv::saturate_cast<uchar>((sample + 1) * 127.5f);
        }
    }
    return result;
}

cv::Mat gray(const cv::Mat& bgr) {
    cv::Mat result;
    cv::cvtColor(bgr, result, cv::COLOR_BGR2GRAY);
    result.convertTo(result, CV_32F, 1. / 255);
    return result;
}

cv::Mat edgeMagnitude(const cv::Mat& image) {
    cv::Mat x, y, result;
    cv::Sobel(image, x, CV_32F, 1, 0);
    cv::Sobel(image, y, CV_32F, 0, 1);
    cv::magnitude(x, y, result);
    return result;
}

double roundScore(double value) {
    return std::round(value * 100) / 100;
}

}

Candidate scoreRestoration(const std::string& model, const cv::Mat& candidate, const cv::Mat& source) {
    auto before = gray(source);
    auto after = gray(candidate);
    cv::Mat beforeBlurred, afterBlurred, difference, smoothed;
    cv::GaussianBlur(before, beforeBlurred, {}, 1.15);
    cv::GaussianBlur(after, afterBlurred, {}, 1.15);
    cv::absdiff(beforeBlurred, afterBlurred, difference);
    const double agreement = std::clamp(1 - 3 * cv::mean(difference)[0], 0., 1.);
    cv::GaussianBlur(before, smoothed, {3, 3}, .6);
    const auto beforeEdges = edgeMagnitude(smoothed);
    const auto afterEdges = edgeMagnitude(after);
    const auto* first = beforeEdges.ptr<float>();
    std::vector<float> values(first, first + beforeEdges.total());
    const auto threshold = values.begin() + values.size() * 7 / 10;
    std::nth_element(values.begin(), threshold, values.end());
    const cv::Mat support = beforeEdges > std::max(.08f, *threshold);
    const double detail = std::clamp((cv::mean(afterEdges, support)[0] /
        std::max(cv::mean(beforeEdges, support)[0], .01) - 1) / 1.2, 0., 1.);
    return {model, {}, roundScore(80 * agreement + 20 * detail),
        roundScore(100 * agreement), roundScore(100 * detail), false};
}

void rankRestorations(Candidates& candidates) {
    std::sort(candidates.begin(), candidates.end(), [](const auto& a, const auto& b) {
        if (a.score == b.score) {
            return a.agreement == b.agreement ? a.model < b.model : a.agreement > b.agreement;
        }
        return a.score > b.score;
    });
}

namespace {

void composite(cv::Mat& original, const Face& face, const cv::Mat& prediction) {
    static const cv::Mat alpha = [] {
        cv::Mat mask(portraitSize, portraitSize, CV_32F);
        for (int row = 0; row < mask.rows; ++row) {
            for (int col = 0; col < mask.cols; ++col) {
                const double x = (2. * col / (mask.cols - 1) - 1) / .83;
                const double y = (2. * row / (mask.rows - 1) - 1.10) / .89;
                mask.at<float>(row, col) = float(std::clamp((1 - x*x - y*y) * 3, 0., 1.));
            }
        }
        return mask;
    }();
    cv::Mat inverse;
    cv::invertAffineTransform(face.transform, inverse);
    const std::vector<cv::Point2f> corners{
        {0, 0}, {portraitSize, 0}, {portraitSize, portraitSize}, {0, portraitSize}
    };
    std::vector<cv::Point2f> limits;
    cv::transform(corners, limits, inverse);
    const auto bounds = cv::boundingRect(limits) & cv::Rect(0, 0, original.cols, original.rows);
    if (bounds.empty()) {
        return;
    }
    inverse.at<float>(0, 2) -= bounds.x;
    inverse.at<float>(1, 2) -= bounds.y;
    cv::Mat weights, pixels;
    cv::warpAffine(alpha, weights, inverse, bounds.size());
    cv::warpAffine(prediction, pixels, inverse, bounds.size(), cv::INTER_LINEAR, cv::BORDER_REFLECT_101);
    auto destination = original(bounds);
    for (int row = 0; row < bounds.height; ++row) {
        for (int col = 0; col < bounds.width; ++col) {
            const float weight = weights.at<float>(row, col);
            auto& dst = destination.at<cv::Vec3b>(row, col);
            const auto src = pixels.at<cv::Vec3b>(row, col);
            for (int channel = 0; channel < 3; ++channel) {
                dst[channel] = cv::saturate_cast<uchar>(dst[channel] * (1 - weight) + src[channel] * weight);
            }
        }
    }
}

cv::Mat restoreInput(const cv::Mat& original) {
    cv::Mat prepared;
    cv::GaussianBlur(original, prepared, {}, 1.35);
    cv::addWeighted(prepared, .72, original, .28, 0, prepared);
    if (cv::mean(gray(prepared)(cv::Rect(160, 220, 192, 176)))[0] < .32) {
        // Slightly open shadows during inference. The source comparison still uses the original crop.
        prepared.convertTo(prepared, -1, 1.06, 7);
    }
    return prepared;
}

cv::Mat emphasizeSourceEdges(const cv::Mat& restored, const cv::Mat& original) {
    cv::Mat sourceSmoothed, sourceEdges, confidence;
    cv::GaussianBlur(gray(original), sourceSmoothed, {}, 2.0);
    cv::min(edgeMagnitude(sourceSmoothed) * 3.0, 1.0, confidence);
    cv::Mat yuv;
    cv::cvtColor(restored, yuv, cv::COLOR_BGR2YCrCb);
    std::vector<cv::Mat> channels;
    cv::split(yuv, channels);
    cv::Mat y, smooth, residual;
    channels[0].convertTo(y, CV_32F);
    cv::GaussianBlur(y, smooth, {}, 1.65);
    cv::max((y - smooth) * .65, -9.0, residual);
    cv::min(residual, 9.0, residual);
    cv::Mat emphasized = y + residual.mul(confidence);
    emphasized.convertTo(channels[0], CV_8U);
    cv::merge(channels, yuv);
    cv::Mat output;
    cv::cvtColor(yuv, output, cv::COLOR_YCrCb2BGR);
    return output;
}

struct PortraitOption {
    Candidate preview;
    std::vector<cv::Mat> structure;
    double fragmentedDetail{};
};

double fragmentedDetail(const cv::Mat& reduced) {
    const auto core = gray(reduced)(cv::Rect(48, 70, 96, 100));
    cv::Mat smoothed, dx, dy, laplacian, xx, xy, yy;
    cv::GaussianBlur(core, smoothed, {}, 1.6);
    cv::Sobel(smoothed, dx, CV_32F, 1, 0, 3, .125);
    cv::Sobel(smoothed, dy, CV_32F, 0, 1, 3, .125);
    cv::Laplacian(core, laplacian, CV_32F);
    cv::GaussianBlur(dx.mul(dx), xx, {}, 2);
    cv::GaussianBlur(dx.mul(dy), xy, {}, 2);
    cv::GaussianBlur(dy.mul(dy), yy, {}, 2);
    cv::Mat magnitude, coherence;
    cv::magnitude(xx - yy, xy * 2, magnitude);
    cv::divide(magnitude, xx + yy + .000001, coherence);
    // Isolated high-frequency fragments should not count as useful detail.
    cv::Mat absoluteLaplacian = cv::abs(laplacian);
    return cv::mean(absoluteLaplacian.mul(1 - coherence))[0];
}

cv::Mat facialStructure(const cv::Mat& pixels) {
    cv::Mat reduced, largeScale;
    cv::resize(gray(pixels), reduced, {128, 128}, 0, 0, cv::INTER_AREA);
    cv::GaussianBlur(reduced, largeScale, {}, 2.0);
    return reduced - largeScale;
}

bool meaningfullyDifferent(const cv::Mat& first, const cv::Mat& second) {
    cv::Mat difference;
    cv::absdiff(first, second, difference);
    const std::array<cv::Rect, 3> regions{
        cv::Rect(41, 54, 18, 13), cv::Rect(70, 54, 18, 13), cv::Rect(46, 86, 36, 16),
    };
    return std::any_of(regions.begin(), regions.end(), [&](const auto& region) {
        const auto patch = difference(region);
        return cv::countNonZero(patch > .038) > patch.total() * .20;
    });
}

bool meaningfullyDifferent(const PortraitOption& first, const PortraitOption& second) {
    for (size_t index = 0; index < first.structure.size(); ++index) {
        if (meaningfullyDifferent(first.structure[index], second.structure[index])) {
            return true;
        }
    }
    return false;
}

PortraitOption produceOption(Inference& model, std::string label, const cv::Mat& original,
                             const std::vector<Face>& faces, bool prepared) {
    PortraitOption option{{label, original.clone(), 101}, {}};
    for (const auto& face : faces) {
        const auto input = prepared ? restoreInput(face.crop) : face.crop;
        const auto restored = emphasizeSourceEdges(inferPortrait(model, input), face.crop);
        auto scored = scoreRestoration(label, restored, face.crop);
        cv::Mat reducedSource, reducedCandidate;
        cv::resize(face.crop, reducedSource, {192, 192}, 0, 0, cv::INTER_AREA);
        cv::resize(restored, reducedCandidate, {192, 192}, 0, 0, cv::INTER_AREA);
        scored.detail = scoreRestoration(label, reducedCandidate, reducedSource).detail;
        const auto fragmented = fragmentedDetail(reducedCandidate);
        scored.score = roundScore(std::max(0., scored.agreement * .86 + scored.detail * .14
                                               - std::min(350 * fragmented, 12.)));
        option.fragmentedDetail = std::max(option.fragmentedDetail, fragmented);
        if (scored.score < option.preview.score) {
            option.preview.score = scored.score;
            option.preview.agreement = scored.agreement;
            option.preview.detail = scored.detail;
        }
        option.structure.push_back(facialStructure(restored));
        composite(option.preview.pixels, face, restored);
    }
    return option;
}

}

std::vector<Candidates> restorePortraits(
    const std::vector<cv::Mat>& images, const std::filesystem::path& modelDirectory,
    std::optional<cv::Rect> region, bool preferRecall) {
    std::vector<std::vector<Face>> aligned;
    {
        Inference detector(modelDirectory / L"scrfd_10g_bnkps.onnx");
        for (const auto& image : images) {
            if (image.empty() || image.type() != CV_8UC3) {
                throw std::runtime_error("Portrait input must be a decoded 8-bit color image.");
            }
            aligned.push_back(findFaces(detector, image, region, preferRecall));
        }
    }
    std::vector<Candidates> candidates(images.size());
    std::vector<std::vector<PortraitOption>> options(images.size());
    constexpr std::array models{
        std::pair{L"pcflow-bfr-fp32.onnx", "PCFlow"},
        std::pair{L"gfpgan-v1.4-fp32.onnx", "GFPGAN v1.4"},
    };
    if (std::none_of(aligned.begin(), aligned.end(), [](const auto& faces) { return !faces.empty(); })) {
        return candidates;
    }
    std::string firstError;
    for (const auto& [file, model] : models) {
        try {
            Inference net(modelDirectory / file);
            for (size_t index = 0; index < images.size(); ++index) {
                if (aligned[index].empty()) {
                    continue;
                }
                options[index].push_back(produceOption(net, model, images[index], aligned[index], false));
                if (preferRecall) {
                    options[index].push_back(produceOption(net, std::string(model) + " · 去屏纹",
                                                           images[index], aligned[index], true));
                }
            }
        } catch (const std::exception& error) {
            if (firstError.empty()) {
                firstError = error.what();
            }
        }
    }
    if (std::none_of(options.begin(), options.end(), [](const auto& items) { return !items.empty(); })) {
        throw std::runtime_error(firstError);
    }
    for (size_t index = 0; index < candidates.size(); ++index) {
        auto& ranked = options[index];
        if (ranked.empty()) {
            continue;
        }
        std::stable_sort(ranked.begin(), ranked.end(), [](const auto& a, const auto& b) {
            return a.preview.score > b.preview.score;
        });
        auto& best = ranked.front();
        const auto firstAlternative = ranked.begin() + 1;
        const auto alternative = std::find_if(firstAlternative, ranked.end(), [&](const auto& option) {
            return best.preview.agreement >= 92 && option.preview.agreement >= 92
                && best.preview.agreement - option.preview.agreement <= 3
                && option.fragmentedDetail <= std::max(best.fragmentedDetail * 1.45, best.fragmentedDetail + .003)
                && meaningfullyDifferent(best, option);
        });
        const bool modelAgreement = std::any_of(firstAlternative, ranked.end(), [&](const auto& option) {
            return option.preview.model.starts_with("GFPGAN") != best.preview.model.starts_with("GFPGAN")
                && option.preview.agreement >= 92 && !meaningfullyDifferent(best, option);
        });
        const bool reliableLandmarks = std::all_of(aligned[index].begin(), aligned[index].end(), [](const auto& face) {
            return face.detectionScore >= .65f;
        });
        best.preview.moderateConfidence = best.preview.agreement >= 94 && reliableLandmarks
                                         && modelAgreement && alternative == ranked.end();
        candidates[index].push_back(std::move(best.preview));
        if (alternative != ranked.end()) {
            candidates[index].push_back(std::move(alternative->preview));
        }
    }
    return candidates;
}

struct VideoPortraitRestorer::State {
    struct Previous {
        Face face;
        cv::Mat residual;
    };

    explicit State(const std::filesystem::path& models)
        : detector(models / L"scrfd_10g_bnkps.onnx"), net(models / L"gfpgan-v1.4-fp32.onnx") {}

    Inference detector;
    Inference net;
    std::vector<Previous> previous;
};

VideoPortraitRestorer::VideoPortraitRestorer(const std::filesystem::path& models)
    : state_(std::make_unique<State>(models)) {}

VideoPortraitRestorer::~VideoPortraitRestorer() = default;

std::pair<cv::Mat, int> VideoPortraitRestorer::restore(const cv::Mat& source) {
    auto result = source.clone();
    std::vector<State::Previous> current;
    for (auto& face : findFaces(state_->detector, source, std::nullopt)) {
        auto pixels = inferPortrait(state_->net, face.crop);
        auto& previous = state_->previous;
        auto match = previous.end();
        double nearest = std::min(face.bounds.width, face.bounds.height) * .45;
        for (auto candidate = previous.begin(); candidate != previous.end(); ++candidate) {
            const auto old = candidate->face.bounds;
            const auto now = face.bounds;
            const auto distance = std::hypot(cv::norm(old.tl() - now.tl()), cv::norm(old.br() - now.br()));
            if (distance < nearest) {
                nearest = distance;
                match = candidate;
            }
        }
        if (match != previous.end()) {
            const auto before = match->face.crop;
            if (cv::norm(face.crop, before, cv::NORM_L1) / (face.crop.total() * 3.) <= 24) {
                cv::Mat nowGray, beforeGray, flow;
                cv::cvtColor(face.crop, nowGray, cv::COLOR_BGR2GRAY);
                cv::cvtColor(before, beforeGray, cv::COLOR_BGR2GRAY);
                cv::calcOpticalFlowFarneback(nowGray, beforeGray, flow, .5, 2, 13, 2, 5, 1.1, 0);
                auto map = flow.clone();
                for (int row = 0; row < map.rows; ++row) {
                    for (int col = 0; col < map.cols; ++col) map.at<cv::Vec2f>(row, col) += cv::Vec2f(float(col), float(row));
                }
                cv::Mat alignedSource, alignedDetail;
                cv::remap(beforeGray, alignedSource, map, {}, cv::INTER_LINEAR);
                cv::remap(match->residual, alignedDetail, map, {}, cv::INTER_LINEAR);
                for (int row = 0; row < pixels.rows; ++row) {
                    for (int col = 0; col < pixels.cols; ++col) {
                        const int delta = int(nowGray.at<uchar>(row, col)) - alignedSource.at<uchar>(row, col);
                        if (std::abs(delta) >= 9 || cv::norm(flow.at<cv::Vec2f>(row, col)) >= 10) continue;
                        const auto original = face.crop.at<cv::Vec3b>(row, col);
                        const auto detail = alignedDetail.at<cv::Vec3f>(row, col);
                        auto& pixel = pixels.at<cv::Vec3b>(row, col);
                        for (int channel = 0; channel < 3; ++channel) {
                            pixel[channel] = cv::saturate_cast<uchar>((original[channel] + detail[channel]) * .2f
                                                                   + pixel[channel] * .8f);
                        }
                    }
                }
            }
            previous.erase(match);
        }
        cv::Mat residual;
        cv::subtract(pixels, face.crop, residual, cv::noArray(), CV_32F);
        composite(result, face, pixels);
        current.push_back({std::move(face), std::move(residual)});
    }
    state_->previous = std::move(current);
    return {std::move(result), static_cast<int>(state_->previous.size())};
}

}
