#include "portrait_selection.h"
#include "duplicate_consensus.h"
#include "selection_replay.hpp"

#include <algorithm>
#include <cmath>
#include <onnxruntime_cxx_api.h>
#include <opencv2/calib3d.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <stdexcept>

namespace portrait_selection {
namespace {

constexpr std::array<int64_t, 4> input_shape{1, 3, 256, 256};
constexpr std::array<int, 10> stable{0, 32, 16, 51, 60, 64, 68, 72, 76, 82};
constexpr std::array<std::pair<int, int>, 5> regions{{{0, 33}, {51, 60}, {60, 68}, {68, 76}, {76, 96}}};

Ort::SessionOptions single_threaded_cpu() {
    Ort::SessionOptions options;
    options.SetIntraOpNumThreads(1);
    options.SetInterOpNumThreads(1);
    options.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
    options.AddConfigEntry("session.intra_op.allow_spinning", "0");
    options.AddConfigEntry("session.inter_op.allow_spinning", "0");
    options.DisableMemPattern();
    options.DisableCpuMemArena();
    return options;
}

std::array<double, 5> regional_shift(const std::vector<cv::Point2f> &candidate,
                                     const std::vector<cv::Point2f> &primary) {
    std::vector<cv::Point2f> from, to;
    for (int index : stable) {
        from.push_back(candidate[index]);
        to.push_back(primary[index]);
    }
    auto aligned = candidate;
    auto similarity = cv::estimateAffinePartial2D(from, to, cv::noArray(), cv::LMEDS);
    if (!similarity.empty())
        cv::transform(candidate, aligned, similarity);
    std::array<double, 5> shift{};
    for (size_t region = 0; region < regions.size(); ++region) {
        auto [begin, end] = regions[region];
        for (int i = begin; i < end; ++i)
            shift[region] += cv::norm(aligned[i] - primary[i]);
        shift[region] /= end - begin;
    }
    return shift;
}

Evidence measure(const selection::Observation &observed, const cv::Mat &face, double raw_clarity) {
    const cv::Mat gray = selection::grayscale(face);
    const double clarity = selection::clarity(gray);
    cv::Mat replay_image;
    const double replay = observed.replay(gray, replay_image);
    return {clarity, clarity / std::max(raw_clarity, 1.e-8),
            std::isfinite(replay) ? std::optional<double>(replay) : std::nullopt, replay_image};
}

} // namespace

class Selector::Model {
  public:
    explicit Model(const std::filesystem::path &model)
        : env_(ORT_LOGGING_LEVEL_WARNING, "HR-noSA"),
          session_(env_, model.c_str(), single_threaded_cpu()),
          memory_(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)) {
        Ort::AllocatorWithDefaultOptions alloc;
        bool has_landmarks = false;
        for (size_t i = 0; i < session_.GetOutputCount(); ++i) {
            if (std::string(session_.GetOutputNameAllocated(i, alloc).get()) == "landmarks")
                has_landmarks = true;
        }
        if (session_.GetInputCount() != 1 ||
            std::string(session_.GetInputNameAllocated(0, alloc).get()) != "pixels" || !has_landmarks)
            throw std::invalid_argument(
                "Expected the HR-noSA WFLW landmarks model with pixels and landmarks tensors.");
    }

    std::vector<cv::Point2f> points(const cv::Mat &image) {
        const cv::Matx23f crop(.8f, 0.f, -77.3f, 0.f, .8f, -117.3f);
        cv::Mat region;
        cv::warpAffine(image, region, crop, {256, 256}, cv::INTER_LINEAR, cv::BORDER_REPLICATE);
        auto blob = cv::dnn::blobFromImage(region, 1. / 127.5, {256, 256}, cv::Scalar::all(127.5), false,
                                           false, CV_32F);
        auto tensor = Ort::Value::CreateTensor<float>(memory_, blob.ptr<float>(), blob.total(),
                                                      input_shape.data(), input_shape.size());
        const char *input = "pixels";
        const char *output = "landmarks";
        auto values = session_.Run(Ort::RunOptions{nullptr}, &input, &tensor, 1, &output, 1);
        auto info = values[0].GetTensorTypeAndShapeInfo();
        if (info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
            info.GetShape() != std::vector<int64_t>{1, 98, 2})
            throw std::runtime_error("Unexpected HR-noSA landmark tensor dimensions");
        const float *pts = values[0].GetTensorData<float>();
        std::vector<cv::Point2f> landmarks;
        landmarks.reserve(98);
        for (int i = 0; i < 98; ++i) {
            if (!std::isfinite(pts[2 * i]) || !std::isfinite(pts[2 * i + 1]))
                throw std::runtime_error("Nonfinite facial landmark");
            landmarks.emplace_back((pts[2 * i] + 1.f) * 127.5f, (pts[2 * i + 1] + 1.f) * 127.5f);
        }
        cv::Mat inverse;
        cv::invertAffineTransform(crop, inverse);
        cv::transform(landmarks, landmarks, inverse);
        return landmarks;
    }

  private:
    Ort::Env env_;
    Ort::Session session_;
    Ort::MemoryInfo memory_;
};

Selector::Selector(const std::filesystem::path &model) : model_(std::make_unique<Model>(model)) {}
Selector::~Selector() = default;

Result Selector::select(const cv::Mat &source, const cv::Mat &codeformer, const cv::Mat &amd,
                        const TextureGuard *guard) {
    for (const cv::Mat *im : {&source, &codeformer, &amd}) {
        if (im->type() != CV_8UC3 || im->size() != cv::Size(512, 512))
            throw std::invalid_argument("Expected an aligned 512x512 CV_8UC3 portrait");
    }
    Result r{};
    const auto raw = selection::grayscale(source);
    const selection::Observation observation(raw);
    observation.observed().convertTo(r.observed_image, CV_8U, 255.);
    r.input_clarity = selection::clarity(raw);
    r.codeformer = measure(observation, codeformer, r.input_clarity);
    r.amd = measure(observation, amd, r.input_clarity);
    r.relative_clarity = r.amd.clarity / std::max(1.e-8, r.codeformer.clarity);
    const auto primary_points = model_->points(codeformer);
    r.regional_shift = regional_shift(model_->points(amd), primary_points);
    const double max_shift = *std::max_element(r.regional_shift.begin(), r.regional_shift.end());
    r.replay_supported = r.amd.replay_error && r.codeformer.replay_error &&
                         *r.amd.replay_error <= *r.codeformer.replay_error + .007;
    r.original_display = {Route::codeformer};
    if (r.replay_supported && max_shift > 3. && r.relative_clarity >= .96)
        r.original_display.push_back(Route::amd);
    r.fitted_display = r.original_display;
    // Compare the existing pixels only after the candidates pass the replay and shape checks.
    r.duplicate_eligible = r.replay_supported && max_shift <= 3.;
    if (r.duplicate_eligible) {
        r.amd_fraction = portrait_consensus::amd_fraction(source, codeformer, amd);
        if (*r.amd_fraction > .5)
            r.fitted_display = {Route::amd};
    }
    if (guard) {
        r.texture_cutoff = guard->cutoff;
        r.guarded_display = r.original_display;
        if (r.fitted_display != r.original_display) {
            r.replacement_texture_score = guard->score(amd);
            if (*r.replacement_texture_score <= guard->cutoff)
                r.guarded_display = r.fitted_display;
        }
    }
    return r;
}

} // namespace portrait_selection
