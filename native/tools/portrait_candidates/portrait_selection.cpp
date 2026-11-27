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
#include <utility>

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

bool has_visible_shape_difference(const cv::Mat &primary, const cv::Mat &candidate,
                                  const std::vector<cv::Point2f> &primary_points,
                                  const std::array<double, 5> &shifts) {
    if (*std::max_element(shifts.begin(), shifts.end()) <= 3.)
        return false;

    const auto structure = [](const cv::Mat &image) -> cv::Mat {
        const cv::Mat gray = selection::grayscale(image);
        return selection::filtered(gray, .85) - selection::filtered(gray, 3.4);
    };
    const cv::Mat primary_band = structure(primary);
    const cv::Mat candidate_band = structure(candidate);
    const cv::Rect image_bounds(0, 0, primary.cols, primary.rows);
    for (size_t region = 0; region < regions.size(); ++region) {
        if (shifts[region] <= 3.)
            continue;
        const auto [begin, end] = regions[region];
        const std::vector<cv::Point2f> points(primary_points.begin() + begin, primary_points.begin() + end);
        const cv::Rect bounds = cv::boundingRect(points);
        const cv::Rect area = cv::Rect(bounds.x - 8, bounds.y - 8, bounds.width + 16, bounds.height + 16) & image_bounds;
        if (area.empty())
            continue;
        const double pixel_count_sqrt = std::sqrt(double(area.area()));
        const cv::Mat a = primary_band(area);
        const cv::Mat b = candidate_band(area);
        if (cv::norm(a, b, cv::NORM_L2) < (1.5 / 255.) * pixel_count_sqrt)
            continue;
        const double a_rms = std::max(.004, cv::norm(a, cv::NORM_L2) / pixel_count_sqrt);
        const double b_rms = std::max(.004, cv::norm(b, cv::NORM_L2) / pixel_count_sqrt);
        if (cv::norm(a / a_rms - b / b_rms, cv::NORM_L2) >= .25 * pixel_count_sqrt)
            return true;
    }
    return false;
}

Evidence measure(const selection::Observation &observed, const cv::Mat &face, double raw_clarity) {
    const cv::Mat gray = selection::grayscale(face);
    const double clarity = selection::clarity(gray);
    cv::Mat replay_image;
    const double replay = observed.replay(gray, replay_image);
    return {clarity, clarity / std::max(raw_clarity, 1.e-8),
            std::isfinite(replay) ? std::optional<double>(replay) : std::nullopt, replay_image};
}

struct MouthGeometry {
    cv::Point2d center;
    cv::Point2d axis;
    cv::Point2d normal;
    double width;
    double inner_height;
};

MouthGeometry mouth_geometry(const std::vector<cv::Point2f> &points) {
    const cv::Point2d span = cv::Point2d(points[82]) - cv::Point2d(points[76]);
    const double width = cv::norm(span);
    if (width < 12.)
        throw std::runtime_error("Mouth corners do not define a stable axis.");
    const cv::Point2d axis = span / width;
    const cv::Point2d normal(-axis.y, axis.x);
    cv::Point2d center{};
    const int inner_begin = 88, inner_end = 96;
    double low = normal.dot(cv::Point2d(points[inner_begin])), high = low;
    for (int i = inner_begin; i < inner_end; ++i) {
        const cv::Point2d point = points[i];
        center += point;
        const double position = normal.dot(point);
        low = std::min(low, position);
        high = std::max(high, position);
    }
    return {center / (inner_end - inner_begin), axis, normal, width, high - low};
}

cv::Mat reshape_mouth(const cv::Mat &face, const MouthGeometry &shape, double opening) {
    const double radius = std::max(1.3, .5 * shape.inner_height);
    const double requested = std::clamp(.5 * shape.width * opening, .43 * radius, 1.9 * radius);
    const double margin = std::max({13., .34 * shape.width, 2. * radius, 2. * requested});
    const double half_width = .58 * shape.width;
    cv::Mat mapping(face.size(), CV_32FC2);
    for (int y = 0; y < face.rows; ++y) {
        auto *row = mapping.ptr<cv::Point2f>(y);
        for (int x = 0; x < face.cols; ++x) {
            row[x] = cv::Point2f(static_cast<float>(x), static_cast<float>(y));
            const cv::Point2d relative = cv::Point2d(x, y) - shape.center;
            const double across = shape.axis.dot(relative) / half_width;
            const double signed_height = shape.normal.dot(relative);
            const double height = std::abs(signed_height);
            if (std::abs(across) >= 1. || height >= margin)
                continue;
            // Map the requested inner lip interval into the original one. The
            // transition to the outer boundary preserves ordering along the normal.
            const double blend = std::pow(1. - across * across, .68);
            const double desired = radius + blend * (requested - radius);
            const double from = height < desired ? height * radius / desired
                                                  : radius + (height - desired) * (margin - radius) / (margin - desired);
            const cv::Point2d displacement = shape.normal * ((from - height) * (signed_height < 0. ? -1. : 1.));
            row[x] += cv::Point2f(displacement);
        }
    }
    cv::Mat image;
    cv::remap(face, image, mapping, {}, cv::INTER_CUBIC, cv::BORDER_REFLECT_101);
    return image;
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

ShapeDescriptor Selector::describe_shape(const cv::Mat &face) {
    if (face.type() != CV_8UC3 || face.size() != cv::Size(512, 512))
        throw std::invalid_argument("Expected an aligned 512x512 CV_8UC3 portrait");
    return {face, model_->points(face)};
}

bool visibly_different(const ShapeDescriptor &primary, const ShapeDescriptor &candidate) {
    if (primary.points.size() != 98 || candidate.points.size() != 98)
        throw std::invalid_argument("Expected two WFLW landmark sets");
    const auto shifts = regional_shift(candidate.points, primary.points);
    return has_visible_shape_difference(primary.image, candidate.image, primary.points, shifts);
}

std::vector<MouthVariant> Selector::mouth_variants(const cv::Mat &face, const std::vector<double> &openings) {
    if (face.type() != CV_8UC3 || face.size() != cv::Size(512, 512))
        throw std::invalid_argument("Expected an aligned 512x512 CV_8UC3 portrait");
    if (openings.empty() || std::any_of(openings.begin(), openings.end(), [](double opening) {
            return !std::isfinite(opening) || opening <= 0. || opening > 1.;
        }))
        throw std::invalid_argument("Expected mouth openings in (0, 1].");
    const auto shape = mouth_geometry(model_->points(face));
    std::vector<MouthVariant> variants;
    variants.reserve(openings.size());
    for (double opening : openings) {
        auto image = reshape_mouth(face, shape, opening);
        const auto actual = mouth_geometry(model_->points(image));
        variants.push_back({std::move(image), opening, actual.inner_height / actual.width});
    }
    return variants;
}

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
    r.visible_shape_difference = has_visible_shape_difference(codeformer, amd, primary_points, r.regional_shift);
    r.replay_supported = r.amd.replay_error && r.codeformer.replay_error &&
                         *r.amd.replay_error <= *r.codeformer.replay_error + .007;
    r.original_display = {Route::codeformer};
    if (r.replay_supported && r.visible_shape_difference && r.relative_clarity >= .96)
        r.original_display.push_back(Route::amd);
    r.fitted_display = r.original_display;
    r.duplicate_eligible = r.replay_supported && !r.visible_shape_difference;
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
