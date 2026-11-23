#pragma once

#include <array>
#include <filesystem>
#include <memory>
#include <opencv2/core/mat.hpp>
#include <optional>
#include <vector>

namespace portrait_selection {

enum class Route { codeformer, amd };
struct Evidence {
    double clarity;
    double clarity_relative_to_input;
    std::optional<double> replay_error;
    cv::Mat replay_image;
};

struct Result {
    Evidence codeformer;
    Evidence amd;
    cv::Mat observed_image;
    double input_clarity;
    double relative_clarity;
    // Ordered jaw, nose, image-left eye, image-right eye, mouth.
    std::array<double, 5> regional_shift;
    bool replay_supported;
    bool duplicate_eligible;
    std::optional<double> amd_fraction;
    std::vector<Route> original_display;
    std::vector<Route> fitted_display;

    const std::vector<Route> &selected(bool fit_duplicate = false) const noexcept {
        return fit_duplicate ? fitted_display : original_display;
    }
};

class Selector {
  public:
    explicit Selector(const std::filesystem::path &landmarks);
    ~Selector();
    Selector(const Selector &) = delete;
    Selector &operator=(const Selector &) = delete;
    // All images are already aligned 512x512 BGR, 8 bits/channel. The order matters.
    Result select(const cv::Mat &source, const cv::Mat &codeformer, const cv::Mat &amd);

  private:
    class Model;
    std::unique_ptr<Model> model_;
};

} // namespace portrait_selection
