#include "screen_restore.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <numbers>
#include <stdexcept>
#include <utility>

#include <opencv2/imgproc.hpp>

namespace visual::screen {
namespace {

constexpr int tileSide = 160;
constexpr int stride = 80;
constexpr int halfTile = tileSide / 2;
constexpr int margin = 64;
constexpr double pi = std::numbers::pi_v<double>;

struct Peak {
    cv::Point point;
    float power;
    float support;
};

void checkBgr(const cv::Mat& source) {
    if (source.empty() || source.type() != CV_8UC3) {
        throw std::invalid_argument("Screen restoration expects an 8-bit BGR image");
    }
}

cv::Mat unit(const cv::Mat& image) {
    cv::Mat result;
    image.convertTo(result, CV_32F, 1. / 255.);
    return result;
}

cv::Mat gaussian(const cv::Mat& image, double radius) {
    cv::Mat result;
    cv::GaussianBlur(image, result, {}, radius, radius, cv::BORDER_REFLECT_101 | cv::BORDER_ISOLATED);
    return result;
}

cv::Mat linearBgr(const cv::Mat& image) {
    static const cv::Mat lut = [] {
        cv::Mat table(1, 256, CV_32F);
        for (int i = 0; i < 256; ++i) {
            const float value = i / 255.f;
            table.at<float>(i) = value <= .04045f ? value / 12.92f : std::pow((value + .055f) / 1.055f, 2.4f);
        }
        return table;
    }();
    cv::Mat linear;
    cv::LUT(image, lut, linear);
    return linear;
}

unsigned char srgbByte(float value) {
    const float clipped = std::clamp(value, 0.f, 1.f);
    const float encoded = clipped <= .0031308f ? 12.92f * clipped : 1.055f * std::pow(clipped, 1.f / 2.4f) - .055f;
    return cv::saturate_cast<unsigned char>(encoded * 255.f);
}

void carrierComponents(cv::Size shape, cv::Point2f frequency, cv::Mat& cosine, cv::Mat& sine) {
    cosine.create(shape, CV_32FC3);
    sine.create(shape, CV_32FC3);
    for (int y = 0; y < shape.height; ++y) {
        auto* real = cosine.ptr<cv::Vec3f>(y);
        auto* imaginary = sine.ptr<cv::Vec3f>(y);
        for (int x = 0; x < shape.width; ++x) {
            const float angle = static_cast<float>(2. * pi) * (frequency.x * x + frequency.y * y);
            real[x] = cv::Vec3f::all(std::cos(angle));
            imaginary[x] = cv::Vec3f::all(std::sin(angle));
        }
    }
}

void subtractCarrier(cv::Mat& scene, cv::Point2f frequency) {
    cv::Mat cosine, sine;
    carrierComponents(scene.size(), frequency, cosine, sine);
    const cv::Mat local = scene - gaussian(scene, 2.);
    const cv::Mat real = 2.f * gaussian(local.mul(cosine), 10.);
    const cv::Mat imaginary = 2.f * gaussian(local.mul(sine), 10.);
    const cv::Mat smoothReal = gaussian(real, 16.);
    const cv::Mat smoothImaginary = gaussian(imaginary, 16.);
    cv::Mat amplitude, smoothAmplitude;
    cv::sqrt(real.mul(real) + imaginary.mul(imaginary), amplitude);
    cv::sqrt(smoothReal.mul(smoothReal) + smoothImaginary.mul(smoothImaginary), smoothAmplitude);
    const cv::Mat localAmplitude = gaussian(amplitude, 16.);
    for (int y = 0; y < scene.rows; ++y) {
        auto* sceneRow = scene.ptr<float>(y);
        const auto* a = real.ptr<float>(y);
        const auto* b = imaginary.ptr<float>(y);
        const auto* c = cosine.ptr<float>(y);
        const auto* s = sine.ptr<float>(y);
        const auto* amp = amplitude.ptr<float>(y);
        const auto* coherenceNumerator = smoothAmplitude.ptr<float>(y);
        const auto* coherenceDenominator = localAmplitude.ptr<float>(y);
        for (int x = 0; x < 3 * scene.cols; ++x) {
            const float coherence = coherenceNumerator[x] / std::max(coherenceDenominator[x], 1.e-6f);
            const float confidence = std::clamp((coherence - .26f) / .49f, 0.f, 1.f)
                * std::clamp((amp[x] - .0015f) / .003f, 0.f, 1.f);
            sceneRow[x] -= .9f * confidence * (a[x] * c[x] + b[x] * s[x]);
        }
    }
}

std::vector<int> positions(int length) {
    std::vector<int> values;
    for (int i = 0; i <= length - tileSide; i += stride) {
        values.push_back(i);
    }
    if (values.back() != length - tileSide) {
        values.push_back(length - tileSide);
    }
    return values;
}

bool exposed(const cv::Mat& patch) {
    cv::Scalar mean, sd;
    cv::meanStdDev(patch, mean, sd);
    const double gray = (mean[0] + mean[1] + mean[2]) / 3.;
    double variance = 0.;
    for (int channel = 0; channel < 3; ++channel) {
        variance += sd[channel] * sd[channel] + std::pow(mean[channel] - gray, 2);
    }
    return gray > .045 && gray < .97 && variance / 3. > .028 * .028;
}

cv::Mat hann(int width, int height) {
    cv::Mat result(height, width, CV_32F);
    for (int y = 0; y < height; ++y) {
        auto* row = result.ptr<float>(y);
        const double vertical = .5 - .5 * std::cos(2. * pi * y / (height - 1));
        for (int x = 0; x < width; ++x) {
            row[x] = static_cast<float>(vertical * (.5 - .5 * std::cos(2. * pi * x / (width - 1))));
        }
    }
    return result;
}

cv::Mat power(const cv::Mat& channel, const cv::Mat& taper) {
    cv::Mat complex, components[2];
    cv::dft(channel.mul(taper), complex, cv::DFT_COMPLEX_OUTPUT);
    cv::split(complex, components);
    return components[0].mul(components[0]) + components[1].mul(components[1]);
}

cv::Mat detailPower(const cv::Mat& patch, const cv::Mat& taper) {
    const cv::Mat detail = patch - gaussian(patch, 1.5);
    cv::Mat bgr[3];
    cv::split(detail, bgr);
    cv::Mat spectrum = power((bgr[0] + bgr[1] + bgr[2]) / 3., taper)
        + .55 * power(bgr[1] - bgr[2], taper) + .55 * power(bgr[0] - bgr[1], taper);
    spectrum += .026;
    cv::log(spectrum, spectrum);
    cv::Mat centered(spectrum.size(), CV_32F);
    for (int y = 0; y < tileSide; ++y) {
        const auto* row = spectrum.ptr<float>((y + halfTile) % tileSide);
        auto* dst = centered.ptr<float>(y);
        std::copy_n(row + halfTile, halfTile, dst);
        std::copy_n(row, halfTile, dst + halfTile);
    }
    return centered - gaussian(centered, 2.2);
}

std::vector<Peak> recurring(const std::vector<cv::Mat>& spectra) {
    cv::Mat mean = cv::Mat::zeros(tileSide, tileSide, CV_32F);
    cv::Mat support = cv::Mat::zeros(tileSide, tileSide, CV_32F);
    cv::Mat raised, mask;
    const cv::Mat voteWindow = cv::Mat::ones(7, 7, CV_8U);
    for (const auto& score : spectra) {
        mean += score;
        cv::dilate(score, raised, voteWindow);
        cv::compare(raised, 1.45, mask, cv::CMP_GE);
        cv::accumulate(mask, support);
    }
    mean /= static_cast<double>(spectra.size());
    support /= 255. * static_cast<double>(spectra.size());
    cv::dilate(mean, raised, cv::Mat::ones(9, 9, CV_8U));
    std::vector<Peak> all;
    for (int y = 0; y < tileSide; ++y) {
        for (int x = 0; x < tileSide; ++x) {
            const cv::Point centered(x - halfTile, y - halfTile);
            const double radius = cv::norm(centered);
            const float strength = mean.at<float>(y, x);
            const float votes = support.at<float>(y, x);
            if (radius > 12. && radius < 72. && (centered.x > 1 || (std::abs(centered.x) <= 1 && centered.y > 0))
                && strength == raised.at<float>(y, x) && strength > .86f && votes >= .60f) {
                all.push_back({{x, y}, strength, votes});
            }
        }
    }
    std::stable_sort(all.begin(), all.end(), [](const Peak& a, const Peak& b) {
        return a.power == b.power ? a.support > b.support : a.power > b.power;
    });
    std::vector<Peak> distinct;
    for (const auto& peak : all) {
        if (std::none_of(distinct.begin(), distinct.end(), [&](const Peak& p) {
            return cv::norm(peak.point - p.point) < 6.;
        })) {
            distinct.push_back(peak);
            if (distinct.size() == 5) {
                break;
            }
        }
    }
    return distinct;
}

float median(std::vector<float>& numbers) {
    std::sort(numbers.begin(), numbers.end());
    const size_t mid = numbers.size() / 2;
    return numbers.size() % 2 ? numbers[mid] : (numbers[mid - 1] + numbers[mid]) * .5f;
}

cv::Point2f refine(const std::vector<cv::Mat>& spectra, cv::Point guess) {
    std::vector<float> horizontal, vertical;
    const cv::Rect box(guess.x - 3, guess.y - 3, 7, 7);
    for (const auto& score : spectra) {
        double top;
        cv::Point at;
        cv::minMaxLoc(score(box), nullptr, &top, nullptr, &at);
        if (top < 1.5) {
            continue;
        }
        at += box.tl();
        auto offset = [&](cv::Point step) {
            const float left = score.at<float>(at - step);
            const float right = score.at<float>(at + step);
            return std::clamp(.5f * (left - right) /
                std::min(left + right - 2.f * static_cast<float>(top), -1.e-4f), -.45f, .45f);
        };
        horizontal.push_back((at.x + offset({1, 0}) - halfTile) / tileSide);
        vertical.push_back((at.y + offset({0, 1}) - halfTile) / tileSide);
    }
    if (horizontal.empty()) {
        return cv::Point2f(guess - cv::Point(halfTile, halfTile)) * (1.f / tileSide);
    }
    return {median(horizontal), median(vertical)};
}

double sinc(double value) {
    return std::abs(value) < 1.e-8 ? 1. : std::sin(pi * value) / (pi * value);
}

std::vector<cv::Point2d> mapFrequencies(const Peaks& peaks, const cv::Matx23d& affine) {
    const cv::Matx22d transform{affine(0, 0), affine(1, 0), affine(0, 1), affine(1, 1)};
    if (std::abs(cv::determinant(transform)) < 1.e-8) {
        throw std::invalid_argument("Screen alignment is singular");
    }
    const auto inverse = transform.inv();
    std::vector<cv::Point2d> values;
    const auto insert = [&](cv::Point2d point) {
        const cv::Vec2d candidate = inverse * cv::Vec2d(point.x, point.y);
        point = {candidate[0] - std::round(candidate[0]), candidate[1] - std::round(candidate[1])};
        if (point.x < 0. || (point.x == 0. && point.y < 0.)) {
            point *= -1.;
        }
        const double norm = cv::norm(point);
        if (norm < .045 || norm > .48 || std::any_of(values.begin(), values.end(), [&](auto p) {
            return cv::norm(p - point) < .012;
        })) {
            return;
        }
        values.push_back(point);
    };
    std::vector<cv::Point2d> observed;
    for (auto peak : peaks.periods) {
        observed.emplace_back(peak);
    }
    for (auto peak : peaks.frequencies) {
        observed.emplace_back(peak);
    }
    for (auto peak : observed) {
        insert(peak);
    }
    for (auto peak : observed) {
        for (int y = -1; y <= 1; ++y) {
            for (int x = -1; x <= 1; ++x) {
                if (x == 0 && y == 0) {
                    continue;
                }
                const cv::Point2d shifted = peak + cv::Point2d(x, y);
                const double fraction = sinc(shifted.x) * sinc(shifted.y) /
                    (sinc(peak.x) * sinc(peak.y));
                if (fraction * fraction >= .13) {
                    insert(shifted);
                }
            }
        }
    }
    return values;
}

cv::Mat band(cv::Size size, const std::vector<cv::Point2d>& frequencies, double radius, bool half) {
    cv::Mat weights = cv::Mat::zeros(size.height, half ? size.width / 2 + 1 : size.width, CV_32F);
    for (auto f : frequencies) {
        for (double sign : {-1., 1.}) {
            for (int y = 0; y < weights.rows; ++y) {
                auto* w = weights.ptr<float>(y);
                const double dy = std::remainder(y / double(size.height) - sign * f.y, 1.);
                for (int x = 0; x < weights.cols; ++x) {
                    const double dx = std::remainder(x / double(size.width) - sign * f.x, 1.);
                    w[x] = std::max(w[x], static_cast<float>(std::exp(-(dx * dx + dy * dy) / (2. * radius * radius))));
                }
            }
        }
    }
    return weights;
}

double bandRms(const cv::Mat& image, const cv::Mat& mask, const cv::Mat& taper) {
    const cv::Mat clip = image(cv::Rect(104, 128, 304, 320));
    cv::Mat values[3];
    cv::split(clip - gaussian(clip, 2.), values);
    const cv::Rect half{0, 0, mask.cols, mask.rows};
    double total = 0.;
    for (const auto& channel : values) {
        total += cv::sum(power(channel, taper)(half).mul(mask))[0];
    }
    return std::sqrt(2. * total / (clip.total() * cv::sum(taper.mul(taper))[0] * 3.));
}

cv::Mat cleanGray(const cv::Mat& bgr, const cv::Mat& response) {
    cv::Mat gray, extended, spectrum, part[2];
    cv::cvtColor(bgr, gray, cv::COLOR_BGR2GRAY);
    cv::copyMakeBorder(gray, extended, margin, margin, margin, margin, cv::BORDER_REFLECT_101);
    cv::dft(extended, spectrum, cv::DFT_COMPLEX_OUTPUT);
    cv::split(spectrum, part);
    part[0] = part[0].mul(response);
    part[1] = part[1].mul(response);
    cv::merge(part, 2, spectrum);
    cv::idft(spectrum, extended, cv::DFT_SCALE | cv::DFT_REAL_OUTPUT);
    return extended(cv::Rect(margin, margin, gray.cols, gray.rows)).clone();
}

double centralRms(const cv::Mat& source) {
    const cv::Mat core = source(cv::Rect(120, 146, 270, 280));
    return cv::norm(core, cv::NORM_L2) / std::sqrt(static_cast<double>(core.total()));
}

}

Peaks detect(const cv::Mat& nativeBgr) {
    checkBgr(nativeBgr);
    Peaks result;
    if (nativeBgr.rows < tileSide || nativeBgr.cols < tileSide) {
        return result;
    }
    const cv::Mat pixels = unit(nativeBgr);
    const cv::Mat window = hann(tileSide, tileSide);
    std::vector<cv::Mat> spectra;
    for (int y : positions(pixels.rows)) {
        for (int x : positions(pixels.cols)) {
            const cv::Mat patch = pixels(cv::Rect(x, y, tileSide, tileSide));
            if (exposed(patch)) {
                spectra.push_back(detailPower(patch, window));
            }
        }
    }
    if (spectra.size() < 5) {
        return result;
    }
    std::vector<std::pair<cv::Point2f, float>> chosen;
    for (const auto& peak : recurring(spectra)) {
        const auto integer = cv::Point2f(peak.point - cv::Point(halfTile, halfTile)) * (1.f / tileSide);
        if (peak.power >= 1.f) {
            result.frequencies.push_back(integer);
        }
        const auto exact = refine(spectra, peak.point);
        if (peak.power >= 1.22f && cv::norm(exact) >= .105) {
            chosen.emplace_back(exact, peak.power);
        }
    }
    if (chosen.empty()) {
        return result;
    }
    result.periods.push_back(chosen.front().first);
    double best = 0.;
    for (size_t i = 0; i < chosen.size(); ++i) {
        for (size_t j = i + 1; j < chosen.size(); ++j) {
            const auto& [a, s] = chosen[i];
            const auto& [b, t] = chosen[j];
            const double oblique = std::abs(a.cross(b)) / (cv::norm(a) * cv::norm(b));
            const double reading = (s + t) * (.82 + .18 * oblique);
            if (oblique >= .64 && reading > best) {
                result.periods = {a, b};
                best = reading;
            }
        }
    }
    return result;
}

cv::Mat averagePeriods(const cv::Mat& image, const Peaks& peaks) {
    checkBgr(image);
    if (peaks.periods.empty()) {
        return image.clone();
    }
    if (peaks.periods.size() > 2) {
        throw std::invalid_argument("At most two screen directions can be sampled");
    }
    cv::Mat measured(static_cast<int>(peaks.periods.size()), 2, CV_32F);
    for (int i = 0; i < measured.rows; ++i) {
        measured.at<float>(i, 0) = peaks.periods[i].x;
        measured.at<float>(i, 1) = peaks.periods[i].y;
    }
    cv::Mat inverse;
    cv::invert(measured, inverse, cv::DECOMP_SVD);
    constexpr std::array<float, 4> phases{-.375f, -.125f, .125f, .375f};
    std::vector<cv::Point2f> shifts{{0.f, 0.f}};
    for (int j = 0; j < inverse.cols; ++j) {
        const cv::Point2f direction(inverse.at<float>(0, j), inverse.at<float>(1, j));
        std::vector<cv::Point2f> next;
        for (const auto& prior : shifts) {
            for (float phase : phases) {
                next.push_back(prior + phase * direction);
            }
        }
        shifts = std::move(next);
    }
    const cv::Mat linear = linearBgr(image);
    cv::Mat sampled;
    cv::Mat accumulated = cv::Mat::zeros(image.size(), CV_32FC3), coordinates(image.size(), CV_32FC2);
    for (auto shift : shifts) {
        for (int y = 0; y < image.rows; ++y) {
            auto* row = coordinates.ptr<cv::Point2f>(y);
            for (int x = 0; x < image.cols; ++x) {
                row[x] = {x + shift.x, y + shift.y};
            }
        }
        cv::remap(linear, sampled, coordinates, cv::noArray(), cv::INTER_LINEAR, cv::BORDER_REFLECT_101);
        accumulated += sampled;
    }
    cv::Mat output(image.size(), CV_8UC3);
    const float count = static_cast<float>(shifts.size());
    for (int y = 0; y < image.rows; ++y) {
        const auto* src = linear.ptr<float>(y);
        const auto* avg = accumulated.ptr<float>(y);
        auto* dst = output.ptr<unsigned char>(y);
        for (int x = 0; x < 3 * image.cols; ++x) {
            dst[x] = srgbByte(src[x] + .86f * (avg[x] / count - src[x]));
        }
    }
    return output;
}

cv::Mat subtractCoherentCarrier(const cv::Mat& image, const Peaks& peaks) {
    checkBgr(image);
    if (peaks.periods.empty()) {
        return image.clone();
    }
    if (peaks.periods.size() > 2) {
        throw std::invalid_argument("At most two screen directions can be subtracted");
    }
    cv::Mat scene = linearBgr(image);
    for (const auto frequency : peaks.periods) {
        if (cv::norm(frequency) >= .105) {
            subtractCarrier(scene, frequency);
        }
    }
    cv::Mat result(image.size(), CV_8UC3);
    for (int y = 0; y < image.rows; ++y) {
        const auto* input = scene.ptr<float>(y);
        auto* output = result.ptr<unsigned char>(y);
        for (int x = 0; x < 3 * image.cols; ++x) {
            output[x] = srgbByte(input[x]);
        }
    }
    return result;
}

Selection selectBase(const cv::Mat& raw, const cv::Mat& previous, const cv::Mat& phase,
                     const Peaks& peaks, const cv::Matx23d& affine) {
    for (const auto& picture : {raw, previous, phase}) {
        checkBgr(picture);
        if (picture.size() != cv::Size(512, 512)) {
            throw std::invalid_argument("The aligned portrait must be 512 by 512");
        }
    }
    const auto bands = mapFrequencies(peaks, affine);
    if (bands.empty() || peaks.periods.empty()) {
        return {};
    }
    const cv::Mat rawF = unit(raw), baseF = unit(previous), phaseF = unit(phase);
    const cv::Mat atBand = band({304, 320}, bands, .010, true);
    const cv::Mat taper = hann(304, 320);
    Selection result;
    result.carrierRatio = bandRms(phaseF, atBand, taper) /
        std::max(bandRms(baseF, atBand, taper), 1.e-8);
    if (result.carrierRatio > .72) {
        return result;
    }
    const cv::Mat pass = 1.f - band({512 + 2 * margin, 512 + 2 * margin}, bands, .009, false);
    const cv::Mat observed = cleanGray(rawF, pass);
    const cv::Mat base = cleanGray(baseF, pass);
    const cv::Mat proposal = cleanGray(phaseF, pass);
    const auto midband = [](const cv::Mat& x) { return centralRms(gaussian(x, 2.) - gaussian(x, 6.5)); };
    const cv::Mat coarse = gaussian(observed, 5.);
    result.midbandRatio = midband(proposal) / std::max(midband(base), 1.e-8);
    result.baseCoarseError = centralRms(gaussian(base, 5.) - coarse);
    result.phaseCoarseError = centralRms(gaussian(proposal, 5.) - coarse);
    result.usePhase = result.midbandRatio >= .75 && result.phaseCoarseError <= result.baseCoarseError + .008;
    return result;
}

}
