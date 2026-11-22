#pragma once

#include <atomic>
#include <filesystem>

namespace visual {

struct VideoProgress {
    std::atomic<bool> cancel{};
    std::atomic<int> frames{};
    std::atomic<int> faces{};
};

void restorePortraitVideo(const std::filesystem::path& source, const std::filesystem::path& target,
                          const std::filesystem::path& models, VideoProgress& progress);

}
