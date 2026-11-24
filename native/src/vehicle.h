#pragma once

#include "vision_models.h"

#include <atomic>

namespace visual {

struct DetectionPreview;

Json analyzeVehicle(const Json& request,const std::filesystem::path& models,
                    const std::filesystem::path& worker,const std::atomic<bool>& cancel,
                    DetectionPreview* preview);

}
