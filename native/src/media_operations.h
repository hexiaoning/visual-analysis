#pragma once

#include "operations.h"

namespace visual {

Json mediaOperation(const std::string& action,const Json& request,const std::filesystem::path& models,
                    const std::atomic<bool>& cancel,DetectionPreview* preview);

}
