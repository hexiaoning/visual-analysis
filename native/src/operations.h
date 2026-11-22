#pragma once

#include "storage.h"

#include <atomic>

namespace visual {

Json runOperation(const std::string& action,const Json& request,const std::filesystem::path& models,
                  const std::filesystem::path& worker,const std::atomic<bool>& cancel);

}
