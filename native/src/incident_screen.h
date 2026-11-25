#pragma once

#include "storage.h"

#include <atomic>

namespace visual {

Json incidentOperation(const std::string& action,const Json& request,const std::atomic<bool>& cancel);

}
