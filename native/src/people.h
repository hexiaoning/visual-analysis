#pragma once

#include "vision_models.h"
#include <atomic>

namespace visual {

struct PersonFilter {
    std::string library,sex,region;
    int minimum=-1,maximum=-1;
    static PersonFilter from(const Json& payload);
    bool matches(std::string idCard) const;
};

Json peopleOperation(const std::string& action,const Json& payload,const std::filesystem::path& models,
                     const std::atomic<bool>& cancel);
Json matchGallery(Database& database, const Feature& query, const FaceSettings& settings,
                  const PersonFilter& filter = {}, bool photoRank = false);
Json rectJson(cv::Rect rect);
cv::Rect readRect(const Json& values);

}
