#include "operations.h"
#include "people.h"
#include "vehicle.h"
#include "media_operations.h"
#include "plate_video.h"

namespace visual {

Json runOperation(const std::string& action,const Json& request,const std::filesystem::path& models,
                  const std::filesystem::path& worker,const std::atomic<bool>& cancel) {
    if(action=="vehicle.analyze") return analyzeVehicle(request,models,worker,cancel);
    if(action=="plate.video") return plateVideo(request,cancel);
    if(action=="semantic.search" || action=="visual.search" || action=="objects.scan" || action=="face.scan" ||
       action=="hits.export") {
        return mediaOperation(action,request,models,cancel);
    }
    return peopleOperation(action,request,models,cancel);
}

}
