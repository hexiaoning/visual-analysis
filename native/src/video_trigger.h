#pragma once

#include <algorithm>
#include <cmath>
#include <map>
#include <optional>
#include <utility>
#include <opencv2/core.hpp>

namespace visual {

struct VideoLine {cv::Point start,end;};

class VideoTrigger {
public:
    VideoTrigger(std::optional<VideoLine> line,cv::Rect area):line_(line),area_(area) {}

    void advance(double seconds) {
        std::erase_if(positions_,[&](const auto& entry){return seconds-entry.second.seconds>2.;});
    }

    bool observe(int category,size_t id,cv::Point2d center,double seconds) {
        auto& state=positions_[{category,id}];
        state.seconds=seconds;
        if(!line_) {
            const auto margin=state.inside?2:0;
            const bool inside=center.x>=area_.x-margin && center.x<area_.br().x+margin &&
                center.y>=area_.y-margin && center.y<area_.br().y+margin;
            const bool entered=inside && !state.inside;
            state.inside=inside;
            return entered;
        }
        const cv::Point2d start=line_->start,end=line_->end;
        const auto direction=end-start;
        const auto side=direction.cross(center-start)/cv::norm(direction);
        if(std::abs(side)<=2.) return false;
        const auto previous=state.position;
        state.position=center;
        if(!previous) return false;
        const auto prior=direction.cross(*previous-start)/cv::norm(direction);
        if((prior>0.)==(side>0.)) return false;
        const auto intersection=*previous+(center-*previous)*(prior/(prior-side));
        const auto projection=(intersection-start).dot(direction);
        return projection>=0. && projection<=direction.dot(direction);
    }

private:
    struct Position {double seconds{}; bool inside{}; std::optional<cv::Point2d> position;};
    std::optional<VideoLine> line_;
    cv::Rect area_;
    std::map<std::pair<int,size_t>,Position> positions_;
};

}
