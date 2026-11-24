#pragma once

#include "vision_models.h"

#include <algorithm>
#include <cstddef>

namespace visual {

struct PersonObservation {
    cv::Rect box;
    Feature appearance;
};

class PersonTracks {
public:
    explicit PersonTracks(double minimumAppearance):minimumAppearance_(minimumAppearance) {}

    std::vector<size_t> update(const std::vector<PersonObservation>& people,double seconds) {
        std::erase_if(active_,[&](const auto& track){return seconds-track.seconds>2.;});
        struct Match {double score;size_t track,person;};
        std::vector<Match> matches;
        for(size_t p=0;p<people.size();++p) {
            const auto& person=people[p];
            for(size_t t=0;t<active_.size();++t) {
                const auto& prior=active_[t].observation;
                const double overlap=(prior.box&person.box).area();
                const double iou=overlap/std::max(1.,prior.box.area()+double(person.box.area())-overlap);
                if(iou<.30) continue;
                double appearance=0.;
                if(!prior.appearance.empty() && !person.appearance.empty()) {
                    appearance=similarity(prior.appearance,person.appearance);
                    if(appearance<minimumAppearance_) continue;
                }
                matches.push_back({iou+.2*appearance,t,p});
            }
        }
        std::stable_sort(matches.begin(),matches.end(),[](const Match& a,const Match& b){return a.score>b.score;});
        std::vector<size_t> ids(people.size());
        std::vector<bool> used(active_.size());
        for(const auto& match:matches) {
            if(ids[match.person] || used[match.track]) continue;
            auto& track=active_[match.track];
            track.observation=people[match.person];
            track.seconds=seconds;
            ids[match.person]=track.id;
            used[match.track]=true;
        }
        for(size_t p=0;p<people.size();++p) {
            if(ids[p]) continue;
            active_.push_back({people[p],seconds,++lastId_});
            ids[p]=lastId_;
        }
        return ids;
    }

private:
    struct Track {PersonObservation observation;double seconds;size_t id;};
    std::vector<Track> active_;
    double minimumAppearance_;
    size_t lastId_{};
};

}
