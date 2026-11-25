#include "plate_recognition.h"
#include "local_images.h"
#include "model_session.h"
#include "people.h"
#include "plate_video.h"
#include "portrait_video.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <iomanip>
#include <map>
#include <numeric>
#include <sstream>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>

namespace visual {
namespace {
namespace fs=std::filesystem;

constexpr int detectorSize=640;
constexpr float locateThreshold=.33f;
constexpr int maxEvidenceFrames=8;
constexpr std::array<std::string_view,78> alphabet{
    "", "'", "0","1","2","3","4","5","6","7","8","9","A","B","C","D","E","F","G","H","J",
    "K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z","云","京","冀","吉","学","宁",
    "川","挂","新","晋","桂","民","沪","津","浙","渝","港","湘","琼","甘","皖","粤","航","苏","蒙","藏",
    "警","豫","贵","赣","辽","鄂","闽","陕","青","鲁","黑","领","使","澳"
};
constexpr std::string_view provinces="云京冀吉宁川新晋桂沪津浙渝湘琼甘皖粤苏蒙藏豫贵赣辽鄂闽陕青鲁黑";

struct Reading {
    std::vector<int> symbols;
    std::vector<float> scores;

    float average() const {
        return scores.empty()?0.f:std::accumulate(scores.begin(),scores.end(),0.f)/static_cast<float>(scores.size());
    }
    float minimum() const {return scores.empty()?0.f:*std::ranges::min_element(scores);}
    std::string text() const {
        std::string result;
        for(const auto i:symbols) result+=alphabet[static_cast<size_t>(i)];
        return result;
    }
    bool standard() const {
        if(symbols.size()!=7 && symbols.size()!=8) return false;
        if(provinces.find(alphabet[static_cast<size_t>(symbols[0])])==std::string_view::npos) return false;
        if(symbols[1]<12 || symbols[1]>36 || symbols[1]==25) return false;
        for(size_t i=2;i<symbols.size();++i) if(symbols[i]<2 || symbols[i]>36 || symbols[i]==25) return false;
        return true;
    }
};

using Corners=std::array<cv::Point2f,4>;
struct Location {
    cv::Rect box;
    Corners corners{};
    float score{};
    bool doubleLine{};
    bool rectifiable{};
};

float overlap(const cv::Rect& a,const cv::Rect& b) {
    const auto common=(a&b).area();
    return common>0?static_cast<float>(common)/(a.area()+b.area()-common):0.f;
}

std::vector<Location> suppress(std::vector<Location> candidates) {
    std::stable_sort(candidates.begin(),candidates.end(),[](const Location& a,const Location& b){return a.score>b.score;});
    std::vector<Location> result;
    for(const auto& candidate:candidates) {
        if(std::ranges::none_of(result,[&](const Location& other){return overlap(other.box,candidate.box)>.45f;})) {
            result.push_back(candidate);
            if(result.size()>=64) break;
        }
    }
    return result;
}

cv::Mat rectify(const cv::Mat& image,const Location& location) {
    if(!location.rectifiable) return image(location.box).clone();
    const auto& corners=location.corners;
    const auto width=std::max(cv::norm(corners[0]-corners[1]),cv::norm(corners[3]-corners[2]));
    const auto height=std::max(cv::norm(corners[0]-corners[3]),cv::norm(corners[1]-corners[2]));
    if(width<10. || height<5. || width/height<.8 || width/height>7. ||
       !cv::isContourConvex(std::vector<cv::Point2f>(corners.begin(),corners.end()))) {
        return image(location.box).clone();
    }
    const cv::Size size(std::clamp(static_cast<int>(width),16,512),std::clamp(static_cast<int>(height),8,256));
    const Corners target{{{0.f,0.f},{static_cast<float>(size.width),0.f},
        {static_cast<float>(size.width),static_cast<float>(size.height)},{0.f,static_cast<float>(size.height)}}};
    cv::Mat straight;
    cv::warpPerspective(image,straight,cv::getPerspectiveTransform(corners.data(),target.data()),size,
                        cv::INTER_CUBIC,cv::BORDER_REPLICATE);
    return straight;
}

class PlateModels {
public:
    explicit PlateModels(const fs::path& directory)
        :recognizer_(directory/L"rpv3_mdict_160_r3.onnx"),root_(directory) {}

    std::vector<Location> locate(const cv::Mat& frame,bool searchTiles) {
        const cv::Rect extent(0,0,frame.cols,frame.rows);
        auto result=inferLocations(frame,extent);
        // A high-resolution source can contain plates with only a few pixels after full-frame scaling.
        if(searchTiles && std::max(frame.cols,frame.rows)>1000 && result.size()<4) {
            const int width=std::max(1,frame.cols*3/5),height=std::max(1,frame.rows*3/5);
            for(const int x:{0,frame.cols-width}) for(const int y:{0,frame.rows-height}) {
                auto local=inferLocations(frame,{x,y,width,height});
                result.insert(result.end(),local.begin(),local.end());
            }
        }
        return suppress(std::move(result));
    }

    std::vector<Location> locateNear(const cv::Mat& frame,const cv::Rect& box) {
        const auto expanded=cv::Rect(box.x-box.width/3,box.y-box.height/2,
            box.width+2*(box.width/3),box.height+2*(box.height/2))&cv::Rect(0,0,frame.cols,frame.rows);
        return suppress(inferLocations(frame,expanded));
    }

    Reading read(const cv::Mat& image,bool doubleLine) {
        if(!doubleLine || image.rows<12) return readLine(image);
        const auto split=std::clamp(image.rows*2/5,1,image.rows-1);
        auto first=readLine(image.rowRange(0,split));
        auto second=readLine(image.rowRange(split,image.rows));
        first.symbols.insert(first.symbols.end(),second.symbols.begin(),second.symbols.end());
        first.scores.insert(first.scores.end(),second.scores.begin(),second.scores.end());
        return first;
    }

private:
    std::vector<Location> inferLocations(const cv::Mat& frame,const cv::Rect& search) {
        if(!detector_) detector_=std::make_unique<ModelSession>(root_/L"y5fu_640x_sim.onnx");
        const auto ratio=std::min(float(detectorSize)/search.width,float(detectorSize)/search.height);
        const auto width=std::clamp(static_cast<int>(search.width*ratio),1,detectorSize);
        const auto height=std::clamp(static_cast<int>(search.height*ratio),1,detectorSize);
        const auto left=(detectorSize-width)/2,top=(detectorSize-height)/2;
        cv::Mat letter(detectorSize,detectorSize,CV_8UC3,cv::Scalar());
        cv::resize(frame(search),letter({left,top,width,height}),{width,height});
        auto output=detector_->run(cv::dnn::blobFromImage(letter,1./255,{}, {},true));
        if(output.size()!=1) throw std::runtime_error("车牌定位输出数量异常。");
        const auto shape=output.front().GetTensorTypeAndShapeInfo().GetShape();
        if(shape!=std::vector<int64_t>{1,25200,15}) throw std::runtime_error("车牌定位模型尺寸不匹配。");
        const auto* tensor=output.front().GetTensorData<float>();
        const auto x=[&](float value){return (value-left)/ratio+search.x;};
        const auto y=[&](float value){return (value-top)/ratio+search.y;};
        const cv::Rect extent(0,0,frame.cols,frame.rows);
        std::vector<Location> result;
        for(int64_t i=0;i<shape[1];++i) {
            const auto* row=tensor+i*shape[2];
            const bool doubleLine=row[14]>row[13];
            const float score=row[4]*row[doubleLine?14:13];
            if(!std::isfinite(score) || score<locateThreshold) continue;
            if(!std::all_of(row,row+15,[](float value){return std::isfinite(value) && std::abs(value)<4096;})) continue;
            const cv::Rect box=cv::Rect(cv::Point(cvFloor(x(row[0]-row[2]*.5f)),cvFloor(y(row[1]-row[3]*.5f))),
                cv::Point(cvCeil(x(row[0]+row[2]*.5f)),cvCeil(y(row[1]+row[3]*.5f))))&extent;
            if(box.width<14 || box.height<6 || box.width<box.height*.8 || box.width>box.height*8.) continue;
            Corners corners;
            for(size_t point=0;point<corners.size();++point) {
                corners[point]={x(row[5+point*2]),y(row[6+point*2])};
            }
            // Tile crops touching the crop boundary can truncate the plate.
            const auto validInterior=(search.x==0 || box.x>search.x+2) &&
                (search.y==0 || box.y>search.y+2) &&
                (search.br().x==frame.cols || box.br().x<search.br().x-2) &&
                (search.br().y==frame.rows || box.br().y<search.br().y-2);
            if(validInterior) result.push_back({box,corners,score,doubleLine,true});
        }
        return result;
    }

    Reading readLine(const cv::Mat& image) {
        constexpr int width=160,height=48;
        const auto resizedWidth=std::clamp(cvCeil(image.cols*static_cast<float>(height)/image.rows),48,width);
        cv::Mat padded(height,width,CV_8UC3,cv::Scalar(128,128,128));
        cv::resize(image,padded({0,0,resizedWidth,height}),{resizedWidth,height});
        auto blob=cv::dnn::blobFromImage(padded,1./127.5,{},cv::Scalar::all(127.5),false);
        for(int channel=0;channel<3;++channel) for(int y=0;y<height;++y) {
            auto* row=blob.ptr<float>(0,channel,y);
            std::fill(row+resizedWidth,row+width,0.f);
        }
        auto output=recognizer_.run(blob);
        if(output.size()!=1) throw std::runtime_error("车牌识读输出数量异常。");
        const auto shape=output.front().GetTensorTypeAndShapeInfo().GetShape();
        if(shape!=std::vector<int64_t>{1,20,static_cast<int64_t>(alphabet.size())}) {
            throw std::runtime_error("车牌识读模型词表不匹配。");
        }
        Reading result;
        const auto* tensor=output.front().GetTensorData<float>();
        size_t previous=0;
        for(int64_t i=0;i<shape[1];++i) {
            const auto* row=tensor+i*shape[2];
            const auto best=std::max_element(row,row+alphabet.size());
            const auto index=static_cast<size_t>(best-row);
            if(!std::isfinite(*best) || *best<0.f || *best>1.001f) continue;
            if(index>1 && index!=previous) {
                result.symbols.push_back(static_cast<int>(index));
                result.scores.push_back(std::min(*best,1.f));
            }
            previous=index;
        }
        return result;
    }

    ModelSession recognizer_;
    fs::path root_;
    std::unique_ptr<ModelSession> detector_;
};

struct FrameEvidence {
    int frame{};
    double seconds{};
    Location location;
    cv::Mat original,rectified;
    Reading reading;
    double quality{};
};

FrameEvidence observe(PlateModels& models,const cv::Mat& image,const Location& location,int index,double seconds) {
    auto rectified=rectify(image,location);
    auto read=models.read(rectified,location.doubleLine);
    cv::Mat gray,laplacian;
    cv::cvtColor(rectified,gray,cv::COLOR_BGR2GRAY);
    cv::Laplacian(gray,laplacian,CV_32F);
    cv::Scalar mean,deviation;
    cv::meanStdDev(laplacian,mean,deviation);
    const auto confidence=read.standard()?read.average():0.f;
    const auto quality=std::sqrt(rectified.cols*static_cast<double>(rectified.rows))*
        (1.+.02*std::min(deviation[0],30.))*(1.+confidence);
    return {index,seconds,location,image(location.box).clone(),std::move(rectified),std::move(read),quality};
}

struct Track {
    int id{},start{},last{};
    cv::Rect box;
    cv::Point2f velocity{};
    std::vector<FrameEvidence> frames;
    size_t observed{};
};

cv::Point2f center(const cv::Rect& box) {
    return {box.x+box.width*.5f,box.y+box.height*.5f};
}

void record(Track& track,FrameEvidence candidate) {
    if(track.observed>0) {
        track.velocity=(center(candidate.location.box)-center(track.box))/static_cast<float>(candidate.frame-track.last);
    } else track.start=candidate.frame;
    track.last=candidate.frame;
    track.box=candidate.location.box;
    ++track.observed;
    track.frames.push_back(std::move(candidate));
    std::stable_sort(track.frames.begin(),track.frames.end(),[](const FrameEvidence& a,const FrameEvidence& b){return a.quality>b.quality;});
    if(track.frames.size()>maxEvidenceFrames) track.frames.pop_back();
}

float association(const Track& track,const FrameEvidence& observation,int interval) {
    const int gap=observation.frame-track.last;
    if(gap<1 || gap>interval*4) return -1.f;
    const auto current=observation.location.box;
    const auto predicted=track.box+cv::Point(cvRound(track.velocity.x*gap),cvRound(track.velocity.y*gap));
    const auto ratio=current.area()/static_cast<float>(predicted.area());
    if(ratio<.35f || ratio>3.f) return -1.f;
    const auto common=std::max(overlap(predicted,current),overlap(track.box,current)*.8f);
    const auto distance=cv::norm(center(predicted)-center(current))/std::max(predicted.width,current.width);
    if(common<.15f && distance>.75) return -1.f;
    const auto& previous=track.frames.front().reading;
    const auto& read=observation.reading;
    const bool conflict=previous.standard() && read.standard() && previous.average()>.90f && read.average()>.90f &&
        previous.symbols!=read.symbols;
    if(conflict && common<.18f) return -1.f;
    return common+.30f*std::max(0.f,1.f-static_cast<float>(distance))-(conflict?.2f:0.f);
}

struct Votes {
    size_t count{};
    double total{};
    const FrameEvidence* best{};
};

Json emit(PlateModels& models,const Track& track,const fs::path& output,const fs::path& source) {
    std::map<std::string,Votes> pool;
    for(const auto& observation:track.frames) {
        const auto& read=observation.reading;
        if(!read.standard() || read.average()<.60f || read.minimum()<.25f) continue;
        auto& vote=pool[read.text()];
        vote.count++;
        vote.total+=read.average();
        if(!vote.best || read.average()>vote.best->reading.average()) vote.best=&observation;
    }
    const auto winner=std::max_element(pool.begin(),pool.end(),[](const auto& a,const auto& b) {
        return a.second.total<b.second.total;
    });
    const auto* sample=winner==pool.end()?&track.frames.front():winner->second.best;
    const auto& read=sample->reading;
    const auto enoughPixels=sample->rectified.cols>=44 && sample->rectified.rows>=12;
    const auto candidate=winner==pool.end()?std::string{}:winner->first;
    const auto contenders=std::count_if(pool.begin(),pool.end(),[&](const auto& item) {
        return item.first!=candidate && item.second.best->reading.average()>=.84f;
    });
    std::vector<cv::Mat> frames;
    for(const auto& frame:track.frames) {
        if(&frame==sample || frame.location.doubleLine!=sample->location.doubleLine) continue;
        if(frame.reading.standard() && frame.reading.average()>.82f && frame.reading.text()!=candidate) continue;
        if(frame.rectified.cols<sample->rectified.cols*.65 || frame.rectified.rows<sample->rectified.rows*.65) continue;
        cv::Mat scaled;
        cv::resize(frame.rectified,scaled,sample->rectified.size(),0,0,cv::INTER_CUBIC);
        frames.push_back(std::move(scaled));
    }

    auto restored=fusePlateFrames(sample->rectified,frames);
    std::string restoration;
    if(!restored.empty()) restoration="多帧一致性融合";
    else {
        const auto& crop=sample->rectified;
        if(crop.cols>=16 && crop.rows>=8 && crop.cols<=1024 && crop.rows<=512) {
            const auto enhanced=restorePlate(crop,{0,0,crop.cols,crop.rows});
            if(!enhanced.empty() && cv::norm(crop,enhanced.front().pixels,cv::NORM_L1)>1.8*crop.total()) {
                restored=enhanced.front().pixels;
                restoration=enhanced.front().model;
            }
        }
    }

    const bool strong=read.average()>=.97f && read.minimum()>=.80f;
    const bool multi=winner!=pool.end() && winner->second.count>=2 &&
        winner->second.total/winner->second.count>=.93 && read.minimum()>=.72f;
    const bool accepted=enoughPixels && !candidate.empty() && contenders==0 && (strong || multi);
    const std::string status=accepted?"已识读":"待核对";
    const auto prefix=L"车牌-"+std::to_wstring(track.id);
    const auto original=output/(prefix+L"-原始.png");
    writePng(original,sample->original);
    Json result{{"name",status+(candidate.empty()?"":" · "+candidate)},{"status",status},{"accepted",accepted},
        {"number",accepted?candidate:std::string{}},{"candidate_number",candidate},{"confidence",read.average()},
        {"minimum_character_confidence",read.minimum()},{"detection_score",sample->location.score},
        {"track_id",track.id},{"observations",track.observed},{"frame_index",sample->frame},{"seconds",sample->seconds},
        {"bbox",rectJson(sample->location.box)},{"source",pathText(source)},{"original",pathText(original)}};
    if(!restored.empty() || sample->location.rectifiable) {
        if(restored.empty()) {
            restored=sample->rectified;
            restoration="车牌透视校正";
        }
        const auto picture=output/(prefix+L"-还原.png");
        writePng(picture,restored);
        result["restored"]=pathText(picture);
        result["restoration"]=restoration;
        const auto after=models.read(restored,sample->location.doubleLine);
        if(after.standard() && after.average()>.70f) result["restored_candidate"]=after.text();
    }
    return result;
}

bool stillImage(const fs::path& path) {
    constexpr std::array formats{L".jpg",L".jpeg",L".png",L".bmp",L".gif",L".webp",L".tif",L".tiff",L".avif"};
    return std::ranges::any_of(formats,[&](const wchar_t* suffix){return _wcsicmp(path.extension().c_str(),suffix)==0;});
}

}

Json recognizePlates(const Json& request,const fs::path& models,const std::atomic<bool>& cancel,DetectionPreview* preview) {
    const auto source=localPath(textPath(request.at("source").get<std::string>()));
    const bool image=stillImage(source);
    const auto start=request.value("frame_index",0);
    const bool manual=request.contains("bbox");
    const auto step=request.value("step",manual?1:3);
    const auto count=request.value("count",24);
    if(start<0 || step<1 || step>120 || count<1 || count>240) throw std::invalid_argument("车牌识别帧参数无效。");
    if(image && start!=0) throw std::invalid_argument("图片的起始帧须为 0。");
    cv::Rect manualBox=manual?readRect(request.at("bbox")):cv::Rect();
    if(manual && (manualBox.width<16 || manualBox.height<8)) throw std::invalid_argument("车牌选区至少需要 16×8 像素。");
    const auto output=request.contains("output")?localPath(textPath(request.at("output").get<std::string>())):
        dataHome()/L"车牌识别结果"/wide(newId());
    PlateModels scanner(models);
    if(!output.parent_path().empty()) fs::create_directories(output.parent_path());
    if(!fs::create_directory(output)) throw std::invalid_argument("请选择新的车牌识别输出目录。");
    std::vector<Track> active;
    Json hits=Json::array();
    int nextTrack=1, sampled=0;
    cv::Mat previousCrop;
    bool limitReached=false;
    const auto finish=[&](const Track& track) {
        if(hits.size()<5000) hits.push_back(emit(scanner,track,output,source));
        else limitReached=true;
    };
    const auto inspect=[&](const cv::Mat& pixels,int frame,double seconds) {
        if(cancel || (preview && !preview->wait(cancel))) return false;
        ++sampled;
        std::vector<Location> locations;
        if(manual) {
            if(sampled==1 && (manualBox&cv::Rect(0,0,pixels.cols,pixels.rows))!=manualBox) {
                throw std::invalid_argument("车牌选区超出画面。");
            }
            if(!previousCrop.empty()) {
                manualBox=trackPlateRegion(pixels,previousCrop,manualBox);
                if(manualBox.empty()) return false;
            }
            auto plausible=scanner.locateNear(pixels,manualBox);
            const auto valid=std::ranges::find_if(plausible,[&](const Location& item){return overlap(item.box,manualBox)>.25f;});
            locations.push_back(valid==plausible.end()?Location{manualBox,{},0.f,false,false}:*valid);
            previousCrop=pixels(manualBox).clone();
        } else {
            locations=scanner.locate(pixels,image || sampled%3==1);
        }

        auto at=active.begin();
        while(at!=active.end()) {
            if(frame-at->last>step*4) {
                finish(*at);
                at=active.erase(at);
            } else ++at;
        }
        for(const auto& location:locations) {
            auto observation=observe(scanner,pixels,location,frame,seconds);
            auto selected=active.end();
            float best=-1.f;
            for(auto item=active.begin();item!=active.end();++item) {
                const auto score=association(*item,observation,step);
                if(score>best) {best=score;selected=item;}
            }
            if(manual && !active.empty()) selected=active.begin();
            if(selected==active.end()) {
                active.emplace_back();
                selected=std::prev(active.end());
                selected->id=nextTrack++;
            }
            record(*selected,std::move(observation));
        }
        if(preview && preview->refreshDue()) {
            auto marked=pixels.clone();
            for(const auto& location:locations) cv::rectangle(marked,location.box,{40,190,255},2);
            preview->update(marked,frame+1,static_cast<int>(locations.size()));
        }
        return !cancel && !limitReached && (!manual || sampled<count);
    };
    if(image) inspect(readImage(source),0,0.);
    else visitVideo(source,inspect,step,cancel,start);
    if(cancel) return {{"cancelled",true}};
    if(sampled==0) throw std::invalid_argument("视频起始帧超出范围。");
    for(const auto& track:active) finish(track);
    std::sort(hits.begin(),hits.end(),[](const Json& a,const Json& b) {
        return a.at("frame_index")<b.at("frame_index");
    });
    std::ostringstream summary;
    summary<<"共检查 "<<sampled<<(image?" 张图片":" 帧视频")<<" · "<<hits.size()<<" 个车牌结果\r\n";
    if(hits.empty()) summary<<"未定位到车牌，可框选清晰画面后重试。\r\n";
    for(const auto& hit:hits) {
        const auto candidate=hit.at("candidate_number").get<std::string>();
        summary<<"\r\n"<<hit.at("name").get<std::string>()<<" · ";
        if(!image) summary<<"第 "<<hit.at("frame_index").get<int>()+1<<" 帧 · "<<hit.at("observations").get<size_t>()<<" 帧观测 · ";
        summary<<"模型分数 "<<std::fixed<<std::setprecision(2)<<hit.at("confidence").get<double>();
        if(candidate.empty()) summary<<" · 未获得可核对号码";
    }
    Json result{{"algorithm","hyperlpr3-landmarks-ctc-track"},{"frames",sampled},{"source",pathText(source)},
        {"output",pathText(output)},{"truncated",limitReached},{"hits",std::move(hits)},{"review_text",summary.str()}};
    writeJson(output/L"车牌识别结果.json",result);
    return result;
}

}
