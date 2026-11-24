#include "media_operations.h"
#include "local_images.h"
#include "people.h"
#include "portrait_video.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cwctype>
#include <map>
#include <numeric>
#include <regex>
#include <set>

#include <opencv2/calib3d.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/imgproc.hpp>

namespace visual {
namespace fs=std::filesystem;
namespace {

std::wstring extension(const fs::path& file) {
    auto text=file.extension().wstring();
    for(auto& c:text) c=static_cast<wchar_t>(std::towlower(c));
    return text;
}

bool isImage(const fs::path& path) {
    constexpr std::array formats{L".jpg",L".jpeg",L".png",L".bmp",L".gif",L".webp",L".tif",L".tiff",L".avif"};
    return std::find(formats.begin(),formats.end(),extension(path))!=formats.end();
}

bool isVideo(const fs::path& path) {
    constexpr std::array formats{L".mp4",L".mkv",L".mov",L".avi",L".webm",L".ts",L".wmv",L".flv",L".m4v",L".mpg",L".mpeg"};
    return std::find(formats.begin(),formats.end(),extension(path))!=formats.end();
}

std::set<fs::path> sources(const Json& input) {
    const auto choices=input.contains("sources") ? input.at("sources").get<std::vector<std::string>>() :
        std::vector{input.at("source").get<std::string>()};
    std::set<fs::path> result;
    for(const auto& value:choices) {
        const auto root=localPath(textPath(value));
        if(fs::is_regular_file(root) && (isImage(root)||isVideo(root))) result.insert(root);
        else if(fs::is_directory(root)) {
            for(auto iter=fs::recursive_directory_iterator(root,fs::directory_options::skip_permission_denied);
                iter!=fs::recursive_directory_iterator();++iter) {
                try {
                    const auto path=localPath(iter->path());
                    if(iter->is_regular_file() && (isImage(path)||isVideo(path))) result.insert(path);
                } catch(const std::exception&) {iter.disable_recursion_pending();}
            }
        } else throw std::invalid_argument("素材路径没有可分析的图片或视频。");
    }
    if(result.empty()) throw std::invalid_argument("没有找到支持的图片或视频。");
    return result;
}

void sample(const fs::path& path,const FrameVisitor& visitor,int interval,const std::atomic<bool>& cancel, int firstFrame=0) {
    if(cancel) return;
    if(isImage(path)) visitor(readImage(path),0,-1.);
    else visitVideo(path,visitor,interval,cancel,firstFrame);
}

std::optional<fs::path> outputDirectory(const Json& request) {
    const auto value=request.value("output",std::string{});
    if(value.empty()) return std::nullopt;
    const auto path=localPath(textPath(value));
    if(!fs::create_directory(path)) throw std::runtime_error("请选择一个尚未存在的结果目录。");
    return path;
}

Json hit(const fs::path& path,int frame,double seconds,cv::Rect box,double score,const std::string& method) {
    return {{"source",pathText(path)},{"frame_index",frame},{"seconds",seconds>=0?Json(seconds):Json(nullptr)},
            {"bbox",rectJson(box)},{"score",score},{"method",method}};
}

void rank(Json& results,size_t limit=500) {
    std::stable_sort(results.begin(),results.end(),[](const auto& a,const auto& b){return a.at("score")>b.at("score");});
    if(results.size()>limit) results.erase(results.begin()+limit,results.end());
}

cv::Mat scaledGray(const cv::Mat& image,int maximum,double& scale) {
    scale=std::min(1.,double(maximum)/std::max(image.rows,image.cols));
    cv::Mat resized,gray;
    if(scale<1.) cv::resize(image,resized,{},scale,scale,cv::INTER_AREA);
    else resized=image;
    cv::cvtColor(resized,gray,cv::COLOR_BGR2GRAY);
    return gray;
}

class AppearanceMatcher {
public:
    explicit AppearanceMatcher(const cv::Mat& image): sift_(cv::SIFT::create(2000)) {
        if(std::min(image.cols,image.rows)<12) throw std::invalid_argument("参考选区至少需要 12×12 像素。");
        double ignored{};
        query_=scaledGray(image,480,ignored);
        cv::Scalar mean,deviation;
        cv::meanStdDev(query_,mean,deviation);
        if(deviation[0]<3) throw std::invalid_argument("参考选区缺少可辨纹理。");
        sift_->detectAndCompute(query_,cv::noArray(),points_,descriptors_);
    }

    struct Match {double score{}; cv::Rect box; const char* method{};};
    Match find(const cv::Mat& image) {
        double scale{};
        const auto frame=scaledGray(image,1600,scale);
        auto best=geometry(frame);
        for(const double factor:{.35,.5,.7,1.,1.4,2.,2.8}) {
            const cv::Size size(cvRound(query_.cols*factor),cvRound(query_.rows*factor));
            if(std::min(size.width,size.height)<12 || size.width>frame.cols || size.height>frame.rows) continue;
            cv::Mat pattern,result;
            cv::resize(query_,pattern,size);
            cv::Scalar mean,deviation;
            cv::meanStdDev(pattern,mean,deviation);
            if(deviation[0]<3) continue;
            cv::matchTemplate(frame,pattern,result,cv::TM_CCOEFF_NORMED);
            double peak{};
            cv::Point point;
            cv::minMaxLoc(result,nullptr,&peak,nullptr,&point);
            if(peak>best.score) best={peak,{point,size},"多尺度外观匹配"};
        }
        if(!best.box.empty()) {
            const auto x=cvRound(best.box.x/scale), y=cvRound(best.box.y/scale);
            best.box=cv::Rect(x,y,cvRound(best.box.br().x/scale)-x,cvRound(best.box.br().y/scale)-y)
                & cv::Rect(0,0,image.cols,image.rows);
        }
        return best;
    }
private:
    Match geometry(const cv::Mat& frame) {
        if(descriptors_.rows<6) return {};
        std::vector<cv::KeyPoint> keypoints;
        cv::Mat descriptors;
        sift_->detectAndCompute(frame,cv::noArray(),keypoints,descriptors);
        if(descriptors.rows<2) return {};
        std::vector<std::vector<cv::DMatch>> neighbours;
        cv::BFMatcher().knnMatch(descriptors_,descriptors,neighbours,2);
        std::vector<cv::Point2f> src,dst;
        for(const auto& pair:neighbours) if(pair.size()==2 && pair[0].distance<.7*pair[1].distance) {
            src.push_back(points_[pair[0].queryIdx].pt);
            dst.push_back(keypoints[pair[0].trainIdx].pt);
        }
        if(src.size()<6) return {};
        cv::Mat inliers;
        const auto matrix=cv::findHomography(src,dst,cv::RANSAC,3.,inliers);
        if(matrix.empty() || !cv::checkRange(matrix)) return {};
        std::vector<cv::Point2f> hull;
        for(size_t i=0;i<src.size();++i) if(inliers.ptr<uchar>()[i]) hull.push_back(src[i]);
        if(hull.size()<6) return {};
        const auto count=hull.size();
        std::vector<cv::Point2f> outline;
        cv::convexHull(hull,outline);
        if(cv::contourArea(outline)<.04*query_.total()) return {};
        const float w=static_cast<float>(query_.cols),h=static_cast<float>(query_.rows);
        const std::vector<cv::Point2f> corners{{0,0},{w,0},{w,h},{0,h}};
        std::vector<cv::Point2f> mapped;
        cv::perspectiveTransform(corners,mapped,matrix);
        for(const auto& pixel:mapped) if(!std::isfinite(pixel.x)||!std::isfinite(pixel.y)) return {};
        const auto area=cv::contourArea(mapped);
        if(!cv::isContourConvex(mapped)||area<144||area>1.1*frame.total()) return {};
        return {std::min(.99,.5+.25*count/src.size()+.25*std::min(count/20.,1.)),
            cv::boundingRect(mapped)&cv::Rect(0,0,frame.cols,frame.rows),"局部特征与几何校验"};
    }
    cv::Ptr<cv::SIFT> sift_;
    cv::Mat query_,descriptors_;
    std::vector<cv::KeyPoint> points_;
};

Json faceConfig() {
    const auto path=dataHome()/L"config.json";
    return fs::exists(path)?readJson(path):Json::object();
}

Json visualSearch(const Json& request,const fs::path& models,const std::atomic<bool>& cancel,
                    DetectionPreview* preview) {
    auto query=readImage(textPath(request.at("input").get<std::string>()));
    if(request.contains("bbox")) {
        const auto rect=readRect(request.at("bbox"));
        if((rect&cv::Rect(0,0,query.cols,query.rows))!=rect) throw std::invalid_argument("选区超出参考图。");
        query=query(rect).clone();
    }
    const bool face=request.value("mode",std::string("appearance"))=="face";
    const double threshold=std::clamp(request.value("threshold",face?.45:.75),0.,1.);
    std::unique_ptr<AppearanceMatcher> appearance;
    std::unique_ptr<FaceEngine> engine;
    Feature target;
    if(face) {
        engine=std::make_unique<FaceEngine>(models,FaceSettings::from(faceConfig(),models));
        target=engine->reference(query).feature;
    } else appearance=std::make_unique<AppearanceMatcher>(query);
    Json matches=Json::array(),failures=Json::array();
    int processed=0;
    for(const auto& source:sources(request)) {
        if(cancel) break;
        try {sample(source,[&](const cv::Mat& pixels,int frame,double seconds){
            const auto prior=matches.size();
            if(face) {
                for(const auto& found:engine->faces(pixels)) {
                    const auto score=similarity(target,found.feature);
                    if(score>=threshold) matches.push_back(hit(source,frame,seconds,found.box,score,"人脸特征相似候选"));
                }
            } else {
                const auto match=appearance->find(pixels);
                if(match.score>=threshold && !match.box.empty()) matches.push_back(hit(source,frame,seconds,match.box,match.score,match.method));
            }
            ++processed;
            if(preview && preview->refreshDue() && seconds>=0 && !cancel) {
                preview->update(pixels,processed,static_cast<int>(matches.size()-prior));
            }
            if(matches.size()>1000) rank(matches);
            return !cancel;
        },request.value("step",1),cancel);} catch(const std::exception& error) {
            failures.push_back({{"source",pathText(source)},{"error",error.what()}});
        }
    }
    rank(matches);
    return {{"hits",matches},{"failures",failures},{"cancelled",cancel.load()}};
}

Json objectScan(const Json& request,const fs::path& models,const std::atomic<bool>& cancel,
                DetectionPreview* preview) {
    ObjectDetector detector(models);
    const auto threshold=std::clamp(request.value("threshold",.5),.05,.99);
    const bool people=request.value("people",true),vehicles=request.value("vehicles",true);
    if(!people && !vehicles) throw std::invalid_argument("请至少选择人或车辆。");
    if(const auto video=request.value("video_output",std::string{});!video.empty()) {
        if(!request.value("output",std::string{}).empty()) throw std::invalid_argument("请分开导出标注视频和目标截图。");
        VideoProgress progress;
        progress.externalCancel=&cancel;
        const auto first=localPath(textPath(request.at("source").get<std::string>()));
        const auto target=localPath(textPath(video));
        renderObjectVideo(first,target,[&](const cv::Mat& frame){
            if(preview && !preview->wait(cancel)) return std::pair{frame,0};
            const auto found=detector.detect(frame,threshold,people,vehicles);
            auto marked=markObjects(frame,found);
            const auto count=static_cast<int>(found.size());
            if(preview && preview->refreshDue()) preview->update(marked,progress.frames.load()+1,count);
            return std::pair{std::move(marked),count};
        },progress);
        return {{"frames",progress.frames.load()},{"detections",progress.faces.load()},{"cancelled",cancel.load()},
                {"output",cancel.load()?Json(nullptr):Json(pathText(target))}};
    }
    const auto output=outputDirectory(request);
    int n=0;
    Json results=Json::array(),failures=Json::array();
    for(const auto& source:sources(request)) {
        if(cancel) break;
        try {sample(source,[&](const cv::Mat& frame,int index,double seconds){
            if(preview && !preview->wait(cancel)) return false;
            ++n;
            auto detections=detector.detect(frame,threshold,people,vehicles);
            for(size_t i=0;i<detections.size();++i) {
                if(cancel) return false;
                const auto& found=detections[i];
                auto item=hit(source,index,seconds,found.box,found.score,objectName(found.category));
                if(output) {
                    const auto path=*output/(std::to_wstring(n)+L"-"+std::to_wstring(i+1)+L".png");
                    writePng(path,frame(found.box));
                    item["image"]=pathText(path);
                }
                if(results.size()<2000) results.push_back(std::move(item));
            }
            const bool saveMarked=output && request.value("marked_frames",false);
            const bool refresh=preview && preview->refreshDue();
            if(refresh || saveMarked) {
                const auto marked=markObjects(frame,detections);
                if(refresh) preview->update(marked,n,static_cast<int>(detections.size()));
                if(saveMarked) writePng(*output/(std::to_wstring(n)+L"-marked.png"),marked);
            }
            return !cancel;
        },request.value("step",1),cancel);} catch(const std::exception& error) {
            failures.push_back({{"source",pathText(source)},{"error",error.what()}});
        }
    }
    if(output) writeJson(*output/L"检测结果.json",results);
    return {{"frames",n},{"hits",results},{"failures",failures},{"cancelled",cancel.load()}};
}

Json faceScan(const Json& request,const fs::path& models,const std::atomic<bool>& cancel,
              DetectionPreview* preview) {
    const auto stored=faceConfig();
    const auto config=FaceSettings::from(stored,models);
    FaceEngine engine(models,config);
    Database db(dataHome()/L"data/faces.db");
    initializePeople(db);
    const auto gallery=PersonFilter::from(request);
    const auto interval=request.value("step",std::max(1,stored.value("frame_interval",1)));
    const auto minimum=std::max(1,stored.value("min_track_observations",3));
    const auto confirmations=std::max(1,stored.value("confirmation_matches",2));
    const auto session=newId();
    const auto snapshots=dataHome()/L"data/annotations"/wide(session);
    Json hits=Json::array(),errors=Json::array();
    int trackId=0,processed=0;
    for(const auto& path:sources(request)) {
        if(cancel) break;
        struct Track {
            cv::Rect box;
            int id{},last{},count{},confirmed{};
            std::string person;
            bool emitted{};
        };
        std::vector<Track> tracks;
        try { sample(path,[&](const cv::Mat& pixels,int frame,double seconds){
            if(preview && !preview->wait(cancel)) return false;
            std::erase_if(tracks,[&](const auto& item){return frame-item.last>12*interval;});
            const auto observations=engine.faces(pixels);
            for(const auto& observation:observations) {
                if(cancel) return false;
                const auto iou=[&](const Track& t){
                    const double intersection=(observation.box&t.box).area();
                    return t.last==frame ? 0. : intersection/std::max(1.,double(observation.box.area())+t.box.area()-intersection);
                };
                auto track=std::max_element(tracks.begin(),tracks.end(),[&](const auto& a,const auto& b){return iou(a)<iou(b);});
                if(track==tracks.end() || iou(*track)<.22) {
                    tracks.push_back({observation.box,++trackId,frame,0,0,{},false});
                    track=std::prev(tracks.end());
                }
                track->last=frame;
                track->box=observation.box;
                ++track->count;
                const auto found=matchGallery(db,observation.feature,config,gallery);
                if(found.empty()) continue;
                const auto& best=found.front();
                const auto person=best.at("person_id").get<std::string>();
                if(track->person!=person) {track->person=person;track->confirmed=0;track->emitted=false;}
                const auto score=best.at("score").get<double>();
                if(best.value("accepted",false)) ++track->confirmed;
                if(track->emitted || (isVideo(path) && track->count<minimum)) continue;
                if(hits.size()>=500) return false;
                auto result=hit(path,frame,seconds,observation.box,score,"人员库相似候选");
                const bool accepted=best.value("accepted",false)&&(isImage(path)||track->confirmed>=confirmations);
                if(accepted) db.query("INSERT INTO recognition_events(person_id,source,track_id,score,quality,occurred_at) VALUES(?,?,?,?,?,?)",
                    {person,pathText(path),track->id,score,observation.quality,utcTimestamp()});
                result["candidate"]=best;
                result["accepted"]=accepted;
                result["session_id"]=session;
                result["track_id"]=track->id;
                result["model_id"]=config.modelId();
                result["quality"]=observation.quality;
                fs::create_directories(snapshots);
                const auto stem=std::to_wstring(track->id)+L"-"+std::to_wstring(frame);
                writePng(snapshots/(stem+L"-frame.png"),pixels);
                writePng(snapshots/(stem+L"-face.png"),pixels(observation.box));
                result["snapshot_path"]=pathText(snapshots/(stem+L"-frame.png"));
                result["face_path"]=pathText(snapshots/(stem+L"-face.png"));
                hits.push_back(std::move(result));
                track->emitted=true;
            }
            ++processed;
            if(preview && preview->refreshDue() && !cancel) {
                auto marked=pixels.clone();
                for(const auto& observation:observations) cv::rectangle(marked,observation.box,{255,200,70},2);
                preview->update(marked,processed,static_cast<int>(observations.size()));
            }
            return !cancel;
        },interval,cancel);} catch(const std::exception& error) {
            errors.push_back({{"source",pathText(path)},{"error",error.what()}});
        }
    }
    return {{"session_id",session},{"hits",hits},{"failures",errors},{"cancelled",cancel.load()}};
}

constexpr std::array colorNames{L"黑色",L"白色",L"灰色",L"蓝色",L"红色",L"绿色",L"黄色",L"棕色",L"紫色",L"粉色",L"橙色"};
using Colors=std::array<float,colorNames.size()>;
using ObjectFeatures=std::array<Feature,4>;

struct Clause {
    size_t color, part;
    Feature text;
};

struct Description {
    Feature full;
    bool person=true;
    std::vector<Clause> clauses;
    std::optional<std::pair<Feature,Feature>> hat;
};

Description describe(const Json& payload,SemanticEncoder& encoder) {
    const auto query=wide(payload.at("text").get<std::string>());
    if(query.empty() || query.size()>160) throw std::invalid_argument("请输入 1～160 字的检测目标描述。");
    const std::wregex negative(L"不穿|不是|没有|不戴|非白|非黑|除了|或者|或是");
    if(std::regex_search(query,negative)) throw std::invalid_argument("请使用一种肯定的衣着或车辆描述。");
    const bool vehicle=std::regex_search(query,std::wregex(L"车|汽车|轿车|货车|公交|摩托"));
    const bool person=std::regex_search(query,std::wregex(L"人|帽|衣|裤|裙|外套|夹克|衬衫|短袖|卫衣|T恤",std::regex_constants::icase));
    const auto type=payload.value("category",std::string("auto"));
    if((type=="auto" && vehicle && person) || (type=="person" && vehicle && !person) || (type=="vehicle" && person && !vehicle)) {
        throw std::invalid_argument("请将人物衣着和车辆分开搜索。");
    }
    if(type!="auto" && type!="person" && type!="vehicle") throw std::invalid_argument("未知语义目标类型。");
    Description result;
    result.person=type=="person" || (type=="auto" && !vehicle);
    const std::wstring keywords=result.person?L"帽子|帽|上衣|衣服|外套|夹克|衬衫|短袖|T恤|卫衣|裤子|长裤|短裤|裤|裙子|裙":
                                              L"车辆|汽车|轿车|货车|公交车|摩托车|自行车|车";
    const std::wregex clauses(L"(黑|白|灰|蓝|红|绿|黄|棕|紫|粉|橙)色?的?\\s*("+keywords+L")",std::regex_constants::icase);
    std::map<size_t,size_t> assigned;
    for(auto occurrence=std::wsregex_iterator(query.begin(),query.end(),clauses);occurrence!=std::wsregex_iterator();++occurrence) {
        const auto color=(*occurrence)[1].str()+L"色";
        const auto noun=(*occurrence)[2].str();
        const auto index=size_t(std::find(colorNames.begin(),colorNames.end(),color)-colorNames.begin());
        const size_t part=!result.person?0:noun.find(L"帽")!=std::wstring::npos?1:
            noun.find_first_of(L"裤裙")!=std::wstring::npos?3:2;
        if(assigned.contains(part) && assigned[part]!=index) throw std::invalid_argument("同一区域出现多种颜色，请分开检索。");
        if(assigned.contains(part)) continue;
        assigned[part]=index;
        result.clauses.push_back({index,part,encoder.text(utf8(color+noun))});
    }
    if(assigned.contains(1)) result.hat=std::pair{encoder.text("戴着帽子的人"),encoder.text("没有戴帽子的人")};
    result.full=encoder.text(utf8(query));
    return result;
}

Colors localColors(const cv::Mat& bgr,const std::array<double,4>& coordinates) {
    const int left=int(coordinates[0]*bgr.cols), top=int(coordinates[2]*bgr.rows);
    const int right=std::max(left+1,int(coordinates[1]*bgr.cols));
    const int bottom=std::max(top+1,int(coordinates[3]*bgr.rows));
    cv::Mat hsv;
    cv::cvtColor(bgr(cv::Rect(left,top,right-left,bottom-top)),hsv,cv::COLOR_BGR2HSV);
    Colors histogram{};
    for(int y=0;y<hsv.rows;++y) for(int x=0;x<hsv.cols;++x) {
        const auto pixel=hsv.at<cv::Vec3b>(y,x);
        const int h=pixel[0];
        const double s=pixel[1]/255.,v=pixel[2]/255.;
        const bool c=s>=.22 && v>.18;
        const std::array masks{v<=.27,s<.22&&v>=.72,s<.22&&v>.27&&v<.72,c&&h>=90&&h<130,
            c&&(h<10||h>=170)&&v>=.35,c&&h>=35&&h<90,c&&h>=20&&h<35,c&&h>=5&&h<30&&v<.65,
            c&&h>=130&&h<155,c&&h>=155&&h<175&&v>=.5,c&&h>=10&&h<20&&v>=.65};
        for(size_t i=0;i<masks.size();++i) if(masks[i]) ++histogram[i];
    }
    for(auto& number:histogram) number/=static_cast<float>(hsv.total());
    return histogram;
}

Feature objectFeatures(const cv::Mat& object,bool person,SemanticEncoder& encoder) {
    constexpr std::array spans{std::pair{0.,1.},std::pair{0.,.30},std::pair{.18,.62},std::pair{.52,1.}};
    constexpr std::array colors{std::array{.1,.9,.1,.9},std::array{.2,.8,.0,.23},
        std::array{.2,.8,.25,.55},std::array{.25,.75,.58,.85}};
    Feature all;
    all.reserve(4*(512+colorNames.size()));
    const auto entire=encoder.image(object);
    const auto entireColors=localColors(object,colors[0]);
    for(size_t i=0;i<4;++i) {
        Feature feature=entire;
        auto cues=entireColors;
        if(i>0 && person) {
            const int start=int(object.rows*spans[i].first);
            const int end=std::max(start+1,int(object.rows*spans[i].second));
            feature=encoder.image(object.rowRange(start,end));
            cues=localColors(object,colors[i]);
        }
        all.insert(all.end(),feature.begin(),feature.end());
        all.insert(all.end(),cues.begin(),cues.end());
    }
    return all;
}

std::optional<double> semanticScore(const Description& query,const Feature& values) {
    constexpr size_t stride=512+colorNames.size();
    const auto region=[&](size_t id) {return std::span<const float>(values.data()+id*stride,512);};
    const auto whole=similarity(query.full,region(0));
    if(query.hat && similarity(query.hat->first,region(1))<similarity(query.hat->second,region(1))+.005) return std::nullopt;
    std::vector<double> pieces;
    for(const auto& clause:query.clauses) {
        const double minimum=clause.color==1&&query.person?.55:clause.color<3?.18:.10;
        if(values[clause.part*stride+512+clause.color]<minimum) return std::nullopt;
        pieces.push_back(similarity(clause.text,region(clause.part)));
    }
    if(pieces.empty()) return whole;
    return .4*whole+.4*std::accumulate(pieces.begin(),pieces.end(),0.)/pieces.size()
        +.2**std::min_element(pieces.begin(),pieces.end());
}

std::string signature(const fs::path& source) {
    return std::to_string(fs::file_size(source))+":"+std::to_string(fs::last_write_time(source).time_since_epoch().count());
}

Json semanticSearch(const Json& request,const fs::path& models,const std::atomic<bool>& cancel,
                    DetectionPreview* preview) {
    SemanticEncoder encoder(models);
    const auto query=describe(request,encoder);
    const auto interval=std::clamp(request.value("step",25),1,10000);
    const auto threshold=std::clamp(request.value("threshold",.2),-1.,1.);
    auto indexPath=dataHome()/L"data/semantic-index.db";
    Database index(indexPath);
    index.script(R"sql(
        CREATE TABLE IF NOT EXISTS sources(id INTEGER PRIMARY KEY,path TEXT NOT NULL,signature TEXT NOT NULL,
            pipeline TEXT NOT NULL,step INTEGER NOT NULL,UNIQUE(path,pipeline,step));
        CREATE TABLE IF NOT EXISTS objects(source_id INTEGER NOT NULL REFERENCES sources(id) ON DELETE CASCADE,
            frame_index INTEGER NOT NULL,seconds REAL,bbox TEXT NOT NULL,category TEXT NOT NULL,features BLOB NOT NULL);
        CREATE INDEX IF NOT EXISTS object_sources ON objects(source_id,category);
    )sql");
    constexpr auto imagePipeline="chinese-clip-b16-felixdu-47080d16-fp32-parts-v1:objects-native-v1";
    const auto videoPipeline=std::string(imagePipeline)+":video-frame-reader-v2";
    std::unique_ptr<ObjectDetector> detector;
    Json results=Json::array(),failures=Json::array();
    int reused=0,indexed=0,processed=0;
    for(const auto& source:sources(request)) {
        if(cancel) break;
        try {
            const std::string pipeline=isVideo(source)?videoPipeline:imagePipeline;
            const auto stamp=signature(source);
            auto cached=index.query("SELECT id,signature FROM sources WHERE path=? AND pipeline=? AND step=?",
                {pathText(source),pipeline,interval});
            int64_t id{};
            if(!cached.empty() && cached.front()["signature"]==stamp) {
                id=cached.front()["id"].get<int64_t>();
                ++reused;
            } else {
                if(!detector) detector=std::make_unique<ObjectDetector>(models);
                Transaction batch(index);
                index.query("DELETE FROM sources WHERE path=? AND pipeline IN (?,?) AND step=?",
                    {pathText(source),pipeline,imagePipeline,interval});
                index.query("INSERT INTO sources(path,signature,pipeline,step) VALUES(?,?,?,?)",
                    {pathText(source),stamp,pipeline,interval});
                id=index.lastId();
                sample(source,[&](const cv::Mat& frame,int frameIndex,double seconds){
                    auto detected=detector->detect(frame,.35,true,true);
                    for(const auto& item:detected) {
                        if(cancel) return false;
                        const auto person=item.category==0;
                        const auto feature=objectFeatures(frame(item.box),person,encoder);
                        index.query("INSERT INTO objects VALUES(?,?,?,?,?,?)",{id,frameIndex,seconds<0?Json(nullptr):Json(seconds),
                            rectJson(item.box).dump(),person?"person":"vehicle",featureBlob(feature)});
                    }
                    ++processed;
                    if(preview && preview->refreshDue() && seconds>=0 && !cancel) {
                        std::erase_if(detected,[&](const auto& item){return (item.category==0)!=query.person;});
                        preview->update(markObjects(frame,detected),processed,static_cast<int>(detected.size()));
                    }
                    return !cancel;
                },interval,cancel);
                if(cancel) break;
                if(signature(source)!=stamp) throw std::runtime_error("分析期间源文件发生变化，请重试。");
                batch.commit();
                ++indexed;
            }
            const auto entries=index.query("SELECT frame_index,seconds,bbox,features FROM objects WHERE source_id=? AND category=?",
                {id,query.person?"person":"vehicle"});
            for(const auto& item:entries) {
                if(cancel) break;
                const auto feature=readFeature(item.at("features"),4*(512+colorNames.size()));
                if(feature.empty()) continue;
                const auto score=semanticScore(query,feature);
                if(score && *score>=threshold) {
                    results.push_back(hit(source,item["frame_index"].get<int>(),item["seconds"].is_number()?item["seconds"].get<double>():-1,
                        readRect(Json::parse(item["bbox"].get<std::string>())),*score,"中文语义与区域颜色"));
                }
                if(results.size()>1000) rank(results);
            }
        } catch(const std::exception& error) {failures.push_back({{"source",pathText(source)},{"error",error.what()}});}
    }
    rank(results);
    return {{"hits",results},{"failures",failures},{"reused_sources",reused},{"indexed_sources",indexed},{"cancelled",cancel.load()}};
}

Json exportHits(const Json& request,const std::atomic<bool>& cancel) {
    const auto directory=outputDirectory(request);
    if(!directory) throw std::invalid_argument("请选择截图输出目录。");
    const auto& results=request.at("hits");
    std::map<fs::path,std::map<int,std::vector<std::pair<size_t,cv::Rect>>>> groups;
    for(size_t i=0;i<results.size();++i) {
        const auto& result=results[i];
        const auto source=localPath(textPath(result.at("source").get<std::string>()));
        groups[source][result.at("frame_index").get<int>()].push_back({i,readRect(result.at("bbox"))});
    }
    size_t count=0;
    auto failures=Json::array();
    for(const auto& [source,frames]:groups) {
        if(cancel) break;
        try {
            sample(source,[&](const cv::Mat& pixels,int frame,double) {
                if(const auto found=frames.find(frame);found!=frames.end()) {
                    for(const auto& [index,rect]:found->second) {
                        const auto bounded=rect&cv::Rect(0,0,pixels.cols,pixels.rows);
                        if(bounded.empty()) continue;
                        writePng(*directory/(L"目标-"+std::to_wstring(index+1)+L".png"),pixels(bounded));
                        ++count;
                    }
                }
                return frame<frames.rbegin()->first && !cancel;
            },1,cancel,frames.begin()->first);
        } catch(const std::exception& error) {failures.push_back({{"source",pathText(source)},{"error",error.what()}});}
    }
    return {{"screenshots",count},{"output",pathText(*directory)},{"failed",failures},{"cancelled",cancel.load()}};
}

}

Json mediaOperation(const std::string& action,const Json& request,const fs::path& models,const std::atomic<bool>& cancel,DetectionPreview* preview) {
    if(action=="visual.search") return visualSearch(request,models,cancel,preview);
    if(action=="objects.scan") return objectScan(request,models,cancel,preview);
    if(action=="face.scan") return faceScan(request,models,cancel,preview);
    if(action=="semantic.search") return semanticSearch(request,models,cancel,preview);
    if(action=="hits.export") return exportHits(request,cancel);
    throw std::invalid_argument("未知多媒体操作。");
}

}
