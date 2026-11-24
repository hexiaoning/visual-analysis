#include "vehicle.h"
#include "local_images.h"
#include "operations.h"
#include "portrait_video.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cwctype>
#include <map>
#include <numeric>

#include <Windows.h>
#include <wincrypt.h>
#include <opencv2/imgcodecs.hpp>

namespace visual {
namespace fs=std::filesystem;
namespace {

constexpr std::array fields{"brand","series","model","year","trim","catalogue_id"};
constexpr std::array starters{
    std::pair{"比亚迪","海鸥"},std::pair{"比亚迪","秦 PLUS"},std::pair{"比亚迪","宋 PLUS"},std::pair{"比亚迪","汉"},
    std::pair{"特斯拉","Model 3"},std::pair{"特斯拉","Model Y"},std::pair{"小米","SU7"},std::pair{"小米","YU7"},
    std::pair{"理想","L6"},std::pair{"理想","L9"},std::pair{"问界","M7"},std::pair{"问界","M9"},
    std::pair{"吉利","星愿"},std::pair{"五菱","宏光 MINIEV"},std::pair{"丰田","卡罗拉"},std::pair{"丰田","凯美瑞"},
    std::pair{"本田","思域"},std::pair{"大众","朗逸"},std::pair{"福特","Mustang"},std::pair{"本田","雅阁"},
    std::pair{"日产","轩逸"},std::pair{"大众","速腾"},std::pair{"宝马","3 系"},std::pair{"奔驰","C 级"},
};

Json candidate(const Json& input) {
    Json result=Json::object();
    for(const auto* key:fields) {
        const auto value=input.value(key,std::string{});
        const auto trimmed=value.find_first_not_of(" \t\r\n");
        if(value.size()>800 || wide(value).size()>200 || value.find('\0')!=std::string::npos) {
            throw std::invalid_argument("车型文字字段超过长度上限。");
        }
        result[key]=trimmed==std::string::npos ? std::string{} : value.substr(trimmed,value.find_last_not_of(" \t\r\n")-trimmed+1);
    }
    if(result["brand"]=="" || result["series"]=="") throw std::invalid_argument("车型资料需要品牌和车系。");
    return result;
}

bool imagePath(const fs::path& path) {
    auto extension=path.extension().wstring();
    std::transform(extension.begin(),extension.end(),extension.begin(),[](wchar_t c){return wchar_t(std::towlower(c));});
    constexpr std::array formats{L".jpg",L".jpeg",L".png",L".bmp",L".webp",L".tif",L".tiff",L".gif",L".avif"};
    return std::find(formats.begin(),formats.end(),extension)!=formats.end();
}

double intersection(cv::Rect a,cv::Rect b) {
    const double overlap=(a&b).area();
    return overlap/std::max(1.,double(a.area())+b.area()-overlap);
}

std::vector<Feature> vehicleViews(const fs::path& path,const fs::path& models,SemanticEncoder& encoder,
                                  const std::atomic<bool>& cancel,DetectionPreview* preview) {
    ObjectDetector detector(models);
    cv::Rect previous;
    int missed=0,processed=0;
    std::vector<Feature> views;
    const bool video=!imagePath(path);
    const auto examine=[&](const cv::Mat& frame,int,double) {
        ++processed;
        auto objects=detector.detect(frame,.5,false,true);
        std::erase_if(objects,[](const auto& object){return object.category!=2 && object.category!=3 && object.category!=5 && object.category!=7;});
        const auto select=[&](const ObjectObservation& object) {return previous.empty()?double(object.box.area()):intersection(previous,object.box);};
        const auto found=std::max_element(objects.begin(),objects.end(),[&](const auto& a,const auto& b){return select(a)<select(b);});
        if(found==objects.end() || (!previous.empty() && select(*found)<.15)) {
            if(video && preview && preview->refreshDue() && !cancel) preview->update(frame,processed,0);
            return !cancel && (previous.empty() || ++missed<=3);
        }
        previous=found->box;
        missed=0;
        views.push_back(encoder.image(frame(previous)));
        if(video && preview && preview->refreshDue() && !cancel) preview->update(frame,processed,1);
        return views.size()<8 && !cancel;
    };
    if(video) visitVideo(path,examine,5,cancel);
    else examine(readImage(path),0,0.);
    return views;
}

double scores(const std::vector<Feature>& views,const std::vector<Feature>& references) {
    double sum{};
    for(const auto& image:views) {
        double best=-1.;
        for(const auto& ref:references) best=std::max(best,similarity(image,ref));
        sum+=best;
    }
    return sum/views.size();
}

Json localMatches(const Json& request,const std::vector<Feature>& views,SemanticEncoder& encoder,const std::atomic<bool>& cancel) {
    Json result=Json::array();
    const auto catalogue=request.value("catalogue",std::string{});
    if(!catalogue.empty()) {
        const auto path=localPath(textPath(catalogue));
        const auto data=readJson(path);
        if(data.value("schema_version",0)!=1 || !data.at("vehicles").is_array()) {
            throw std::invalid_argument("车型参考库须使用 schema_version 1。");
        }
        for(const auto& entry:data.at("vehicles")) {
            if(cancel) return Json::array();
            const auto vehicle=candidate(entry);
            std::vector<Feature> features;
            for(const auto& relative:entry.at("images")) {
                if(cancel) return Json::array();
                const auto fragment=textPath(relative.get<std::string>());
                if(fragment.has_root_path()) throw std::invalid_argument("参考图片须在参考库目录内。");
                const auto file=localPath(path.parent_path()/fragment);
                const auto suffix=file.lexically_relative(path.parent_path());
                if(suffix.empty() || *suffix.begin()==L".." || !imagePath(file)) throw std::invalid_argument("参考图片须在参考库目录内。");
                features.push_back(encoder.image(readImage(file)));
            }
            if(!features.empty()) result.push_back({{"candidate",vehicle},{"similarity",scores(views,features)}});
        }
    } else if(!request.contains("manual")) {
        for(const auto& [brand,series]:starters) {
            if(cancel) return Json::array();
            const auto score=scores(views,{encoder.text(std::string("一辆")+brand+" "+series+"汽车")});
            result.push_back({{"candidate",candidate({{"brand",brand},{"series",series}})},{"similarity",score}});
        }
    }
    std::stable_sort(result.begin(),result.end(),[](auto& a,auto& b){return a["similarity"]>b["similarity"];});
    if(result.size()>3) result.erase(result.begin()+3,result.end());
    if(request.contains("manual")) {
        const auto manual=candidate(request.at("manual"));
        std::erase_if(result.get_ref<Json::array_t&>(),[&](const auto& entry){return entry.at("candidate")==manual;});
        result.insert(result.begin(),Json{{"candidate",manual},{"similarity",nullptr}});
        if(result.size()>3) result.erase(result.begin()+3,result.end());
    }
    return result;
}

std::wstring quoted(const fs::path& value) {
    const auto path=value.wstring();
    if(path.find(L'"')!=std::wstring::npos || path.ends_with(L"\\")) throw std::invalid_argument("进程参数路径无效。");
    return L"\""+path+L"\"";
}

Json sourceFailures(const char* status,const std::string& time={}) {
    Json sources=Json::array();
    for(const auto* name:{"wikimedia","nhtsa_vpic"}) sources.push_back(
        {{"source",name},{"queried_at",time},{"status",status},{"records",Json::array()}});
    return sources;
}

Json queryWorker(const fs::path& worker,const Json& matches,bool broad,const std::atomic<bool>& cancel) {
    if(matches.empty()) return sourceFailures("no_candidate");
    const auto root=dataHome()/L"data";
    fs::create_directories(root);
    auto settings=Json::object();
    const auto options=dataHome()/L"vehicle-providers.json";
    if(fs::is_regular_file(options)) {
        try {settings=readJson(options,65536);} catch(const std::exception&) {}
    }
    Json candidates=Json::array();
    for(const auto& match:matches) candidates.push_back(match["candidate"]);
    auto publicCandidates=candidates;
    if(broad) {
        for(const auto& [brand,series]:starters) {
            const auto record=candidate({{"brand",brand},{"series",series}});
            if(std::find(publicCandidates.begin(),publicCandidates.end(),record)==publicCandidates.end()) publicCandidates.push_back(record);
        }
    }
    const auto work=localPath(root/(L".catalogue-"+wide(newId())));
    if(!fs::create_directory(work)) throw std::runtime_error("创建车型核验目录失败。");
    const auto request=work/L"request.json", reply=work/L"response.json";
    struct Cleanup {
        fs::path request,reply,root;
        ~Cleanup() {
            std::error_code ignored;
            fs::remove(request,ignored);fs::remove(reply,ignored);fs::remove(root,ignored);
        }
    } cleanup{request,reply,work};
    writeJson(request,{{"settings",settings},{"candidates",candidates},{"public_candidates",publicCandidates}});
    const auto executable=localPath(worker);
    auto command=quoted(executable)+L" "+quoted(request)+L" "+quoted(reply);
    STARTUPINFOW startup{};
    startup.cb=sizeof(startup);
    PROCESS_INFORMATION process{};
    const auto time=utcTimestamp();
    if(!CreateProcessW(executable.c_str(),command.data(),nullptr,nullptr,FALSE,CREATE_NO_WINDOW,
                       nullptr,work.c_str(),&startup,&process)) return sourceFailures("unavailable",time);
    CloseHandle(process.hThread);
    const auto deadline=GetTickCount64()+65000;
    DWORD state{};
    while((state=WaitForSingleObject(process.hProcess,100))==WAIT_TIMEOUT) {
        if(cancel || GetTickCount64()>deadline) {
            TerminateProcess(process.hProcess,2);
            WaitForSingleObject(process.hProcess,1000);
            break;
        }
    }
    DWORD status=2;
    GetExitCodeProcess(process.hProcess,&status);
    CloseHandle(process.hProcess);
    if(state!=WAIT_OBJECT_0 || status!=0) return sourceFailures("unavailable",time);
    try { return readJson(reply,28*1024*1024); } catch(const std::exception&) {return sourceFailures("unavailable",time);}
}

std::vector<Feature> decodeReferences(const Json& record,SemanticEncoder& encoder) {
    std::vector<Feature> result;
    for(const auto& item:record.value("image_jpeg_base64",Json::array())) {
        if(!item.is_string()) continue;
        const auto& encoded=item.get_ref<const std::string&>();
        if(encoded.size()>1024*1024) continue;
        DWORD bytes{};
        if(!CryptStringToBinaryA(encoded.data(),DWORD(encoded.size()),CRYPT_STRING_BASE64|CRYPT_STRING_STRICT,
                                nullptr,&bytes,nullptr,nullptr) || bytes<3 || bytes>768*1024) continue;
        std::vector<uchar> image(bytes);
        if(!CryptStringToBinaryA(encoded.data(),DWORD(encoded.size()),CRYPT_STRING_BASE64|CRYPT_STRING_STRICT,
           image.data(),&bytes,nullptr,nullptr) || image[0]!=0xff || image[1]!=0xd8) continue;
        try {
            const auto pixels=cv::imdecode(image,cv::IMREAD_REDUCED_COLOR_2);
            if(!pixels.empty() && pixels.total()<=16*1024*1024) result.push_back(encoder.image(pixels));
        } catch(const std::exception&) {}
    }
    return result;
}

}

Json analyzeVehicle(const Json& request,const fs::path& models,const fs::path& worker,
                    const std::atomic<bool>& cancel,DetectionPreview* preview) {
    auto source=localPath(textPath(request.at("input").get<std::string>()));
    SemanticEncoder encoder(models);
    const auto views=vehicleViews(source,models,encoder,cancel,preview);
    if(views.empty() && !cancel) throw std::runtime_error("未找到可用于比对的车辆，请使用车身可见的图片或视频。");
    if(cancel) return Json::object();
    auto matches=localMatches(request,views,encoder,cancel);
    if(cancel) return Json::object();
    const bool broad=!request.contains("manual") && request.value("catalogue",std::string{}).empty();
    auto sources=queryWorker(worker,matches,broad,cancel);
    if(cancel) return Json::object();
    int matching=0;
    bool reached=false;
    int references=0;
    Json visualMatches=Json::array();
    for(auto& sourceResult:sources) {
        reached|=sourceResult.at("status")=="ok" || sourceResult.at("status")=="no_match";
        bool matched=false;
        for(auto& record:sourceResult.at("records")) {
            if(cancel) return Json::object();
            const auto images=decodeReferences(record,encoder);
            references+=static_cast<int>(images.size());
            record.erase("image_jpeg_base64");
            record["image_similarity"]=images.empty() ? Json(nullptr) : Json(scores(views,images));
            if(sourceResult["source"]=="wikimedia" && !images.empty()) {
                const auto vehicle=candidate({{"brand",record["brand"]},{"series",record["series"]}});
                visualMatches.push_back({{"candidate",vehicle},{"similarity",record["image_similarity"]},
                                        {"method","公开参考图比对"}});
            }
            for(const auto& match:matches) {
                const auto& car=match.at("candidate");
                if(record["brand"]==car["brand"] && record["series"]==car["series"]) {
                    record["matched_level"]="车系";
                    matched=true;
                    break;
                }
            }
        }
        if(matched) ++matching;
    }
    if(broad && !visualMatches.empty()) {
        std::stable_sort(visualMatches.begin(),visualMatches.end(),[](const auto& a,const auto& b){
            return a.at("similarity")>b.at("similarity");
        });
        if(visualMatches.size()>3) visualMatches.erase(visualMatches.begin()+3,visualMatches.end());
        matches=std::move(visualMatches);
        for(auto& sourceResult:sources) {
            if(sourceResult["source"]!="wikimedia") continue;
            std::erase_if(sourceResult["records"].get_ref<Json::array_t&>(),[&](const auto& row){
                return std::none_of(matches.begin(),matches.end(),[&](const auto& match){
                    const auto& car=match["candidate"];
                    return row["brand"]==car["brand"] && row["series"]==car["series"];
                });
            });
        }
        matching=std::max(1,matching);
    }
    const std::string status=matching ? "已查询线上资料 · 车系候选，见来源和比对分"
        : reached ? "已查线上资料，暂未找到相符车系" : "本地初判，未联网核验";
    return {{"status",status},{"source",pathText(source)},{"local_method",request.contains("manual")?"人工候选":
            broad&&references>0?"公开参考图比对":request.value("catalogue",std::string{}).empty()?"内置车系词表":"本地参考图"},
            {"candidates",matches},{"catalogues",sources},{"reference_images",references},{"views",views.size()}};
}

}
