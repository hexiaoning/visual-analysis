#include "people.h"
#include "local_images.h"
#include "portrait_video.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cwctype>
#include <cmath>
#include <map>
#include <numeric>
#include <set>
#include <Windows.h>

namespace visual {
namespace fs = std::filesystem;

Json rectJson(cv::Rect rect) { return Json::array({rect.x,rect.y,rect.br().x,rect.br().y}); }

cv::Rect readRect(const Json& values) {
    if (!values.is_array() || values.size()!=4) throw std::invalid_argument("选区需要四个坐标。");
    const auto x=values[0].get<int>(), y=values[1].get<int>();
    const auto x2=values[2].get<int>(), y2=values[3].get<int>();
    if (x<0 || y<0 || x2<=x || y2<=y || x2>100000 || y2>100000) throw std::invalid_argument("选区坐标无效。");
    return {x,y,x2-x,y2-y};
}

namespace {

std::string text(const Json& value,const char* field,size_t limit=200) {
    const auto content=value.value(field,std::string{});
    const auto first=content.find_first_not_of(" \r\n\t");
    if(first==std::string::npos) return {};
    auto result=content.substr(first,content.find_last_not_of(" \r\n\t")-first+1);
    if (wide(result).size()>limit || result.find('\0')!=std::string::npos) throw std::invalid_argument("文本字段无效。");
    return result;
}

Json settings(const fs::path& root) {
    const auto config=root/L"config.json";
    return fs::exists(config) ? readJson(config) : Json::object();
}

Json queryPeopleInScope(Database& db,std::string sql,const PersonFilter& filter,
                        std::vector<Json> parameters={},std::string_view order={}) {
    if(filter.libraries) {
        if(filter.libraries->empty()) return Json::array();
        sql+=" AND EXISTS(SELECT 1 FROM person_libraries l WHERE p.id=l.person_id AND l.library_id IN (";
        const char* placeholder="?";
        for(const auto& id:*filter.libraries) {
            sql+=placeholder;
            parameters.push_back(id);
            placeholder=",?";
        }
        sql+="))";
    }
    sql+=order;
    return db.query(sql.c_str(),std::span<const Json>(parameters));
}

Json listPeople(Database& db,const PersonFilter& filter) {
    auto people=queryPeopleInScope(db,R"sql(
        SELECT p.*,COUNT(s.id) AS photo_count FROM persons p LEFT JOIN face_samples s ON p.id=s.person_id
        WHERE 1
    )sql",filter,{}," GROUP BY p.id ORDER BY p.name,p.id");
    std::erase_if(people.get_ref<Json::array_t&>(),[&](const auto& person){
        return !filter.matches(person.at("id_card").template get<std::string>());
    });
    for(auto& p:people) {
        p["libraries"]=db.query(R"sql(SELECT l.id,l.name FROM portrait_libraries l JOIN person_libraries b
            ON l.id=b.library_id WHERE b.person_id=? ORDER BY l.name)sql",{p.at("id")});
    }
    return people;
}

void memberships(Database& db,const Json& id,const Json& raw) {
    if (!raw.is_array()) throw std::invalid_argument("人员库列表无效。");
    auto ids=raw.get<std::set<std::string>>();
    if(ids.empty()) ids.insert("default");
    db.query("DELETE FROM person_libraries WHERE person_id=?",{id});
    for(const auto& library:ids) db.query("INSERT INTO person_libraries VALUES(?,?)",{id,library});
}

struct PreparedSample { std::string id; cv::Mat pixels; FaceObservation observation; };

void enroll(Database& db,const fs::path& home,FaceEngine& engine,const Json& payload,bool newPerson) {
    const auto paths=payload.at("photos").get<std::vector<std::string>>();
    if (paths.empty() || paths.size()>100) throw std::invalid_argument("请选择 1～100 张用于录入的照片。");
    std::vector<PreparedSample> prepared;
    for(const auto& path:paths) {
        auto image=readImage(textPath(path));
        auto feature=engine.reference(image);
        prepared.push_back({newId(),std::move(image),std::move(feature)});
    }
    const auto personId=newPerson ? newId() : text(payload,"id");
    auto component=textPath(personId);
    if(component.empty() || component!=component.filename() || component==L"." || component==L"..") {
        throw std::invalid_argument("人员标识无效。");
    }
    const auto folder=localPath(home/L"data/face_images"/component);
    fs::create_directories(folder);
    std::vector<fs::path> copied;
    try {
        Transaction transaction(db);
        const auto stamp=utcTimestamp();
        if(newPerson) {
            const auto name=text(payload,"name"), number=text(payload,"id_card");
            if(name.empty() || number.empty()) throw std::invalid_argument("请填写姓名和身份证号。");
            db.query("INSERT INTO persons VALUES(?,?,?,?,?)",{personId,name,number,stamp,stamp});
            memberships(db,personId,payload.value("library_ids",Json::array({"default"})));
        } else if(db.query("SELECT id FROM persons WHERE id=?",{personId}).empty()) {
            throw std::invalid_argument("人员不存在。");
        }
        for(const auto& sample:prepared) {
            const auto file=folder/(wide(sample.id)+L".png");
            writePng(file,sample.pixels);
            copied.push_back(file);
            db.query("INSERT INTO face_samples VALUES(?,?,?,?,?,?,?,?)",
                {sample.id,personId,pathText(file),featureBlob(sample.observation.feature),
                 sample.observation.feature.size(),engine.settings().modelId(),sample.observation.quality,stamp});
        }
        db.query("UPDATE persons SET updated_at=? WHERE id=?",{stamp,personId});
        transaction.commit();
    } catch(...) {
        for(const auto& file:copied) {
            std::error_code ignored;
            fs::remove(file,ignored);
        }
        throw;
    }
}

void deleteOwnedFile(const fs::path& home,const std::string& value) {
    try {
        const auto root=localPath(home/L"data/face_images");
        const auto path=localPath(textPath(value));
        const auto suffix=path.lexically_relative(root);
        if(!suffix.empty() && *suffix.begin()!=L".." && fs::is_regular_file(path)) {
            std::error_code ignored;
            fs::remove(path,ignored);
        }
    } catch(const std::exception&) {}
}

void zipAnnotations(const fs::path& contents,const fs::path& archive,const std::atomic<bool>& cancel) {
    const auto executable=fs::path(environment(L"SystemRoot"))/L"System32/tar.exe";
    if(!fs::is_regular_file(executable)) throw std::runtime_error("Windows ZIP 导出组件不可用。");
    auto command=L"\""+executable.wstring()+L"\" -a -cf \""+archive.wstring()+L"\" -C \""+contents.wstring()+L"\" .";
    STARTUPINFOW startup{};
    startup.cb=sizeof(startup);
    PROCESS_INFORMATION process{};
    if(!CreateProcessW(executable.c_str(),command.data(),nullptr,nullptr,FALSE,CREATE_NO_WINDOW,
                       nullptr,contents.c_str(),&startup,&process)) throw std::runtime_error("无法启动 Windows ZIP 导出。");
    CloseHandle(process.hThread);
    DWORD result=WAIT_TIMEOUT;
    while((result=WaitForSingleObject(process.hProcess,100))==WAIT_TIMEOUT) {
        if(cancel) {
            TerminateProcess(process.hProcess,2);
            WaitForSingleObject(process.hProcess,1000);
            break;
        }
    }
    DWORD code=2;
    GetExitCodeProcess(process.hProcess,&code);
    CloseHandle(process.hProcess);
    if(result!=WAIT_OBJECT_0 || code!=0 || !fs::is_regular_file(archive)) {
        std::error_code ignored;
        fs::remove(archive,ignored);
        throw std::runtime_error("标注 ZIP 未完成。");
    }
}

Json exportAnnotations(Database& db,const Json& payload,const std::atomic<bool>& cancel) {
    auto data=db.query("SELECT * FROM recognition_annotations ORDER BY id");
    if(payload.contains("annotation_ids")) {
        const auto ids=payload["annotation_ids"].get<std::set<int64_t>>();
        std::erase_if(data.get_ref<Json::array_t&>(),[&](const auto& row){return !ids.contains(row.at("id").template get<int64_t>());});
        if(data.size()!=ids.size()) throw std::invalid_argument("选定的部分标注已不存在。");
    }
    if(data.empty()) throw std::invalid_argument("没有可导出的标注。");
    const auto target=localPath(textPath(payload.at("output").get<std::string>()));
    if(!fs::create_directory(target)) throw std::runtime_error("请选择一个尚未存在的导出目录。");
    const auto contents=target/L"内容";
    fs::create_directory(contents);
    const bool media=payload.value("include_media",true),full=payload.value("full_video",false);
    const bool portraits=payload.value("include_portraits",false),info=payload.value("include_person_info",false);
    const double before=payload.value("seconds_before",2.),after=payload.value("seconds_after",2.);
    if(!(before>=0 && before<=600 && after>=0 && after<=600) || (media && !full && before+after==0.)) {
        throw std::invalid_argument("片段前后时长不在允许范围。");
    }
    auto warnings=Json::array(),persons=Json::array();
    std::set<std::string> included;
    std::map<std::pair<fs::path,int>,std::wstring> sources;
    for(auto& row:data) {
        if(cancel) return Json::object();
        const auto id=row.at("id").get<int64_t>();
        const auto prefix=std::to_wstring(id);
        if(row["bbox"].is_string()) {
            auto box=Json::parse(row["bbox"].get<std::string>(),nullptr,false);
            row["bbox"]=box.is_discarded()?Json(nullptr):std::move(box);
        }
        for(const auto* key:{"snapshot_path","face_path"}) {
            if(!row[key].is_string() || row[key].get_ref<const std::string&>().empty()) continue;
            if(!media) {row[key]=nullptr;continue;}
            try {
                const auto source=localPath(textPath(row[key].get<std::string>()));
                if(!fs::is_regular_file(source)) throw std::runtime_error("文件不存在");
                const auto filename=prefix+L"-"+wide(key)+source.extension().wstring();
                fs::copy_file(source,contents/filename);
                row[key]=utf8(filename);
            } catch(const std::exception&) {row[key]=nullptr;warnings.push_back("标注 "+std::to_string(id)+" 的截图不可用。");}
        }
        if(media) {
            try {
                const auto source=localPath(textPath(row.at("source").get<std::string>()));
                auto ext=source.extension().wstring();
                for(auto& ch:ext) ch=static_cast<wchar_t>(std::towlower(ch));
                constexpr std::array images{L".jpg",L".jpeg",L".png",L".bmp",L".webp",L".tif",L".tiff",L".gif"};
                const bool image=std::find(images.begin(),images.end(),ext)!=images.end();
                const auto frame=std::max(0,row.value("frame_index",0));
                const auto index=std::pair{source,image||full?-1:frame};
                if(!sources.contains(index)) {
                    // Stable ASCII ZIP entries are readable regardless of the system code page.
                    const auto filename=prefix+(image||full?L"-source"+ext:L"-source-clip.mp4");
                    if(image||full) {
                        fs::copy_file(source,contents/filename);
                    } else {
                        double when=-1;
                        visitVideo(source,[&](const cv::Mat&,int current,double seconds){
                            if(current!=frame) return true;
                            when=seconds;return false;
                        },1,cancel);
                        if(when<0) throw std::runtime_error("标注帧超出源视频。");
                        VideoProgress progress;
                        progress.externalCancel=&cancel;
                        renderReviewClip(source,contents/filename,std::max(0.,when-before),when+after,progress);
                        if(!progress.frames || cancel) throw std::runtime_error("无法导出标注片段。");
                    }
                    sources[index]=filename;
                }
                row["media"]={{"path",utf8(sources.at(index))},{"mode",image?"image":full?"full_video":"clip"}};
            } catch(const std::exception&) {
                warnings.push_back("标注 "+std::to_string(id)+" 的源文件或相应片段不可用。");
                row["media"]=nullptr;
            }
        }
        if((info||portraits) && row["person_id"].is_string()) {
            const auto personId=row["person_id"].get<std::string>();
            if(!included.insert(personId).second) continue;
            Json entry={{"person_id",personId}};
            if(info) {
                const auto people=db.query("SELECT name,id_card,created_at,updated_at FROM persons WHERE id=?",{personId});
                if(!people.empty()) entry.update(people.front());
            }
            if(portraits) {
                auto pictures=Json::array();
                for(const auto& record:db.query("SELECT image_path FROM face_samples WHERE person_id=?",{personId})) {
                    try {
                        const auto file=localPath(textPath(record.at("image_path").get<std::string>()));
                        const auto filename=L"person-"+std::to_wstring(persons.size()+1)+L"-"+std::to_wstring(pictures.size()+1)+file.extension().wstring();
                        fs::copy_file(file,contents/filename);
                        pictures.push_back(utf8(filename));
                    } catch(const std::exception&) {}
                }
                entry["portraits"]=std::move(pictures);
            }
            persons.push_back(std::move(entry));
        }
    }
    if(cancel) return Json::object();
    writeJson(contents/L"annotations.json",{{"schema_version",1},{"annotations",data},{"people",persons},{"issues",warnings}});
    const auto output=target/L"标注库.zip";
    zipAnnotations(contents,output,cancel);
    return {{"output",pathText(output)},{"count",data.size()},{"issues",warnings}};
}

Json annotationOperation(Database& db,const std::string& action,const Json& payload,const std::atomic<bool>& cancel) {
    if(action=="annotations") return db.query("SELECT * FROM recognition_annotations ORDER BY id DESC");
    if(action=="annotation.export") return exportAnnotations(db,payload,cancel);
    const auto mode=payload.at("action").get<std::string>();
    if(mode!="associate" && mode!="reject" && mode!="undo_rejection" && mode!="remove_association") {
        throw std::invalid_argument("未知标注操作。");
    }
    const auto session=payload.at("session_id");
    const auto track=payload.at("track_id");
    const auto person=payload.value("person_id",Json(nullptr));
    const auto record=[&](const std::string& event) {
        db.query(R"sql(INSERT INTO recognition_annotations
            (session_id,source,track_id,frame_index,bbox,snapshot_path,face_path,score,quality,model_id,action,person_id,occurred_at)
            VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?))sql",
            {session,payload.at("source"),track,payload.value("frame_index",0),
             payload.contains("bbox") ? Json(payload.at("bbox").dump()) : Json(nullptr),
             payload.value("snapshot_path",Json(nullptr)),payload.value("face_path",Json(nullptr)),
             payload.value("score",Json(nullptr)),payload.value("quality",Json(nullptr)),
             payload.value("model_id",Json(nullptr)),event,person,utcTimestamp()});
    };
    Transaction transaction(db);
    if(mode=="associate") {
        const auto previous=db.query("SELECT action FROM recognition_annotations WHERE session_id=? AND track_id=? AND person_id=? "
            "AND action IN ('reject','undo_rejection') ORDER BY id DESC LIMIT 1",{session,track,person});
        if(!previous.empty() && previous.front()["action"]=="reject") record("undo_rejection");
    } else if(mode=="reject") {
        const auto previous=db.query("SELECT action,person_id FROM recognition_annotations WHERE session_id=? AND track_id=? "
            "AND action IN ('associate','remove_association') ORDER BY id DESC LIMIT 1",{session,track});
        if(!previous.empty() && previous.front()["action"]=="associate" && previous.front()["person_id"]==person) record("remove_association");
    }
    record(mode);
    transaction.commit();
    return {{"recorded",true}};
}

}

PersonFilter PersonFilter::from(const Json& request) {
    PersonFilter filter{std::nullopt,text(request,"sex"),text(request,"region_prefix"),
                        request.value("min_age",-1),request.value("max_age",-1)};
    if(request.contains("library_ids")) {
        const auto& ids=request.at("library_ids");
        if(!ids.is_array()) throw std::invalid_argument("目标人脸库列表无效。");
        std::set<std::string> unique;
        for(const auto& id:ids) {
            if(!id.is_string()) throw std::invalid_argument("目标人脸库列表无效。");
            const auto& value=id.get_ref<const std::string&>();
            if(value.empty() || value.find('\0')!=std::string::npos || wide(value).size()>200) {
                throw std::invalid_argument("目标人脸库编号无效。");
            }
            unique.insert(value);
        }
        filter.libraries.emplace(unique.begin(),unique.end());
    } else if(auto id=text(request,"library_id");!id.empty()) {
        filter.libraries=std::vector<std::string>{std::move(id)};
    }
    const auto digits=[](char c){return c>='0'&&c<='9';};
    const auto length=filter.region.size();
    if((!filter.sex.empty() && filter.sex!="男" && filter.sex!="女") ||
       (length!=0 && length!=2 && length!=4 && length!=6) || !std::all_of(filter.region.begin(),filter.region.end(),digits)
       || filter.minimum< -1 || filter.minimum>150 || filter.maximum< -1 || filter.maximum>150
       || (filter.minimum>=0 && filter.maximum>=0 && filter.minimum>filter.maximum)) {
        throw std::invalid_argument("地区码、性别或年龄范围有误。");
    }
    return filter;
}

bool PersonFilter::matches(std::string idCard) const {
    if(sex.empty() && region.empty() && minimum<0 && maximum<0) return true;
    const auto first=idCard.find_first_not_of(" \r\n\t");
    if(first==std::string::npos) return false;
    idCard=idCard.substr(first,idCard.find_last_not_of(" \r\n\t")-first+1);
    if(idCard.size()!=18 || idCard.front()=='0' || idCard.substr(14,3)=="000" || !idCard.starts_with(region)) return false;
    constexpr std::array weights{7,9,10,5,8,4,2,1,6,3,7,9,10,5,8,4,2};
    int checksum=0;
    for(size_t i=0;i<weights.size();++i) {
        if(idCard[i]<'0' || idCard[i]>'9') return false;
        checksum+=(idCard[i]-'0')*weights[i];
    }
    if("10X98765432"[checksum%11]!=(idCard[17]=='x'?'X':idCard[17])) return false;
    if(!sex.empty() && sex!=((idCard[16]-'0')%2?"男":"女")) return false;
    using namespace std::chrono;
    const year_month_day birth{year{std::stoi(idCard.substr(6,4))},month{static_cast<unsigned>(std::stoi(idCard.substr(10,2)))},
        day{static_cast<unsigned>(std::stoi(idCard.substr(12,2)))}};
    const year_month_day today{floor<days>(system_clock::now()+hours(8))};
    if(!birth.ok() || birth.year()<year(1) || birth>today) return false;
    const int age=int(today.year())-int(birth.year())-(month_day{today.month(),today.day()}<month_day{birth.month(),birth.day()});
    return (minimum<0 || age>=minimum) && (maximum<0 || age<=maximum);
}

PreparedFaceGallery::PreparedFaceGallery(Database& database,const FaceSettings& config,const PersonFilter& filter)
    :threshold_(config.matchThreshold),margin_(config.margin) {
    const auto rows=queryPeopleInScope(database,R"sql(
        SELECT p.id,p.name,p.id_card,s.image_path,s.embedding,s.embedding_dim,s.quality FROM face_samples s
        JOIN persons p ON p.id=s.person_id WHERE s.model_id=?
    )sql",filter,{config.modelId()});
    for (const auto& row:rows) {
        if(!filter.matches(row.at("id_card").get<std::string>())) continue;
        auto value=readFeature(row.at("embedding"),row.at("embedding_dim").get<size_t>());
        if(value.empty()) continue;
        try {value=normalized(std::move(value));} catch(const std::exception&) {continue;}
        people_[row.at("id").get<std::string>()].push_back({{{"person_id",row.at("id")},{"name",row.at("name")},
            {"id_card",row.at("id_card")},{"image_path",row.at("image_path")}},std::move(value),
            row.at("quality").get<double>()});
    }
}

Json PreparedFaceGallery::match(const Feature& query,bool photoRank) const {
    struct Ranked {const Sample* sample;double score;};
    Json ranked=Json::array();
    for(const auto& [id,records]:people_) {
        std::vector<Ranked> samples;
        samples.reserve(records.size());
        for(const auto& sample:records) {
            if(sample.feature.size()==query.size()) samples.push_back({&sample,similarity(query,sample.feature)});
        }
        if(samples.empty()) continue;
        std::stable_sort(samples.begin(),samples.end(),[](const auto& a,const auto& b){return a.score>b.score;});
        auto output=samples.front().sample->record;
        double score=samples.front().score;
        if(!photoRank) {
            Feature centroid(query.size(),0);
            double mean=0, weights=0;
            for(size_t i=0;i<samples.size();++i) {
                const auto& sample=*samples[i].sample;
                for(size_t k=0;k<centroid.size();++k) centroid[k]+=sample.feature[k]*static_cast<float>(std::max(.1,sample.quality));
                if(i<3) {const auto weight=std::max(.05,sample.quality); mean+=weight*samples[i].score; weights+=weight;}
            }
            score=.70*score+.25*mean/weights+.05*similarity(query,normalized(std::move(centroid)));
        }
        output["score"]=score;
        ranked.push_back(std::move(output));
    }
    std::sort(ranked.begin(),ranked.end(),[](const auto& a,const auto& b){return a.at("score")>b.at("score");});
    if(photoRank && ranked.size()>50) ranked.erase(ranked.begin()+50,ranked.end());
    if(!photoRank && !ranked.empty()) {
        auto& best=ranked.front();
        const auto score=best.at("score").get<double>();
        const double margin=ranked.size()>1 ? score-ranked[1]["score"].get<double>() : 1.;
        best["margin"]=margin;
        best["accepted"]=score>=threshold_ && margin>=margin_;
    }
    return ranked;
}

Json matchGallery(Database& database,const Feature& query,const FaceSettings& config,const PersonFilter& filter,bool photoRank) {
    return PreparedFaceGallery(database,config,filter).match(query,photoRank);
}

Json peopleOperation(const std::string& action,const Json& payload,const fs::path& models,const std::atomic<bool>& cancel) {
    const auto home=dataHome();
    const auto getConfig=[&]{return FaceSettings::from(settings(home),models);};
    if(action=="settings.load") return settings(home);
    if(action=="settings.save") {
        auto config=settings(home);
        const auto& newValues=payload.at("settings");
        if(!newValues.is_object()) throw std::invalid_argument("设置应为 JSON 对象。");
        config.update(newValues);
        (void)FaceSettings::from(config,models);
        writeJson(home/L"config.json",config);
        return config;
    }
    Database db(home/L"data/faces.db");
    initializePeople(db);
    const auto id=text(payload,"id");
    if(action=="people.list") return listPeople(db,PersonFilter::from(payload));
    if(action=="libraries.list") return db.query("SELECT * FROM portrait_libraries ORDER BY name");
    if(action=="person.samples") return db.query("SELECT id,image_path,model_id,quality,created_at FROM face_samples WHERE person_id=? ORDER BY created_at",{id});
    if(action=="library.add" || action=="library.rename") {
        const auto name=text(payload,"name");
        if(name.empty()) throw std::invalid_argument("请填写库名。");
        if(action=="library.add") db.query("INSERT INTO portrait_libraries VALUES(?,?)",{newId(),name});
        else db.query("UPDATE portrait_libraries SET name=? WHERE id=?",{name,id});
        return db.query("SELECT * FROM portrait_libraries ORDER BY name");
    }
    if(action=="library.remove") {
        if(id=="default") throw std::invalid_argument("默认库不能删除。");
        Transaction tx(db);
        db.query("DELETE FROM portrait_libraries WHERE id=?",{id});
        db.script("INSERT INTO person_libraries SELECT id,'default' FROM persons p WHERE NOT EXISTS(SELECT 1 FROM person_libraries l WHERE l.person_id=p.id)");
        tx.commit();
        return db.query("SELECT * FROM portrait_libraries ORDER BY name");
    }
    if(action=="person.add" || action=="person.photos") {
        FaceEngine engine(models,getConfig());
        enroll(db,home,engine,payload,action=="person.add");
        return listPeople(db,{});
    }
    if(action=="person.edit") {
        const auto name=text(payload,"name"),number=text(payload,"id_card");
        if(name.empty() || number.empty()) throw std::invalid_argument("请填写姓名和身份证号。");
        Transaction tx(db);
        db.query("UPDATE persons SET name=?,id_card=?,updated_at=? WHERE id=?",{name,number,utcTimestamp(),id});
        if(!db.changes()) throw std::invalid_argument("人员不存在。");
        if(payload.contains("library_ids")) memberships(db,id,payload.at("library_ids"));
        tx.commit();
        return listPeople(db,{});
    }
    if(action=="person.remove" || action=="person.sample.remove") {
        const bool person=action=="person.remove";
        Transaction tx(db);
        const auto files=db.query(person ? "SELECT image_path FROM face_samples WHERE person_id=?"
            : "SELECT image_path FROM face_samples WHERE id=?",{id});
        db.query(person ? "DELETE FROM persons WHERE id=?" : "DELETE FROM face_samples WHERE id=?",{id});
        tx.commit();
        for(const auto& file:files) deleteOwnedFile(home,file.at("image_path").get<std::string>());
        return listPeople(db,{});
    }
    if(action=="person.rebuild") {
        FaceEngine engine(models,getConfig());
        const auto existing=db.query("SELECT id,image_path FROM face_samples WHERE (?='' OR person_id=?)",{id,id});
        size_t rebuilt=0;
        auto failures=Json::array();
        for(const auto& sample:existing) {
            try {
                const auto feature=engine.reference(readImage(textPath(sample.at("image_path").get<std::string>())));
                db.query("UPDATE face_samples SET embedding=?,embedding_dim=?,model_id=?,quality=? WHERE id=?",
                    {featureBlob(feature.feature),feature.feature.size(),engine.settings().modelId(),feature.quality,sample.at("id")});
                ++rebuilt;
            } catch(const std::exception& error) {failures.push_back({{"id",sample["id"]},{"error",error.what()}});}
        }
        return {{"rebuilt",rebuilt},{"failures",failures}};
    }
    if(action=="face.compare" || action=="face.gallery") {
        FaceEngine engine(models,getConfig());
        const auto a=engine.reference(readImage(textPath(payload.at("input").get<std::string>())));
        if(action=="face.compare") {
            const auto b=engine.reference(readImage(textPath(payload.at("reference").get<std::string>())));
            return {{"similarity",similarity(a.feature,b.feature)},{"quality",Json::array({a.quality,b.quality})}};
        }
        return matchGallery(db,a.feature,engine.settings(),PersonFilter::from(payload),true);
    }
    if(action=="annotations" || action=="annotation.export" || action=="annotation.write") {
        return annotationOperation(db,action,payload,cancel);
    }
    throw std::invalid_argument("未知人员库或标注操作。");
}

}
