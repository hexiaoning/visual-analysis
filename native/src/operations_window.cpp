#include "operations_window.h"
#include "operations.h"
#include "local_images.h"
#include "people.h"
#include "portrait_video.h"
#include "preview_native.h"
#include "workspace_theme.h"

#include <algorithm>
#include <array>
#include <future>
#include <iomanip>
#include <memory>
#include <sstream>
#include <utility>

#include <commdlg.h>
#include <shlobj.h>
#include <opencv2/imgproc.hpp>

namespace visual {
namespace fs=std::filesystem;
namespace {

constexpr wchar_t windowClass[]=L"VisualAnalysisWorkflow";
constexpr UINT_PTR timer=1;
constexpr UINT_PTR sourceTimer=2;
constexpr UINT chooseBase=200, actionBase=300, cancelAction=500, saveAction=501, exportHitsAction=502,
    viewAction=503, playAction=504, pauseAction=505, restoreHitAction=506,
    libraryList=507, selectLibraries=508, clearLibraries=509;
constexpr char libraryRequired[]="请至少勾选一个目标人脸库。";

enum class Box {Text,Media,Images,Json,Directory,Region,Line};
struct Field {const char* key;const wchar_t* title;Box type=Box::Text;};
struct Action {const char* name;const wchar_t* label;};
struct Workflow {
    const wchar_t* name;
    std::vector<Field> fields;
    std::vector<Action> actions;
};

const std::array workflows{
    Workflow{L"车型识别",{{"input",L"车辆图片／视频",Box::Media},{"catalogue",L"本机参考库（可选）",Box::Json},
            {"brand",L"已知品牌（可选）"},{"series",L"已知车系（可选）"}},
        {{"vehicle.analyze",L"识别并查询线上资料"}}},
    Workflow{L"人脸检索",{{"input",L"探测照片",Box::Images},{"reference",L"对比照片",Box::Images},
            {"source",L"图片／视频（可混选）",Box::Media},{"step",L"抽帧间隔（可选）"},
            {"sex",L"性别（男／女）"},{"region_prefix",L"地区码"},
            {"min_age",L"最小年龄（可选）"},{"max_age",L"最大年龄（可选）"},{"camera",L"摄像头编号（默认 0）"}},
        {{"face.compare",L"两张照片对比"},{"face.gallery",L"照片检索人物"},{"face.scan",L"开始检索"},
         {"face.camera",L"打开摄像头"}}},
    Workflow{L"人车检测",{{"source",L"本机素材路径",Box::Media},{"threshold",L"置信度门槛（可选）"},
            {"output",L"截图保存目录",Box::Directory},{"target",L"类别（人／车／物体／全部）"},
            {"trigger_line",L"触发线（可选）",Box::Line},{"trigger_area",L"触发区域（可选）",Box::Region},
            {"video_output",L"标注视频 MP4 路径"}},
        {{"objects.scan",L"开始检测"}}},
    Workflow{L"以图搜索",{{"input",L"本机参考图",Box::Images},{"source",L"本机素材或文件夹",Box::Media},
            {"bbox",L"选区 x1 y1 x2 y2",Box::Region},{"mode",L"模式（人脸／外观）"},{"threshold",L"相似度门槛"},
            {"step",L"抽帧间隔"}},
        {{"visual.search",L"执行搜索"},{"sources.folder",L"添加文件夹…"},{"sources.clear",L"清空范围"}}},
    Workflow{L"语义搜索",{{"text",L"目标描述"},{"source",L"本机素材或文件夹",Box::Media},
            {"category",L"类别（自动/人物/车辆）"},{"step",L"抽帧间隔（默认 25）"},{"threshold",L"相关度门槛"}},
        {{"semantic.search",L"检索目标"},{"sources.folder",L"添加文件夹…"},{"sources.clear",L"清空范围"}}},
    Workflow{L"视频车牌",{{"source",L"本机车辆视频",Box::Media},{"frame_index",L"起始帧（从 0 开始）"},
            {"bbox",L"选区 x1 y1 x2 y2",Box::Region},{"count",L"跟踪帧数（默认 12）"},{"blur_length",L"拖影长度（像素）"},
            {"blur_angle",L"角度（0～179）"},{"output",L"输出目录",Box::Directory}},
        {{"plate.preview",L"导出起始帧"},{"plate.video",L"还原车牌"}}},
    Workflow{L"人员库",{{"id",L"人员或库的 ID"},{"name",L"姓名／库名"},{"id_card",L"身份证号"},
            {"photos",L"录入照片（可多选）",Box::Images},{"library_ids",L"所属库 ID（空格分隔）"},
            {"library_id",L"检索库 ID"},{"sex",L"性别（男／女）"},{"region_prefix",L"地区码"},
            {"min_age",L"最小年龄（可选）"},{"max_age",L"最大年龄（可选）"}},
        {{"people.list",L"查看人员"},{"person.add",L"新增人员"},{"person.edit",L"修改人员"},
         {"person.photos",L"添加照片"},{"person.samples",L"查看照片"},{"person.sample.remove",L"移除照片"},
         {"person.remove",L"移除人员"},{"libraries.list",L"查看库"},{"library.add",L"新增库"},
         {"library.rename",L"改库名"},{"library.remove",L"移除库"},{"person.rebuild",L"重建特征"}}},
    Workflow{L"标注库",{{"session_id",L"会话 ID"},{"source",L"源文件",Box::Media},{"track_id",L"轨迹号"},
            {"frame_index",L"帧编号（从 0 开始）"},{"person_id",L"人员 ID"},{"output",L"导出目录",Box::Directory},
            {"export_mode",L"媒体（无／片段／整段）"},{"seconds_before",L"片段前（秒）"},{"seconds_after",L"片段后（秒）"},
            {"include_portraits",L"含人员照片（1 或 0）"},{"include_person_info",L"含身份字段（1 或 0）"}},
        {{"annotations",L"查看标注"},{"associate",L"关联人员"},{"reject",L"排除人员"},
         {"undo_rejection",L"撤销排除"},{"remove_association",L"取消关联"},{"annotation.export",L"导出标注"}}},
    Workflow{L"设置",{{"match_threshold",L"人脸相似度阈值"},{"match_margin",L"候选最小分差"},
            {"detector_threshold",L"人脸检测阈值"},{"min_face_size",L"最小人脸尺寸"},
            {"min_quality",L"最低人脸质量"},{"enrollment_min_quality",L"最低录入质量"},
            {"mirror_augmentation",L"镜像增强（1 或 0）"}},
        {{"settings.load",L"读取现有设置"},{"settings.save",L"保存设置"}}},
};

struct Form {
    fs::path models,worker;
    size_t index{};
    bool adopted=false;
    std::vector<HWND> captions,fields,choosers,buttons;
    std::map<std::string,std::vector<fs::path>> selections;
    HWND result{},activity{},cancelButton{},save{},exportHits{},view{},play{},hitChoice{};
    Json last;
    HWND pause{};
    std::shared_ptr<FaceScanSession> faceSession,faceQueue;
    std::vector<fs::path> deferredFaceSources;
    std::wstring lastFaceSource;
    fs::path galleryQuery;
    HWND restoreHit{};
    std::shared_ptr<std::atomic<bool>> cancelFlag;
    std::shared_ptr<DetectionPreview> preview;
    int displayedFrame{};
    std::future<Json> pending;
    std::string operation;
    HWND libraryCaption{},libraryChoices{},selectAll{},clearAll{};
    std::vector<std::string> libraryIds;

};

HWND child(HWND root,const wchar_t* name,const wchar_t* caption,DWORD style,UINT_PTR id) {
    const auto type=std::wstring_view(name);
    if(type==L"EDIT") style|=WS_BORDER;
    if(type==L"BUTTON") style|=BS_OWNERDRAW;
    auto handle=CreateWindowExW(0,name,caption,
        WS_CHILD|WS_VISIBLE|style,0,0,0,0,root,reinterpret_cast<HMENU>(id),nullptr,nullptr);
    theme::styleControl(handle);
    return handle;
}

std::wstring content(HWND edit) {
    auto length=GetWindowTextLengthW(edit);
    std::wstring text(length+1,L'\0');
    GetWindowTextW(edit,text.data(),length+1);
    text.resize(length);
    return text;
}

std::wstring fieldContent(const Form& form,std::string_view key) {
    const auto& fields=workflows[form.index].fields;
    for(size_t i=0;i<fields.size();++i) if(fields[i].key==key) return content(form.fields[i]);
    return {};
}

cv::Mat frameAt(const fs::path& path,int index) {
    if(index<0) throw std::invalid_argument("帧编号须从 0 开始。");
    const auto source=localPath(path);
    const auto extension=source.extension().wstring();
    constexpr std::array types{L".jpg",L".jpeg",L".png",L".bmp",L".webp",L".tif",L".tiff",L".gif",L".avif"};
    if(std::any_of(types.begin(),types.end(),[&](const auto* type){return _wcsicmp(type,extension.c_str())==0;})) {
        return readImage(source);
    }
    std::atomic<bool> cancel=false;
    cv::Mat result;
    visitVideo(source,[&](const cv::Mat& frame,int position,double){
        if(index!=position) return true;
        result=frame.clone();
        return false;
    },1,cancel,index);
    if(result.empty()) throw std::runtime_error("源文件没有对应的视频画面。");
    return result;
}

Json viewable(const Form& form) {
    if(form.last.contains("hits") && form.last["hits"].is_array()) return form.last["hits"];
    if(form.operation=="face.gallery") return form.last;
    if(form.operation=="plate.video" && form.last.contains("candidates")) return form.last["candidates"];
    return Json::array();
}

void previewComparison(HWND owner,cv::Mat left,cv::Mat right,const wchar_t* caption) {
    cv::resize(left,left,{std::max(1,cvRound(left.cols*(420./left.rows))),420});
    cv::resize(right,right,{std::max(1,cvRound(right.cols*(420./right.rows))),420});
    cv::Mat combined;
    cv::hconcat(left,right,combined);
    previewNative(owner,combined,caption,false);
}

void viewResult(HWND owner,const Form& form,bool play) {
    const auto rows=viewable(form);
    auto index=SendMessageW(form.hitChoice,CB_GETCURSEL,0,0);
    if(index<0) index=0;
    if(index<static_cast<LRESULT>(rows.size())) {
        const auto& row=rows[index];
        if(row.contains("source") && row.contains("bbox")) {
            const auto source=textPath(row["source"].get<std::string>());
            const auto first=row.value("frame_index",0);
            const auto frame=row.contains("camera")?readImage(textPath(row.at("snapshot_path").get<std::string>())):frameAt(source,first);
            const auto area=readRect(row["bbox"]);
            if(play && !row.contains("camera") && row.contains("seconds") && row["seconds"].is_number()) playNativeVideo(owner,source,first,frame,area);
            else previewNative(owner,frame,L"命中的原始画面",false,area);
        } else if(row.contains("image_path")) {
            auto photo=readImage(textPath(row["image_path"].get<std::string>()));
            if(form.operation=="face.gallery" && !form.galleryQuery.empty()) {
                previewComparison(owner,readImage(form.galleryQuery),std::move(photo),L"探测照片与人员库候选");
            } else previewNative(owner,photo,L"人员库候选原照",false);
        } else if(row.contains("file")) {
            previewNative(owner,readImage(textPath(row["file"].get<std::string>())),L"车牌还原候选",false);
        }
    } else if(form.last.contains("preview")) {
        previewNative(owner,readImage(textPath(form.last["preview"].get<std::string>())),L"车牌视频起始帧",false);
    } else if(form.operation=="face.compare") {
        previewComparison(owner,readImage(fieldContent(form,"input")),readImage(fieldContent(form,"reference")),
                          L"两张原始照片并排");
    }
}

std::vector<fs::path> chooseFiles(HWND owner,Box mode,bool save=false,bool multiple=false) {
    if(mode==Box::Directory) {
        const auto apartment=CoInitializeEx(nullptr,COINIT_APARTMENTTHREADED);
        if(FAILED(apartment) && apartment!=RPC_E_CHANGED_MODE) throw std::runtime_error("无法初始化目录选择器。");
        struct Uninitialize {HRESULT status;~Uninitialize(){if(SUCCEEDED(status))CoUninitialize();}} cleanup{apartment};
        IFileOpenDialog* raw{};
        if(FAILED(CoCreateInstance(CLSID_FileOpenDialog,nullptr,CLSCTX_INPROC_SERVER,IID_PPV_ARGS(&raw)))) {
            throw std::runtime_error("无法打开目录选择器。");
        }
        const auto release=[](auto* object){if(object)object->Release();};
        std::unique_ptr<IFileOpenDialog,decltype(release)> dialog(raw,release);
        FILEOPENDIALOGOPTIONS flags{};
        dialog->GetOptions(&flags);
        dialog->SetOptions(flags|FOS_PICKFOLDERS|FOS_FORCEFILESYSTEM|FOS_PATHMUSTEXIST);
        if(FAILED(dialog->Show(owner))) return {};
        IShellItem* selected{};
        if(FAILED(dialog->GetResult(&selected))) return {};
        std::unique_ptr<IShellItem,decltype(release)> item(selected,release);
        PWSTR path{};
        if(FAILED(item->GetDisplayName(SIGDN_FILESYSPATH,&path))) return {};
        auto folder=fs::path(path);
        CoTaskMemFree(path);
        return {localPath(folder)};
    }
    std::wstring choices(32768,L'\0');
    OPENFILENAMEW dlg{};
    dlg.lStructSize=sizeof(dlg);
    dlg.hwndOwner=owner;
    dlg.lpstrFile=choices.data();
    dlg.nMaxFile=static_cast<DWORD>(choices.size());
    dlg.Flags=OFN_EXPLORER|OFN_NOCHANGEDIR|OFN_PATHMUSTEXIST;
    dlg.lpstrFilter=mode==Box::Json ? L"JSON 文件\0*.json\0\0" : mode==Box::Images ?
        L"图片\0*.jpg;*.jpeg;*.png;*.webp;*.bmp;*.tif;*.tiff\0\0" :
        L"图片和视频\0*.jpg;*.jpeg;*.png;*.webp;*.bmp;*.mp4;*.mkv;*.mov;*.avi;*.webm;*.ts;*.wmv\0所有文件\0*.*\0\0";
    if(save) {dlg.lpstrDefExt=L"json";dlg.Flags|=OFN_OVERWRITEPROMPT;}
    else {
        dlg.Flags|=OFN_FILEMUSTEXIST;
        if(multiple) dlg.Flags|=OFN_ALLOWMULTISELECT;
    }
    if(!(save?GetSaveFileNameW(&dlg):GetOpenFileNameW(&dlg))) return {};
    auto first=fs::path(choices.c_str());
    auto position=choices.c_str()+first.native().size()+1;
    if(!*position) return {first};
    std::vector<fs::path> selected;
    while(*position) {
        auto path=std::wstring(position);
        selected.push_back(first/path);
        position+=path.size()+1;
    }
    return selected;
}

std::wstring vehicleText(const Json& result) {
    std::wostringstream out;
    out<<wide(result.value("status",std::string{}))<<L"\r\n\r\n";
    for(const auto& match:result.value("candidates",Json::array())) {
        const auto& car=match.at("candidate");
        out<<wide(match.value("method",std::string("本地候选")))<<L"  "
           <<wide(car.value("brand",std::string{}))<<L" "<<wide(car.value("series",std::string{}));
        if(match["similarity"].is_number()) out<<L"  @ "<<std::fixed<<std::setprecision(3)<<match["similarity"].get<double>();
        out<<L"\r\n";
    }
    for(const auto& entry:result.value("catalogues",Json::array())) {
        const auto key=entry.at("source").get<std::string>();
        const std::map<std::string,std::wstring> names{{"wikimedia",L"维基百科／Wikimedia"},{"nhtsa_vpic",L"NHTSA vPIC（美国）"},
                                                    {"miit",L"工信部"},{"dongchedi",L"懂车帝"}};
        const std::map<std::string,std::wstring> statuses{{"ok",L"有相符资料"},{"no_match",L"未找到相符车系"},
                {"unavailable",L"暂时无法访问"},{"unauthorized",L"授权失效"},{"invalid_response",L"资料响应异常"},
                {"no_candidate",L"未提供候选"}};
        const auto flag=statuses.find(entry.value("status",std::string{}));
        out<<L"\r\n"<<(names.contains(key)?names.at(key):wide(key))<<L"  ·  "
           <<(flag==statuses.end()?L"未查询":flag->second)<<L"\r\n";
        out<<L"查询时间  "<<wide(entry.value("queried_at",std::string{}))<<L"\r\n";
        for(const auto& record:entry.value("records",Json::array())) {
            out<<wide(record.value("model",std::string{}));
            if(record.contains("image_similarity") && record["image_similarity"].is_number())
                out<<L"  ·  参考图相似度 "<<std::fixed<<std::setprecision(3)<<record["image_similarity"].get<double>();
            out<<L"\r\n"<<wide(record.value("source_url",std::string{}))<<L"\r\n";
            if(record.contains("image_license_page")) out<<L"参考图来源  "<<wide(record["image_license_page"].get<std::string>())<<L"\r\n";
        }
    }
    return out.str();
}

std::wstring selectionLabel(const std::vector<fs::path>& paths) {
    return paths.size()==1 ? paths.front().wstring() : L"已选择 "+std::to_wstring(paths.size())+L" 个文件";
}

bool librarySelected(const Form& form,size_t index) {
    return SendMessageW(form.libraryChoices,LB_GETSEL,index,0)>0;
}

void reloadLibraries(Form& form) {
    const std::atomic<bool> cancel=false;
    const auto rows=peopleOperation("libraries.list",Json::object(),form.models,cancel);
    std::map<std::string,bool> existing;
    bool allSelected=true;
    for(size_t index=0;index<form.libraryIds.size();++index) {
        const bool checked=librarySelected(form,index);
        existing.emplace(form.libraryIds[index],checked);
        allSelected=allSelected && checked;
    }
    SendMessageW(form.libraryChoices,LB_RESETCONTENT,0,0);
    form.libraryIds.clear();
    for(const auto& row:rows) {
        const auto id=row.at("id").get<std::string>();
        const auto name=wide(row.at("name").get<std::string>());
        const auto rowIndex=SendMessageW(form.libraryChoices,LB_ADDSTRING,0,reinterpret_cast<LPARAM>(name.c_str()));
        if(rowIndex<0) throw std::runtime_error("无法显示目标人脸库。");
        form.libraryIds.push_back(id);
        const auto prior=existing.find(id);
        SendMessageW(form.libraryChoices,LB_SETSEL,prior==existing.end()?allSelected:prior->second,rowIndex);
    }
}

void drawLibraryChoice(const DRAWITEMSTRUCT& item) {
    const auto area=item.rcItem;
    theme::fill(item.hDC,area,theme::white);
    if(item.itemID==static_cast<UINT>(-1)) return;
    const bool checked=(item.itemState&ODS_SELECTED)!=0;
    const bool disabled=(item.itemState&ODS_DISABLED)!=0 || !IsWindowEnabled(item.hwndItem);
    const LONG boxTop=(area.top+area.bottom-14)/2;
    const RECT box{area.left+6,boxTop,area.left+20,boxTop+14};
    if(checked) {
        theme::fill(item.hDC,box,disabled?theme::muted:theme::primaryBlue);
        const auto pen=CreatePen(PS_SOLID,1,theme::white);
        const auto old=SelectObject(item.hDC,pen);
        MoveToEx(item.hDC,box.left+3,box.top+7,nullptr);
        LineTo(item.hDC,box.left+6,box.top+10);
        LineTo(item.hDC,box.left+11,box.top+4);
        SelectObject(item.hDC,old);
        DeleteObject(pen);
    } else {
        theme::card(item.hDC,box);
    }
    const auto length=SendMessageW(item.hwndItem,LB_GETTEXTLEN,item.itemID,0);
    if(length!=LB_ERR) {
        std::wstring name(length+1,L'\0');
        SendMessageW(item.hwndItem,LB_GETTEXT,item.itemID,reinterpret_cast<LPARAM>(name.data()));
        theme::text(item.hDC,name.c_str(),{area.left+26,area.top,area.right-5,area.bottom},theme::Font::Body,
            disabled?theme::muted:theme::textColor,DT_LEFT|DT_VCENTER|DT_SINGLELINE|DT_END_ELLIPSIS);
    }
    if((item.itemState&(ODS_FOCUS|ODS_NOFOCUSRECT))==ODS_FOCUS) {
        auto focus=area;
        DrawFocusRect(item.hDC,&focus);
    }
}

Json input(Form& form,const std::string& action) {
    auto result=Json::object();
    const auto& selected=workflows[form.index];
    for(size_t index=0;index<selected.fields.size();++index) {
        const auto key=std::string(selected.fields[index].key);
        auto text=content(form.fields[index]);
        if(text.empty()) continue;
        if(key=="camera" && action!="face.camera") continue;
        if(key=="photos") {
            auto paths=Json::array();
            const auto found=form.selections.find(key);
            if(found!=form.selections.end() && found->second.size()>1 && text==selectionLabel(found->second)) {
                for(const auto& file:found->second) paths.push_back(pathText(file));
            } else paths.push_back(utf8(text));
            result[key]=std::move(paths);
        } else if(key=="source" && form.selections.contains(key) && form.selections.at(key).size()>1 &&
                  text==selectionLabel(form.selections.at(key))) {
            if(action=="objects.scan" && !fieldContent(form,"video_output").empty()) {
                throw std::invalid_argument("导出标注视频时请选择一个视频文件。");
            }
            result["sources"]=Json::array();
            for(const auto& file:form.selections.at(key)) result["sources"].push_back(pathText(file));
        } else if(key=="track_id" || key=="frame_index" || key=="step" || key=="count" ||
                key=="blur_length" || key=="blur_angle" || key=="min_age" || key=="max_age" || key=="camera") {
            result[key]=std::stoi(text);
        } else if(key=="threshold") {
            result[key]=std::stod(text);
        } else if(key=="seconds_before" || key=="seconds_after") {
            result[key]=std::stod(text);
        } else if(key=="include_portraits" || key=="include_person_info") {
            result[key]=std::stoi(text)!=0;
        } else if(key=="export_mode") {
            if(text!=L"无" && text!=L"片段" && text!=L"整段") throw std::invalid_argument("导出媒体请选择无、片段或整段。");
            result["include_media"]=text!=L"无";
            result["full_video"]=text==L"整段";
        } else if(key=="category") {
            if(text!=L"自动" && text!=L"人物" && text!=L"车辆") {
                throw std::invalid_argument("语义搜索类别请选择自动、人物或车辆。");
            }
            result[key]=text==L"人物"?"person":text==L"车辆"?"vehicle":"auto";
        } else if(key=="mode") {
            result[key]=text==L"人脸" ? "face" : "appearance";
        } else if(key=="target") {
            if(text!=L"人" && text!=L"车" && text!=L"物体" && text!=L"全部") {
                throw std::invalid_argument("类别请选择人、车、物体或全部。");
            }
            result["people"]=text==L"人" || text==L"全部";
            result["vehicles"]=text==L"车" || text==L"全部";
            result["other_objects"]=text==L"物体" || text==L"全部";
        } else if(key=="bbox" || key=="trigger_area" || key=="trigger_line") {
            std::wistringstream coordinates(text);
            auto array=Json::array();
            for(int value;coordinates>>value;) array.push_back(value);
            if(array.size()!=4 || !coordinates.eof()) throw std::invalid_argument("坐标需要四个空格分隔的整数。");
            result[key]=std::move(array);
        } else if(key=="library_ids") {
            std::wistringstream names(text);
            auto array=Json::array();
            for(std::wstring id;names>>id;) array.push_back(utf8(id));
            result[key]=array;
        } else result[key]=utf8(text);
    }
    if(action=="face.gallery" || action=="face.scan" || action=="face.camera") {
        auto selectedIds=Json::array();
        for(size_t index=0;index<form.libraryIds.size();++index) {
            if(librarySelected(form,index)) selectedIds.push_back(form.libraryIds[index]);
        }
        if(selectedIds.empty()) throw std::invalid_argument(libraryRequired);
        result["library_ids"]=std::move(selectedIds);
    }
    if(action=="face.camera" && !result.contains("camera")) {
        result["camera"]=0;
    } else if(action=="vehicle.analyze" && (result.contains("brand") || result.contains("series"))) {
        result["manual"]={{"brand",result.value("brand",std::string{})},{"series",result.value("series",std::string{})}};
    } else if(result.contains("output") && fs::is_directory(localPath(textPath(result["output"].get<std::string>())))) {
        result["output"]=pathText(textPath(result["output"].get<std::string>())/(L"分析结果-"+wide(newId()).substr(0,8)));
    } else if(action=="settings.save") {
        auto values=Json::object();
        for(const auto& [key,value]:result.items()) {
            const auto text=wide(value.get<std::string>());
            if(key=="mirror_augmentation") values[key]=std::stoi(text)!=0;
            else if(key=="min_face_size") values[key]=std::stoi(text);
            else values[key]=std::stod(text);
        }
        result["settings"]=std::move(values);
    }
    return result;
}

constexpr int panelGap=12;
constexpr int fieldInset=12;
constexpr int controlsTop=theme::panelHeaderHeight+9;
constexpr int resultsTop=140;

int formWidth(const RECT& client) {
    return std::clamp(int(client.right)*36/100,398,510);
}

RECT resultArea(const RECT& client) {
    return {formWidth(client)+panelGap+fieldInset,resultsTop,client.right-fieldInset,client.bottom-fieldInset};
}

void layout(HWND handle,const Form& state) {
    RECT client{};
    GetClientRect(handle,&client);
    const int leftWidth=formWidth(client);
    const int rightStart=leftWidth+panelGap;
    const int rightWidth=client.right-rightStart;
    constexpr int labelWidth=174;
    int y=controlsTop;
    for(size_t index=0;index<state.fields.size();++index) {
        MoveWindow(state.captions[index],fieldInset,y+3,labelWidth-6,24,TRUE);
        const bool picker=state.choosers[index]!=nullptr;
        MoveWindow(state.fields[index],fieldInset+labelWidth,y,leftWidth-2*fieldInset-labelWidth-(picker?74:0),25,TRUE);
        if(picker) MoveWindow(state.choosers[index],leftWidth-fieldInset-68,y,68,25,TRUE);
        y+=29;
    }
    const int columns=3;
    const int buttonWidth=(leftWidth-2*fieldInset-12)/columns;
    y+=8;
    for(size_t index=0;index<state.buttons.size();++index) {
        MoveWindow(state.buttons[index],fieldInset+int(index%columns)*(buttonWidth+6),y+int(index/columns)*30,buttonWidth,26,TRUE);
    }
    if(state.libraryChoices) {
        const int titleY=y+30*static_cast<int>((state.buttons.size()+columns-1)/columns)+8;
        const int listY=titleY+29;
        MoveWindow(state.libraryCaption,fieldInset,titleY+4,120,22,TRUE);
        MoveWindow(state.selectAll,leftWidth-fieldInset-114,titleY,54,24,TRUE);
        MoveWindow(state.clearAll,leftWidth-fieldInset-54,titleY,54,24,TRUE);
        MoveWindow(state.libraryChoices,fieldInset,listY,leftWidth-2*fieldInset,
            std::max(int(client.bottom)-57-listY,26),TRUE);
    }
    MoveWindow(state.activity,fieldInset,client.bottom-36,leftWidth-(state.pause?180:108),23,TRUE);
    MoveWindow(state.cancelButton,leftWidth-fieldInset-80,client.bottom-40,80,27,TRUE);
    if(state.pause) MoveWindow(state.pause,leftWidth-fieldInset-145,client.bottom-40,59,27,TRUE);

    MoveWindow(state.hitChoice,rightStart+fieldInset,controlsTop,rightWidth-2*fieldInset,170,TRUE);
    const std::array resultButtons{state.view,state.play,state.exportHits,state.save,state.restoreHit};
    const int itemWidth=(rightWidth-2*fieldInset-12)/columns;
    for(size_t index=0;index<resultButtons.size() && resultButtons[index];++index) {
        MoveWindow(resultButtons[index],rightStart+fieldInset+int(index%columns)*(itemWidth+6),
            74+int(index/columns)*30,itemWidth,26,TRUE);
    }
    const auto result=resultArea(client);
    MoveWindow(state.result,result.left,result.top,result.right-result.left,std::max<LONG>(result.bottom-result.top,40),TRUE);
    InvalidateRect(handle,nullptr,TRUE);
}

void buttons(Form& state,bool running) {
    for(auto button:state.buttons) EnableWindow(button,!running);
    const auto& fields=workflows[state.index].fields;
    for(size_t i=0;i<state.choosers.size();++i) if(state.choosers[i]) {
        EnableWindow(state.choosers[i],!running || (state.faceQueue && fields[i].key==std::string_view("source")));
    }
    for(auto field:state.fields) EnableWindow(field,!running);
    if(state.libraryChoices) {
        for(const auto control:{state.libraryChoices,state.selectAll,state.clearAll}) EnableWindow(control,!running);
    }
    EnableWindow(state.cancelButton,running);
    if(state.pause) EnableWindow(state.pause,running && state.preview!=nullptr);
    EnableWindow(state.save,!running && !state.last.is_null());
    EnableWindow(state.exportHits,!running && state.last.contains("hits") && !state.last["hits"].empty());
    EnableWindow(state.view,!running && (!viewable(state).empty() || state.last.contains("preview") || state.operation=="face.compare"));
    EnableWindow(state.play,!running && state.last.contains("hits") && !state.last["hits"].empty());
    if(state.restoreHit) EnableWindow(state.restoreHit,!running && !viewable(state).empty());
}

bool showsProcessingPreview(std::string_view action) {
    return action=="face.scan" || action=="objects.scan" || action=="visual.search" ||
        action=="semantic.search" || action=="vehicle.analyze" || action=="plate.video";
}

void setSearchScope(Form& form,std::vector<fs::path> additions) {
    auto& scope=form.selections["source"];
    if(additions.empty() || fieldContent(form,"source")!=selectionLabel(scope)) scope.clear();
    for(auto& path:additions) {
        if(std::find(scope.begin(),scope.end(),path)==scope.end()) scope.push_back(std::move(path));
    }
    const auto& fields=workflows[form.index].fields;
    for(size_t i=0;i<fields.size();++i) if(fields[i].key==std::string_view("source")) {
        SetWindowTextW(form.fields[i],scope.empty()?L"":selectionLabel(scope).c_str());
    }
}

void beginTask(HWND handle,Form& state,std::string action,Json request) {
    state.galleryQuery=action=="face.gallery"?textPath(request.value("input",std::string{})):fs::path{};
    auto flag=std::make_shared<std::atomic<bool>>(false);
    state.preview=showsProcessingPreview(action)?std::make_shared<DetectionPreview>():nullptr;
    state.faceQueue.reset();
    if(action=="face.scan" && !request.contains("camera")) {
        if(!state.faceSession) state.faceSession=std::make_shared<FaceScanSession>();
        state.faceSession->beginBatch();
        state.faceQueue=state.faceSession;
        state.lastFaceSource=fieldContent(state,"source");
    }
    state.displayedFrame=0;
    state.operation=action;
    state.pending=std::async(std::launch::async,[models=state.models,worker=state.worker,action=std::move(action),
        request=std::move(request),flag,preview=state.preview,faceQueue=state.faceQueue] {
        return runOperation(action,request,localPath(models),worker,*flag,preview.get(),faceQueue.get());
    });
    state.cancelFlag=flag;
    SetWindowTextW(state.activity,L"正在处理…");
    if(state.preview) {
        if(state.pause) SetWindowTextW(state.pause,L"暂停");
        ShowWindow(state.result,SW_HIDE);
        InvalidateRect(handle,nullptr,FALSE);
    }
    buttons(state,true);
    SetTimer(handle,timer,state.preview?67:150,nullptr);
}

std::vector<fs::path> typedFaceSources(const std::wstring& text) {
    const auto trim=[](std::wstring value) {
        const auto first=value.find_first_not_of(L" \t\r\n\"");
        const auto last=value.find_last_not_of(L" \t\r\n\"");
        return first==std::wstring::npos?std::wstring{}:value.substr(first,last-first+1);
    };
    const auto value=trim(text);
    if(value.empty()) return {};
    if(fs::is_regular_file(localPath(value))) return {value};
    std::vector<fs::path> paths;
    size_t start=0;
    for(size_t i=0;i<=value.size();++i) {
        if(i<value.size() && value[i]!=L';' && value[i]!=L'\n') continue;
        const auto part=trim(value.substr(start,i-start));
        start=i+1;
        if(part.empty()) continue;
        if(!fs::is_regular_file(localPath(part))) return {};
        paths.emplace_back(part);
    }
    return paths;
}

void submitFaceSources(HWND handle,Form& state,const std::vector<fs::path>& paths) {
    std::vector<fs::path> local;
    for(const auto& path:paths) local.push_back(localPath(path));
    if(state.pending.valid()) {
        if(!state.faceQueue || *state.cancelFlag) return;
        if(!state.faceQueue->append(local)) {
            state.deferredFaceSources.insert(state.deferredFaceSources.end(),local.begin(),local.end());
        }
        SetWindowTextW(state.activity,L"已将素材加入检索队列。");
        return;
    }
    auto request=input(state,"face.scan");
    request.erase("source");
    request["sources"]=Json::array();
    for(const auto& path:local) request["sources"].push_back(pathText(path));
    beginTask(handle,state,"face.scan",std::move(request));
}

void paintDetection(HDC dc,RECT area,const DetectionPreview::Frame& frame) {
    if(frame.image.empty()) {
        theme::text(dc,L"正在读取画面…",area,theme::Font::Body,theme::muted,DT_CENTER|DT_VCENTER|DT_SINGLELINE);
        return;
    }
    const auto ratio=std::min(double(area.right-area.left)/frame.image.cols,
                              double(area.bottom-area.top)/frame.image.rows);
    if(ratio<=0.) return;
    const int width=cvRound(frame.image.cols*ratio), height=cvRound(frame.image.rows*ratio);
    BITMAPINFO canvas{};
    canvas.bmiHeader.biSize=sizeof(BITMAPINFOHEADER);
    canvas.bmiHeader.biWidth=frame.image.cols;
    canvas.bmiHeader.biHeight=-frame.image.rows;
    canvas.bmiHeader.biPlanes=1;
    canvas.bmiHeader.biBitCount=32;
    SetStretchBltMode(dc,HALFTONE);
    StretchDIBits(dc,(area.left+area.right-width)/2,(area.top+area.bottom-height)/2,width,height,
                  0,0,frame.image.cols,frame.image.rows,frame.image.data,&canvas,DIB_RGB_COLORS,SRCCOPY);
}

LRESULT CALLBACK handler(HWND handle,UINT message,WPARAM wparam,LPARAM lparam) {
    auto* state=reinterpret_cast<Form*>(GetWindowLongPtrW(handle,GWLP_USERDATA));
    if(message==WM_NCCREATE) {
        state=static_cast<Form*>(reinterpret_cast<const CREATESTRUCTW*>(lparam)->lpCreateParams);
        SetWindowLongPtrW(handle,GWLP_USERDATA,reinterpret_cast<LONG_PTR>(state));
        return TRUE;
    }
    if(!state) return DefWindowProcW(handle,message,wparam,lparam);
    switch(message) {
    case WM_CREATE: {
        const auto& workflow=workflows[state->index];
        for(size_t index=0;index<workflow.fields.size();++index) {
            const auto& field=workflow.fields[index];
            state->captions.push_back(child(handle,L"STATIC",field.title,0,0));
            state->fields.push_back(child(handle,L"EDIT",L"",ES_AUTOHSCROLL|WS_TABSTOP,700+index));
            state->choosers.push_back(field.type==Box::Text ? nullptr :
                child(handle,L"BUTTON",field.type==Box::Line?L"画线…":field.type==Box::Region?L"框选…":L"选择…",WS_TABSTOP,chooseBase+index));
        }
        for(size_t index=0;index<workflow.actions.size();++index) {
            state->buttons.push_back(child(handle,L"BUTTON",workflow.actions[index].label,WS_TABSTOP,actionBase+index));
        }
        if(state->index==1) {
            state->libraryCaption=child(handle,L"STATIC",L"目标人脸库",0,0);
            SendMessageW(state->libraryCaption,WM_SETFONT,reinterpret_cast<WPARAM>(theme::font(theme::Font::Heading)),TRUE);
            state->selectAll=child(handle,L"BUTTON",L"全选",WS_TABSTOP,selectLibraries);
            state->clearAll=child(handle,L"BUTTON",L"清空",WS_TABSTOP,clearLibraries);
            state->libraryChoices=child(handle,L"LISTBOX",L"",WS_TABSTOP|WS_BORDER|WS_VSCROLL|LBS_NOINTEGRALHEIGHT|
                LBS_MULTIPLESEL|LBS_OWNERDRAWFIXED|LBS_HASSTRINGS|LBS_NOTIFY,libraryList);
            SendMessageW(state->libraryChoices,LB_SETITEMHEIGHT,0,24);
        }
        state->activity=child(handle,L"STATIC",L"",0,0);
        state->cancelButton=child(handle,L"BUTTON",L"取消任务",WS_TABSTOP,cancelAction);
        if(state->index==1 || state->index==2) state->pause=child(handle,L"BUTTON",L"暂停",WS_TABSTOP,pauseAction);
        state->save=child(handle,L"BUTTON",L"导出结果 JSON",WS_TABSTOP,saveAction);
        state->exportHits=child(handle,L"BUTTON",L"导出目标截图",WS_TABSTOP,exportHitsAction);
        state->view=child(handle,L"BUTTON",L"预览选中画面",WS_TABSTOP,viewAction);
        if(state->index==3) state->restoreHit=child(handle,L"BUTTON",L"送去还原",WS_TABSTOP,restoreHitAction);
        state->play=child(handle,L"BUTTON",L"从命中帧播放",WS_TABSTOP,playAction);
        state->hitChoice=child(handle,L"COMBOBOX",L"",CBS_DROPDOWNLIST|WS_TABSTOP|WS_VSCROLL,600);
        state->result=child(handle,L"EDIT",L"",ES_MULTILINE|ES_READONLY|ES_AUTOVSCROLL|WS_VSCROLL|WS_TABSTOP,800);
        SendMessageW(state->result,EM_SETLIMITTEXT,4*1024*1024,0);
        buttons(*state,false);
        layout(handle,*state);
        return 0;
    }
    case WM_SIZE:
        if(state->result) layout(handle,*state);
        return 0;
    case WM_SHOWWINDOW:
        if(wparam && state->libraryChoices && !state->pending.valid()) {
            try {reloadLibraries(*state);}
            catch(const std::exception& error) {SetWindowTextW(state->activity,wide(error.what()).c_str());}
        }
        break;
    case WM_DRAWITEM: {
        const auto* item=reinterpret_cast<const DRAWITEMSTRUCT*>(lparam);
        if(item->CtlID==libraryList) drawLibraryChoice(*item);
        else {
            const UINT primaryAction=state->index==1?actionBase+1:actionBase;
            theme::drawButton(*item,item->CtlID==primaryAction?theme::Button::Primary:theme::Button::Quiet);
        }
        return TRUE;
    }
    case WM_CTLCOLORSTATIC:
    case WM_CTLCOLOREDIT:
    case WM_CTLCOLORLISTBOX:
    case WM_CTLCOLORBTN: {
        auto dc=reinterpret_cast<HDC>(wparam);
        SetBkColor(dc,theme::white);
        SetTextColor(dc,theme::textColor);
        return reinterpret_cast<LRESULT>(theme::surface());
    }
    case WM_PAINT: {
        PAINTSTRUCT paint{};
        const auto dc=BeginPaint(handle,&paint);
        RECT client{};
        GetClientRect(handle,&client);
        const int leftWidth=formWidth(client);
        const int rightStart=leftWidth+panelGap;
        theme::fill(dc,client,theme::canvas);
        theme::panel(dc,{0,0,leftWidth,client.bottom},workflows[state->index].name);
        theme::panel(dc,{rightStart,0,client.right,client.bottom},state->preview?L"实时预览":L"结果");
        theme::fill(dc,{1,client.bottom-50,leftWidth-1,client.bottom-49},theme::border);
        if(state->preview) paintDetection(dc,resultArea(client),state->preview->current());
        EndPaint(handle,&paint);
        return 0;
    }
    case WM_COMMAND: {
        const auto id=LOWORD(wparam);
        try {
            const auto& fields=workflows[state->index].fields;
            if(state->index==1 && HIWORD(wparam)==EN_CHANGE && id>=700 && id<700+fields.size() &&
               fields[id-700].key==std::string_view("source")) {
                SetTimer(handle,sourceTimer,400,nullptr);
                return 0;
            }
            if((id==selectLibraries || id==clearLibraries) && !state->pending.valid()) {
                SendMessageW(state->libraryChoices,LB_SETSEL,id==selectLibraries,-1);
                if(id==selectLibraries && content(state->activity)==wide(libraryRequired)) SetWindowTextW(state->activity,L"");
            } else if(id==libraryList && HIWORD(wparam)==LBN_SELCHANGE) {
                if(content(state->activity)==wide(libraryRequired) && SendMessageW(state->libraryChoices,LB_GETSELCOUNT,0,0)>0) {
                    SetWindowTextW(state->activity,L"");
                }
            } else if(id==cancelAction && state->cancelFlag) {
                *state->cancelFlag=true;
                state->deferredFaceSources.clear();
                if(state->pause) EnableWindow(state->pause,FALSE);
                SetWindowTextW(state->activity,L"等待当前操作结束…");
            } else if(id==pauseAction && state->preview && !*state->cancelFlag) {
                const bool paused=!state->preview->paused.load();
                state->preview->paused=paused;
                SetWindowTextW(state->pause,paused?L"继续":L"暂停");
                SetWindowTextW(state->activity,paused?L"已暂停。":L"继续检测…");
            } else if(id==saveAction && !state->last.is_null()) {
                if(const auto paths=chooseFiles(handle,Box::Json,true);!paths.empty()) writeJson(paths.front(),state->last);
            } else if(id==restoreHitAction && !state->pending.valid()) {
                const auto hits=viewable(*state);
                const auto index=std::max<LRESULT>(0,SendMessageW(state->hitChoice,CB_GETCURSEL,0,0));
                if(static_cast<size_t>(index)<hits.size()) {
                    const auto& hit=hits[index];
                    const SearchRestorationInput result{frameAt(textPath(hit.at("source").get<std::string>()),
                        hit.value("frame_index",0)),readRect(hit.at("bbox"))};
                    const auto root=GetAncestor(handle,GA_ROOTOWNER);
                    if(!SendMessageW(root,loadSearchRestoration,0,reinterpret_cast<LPARAM>(&result))) {
                        throw std::runtime_error("图像还原窗口当前忙，请稍后重试。");
                    }
                    SetForegroundWindow(root);
                    if(GetAncestor(handle,GA_ROOT)==handle) DestroyWindow(handle);
                    return 0;
                }
            } else if((id==viewAction || id==playAction) && !state->pending.valid()) {
                viewResult(handle,*state,id==playAction);
            } else if(id==exportHitsAction && state->last.contains("hits") && !state->pending.valid()) {
                if(const auto paths=chooseFiles(handle,Box::Directory);!paths.empty()) {
                    const auto folder=paths.front()/(L"目标截图-"+wide(newId()).substr(0,8));
                    beginTask(handle,*state,"hits.export",{{"hits",state->last.at("hits")},{"output",pathText(folder)}});
                }
            } else if(id>=chooseBase && id<chooseBase+state->fields.size()) {
                const auto field=workflows[state->index].fields[id-chooseBase];
                if(field.type==Box::Region || field.type==Box::Line) {
                    const auto image=fieldContent(*state,"input");
                    const auto time=fieldContent(*state,"frame_index");
                    const auto path=image.empty()?fieldContent(*state,"source"):image;
                    if(path.empty()) throw std::invalid_argument("请先选择参考图或源视频。");
                    const auto pixels=frameAt(path,time.empty()?0:std::stoi(time));
                    std::optional<VideoLine> points;
                    if(field.type==Box::Line) points=selectNativeLine(handle,pixels);
                    else if(const auto region=previewNative(handle,pixels,L"在画面上拖动鼠标框选",true)) {
                        points=VideoLine{region->tl(),region->br()};
                    }
                    if(points) {
                        const auto coordinates=std::to_wstring(points->start.x)+L" "+std::to_wstring(points->start.y)+L" "
                            +std::to_wstring(points->end.x)+L" "+std::to_wstring(points->end.y);
                        SetWindowTextW(state->fields[id-chooseBase],coordinates.c_str());
                        const std::string_view counterpart=field.type==Box::Line?"trigger_area":"trigger_line";
                        if(std::string_view(field.key)!="bbox") {
                            for(size_t i=0;i<fields.size();++i) {
                                if(fields[i].key==counterpart) SetWindowTextW(state->fields[i],L"");
                            }
                        }
                    }
                    return 0;
                }
                const auto key=std::string_view(field.key);
                const bool multiple=key=="photos" || (key=="source" && state->index>=1 && state->index<=4);
                auto paths=chooseFiles(handle,field.type,false,multiple);
                if(!paths.empty()) {
                    if(key=="source" && (state->index==3 || state->index==4)) {
                        setSearchScope(*state,std::move(paths));
                        return 0;
                    }
                    state->selections[field.key]=paths;
                    const auto shown=selectionLabel(paths);
                    SetWindowTextW(state->fields[id-chooseBase],shown.c_str());
                    if(state->index==1 && key=="source") submitFaceSources(handle,*state,paths);
                }
            } else if(id>=actionBase && id<actionBase+state->buttons.size() && !state->pending.valid()) {
                auto action=std::string(workflows[state->index].actions[id-actionBase].name);
                if(action=="sources.folder" || action=="sources.clear") {
                    if(action=="sources.clear") setSearchScope(*state,{});
                    else if(auto folders=chooseFiles(handle,Box::Directory);!folders.empty()) setSearchScope(*state,std::move(folders));
                    return 0;
                }
                if(action=="person.remove" || action=="person.sample.remove" || action=="library.remove") {
                    if(MessageBoxW(handle,L"确认移除所填 ID 对应的记录？",L"人员库",MB_OKCANCEL|MB_ICONQUESTION)!=IDOK) return 0;
                }
                auto request=input(*state,action);
                if(action=="face.camera") action="face.scan";
                if(action=="plate.preview") {request.erase("bbox"); action="plate.video";}
                if(action=="associate" || action=="reject" || action=="undo_rejection" || action=="remove_association") {
                    request["action"]=action;
                    action="annotation.write";
                }
                beginTask(handle,*state,std::move(action),std::move(request));
            }
        } catch(const std::exception& error) {
            SetWindowTextW(state->activity,wide(error.what()).c_str());
        }
        return 0;
    }
    case WM_TIMER:
        if(wparam==sourceTimer) {
            KillTimer(handle,sourceTimer);
            if(!state->pending.valid()) {
                const auto source=fieldContent(*state,"source");
                try {
                    if(source!=state->lastFaceSource) {
                        const auto inputs=typedFaceSources(source);
                        if(!inputs.empty()) submitFaceSources(handle,*state,inputs);
                    }
                } catch(const std::exception& error) {
                    SetWindowTextW(state->activity,wide(error.what()).c_str());
                }
            }
            return 0;
        }
        if(wparam==timer && state->preview) {
            const auto frame=state->preview->current();
            if(frame.index!=state->displayedFrame) {
                state->displayedFrame=frame.index;
                if(!state->preview->paused && !*state->cancelFlag) {
                    const auto text=L"已处理 "+std::to_wstring(frame.index)+L" 帧 · 当前 "+std::to_wstring(frame.detections)+L" 个目标";
                    SetWindowTextW(state->activity,text.c_str());
                }
                RECT client{};
                GetClientRect(handle,&client);
                const auto area=resultArea(client);
                InvalidateRect(handle,&area,FALSE);
            }
        }
        if(wparam==timer && state->pending.valid() &&
           state->pending.wait_for(std::chrono::seconds(0))==std::future_status::ready) {
            KillTimer(handle,timer);
            try {
                state->last=state->pending.get();
                SendMessageW(state->hitChoice,CB_RESETCONTENT,0,0);
                const auto rows=viewable(*state);
                for(size_t i=0;i<rows.size();++i) {
                    const auto& row=rows[i];
                    std::wostringstream text;
                    text<<i+1<<L"  "<<wide(row.value("name",row.value("model",row.value("method",std::string("候选")))));
                    if(row.contains("score")) text<<L"  "<<std::fixed<<std::setprecision(3)<<row.at("score").get<double>();
                    if(row.contains("frame_index")) text<<L"  第 "<<row.at("frame_index").get<int>()+1<<L" 帧";
                    const auto label=text.str();
                    SendMessageW(state->hitChoice,CB_ADDSTRING,0,reinterpret_cast<LPARAM>(label.c_str()));
                }
                if(!rows.empty()) SendMessageW(state->hitChoice,CB_SETCURSEL,0,0);
                const auto description=state->operation=="vehicle.analyze" ? vehicleText(state->last) : wide(state->last.dump(2));
                SetWindowTextW(state->result,description.c_str());
                SetWindowTextW(state->activity,*state->cancelFlag?L"操作已结束。":L"处理完成。");
                if(state->operation=="settings.load") {
                    const auto& fields=workflows[state->index].fields;
                    for(size_t index=0;index<fields.size();++index) {
                        const auto key=fields[index].key;
                        if(state->last.contains(key)) {
                            const auto& value=state->last[key];
                            const auto formatted=value.is_boolean() ? (value.get<bool>()?L"1":L"0") : wide(value.dump());
                            SetWindowTextW(state->fields[index],formatted.c_str());
                        }
                    }
                }
            } catch(const std::exception& error) {
                SetWindowTextW(state->activity,L"操作未完成。");
                SetWindowTextW(state->result,wide(error.what()).c_str());
            }
            if(state->preview) {
                state->preview.reset();
                ShowWindow(state->result,SW_SHOW);
                InvalidateRect(handle,nullptr,FALSE);
            }
            state->cancelFlag.reset();
            state->faceQueue.reset();
            buttons(*state,false);
            if(!state->deferredFaceSources.empty()) {
                const auto paths=std::exchange(state->deferredFaceSources,{});
                try {submitFaceSources(handle,*state,paths);}
                catch(const std::exception& error) {SetWindowTextW(state->activity,wide(error.what()).c_str());}
            }
        }
        return 0;
    case WM_NCDESTROY:
        if(state->cancelFlag) *state->cancelFlag=true;
        SetWindowLongPtrW(handle,GWLP_USERDATA,0);
        if(state->adopted) delete state;
        break;
    }
    return DefWindowProcW(handle,message,wparam,lparam);
}

}

bool requestWorkflowClose(HWND page) {
    auto* state=reinterpret_cast<Form*>(GetWindowLongPtrW(page,GWLP_USERDATA));
    if(!state || !state->pending.valid()) return true;
    *state->cancelFlag=true;
    state->deferredFaceSources.clear();
    SetWindowTextW(state->activity,L"正在结束当前操作，请稍候。");
    return false;
}

HWND createWorkflowPage(HWND parent,const fs::path& models,const fs::path& worker,size_t workflow) {
    if(workflow>=workflows.size()) throw std::invalid_argument("未知功能入口。");
    WNDCLASSW type{};
    type.hInstance=GetModuleHandleW(nullptr);
    type.lpfnWndProc=handler;
    type.lpszClassName=windowClass;
    type.hCursor=LoadCursorW(nullptr,IDC_ARROW);
    type.hbrBackground=theme::background();
    type.style=CS_HREDRAW|CS_VREDRAW;
    if(!RegisterClassW(&type) && GetLastError()!=ERROR_CLASS_ALREADY_EXISTS) throw std::runtime_error("无法注册功能窗口。");
    auto state=std::make_unique<Form>();
    state->models=models;
    state->worker=worker;
    state->index=workflow;
    const auto handle=CreateWindowExW(WS_EX_CONTROLPARENT,windowClass,workflows[workflow].name,WS_CHILD|WS_CLIPCHILDREN,
        0,0,1000,580,parent,nullptr,type.hInstance,state.get());
    if(!handle) throw std::runtime_error("无法打开功能窗口。");
    state.release()->adopted=true;
    return handle;
}

}
