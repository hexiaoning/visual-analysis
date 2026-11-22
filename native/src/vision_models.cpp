#include "vision_models.h"
#include "model_session.h"
#include "local_images.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <numeric>
#include <sstream>
#include <unordered_map>

#include <Windows.h>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>

namespace visual {
namespace fs = std::filesystem;

Feature normalized(Feature value) {
    const double norm = std::sqrt(std::inner_product(value.begin(),value.end(),value.begin(),0.0));
    if (!std::isfinite(norm) || norm < 1e-12) throw std::runtime_error("模型返回无效的零向量。");
    for (auto& number : value) number /= static_cast<float>(norm);
    return value;
}

double similarity(std::span<const float> first, std::span<const float> second) {
    if (first.size() != second.size() || first.empty()) throw std::runtime_error("特征维度不一致。");
    return std::clamp(std::inner_product(first.begin(),first.end(),second.begin(),0.0),-1.0,1.0);
}

FaceSettings FaceSettings::from(const Json& config, const fs::path& models) {
    FaceSettings result;
    if (config.value("model_profile",std::string("auraface")) == "lvface-b"
        && fs::is_regular_file(models / L"LVFace-B_Glint360K.onnx")) result.profile = "lvface-b";
    result.mirror = config.value("mirror_augmentation",true);
    result.threshold = std::clamp(config.value("detector_threshold",.4f),.05f,.99f);
    result.minimumSize = std::clamp(config.value("min_face_size",32),12,4096);
    result.minimumQuality = std::clamp(config.value("min_quality",.22f),0.f,1.f);
    result.enrollmentQuality = std::clamp(config.value("enrollment_min_quality",.30f),0.f,1.f);
    result.matchThreshold = std::clamp(config.value("match_threshold",.45),0.,1.);
    result.margin = std::clamp(config.value("match_margin",.06),0.,.5);
    return result;
}

std::string FaceSettings::modelId() const {
    return std::string(profile == "lvface-b" ? "lvface-b-glint360k-v1" : "auraface-r100-v1")
        + (mirror ? "-tta" : "-single");
}

namespace {
struct ObjectClass { int id; const char* english; const char* chinese; };
constexpr std::array classes{
    ObjectClass{0,"person","人"},ObjectClass{1,"bicycle","自行车"},ObjectClass{2,"car","汽车"},
    ObjectClass{3,"motorcycle","摩托车"},ObjectClass{5,"bus","公交车"},
    ObjectClass{6,"train","列车"},ObjectClass{7,"truck","货车"},
};

const ObjectClass* objectClass(int id) {
    const auto element = std::find_if(classes.begin(),classes.end(),[&](auto item){ return item.id == id; });
    return element == classes.end() ? nullptr : &*element;
}

Feature semanticOutput(const std::vector<Ort::Value>& output) {
    const auto& tensor = output.at(0);
    const auto size = tensor.GetTensorTypeAndShapeInfo().GetElementCount();
    if (size != 512) throw std::runtime_error("语义模型输出须包含 512 维特征。");
    const auto* data = tensor.GetTensorData<float>();
    return normalized(Feature(data,data+size));
}

std::vector<std::wstring> basicTokens(const std::string& text) {
    auto source = wide(text);
    if (source.size() > 500) throw std::runtime_error("描述过长。");
    const int lowerLength = LCMapStringEx(LOCALE_NAME_INVARIANT,LCMAP_LOWERCASE,source.data(),
                                         static_cast<int>(source.size()),nullptr,0,nullptr,nullptr,0);
    if (lowerLength <= 0) throw std::runtime_error("请输入文字描述。");
    std::wstring lower(lowerLength,L'\0');
    LCMapStringEx(LOCALE_NAME_INVARIANT,LCMAP_LOWERCASE,source.data(),static_cast<int>(source.size()),
                  lower.data(),lowerLength,nullptr,nullptr,0);
    const auto length = NormalizeString(NormalizationD,lower.data(),lowerLength,nullptr,0);
    if (length <= 0) throw std::runtime_error("无法标准化描述。");
    std::wstring decomposed(length,L'\0');
    const auto written = NormalizeString(NormalizationD,lower.data(),lowerLength,decomposed.data(),length);
    if (written <= 0) throw std::runtime_error("无法标准化描述。");
    decomposed.resize(written);
    std::vector<std::wstring> result;
    std::wstring word;
    const auto flush = [&]{
        if (!word.empty()) result.push_back(std::exchange(word,{}));
    };
    for (size_t index=0; index<decomposed.size(); ++index) {
        const auto ch = decomposed[index];
        WORD type1{},type3{};
        GetStringTypeW(CT_CTYPE1,&ch,1,&type1);
        GetStringTypeW(CT_CTYPE3,&ch,1,&type3);
        if (type3 & C3_NONSPACING) continue;
        if (type1 & C1_SPACE) {
            flush();
            continue;
        }
        if (ch == 0 || ch == 0xfffd || (type1 & C1_CNTRL)) continue;
        const bool chinese = (ch >= 0x3400 && ch <= 0x9fff) || (ch >= 0xf900 && ch <= 0xfaff);
        const bool supplementary = ch >= 0xd840 && ch <= 0xd87e && index+1 < decomposed.size()
            && decomposed[index+1] >= 0xdc00 && decomposed[index+1] <= 0xdfff;
        const bool punctuation = (type1 & C1_PUNCT) || (ch >= 33 && ch <= 47) || (ch >= 58 && ch <= 64)
            || (ch >= 91 && ch <= 96) || (ch >= 123 && ch <= 126);
        if (chinese || supplementary || punctuation) {
            flush();
            std::wstring token(1,ch);
            if (supplementary) token += decomposed[++index];
            result.push_back(std::move(token));
        } else word += ch;
    }
    flush();
    return result;
}
}

std::string objectName(int category) {
    const auto* found = objectClass(category);
    return found ? found->chinese : std::string{};
}

cv::Mat markObjects(const cv::Mat& frame,std::span<const ObjectObservation> objects) {
    auto output = frame.clone();
    for (const auto& item : objects) {
        const cv::Scalar color = item.category == 0 ? cv::Scalar(0,0,255) : cv::Scalar(0,255,255);
        cv::rectangle(output,item.box,color,std::max(2,std::min(output.rows,output.cols)/360));
        const auto* type = objectClass(item.category);
        if (type) cv::putText(output,std::string(type->english)+" "+cv::format("%.2f",item.score),
            {item.box.x,std::max(15,item.box.y-4)},cv::FONT_HERSHEY_SIMPLEX,.5,color,1,cv::LINE_AA);
    }
    return output;
}

struct ObjectDetector::State {
    explicit State(const fs::path& models) : model(models / L"rtdetr_v2_r50vd.onnx") {}
    ModelSession model;
};
ObjectDetector::ObjectDetector(const fs::path& models) : state_(std::make_unique<State>(models)) {}
ObjectDetector::~ObjectDetector() = default;

std::vector<ObjectObservation> ObjectDetector::detect(const cv::Mat& image,double threshold,bool people,bool vehicles) {
    if (image.empty()) return {};
    auto outputs = state_->model.run(cv::dnn::blobFromImage(image,1./255,{640,640},{},true));
    const auto& names = state_->model.outputNames();
    const auto get = [&](std::string_view name) -> const Ort::Value& {
        const auto match = std::find(names.begin(),names.end(),name);
        if (match == names.end()) throw std::runtime_error("人车模型缺少输出张量。");
        return outputs[match-names.begin()];
    };
    const auto& scores = get("logits");
    const auto& boxes = get("pred_boxes");
    if (scores.GetTensorTypeAndShapeInfo().GetShape() != std::vector<int64_t>{1,300,80}
        || boxes.GetTensorTypeAndShapeInfo().GetShape() != std::vector<int64_t>{1,300,4}) {
        throw std::runtime_error("人车检测模型张量格式有误。");
    }
    const auto* values = scores.GetTensorData<float>();
    const auto* coordinates = boxes.GetTensorData<float>();
    std::vector<ObjectObservation> result;
    for (int row=0; row<300; ++row) {
        const auto* score = values+row*80;
        if (!std::all_of(score,score+80,[](float n){return std::isfinite(n);})) continue;
        const auto category = int(std::max_element(score,score+80)-score);
        if (!objectClass(category) || !(category == 0 ? people : vehicles)) continue;
        const double confidence = 1/(1+std::exp(-std::clamp(double(score[category]),-80.,80.)));
        if (confidence < threshold) continue;
        const auto* p = coordinates+row*4;
        if (!std::all_of(p,p+4,[](float n){return std::isfinite(n);})) continue;
        const auto x = [&](float point){ return std::clamp(double(point)*image.cols,0.,image.cols-1.); };
        const auto y = [&](float point){ return std::clamp(double(point)*image.rows,0.,image.rows-1.); };
        const int x1 = int(std::floor(x(p[0]-p[2]/2))), x2 = int(std::ceil(x(p[0]+p[2]/2)));
        const int y1 = int(std::floor(y(p[1]-p[3]/2))), y2 = int(std::ceil(y(p[1]+p[3]/2)));
        if (x2 > x1 && y2 > y1) result.push_back({category,confidence,{x1,y1,x2-x1,y2-y1}});
    }
    std::stable_sort(result.begin(),result.end(),[](const auto& a,const auto& b){return a.score>b.score;});
    return result;
}

struct SemanticEncoder::State {
    explicit State(fs::path folder) : models(std::move(folder)) {
        std::ifstream input(models/L"chinese_clip_vocab.txt",std::ios::binary);
        int64_t index=0;
        for (std::string token; std::getline(input,token); ++index) {
            if (!token.empty() && token.back()=='\r') token.pop_back();
            vocab.emplace(wide(token),index);
        }
        for (const auto* key:{L"[CLS]",L"[SEP]",L"[PAD]",L"[UNK]"}) {
            if (!vocab.contains(key)) throw std::runtime_error("缺少完整的 Chinese-CLIP 本地词表。");
        }
    }
    fs::path models;
    std::unique_ptr<ModelSession> vision, text;
    std::unordered_map<std::wstring,int64_t> vocab;
    std::unordered_map<std::string,Feature> cached;
};
SemanticEncoder::SemanticEncoder(const fs::path& models) : state_(std::make_unique<State>(models)) {}
SemanticEncoder::~SemanticEncoder() = default;

Feature SemanticEncoder::image(const cv::Mat& image) {
    if (!state_->vision) state_->vision=std::make_unique<ModelSession>(state_->models/L"chinese_clip_vision.onnx");
    cv::Mat reduced;
    cv::resize(image,reduced,{224,224},0,0,cv::INTER_CUBIC);
    auto blob=cv::dnn::blobFromImage(reduced,1./255,{}, {},true);
    constexpr std::array mu{.48145466f,.4578275f,.40821073f}, sigma{.26862954f,.26130258f,.27577711f};
    auto* values=blob.ptr<float>();
    for (int c=0;c<3;++c) {
        for (int index=0;index<224*224;++index) {
            auto& value=values[c*224*224+index];
            value=(value-mu[c])/sigma[c];
        }
    }
    return semanticOutput(state_->vision->run(blob));
}

Feature SemanticEncoder::text(const std::string& text) {
    if (const auto cached=state_->cached.find(text); cached!=state_->cached.end()) return cached->second;
    std::vector<int64_t> tokens{state_->vocab.at(L"[CLS]")};
    for (const auto& token:basicTokens(text)) {
        std::vector<int64_t> words;
        size_t index=0;
        while (index<token.size() && token.size()<=100) {
            size_t end=token.size();
            for (;end>index;--end) {
                const auto part=(index ? L"##" : L"")+token.substr(index,end-index);
                if (const auto found=state_->vocab.find(part);found!=state_->vocab.end()) {
                    words.push_back(found->second);
                    break;
                }
            }
            if (end==index) break;
            index=end;
        }
        if (index!=token.size()) tokens.push_back(state_->vocab.at(L"[UNK]"));
        else tokens.insert(tokens.end(),words.begin(),words.end());
    }
    if (tokens.size()>51) throw std::runtime_error("描述请缩短到 50 个词元以内。");
    tokens.push_back(state_->vocab.at(L"[SEP]"));
    std::array<int64_t,52> mask{}, types{};
    std::fill_n(mask.begin(),tokens.size(),1);
    tokens.resize(mask.size(),state_->vocab.at(L"[PAD]"));
    if (!state_->text) state_->text=std::make_unique<ModelSession>(state_->models/L"chinese_clip_text.onnx");
    constexpr std::array<int64_t,2> shape{1,52};
    std::vector<Ort::Value> input;
    for (const auto& name:state_->text->inputNames()) {
        if (name=="input_ids") input.push_back(ModelSession::tensor<int64_t>(tokens,shape));
        else if (name=="attention_mask") input.push_back(ModelSession::tensor<int64_t>(mask,shape));
        else if (name=="token_type_ids") input.push_back(ModelSession::tensor<int64_t>(types,shape));
        else throw std::runtime_error("语义文本模型有未识别的输入。");
    }
    auto result=semanticOutput(state_->text->run(input));
    if (state_->cached.size()<1024) state_->cached.emplace(text,result);
    return result;
}

}
