#include "plate_video.h"
#include "local_images.h"
#include "people.h"
#include "portrait_video.h"

#include <algorithm>
#include <cmath>
#include <opencv2/imgproc.hpp>
#include <opencv2/video/tracking.hpp>

namespace visual {
namespace fs=std::filesystem;
namespace {

cv::Mat gray(const cv::Mat& color) {
    cv::Mat result;
    cv::cvtColor(color,result,cv::COLOR_BGR2GRAY);
    return result;
}

struct Observation {
    int frame{};
    double time{};
    cv::Rect box;
    cv::Mat pixels;
    double sharpness{};
};

Observation observe(const cv::Mat& pixels,cv::Rect box,int frame,double time) {
    auto crop = pixels(box).clone();
    cv::Mat laplacian;
    auto luma=gray(crop);
    cv::GaussianBlur(luma,luma,{3,3},.5);
    cv::Laplacian(luma,laplacian,CV_32F);
    cv::Scalar mean,sd;
    cv::meanStdDev(laplacian,mean,sd);
    return {frame,time,box,std::move(crop),sd[0]*sd[0]};
}

cv::Rect track(const cv::Mat& image,const cv::Mat& target,cv::Rect previous) {
    const int radius=std::clamp(target.cols,16,192);
    const auto expanded=cv::Rect(previous.x-radius,previous.y-radius,previous.width+2*radius,previous.height+2*radius)
        & cv::Rect(0,0,image.cols,image.rows);
    if(expanded.width<target.cols || expanded.height<target.rows) return {};
    cv::Mat correlation;
    cv::matchTemplate(gray(image(expanded)),target,correlation,cv::TM_CCOEFF_NORMED);
    double best{},runner{};
    cv::Point point;
    cv::minMaxLoc(correlation,nullptr,&best,nullptr,&point);
    if(!std::isfinite(best) || best<.70) return {};
    const auto neighborhood=cv::Rect(point.x-target.cols/2,point.y-target.rows/2,target.cols+1,target.rows+1)
        & cv::Rect(0,0,correlation.cols,correlation.rows);
    correlation(neighborhood).setTo(-1.f);
    cv::minMaxLoc(correlation,nullptr,&runner);
    if(runner>=best-.07) return {};
    return {expanded.tl()+point,target.size()};
}

bool fuse(const cv::Mat& reference,const cv::Mat& next,cv::Mat& sums,cv::Mat& weights,float weight) {
    const auto ref=gray(reference);
    auto warp=cv::Mat::eye(2,3,CV_32F);
    cv::Mat transform=warp;
    double ecc{};
    try {ecc=cv::findTransformECC(ref,gray(next),transform,cv::MOTION_EUCLIDEAN,
          {cv::TermCriteria::COUNT|cv::TermCriteria::EPS,35,.001});} catch(const cv::Exception&) {return false;}
    if(!std::isfinite(ecc) || ecc<.88 || !cv::checkRange(transform) || std::abs(transform.at<float>(0,1))>.15
       || cv::norm(transform.col(2))>std::min(ref.cols,ref.rows)/4.) return false;
    cv::Mat aligned,mask,coverage;
    cv::warpAffine(next,aligned,transform,ref.size(),cv::INTER_LINEAR|cv::WARP_INVERSE_MAP);
    const cv::Mat inside(ref.size(),CV_8U,cv::Scalar(255));
    cv::warpAffine(inside,mask,transform,ref.size(),cv::INTER_NEAREST|cv::WARP_INVERSE_MAP);
    if(std::min(ref.cols,ref.rows)>=24) {
        const auto donor=gray(aligned);
        cv::Mat forward,backward;
        cv::calcOpticalFlowFarneback(ref,donor,forward,.5,2,13,3,5,1.2,0);
        cv::calcOpticalFlowFarneback(donor,ref,backward,.5,2,13,3,5,1.2,0);
        cv::Mat map=forward.clone();
        for(int y=0;y<map.rows;++y) for(int x=0;x<map.cols;++x) map.at<cv::Vec2f>(y,x)+=cv::Vec2f(float(x),float(y));
        cv::Mat reverse,remapped;
        cv::remap(backward,reverse,map,{},cv::INTER_LINEAR,cv::BORDER_CONSTANT,cv::Scalar(100,100));
        cv::remap(aligned,remapped,map,{},cv::INTER_LINEAR);
        cv::remap(mask,coverage,map,{},cv::INTER_NEAREST);
        aligned=std::move(remapped);
        mask=std::move(coverage);
        for(int y=0;y<mask.rows;++y) for(int x=0;x<mask.cols;++x) {
            const auto movement=forward.at<cv::Vec2f>(y,x);
            if(cv::norm(movement)>2.5 || cv::norm(movement+reverse.at<cv::Vec2f>(y,x))>.85) mask.at<uchar>(y,x)=0;
        }
    }
    for(int y=0;y<mask.rows;++y) for(int x=0;x<mask.cols;++x) {
        const auto a=reference.at<cv::Vec3b>(y,x),b=aligned.at<cv::Vec3b>(y,x);
        if(std::max({std::abs(int(a[0])-b[0]),std::abs(int(a[1])-b[1]),std::abs(int(a[2])-b[2])})>28) mask.at<uchar>(y,x)=0;
    }
    if(cv::countNonZero(mask)<.55*static_cast<double>(mask.total())) return false;
    for(int y=0;y<mask.rows;++y) for(int x=0;x<mask.cols;++x) if(mask.at<uchar>(y,x)) {
        const auto val=aligned.at<cv::Vec3b>(y,x);
        sums.at<cv::Vec3f>(y,x)+=cv::Vec3f(val[0],val[1],val[2])*weight;
        weights.at<float>(y,x)+=weight;
    }
    return true;
}

}

Json plateVideo(const Json& request,const std::atomic<bool>& cancel,DetectionPreview* processingPreview) {
    const auto source=localPath(textPath(request.at("source").get<std::string>()));
    const int anchor=request.value("frame_index",0), count=request.value("count",12);
    const int blur=request.value("blur_length",0), angle=request.value("blur_angle",0);
    if(anchor<0 || count<2 || count>60 || (blur!=0 && (blur<3 || blur>21)) || angle<0 || angle>=180) {
        throw std::invalid_argument("车牌视频参数超出范围。");
    }
    const bool preview=!request.contains("bbox");
    cv::Rect box=preview ? cv::Rect{} : readRect(request.at("bbox"));
    if(!preview && (box.width<16 || box.width>640 || box.height<8 || box.height>320)) throw std::invalid_argument("车牌选区尺寸无效。");
    cv::Mat anchorFrame,pattern;
    std::vector<Observation> frames;
    std::string notice;
    visitVideo(source,[&](const cv::Mat& frame,int index,double seconds){
        if(index==anchor) {
            anchorFrame=frame.clone();
            if(preview) return false;
            if((box&cv::Rect(0,0,frame.cols,frame.rows))!=box) throw std::invalid_argument("车牌选区超出画面。");
            pattern=gray(frame(box));
            frames.push_back(observe(frame,box,index,seconds));
            cv::Scalar average,deviation;
            cv::meanStdDev(pattern,average,deviation);
            if(deviation[0]<4.) {notice="起始选区纹理不足。";return false;}
        } else {
            box=track(frame,pattern,box);
            if(box.empty()) {notice="目标遮挡、切换或跟踪不明确。";return false;}
            frames.push_back(observe(frame,box,index,seconds));
        }
        if(processingPreview && processingPreview->refreshDue() && !cancel) {
            auto marked=frame.clone();
            cv::rectangle(marked,box,{255,200,70},2);
            processingPreview->update(marked,index-anchor+1,1);
        }
        return !cancel && index-anchor+1<count;
    },1,cancel,anchor);
    if(cancel) return Json::object();
    if(anchorFrame.empty()) throw std::runtime_error("起始帧超出视频长度。");
    const auto target=localPath(textPath(request.at("output").get<std::string>()));
    if(!fs::create_directory(target)) throw std::invalid_argument("请选择新的车牌输出目录。");
    writePng(target/L"原始画面.png",anchorFrame);
    if(preview) return {{"preview",pathText(target/L"原始画面.png")},{"frame_index",anchor},
                       {"width",anchorFrame.cols},{"height",anchorFrame.rows}};
    std::stable_sort(frames.begin(),frames.end(),[](const auto& a,const auto& b){return a.sharpness>b.sharpness;});
    const auto& best=frames.front();
    cv::Mat sums;
    best.pixels.convertTo(sums,CV_32F);
    cv::Mat weights(best.pixels.size(),CV_32F,cv::Scalar(1));
    std::vector<size_t> used{0};
    for(size_t i=1;i<frames.size() && used.size()<8 && !cancel;++i) {
        if(frames[i].sharpness<best.sharpness*.65) break;
        if(fuse(best.pixels,frames[i].pixels,sums,weights,float(.5/std::sqrt(used.size())))) used.push_back(i);
    }
    if(cancel) return Json::object();
    auto fused=best.pixels.clone();
    for(int y=0;y<fused.rows;++y) for(int x=0;x<fused.cols;++x) {
        const auto rgb=sums.at<cv::Vec3f>(y,x)/weights.at<float>(y,x);
        for(int c=0;c<3;++c) fused.at<cv::Vec3b>(y,x)[c]=cv::saturate_cast<uchar>(rgb[c]);
    }
    Candidates options;
    const auto add=[&](const char* model,const cv::Mat& pixels) {
        auto result=scoreRestoration(model,pixels,best.pixels);
        result.pixels=pixels;
        options.push_back(std::move(result));
    };
    if(used.size()>1) {
        add("多帧一致性融合",fused);
        if(cv::norm(fused,best.pixels,cv::NORM_L1)>3.*fused.total()) add("参考清晰帧",best.pixels);
    }
    if(blur>=3) add("约束去拖影",deblurPlate(fused,blur,angle));
    if(options.empty()) options=restorePlate(best.pixels,{0,0,best.pixels.cols,best.pixels.rows});
    rankRestorations(options);
    Json candidates=Json::array(),originals=Json::array();
    for(size_t i=0;i<options.size();++i) {
        const auto& option=options[i];
        const auto file=target/(L"候选-"+std::to_wstring(i+1)+L".png");
        writePng(file,option.pixels);
        candidates.push_back({{"model",option.model},{"score",option.score},{"agreement",option.agreement},{"file",pathText(file)}});
    }
    for(const auto i:used) {
        const auto& frame=frames[i];
        const auto file=target/(L"所用原始帧-"+std::to_wstring(frame.frame)+L".png");
        writePng(file,frame.pixels);
        originals.push_back({{"frame_index",frame.frame},{"seconds",frame.time},{"bbox",rectJson(frame.box)},{"file",pathText(file)}});
    }
    Json manifest={{"schema_version",1},{"algorithm","plate-video-ecc-flow-v2-native"},
        {"source",pathText(source)},{"anchor_index",anchor},{"tracked",frames.size()},
        {"notice",notice},{"observations",originals},{"candidates",candidates}};
    writeJson(target/L"车牌还原报告.json",manifest);
    return manifest;
}

}
