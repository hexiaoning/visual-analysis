#include "learned_exposure.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <limits>
#include <numbers>
#include <numeric>
#include <stdexcept>
#include <tuple>

#include <opencv2/imgproc.hpp>

namespace portrait_selection {
namespace {

constexpr int side = 512;
constexpr int grid = 128;
constexpr int feature_size = 63;
const cv::Size tile{38, 32};
constexpr double gamma = 2.2;
constexpr double light_limit = .75;
constexpr double pi = std::numbers::pi_v<double>;
using Weights = std::array<int, 4>;

struct FaceWindows {
    std::array<cv::Mat, 4> parts;
    std::array<cv::Rect, 4> bounds;
    cv::Mat combined = cv::Mat::zeros(side, side, CV_32FC1);
    cv::Mat visible;
    std::array<cv::Mat, 2> fit_weights;
    std::array<std::vector<cv::Point>, 2> outside;
    std::vector<cv::Point> inner;

    FaceWindows() {
        const int coordinates[4][4]{{198,235,66,40}, {324,235,66,40}, {261,306,67,62}, {265,375,90,45}};
        for (std::size_t i=0; i<parts.size(); ++i) {
            parts[i].create(side, side, CV_32FC1);
            const auto& p = coordinates[i];
            for (int y=0; y<side; ++y) {
                auto* dst = parts[i].ptr<float>(y);
                for (int x=0; x<side; ++x) {
                    const float radius = std::hypot((x-p[0])/float(p[2]), (y-p[1])/float(p[3]));
                    const float t = std::clamp((radius-.48f)/.52f, 0.f, 1.f);
                    dst[x] = .5f+.5f*std::cos(float(pi)*t);
                }
            }
            combined += parts[i];
        }
        cv::Mat norm;
        cv::max(combined, 1., norm);
        combined.setTo(0);
        cv::Mat strongest = combined.clone();
        for (std::size_t i=0; i<parts.size(); ++i) {
            cv::divide(parts[i], norm, parts[i]);
            combined += parts[i];
            bounds[i] = cv::boundingRect(parts[i] > 0.f);
            cv::max(strongest, parts[i], strongest);
        }
        visible = combined > 0.f;
        cv::findNonZero(visible, inner);
        cv::resize(strongest, strongest, {grid,grid}, 0., 0., cv::INTER_AREA);
        for (int y=0; y<grid; ++y) {
            for (int x=0; x<grid; ++x) {
                if (strongest.at<float>(y,x) < .06f && std::hypot((x-64.)/43.,(y-71.)/48.)<1.)
                    outside[(y/5+x/6)%2].emplace_back(x,y);
            }
        }
        for (int fold=0; fold<2; ++fold) {
            fit_weights[fold] = cv::Mat::zeros(side, side, CV_32F);
            for (int y=0; y<side; ++y)
                for (int x=0; x<side; ++x)
                    if ((y/23+x/21)%2==fold)
                        fit_weights[fold].at<float>(y,x) = .14f+.86f*combined.at<float>(y,x);
        }
    }
};

struct Example {
    int cohort;
    std::array<cv::Mat,4> observed;
    std::array<cv::Mat,4> residual;
};

std::vector<Example> read_dictionary(const std::filesystem::path& path) {
    std::ifstream stream(path,std::ios::binary);
    const auto read = [&](auto* item, std::size_t count) {
        stream.read(reinterpret_cast<char*>(item), static_cast<std::streamsize>(count*sizeof(*item)));
        if (!stream)
            throw std::invalid_argument("Incomplete learned exposure dictionary");
    };
    std::array<char,8> magic{};
    read(magic.data(),magic.size());
    if (magic!=std::array<char,8>{'V','A','D','I','R','0','1','\0'})
        throw std::invalid_argument("Unsupported learned exposure dictionary format");
    std::uint32_t count{};
    read(&count,1);
    if (count!=168)
        throw std::invalid_argument("Unexpected number of learned exposure examples");
    std::vector<Example> result;
    result.reserve(count);
    for (std::uint32_t i=0; i<count; ++i) {
        Example item{};
        std::uint32_t cohort{};
        read(&cohort,1);
        if (cohort>=6)
            throw std::invalid_argument("Invalid learned exposure cohort");
        item.cohort=static_cast<int>(cohort);
        for (auto& v : item.observed) {
            v.create(1,feature_size,CV_32F);
            read(v.ptr<float>(),v.total());
            if (!cv::checkRange(v,true,nullptr,-256.,256.))
                throw std::invalid_argument("Invalid learned observation");
        }
        for (auto& v : item.residual) {
            v.create(tile,CV_32F);
            read(v.ptr<float>(),v.total());
            if (!cv::checkRange(v,true,nullptr,-39.,39.))
                throw std::invalid_argument("Invalid learned residual");
        }
        result.push_back(std::move(item));
    }
    if (stream.peek()!=std::char_traits<char>::eof())
        throw std::invalid_argument("Trailing data in learned exposure dictionary");
    return result;
}

cv::Mat gray(const cv::Mat& image) {
    if (image.type()!=CV_8UC3 || image.size()!=cv::Size(side,side))
        throw std::invalid_argument("Expected 512x512 BGR exposure input");
    cv::Mat y;
    cv::cvtColor(image,y,cv::COLOR_BGR2GRAY);
    y.convertTo(y,CV_32F);
    return y;
}

cv::Mat smooth(const cv::Mat& image, double sigma) {
    cv::Mat result;
    cv::GaussianBlur(image,result,{},sigma,0.,cv::BORDER_REFLECT_101);
    return result;
}

cv::Mat band(const cv::Mat& image) {
    return image-smooth(image,8.);
}

cv::Mat clip(const cv::Mat& image,double low,double high) {
    cv::Mat result;
    cv::min(image,high,result);
    cv::max(result,low,result);
    return result;
}

cv::Mat render(const cv::Mat& base, const cv::Mat& delta) {
    cv::Mat output=base.clone();
    for (int y=0; y<base.rows; ++y) {
        const auto* change=delta.ptr<float>(y);
        auto* row=output.ptr<cv::Vec3b>(y);
        for (int x=0; x<base.cols; ++x)
            for (auto& value : row[x].val) {
                const auto proposal=std::nearbyintf(value + change[x]);
                value=static_cast<std::uint8_t>(std::clamp(proposal,0.f,255.f));
            }
    }
    return output;
}

struct Camera {
    double error = std::numeric_limits<double>::infinity();
    double sigma{}, length{}, angle{}, dx{}, dy{}, gain{1.}, bias{};
    auto rank() const { return std::tuple(error,sigma,length,angle,dx,dy,gain,bias); }
};

cv::Mat response(int size, Camera c) {
    cv::Mat sample=cv::Mat::zeros(size,size,CV_32F);
    const int samples=c.length>0. ? 13 : 1;
    for (int i=0; i<samples; ++i) {
        const double t=samples==1 ? 0. : -.5+i/12.;
        const double x=-c.dx+c.length*t*std::cos(c.angle);
        const double y=-c.dy+c.length*t*std::sin(c.angle);
        const int xx=static_cast<int>(std::floor(x)), yy=static_cast<int>(std::floor(y));
        const double fx=x-xx, fy=y-yy;
        for (int k=0; k<2; ++k)
            for (int j=0; j<2; ++j)
                sample.at<float>((yy+k+size)%size,(xx+j+size)%size)+=
                    static_cast<float>((k?fy:1.-fy)*(j?fx:1.-fx));
    }
    sample /= cv::sum(sample)[0];
    cv::Mat freq;
    cv::dft(sample,freq,cv::DFT_COMPLEX_OUTPUT);
    const auto normalized=[size](int i) { return (i<size/2 ? i : i-size)/double(size); };
    for (int y=0; y<size; ++y)
        for (int x=0; x<size; ++x) {
            const double fy=normalized(y), fx=normalized(x);
            freq.at<cv::Vec2f>(y,x)*=static_cast<float>(std::exp(-2.*pi*pi*c.sigma*c.sigma*(fy*fy+fx*fx)));
        }
    return freq;
}

cv::Mat project(const cv::Mat& image,const cv::Mat& kernel,bool adjoint=false) {
    cv::Mat freq, result;
    cv::dft(image,freq,cv::DFT_COMPLEX_OUTPUT);
    cv::mulSpectrums(freq,kernel,freq,0,adjoint);
    cv::idft(freq,result,cv::DFT_SCALE|cv::DFT_REAL_OUTPUT);
    return result;
}

Camera calibrate(const cv::Mat& anchor,const cv::Mat& source, const FaceWindows& windows) {
    cv::Mat model,raw;
    cv::resize(anchor,model,{grid,grid},0.,0.,cv::INTER_AREA);
    cv::resize(source,raw,{grid,grid},0.,0.,cv::INTER_AREA);
    Camera best;
    const auto& fit=windows.outside[0];
    const auto& check=windows.outside[1];
    double ymean=0.;
    for (cv::Point p : fit) ymean+=raw.at<float>(p);
    ymean/=fit.size();
    const auto test=[&](double sigma,double length,double angle) {
        Camera candidate;
        candidate.sigma=sigma;
        candidate.length=length;
        candidate.angle=angle;
        const auto shifted=project(model,response(grid,candidate));
        for (int dx=-2; dx<=2; ++dx)
            for (int dy=-2; dy<=2; ++dy) {
                const auto value=[&](cv::Point p) { return shifted.at<float>((p.y+dy+grid)%grid,(p.x+dx+grid)%grid); };
                double xmean=0., variance=0., covariance=0.;
                for (cv::Point p : fit) xmean+=value(p);
                xmean/=fit.size();
                for (cv::Point p : fit) {
                    const double zero=value(p)-xmean;
                    variance+=zero*zero;
                    covariance+=zero*(raw.at<float>(p)-ymean);
                }
                candidate.gain=std::clamp(covariance/std::max(variance,.01),.70,1.40);
                candidate.bias=std::clamp(ymean-candidate.gain*xmean,-35.,35.);
                double error=0.;
                for (cv::Point p : check)
                    error+=std::abs(candidate.gain*value(p)+candidate.bias-raw.at<float>(p));
                candidate.error=error/check.size();
                candidate.dx=dx;
                candidate.dy=dy;
                if (candidate.rank()<best.rank())
                    best=candidate;
            }
    };
    for (double sigma : {.16,.55,1.1,1.85}) test(sigma,0.,0.);
    for (double sigma : {.16,.6})
        for (double length : {2.,4.,7.})
            for (int angle=0; angle<4; ++angle) test(sigma,length,angle*pi/4.);
    best.sigma*=4.;
    best.length*=4.;
    best.dx*=4.;
    best.dy*=4.;
    return best;
}

std::array<double,2> score(const cv::Mat& residual, const FaceWindows& masks) {
    std::array<double,2> result{};
    for (std::size_t i=0; i<masks.parts.size(); ++i) {
        std::array<double,2> numerator{}, denominator{};
        const auto rect=masks.bounds[i];
        for (int y=rect.y; y<rect.br().y; ++y) {
            const auto* w=masks.parts[i].ptr<float>(y);
            const auto* data=residual.ptr<float>(y);
            for (int x=rect.x; x<rect.br().x; ++x) {
                const int j=(y/23+x/21)%2;
                numerator[j]+=w[x]*std::abs(data[x]);
                denominator[j]+=w[x];
            }
        }
        for (int j=0; j<2; ++j) result[j]+=.25*numerator[j]/denominator[j];
    }
    return result;
}

double average(std::array<double,2> s) { return .5*(s[0]+s[1]); }

cv::Mat weighted(const ExposureFrames& values, const Weights& weights) {
    return values[0]*(weights[0]/16.)+values[1]*(weights[1]/16.)+
           values[2]*(weights[2]/16.)+values[3]*(weights[3]/16.);
}

std::pair<Weights,double> fit_weights(const ExposureFrames& values, const cv::Mat& measured, const FaceWindows& mask) {
    Weights center{4,4,4,4};
    std::array<Weights,2> reference{center,center};
    std::array<double,2> error{std::numeric_limits<double>::infinity(),std::numeric_limits<double>::infinity()};
    std::array<int,2> distances{0,0};
    for (int a=1;a<=4;++a) for (int b=1;b<=4;++b) for (int c=1;c<=4;++c) {
        const int d=8-a-b-c;
        if (d<1 || d>4)
            continue;
        Weights weights{a*2,b*2,c*2,d*2};
        const auto loss=score(measured-weighted(values,weights),mask);
        int steps=0;
        for (int x : weights) steps+=std::abs(x-4);
        for (int f=0;f<2;++f)
            if (std::tuple(loss[f],steps,weights)<std::tuple(error[f],distances[f],reference[f])) {
                reference[f]=weights;
                error[f]=loss[f];
                distances[f]=steps;
            }
    }
    Weights proposed{};
    for (int i=0;i<4;++i)
        proposed[i]=(reference[0][i]+reference[1][i])/2;
    const double loss=average(score(measured-weighted(values,proposed),mask));
    const double uniform=average(score(measured-weighted(values,center),mask));
    return loss>uniform ? std::pair(center,uniform) : std::pair(proposed,loss);
}

cv::Mat correction(const cv::Mat& base,const cv::Mat& target,const cv::Mat& kernel,const Camera& camera,
                   const FaceWindows& windows) {
    const auto residual=clip(target-camera.gain*project(base,kernel)-camera.bias,-28.,28.);
    cv::Mat result=cv::Mat::zeros(side,side,CV_32F);
    for (const auto& weight : windows.fit_weights) {
        const auto adjoint=[&](const cv::Mat& pixels)->cv::Mat {
            return camera.gain*windows.combined.mul(project(pixels,kernel,true));
        };
        cv::Mat value=cv::Mat::zeros(side,side,CV_32F);
        auto r=adjoint(weight.mul(residual));
        auto direction=r.clone();
        const double initial=r.dot(r);
        double norm=initial;
        for (int iteration=0; iteration<14 && norm>=std::max(1.e-8,initial*1.e-8); ++iteration) {
            cv::Mat normal=adjoint(weight.mul(camera.gain*project(windows.combined.mul(direction),kernel)))+
                           .016*direction;
            const double step=norm/direction.dot(normal);
            value += step*direction;
            r -= step*normal;
            double next=r.dot(r);
            direction=r+(next/norm)*direction;
            norm=next;
        }
        result+=clip(windows.combined.mul(value),-27.,27.);
    }
    return result*.5;
}

double spread(const ExposureFrames& values,const FaceWindows& windows) {
    const cv::Mat center=(values[0]+values[1]+values[2]+values[3])*.25;
    double sum=0., squares=0.;
    for (const auto& v : values) {
        const auto rolled=band(v-center);
        for (const auto p : windows.inner) {
            const auto x=rolled.at<float>(p);
            sum+=x;
            squares+=x*x;
        }
    }
    const double size=4.*windows.inner.size();
    return std::sqrt(std::max(0.,squares/size-std::pow(sum/size,2.)));
}

std::vector<int> nearest(const std::vector<Example>& dictionary,int region,const cv::Mat& query) {
    std::vector<std::pair<double,int>> ranks;
    for (std::size_t i=0; i<dictionary.size(); ++i) {
        const auto& observed=dictionary[i].observed[region];
        const double norm=std::max(3.,cv::norm(observed)/std::sqrt(double(feature_size)));
        ranks.emplace_back(cv::norm(query,observed,cv::NORM_L2SQR)/(norm*feature_size),static_cast<int>(i));
    }
    std::ranges::sort(ranks);
    std::array<int,6> used{};
    std::array<bool,28> captures{};
    std::vector<int> selected;
    for (const auto& rank : ranks) {
        const int i=rank.second;
        const int cohort=dictionary[i].cohort;
        if (captures[i/6] || used[cohort]>=3)
            continue;
        selected.push_back(i);
        captures[i/6]=true;
        ++used[cohort];
        if (selected.size()==12) break;
    }
    if (selected.size()<8)
        throw std::invalid_argument("Too few independent learned captures");
    return selected;
}

cv::Mat state_span(const ExposureFrames& y,const FaceWindows& mask) {
    const auto mean=weighted(y,{4,4,4,4});
    cv::Mat matrix(4,static_cast<int>(mask.inner.size()),CV_32F);
    for (int i=0;i<4;++i)
        for (int j=0; j<matrix.cols; ++j)
            matrix.at<float>(i,j) = y[i].at<float>(mask.inner[j])-mean.at<float>(mask.inner[j]);
    cv::Mat s,u,v;
    cv::SVD::compute(matrix,s,u,v);
    int rank=0;
    while (rank<s.rows && s.at<float>(rank)>std::max(.001f,s.at<float>(0)*.0002f))
        ++rank;
    return v.rowRange(0,rank).clone();
}

std::array<cv::Mat,2> local_axes(const std::vector<Example>& dict,int region,const cv::Mat& desc,
                                 const cv::Mat& span,const FaceWindows& mask) {
    const auto nearest_ids=nearest(dict,region,desc);
    cv::Mat matrix(static_cast<int>(nearest_ids.size()),tile.area(),CV_32F);
    for (std::size_t i=0; i<nearest_ids.size(); ++i)
        dict[nearest_ids[i]].residual[region].reshape(1,1).copyTo(matrix.row(static_cast<int>(i)));
    cv::Mat s,u,v;
    cv::SVD::compute(matrix,s,u,v);
    std::array<cv::Mat,2> output;
    for (int i=0;i<2;++i) {
        cv::Mat field=cv::Mat::zeros(side,side,CV_32F);
        const auto box=mask.bounds[region];
        cv::Mat small=v.row(i).reshape(1,tile.height)*(1.2*s.at<float>(i)/std::sqrt(double(nearest_ids.size())));
        cv::Mat expanded;
        cv::resize(small,expanded,box.size(),0.,0.,cv::INTER_CUBIC);
        cv::Mat windowed=expanded.mul(mask.parts[region](box));
        windowed.copyTo(field(box));
        cv::Mat samples(1,static_cast<int>(mask.inner.size()),CV_32F);
        for (int j=0;j<samples.cols;++j)
            samples.at<float>(j)=field.at<float>(mask.inner[j]);
        if (!span.empty())
            samples-=samples*span.t()*span;
        for (int j=0;j<samples.cols;++j)
            field.at<float>(mask.inner[j])=samples.at<float>(j);
        output[i]=std::move(field);
    }
    return output;
}

std::pair<int,int> endpoints(const cv::Mat& delta,const ExposureFrames& luminance,const Weights& weights,
                             const FaceWindows& mask,int axis) {
    std::vector<std::tuple<bool,double,int,int>> ranks;
    for (int a=0;a<4;++a) for (int b=a+1;b<4;++b) {
        const int scale=std::max(weights[a],weights[b])/std::gcd(weights[a],weights[b]);
        double difference=0., magnitude=0.;
        for (const auto p : mask.inner) {
            const double q=std::abs(delta.at<float>(p));
            magnitude+=q;
            difference+=q*std::abs(luminance[a].at<float>(p)-luminance[b].at<float>(p));
        }
        ranks.emplace_back(scale>3,scale*.75-.12*difference/std::max(.01,magnitude),a,b);
    }
    std::ranges::sort(ranks);
    const auto& choice=axis==1 && std::get<0>(ranks[1])==std::get<0>(ranks[0]) ? ranks[1] : ranks[0];
    return {std::get<2>(choice),std::get<3>(choice)};
}

std::pair<std::array<cv::Mat,2>,double> exchange(const ExposureFrames& frames,const Weights& w,
                                               std::pair<int,int> pair,const cv::Mat& delta,const FaceWindows& windows) {
    static const auto table=[] {
        std::array<double,256> values{};
        for (int i=0;i<256;++i) values[i]=std::pow(i/255.,gamma);
        return values;
    }();
    const auto [first,other]=pair;
    const int u=w[first], v=w[other];
    const int m=v/std::gcd(u,v), n=u/std::gcd(u,v), weight_step=std::max(m,n);
    std::array<cv::Mat,2> images{frames[first].clone(),frames[other].clone()};
    double peak=0.;
    for (const auto p : windows.inner) {
        const auto orig=frames[first].at<cv::Vec3b>(p), alt=frames[other].at<cv::Vec3b>(p);
        int low=-34/weight_step, high=34/weight_step;
        for (int c=0;c<3;++c) {
            low=std::max({low,-int(orig[c])/m,-(255-alt[c])/n});
            high=std::min({high,(255-orig[c])/m,alt[c]/n});
        }
        int step=std::clamp(static_cast<int>(std::nearbyintf(delta.at<float>(p)/weight_step)),low,high);
        if (!step) continue;
        std::array<double,3> old{}, encoded{}, lower{}, upper{};
        for (int c=0;c<3;++c) {
            for (int f=0;f<4;++f) old[c]+=(w[f]/16.)*table[frames[f].at<cv::Vec3b>(p)[c]];
            encoded[c]=255.*std::pow(old[c],1./gamma);
            lower[c]=std::pow(std::clamp((encoded[c]-light_limit)/255.,0.,1.),gamma);
            upper[c]=std::pow(std::clamp((encoded[c]+light_limit)/255.,0.,1.),gamma);
        }
        const auto candidate=[&](int c) {
            return old[c]+(u/16.)*(table[orig[c]+m*step]-table[orig[c]])+
                          (v/16.)*(table[alt[c]-n*step]-table[alt[c]]);
        };
        while (step) {
            bool valid=true;
            for (int c=0;c<3;++c) {
                const double x=candidate(c);
                valid=valid && x>=lower[c] && x<=upper[c];
            }
            if (valid) break;
            step+=(step<0?1:-1);
        }
        for (int c=0;c<3;++c) {
            auto& x=images[0].at<cv::Vec3b>(p)[c];
            auto& y=images[1].at<cv::Vec3b>(p)[c];
            x=static_cast<std::uint8_t>(orig[c]+m*step);
            y=static_cast<std::uint8_t>(alt[c]-n*step);
            if (u*x+v*y!=u*orig[c]+v*alt[c])
                throw std::logic_error("Encoded exposure changed");
            peak=std::max(peak,std::abs(255.*std::pow(std::clamp(candidate(c),0.,1.),1./gamma)-encoded[c]));
        }
    }
    return {images,peak};
}

}

std::vector<DirectionGroup> learned_exposure_directions(
    const std::filesystem::path& dictionary,const cv::Mat& input,
    const std::array<cv::Mat,2>& anchors,const std::array<ExposureFrames,2>& families) {
    const auto corpus=read_dictionary(dictionary);
    const FaceWindows masks;
    const auto observed=gray(input);
    const cv::Mat signed_edges=smooth(observed,.82)-smooth(observed,6.);
    std::array<cv::Mat,4> description;
    for (int i=0;i<4;++i) {
        cv::resize(signed_edges(masks.bounds[i]),description[i],{9,7},0.,0.,cv::INTER_AREA);
        description[i]=description[i].reshape(1,1);
    }
    std::vector<DirectionGroup> report;
    for (const auto& anchor : anchors) {
        const auto base=gray(anchor);
        const auto camera=calibrate(base,observed,masks);
        const auto kernel=response(side,camera);
        const auto measured=band(observed-camera.bias);
        ExposureFrames reference{};
        DirectionGroup result;
        double best=std::numeric_limits<double>::infinity();
        for (std::size_t i=0;i<families.size();++i) {
            auto states=families[i];
            ExposureFrames middle{},simulated{};
            for (int j=0;j<4;++j) {
                states[j]=render(anchor,masks.combined.mul(clip(gray(states[j])-base,-27.,27.)));
                middle[j]=gray(states[j]);
                simulated[j]=band(camera.gain*project(middle[j],kernel));
            }
            auto [weights,error]=fit_weights(simulated,measured,masks);
            const double deviation=spread(middle,masks);
            if (error-.035*deviation<best) {
                best=error-.035*deviation;
                result.weight_units=weights;
                result.seed_family=static_cast<int>(i);
                result.initial_replay=error;
                reference=middle;
            }
        }
        const auto mean=weighted(reference,result.weight_units);
        const auto change=correction(mean,observed,kernel,camera,masks);
        ExposureFrames states{}, moments{};
        for (int i=0;i<4;++i) {
            states[i]=render(anchor,clip(reference[i]+change-base,-40.,40.));
            moments[i]=gray(states[i]);
        }
        result.fitted_replay=average(score(band(observed-camera.gain*project(
                                             weighted(moments,result.weight_units),kernel)-camera.bias),masks));
        result.seed_spread=spread(moments,masks);
        result.eligible=result.initial_replay-result.fitted_replay>=.12 && result.seed_spread>=.45;
        result.fitted_states=states;
        const auto span=state_span(moments,masks);
        for (int region=0;region<4;++region) {
            const auto axes=local_axes(corpus,region,description[region],span,masks);
            for (int axis=0;axis<2;++axis) {
                const auto pair=endpoints(axes[axis],moments,result.weight_units,masks,axis);
                for (int sign : {1,-1}) {
                    const auto [images,peak]=exchange(states,result.weight_units,pair,sign*axes[axis],masks);
                    result.candidates.push_back({images[0],region,axis,sign,pair.first,peak});
                    result.candidates.push_back({images[1],region,axis,sign,pair.second,peak});
                }
            }
        }
        report.push_back(std::move(result));
    }
    return report;
}

}
