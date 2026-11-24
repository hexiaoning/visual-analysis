#include "portrait_video.h"
#include "local_images.h"

#include <array>
#include <climits>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <optional>
#include <stdexcept>

#include <Windows.h>
#include <opencv2/imgproc.hpp>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/display.h>
#include <libavutil/error.h>
#include <libswscale/swscale.h>
}

namespace visual {
namespace fs = std::filesystem;
namespace {

void check(int error, const char* context) {
    if (error < 0) {
        std::array<char, AV_ERROR_MAX_STRING_SIZE> reason{};
        av_strerror(error, reason.data(), reason.size());
        throw std::runtime_error(std::string(context) + " (" + reason.data() + ")");
    }
}

std::string utf8(const fs::path& path) {
    const auto value = path.u8string();
    return {value.begin(), value.end()};
}

struct Input {
    AVFormatContext* context{};
    ~Input() { avformat_close_input(&context); }
};

struct Output {
    AVFormatContext* context{};
    ~Output() {
        if (context) {
            avio_closep(&context->pb);
            avformat_free_context(context);
        }
    }
};

struct Codec {
    AVCodecContext* context{};
    ~Codec() { avcodec_free_context(&context); }
};

struct Packet {
    AVPacket* data{av_packet_alloc()};
    ~Packet() { av_packet_free(&data); }
};

struct Frame {
    AVFrame* data{av_frame_alloc()};
    ~Frame() { av_frame_free(&data); }
};

struct Scaler {
    SwsContext* context{};
    ~Scaler() { sws_freeContext(context); }
};

struct Interrupted {};

int interrupt(void* state) {
    return static_cast<VideoProgress*>(state)->cancelled() ? 1 : 0;
}

void requireRunning(const VideoProgress& progress) {
    if (progress.cancelled()) throw Interrupted{};
}

struct StagingArea {
    fs::path folder;
    ~StagingArea() {
        std::error_code error;
        fs::remove(folder / L"video.mkv", error);
        fs::remove(folder, error);
    }
};

fs::path temporaryFolder(const fs::path& output) {
    using namespace std::chrono;
    const auto tick = steady_clock::now().time_since_epoch().count();
    const auto name = L".visual-restore-" + std::to_wstring(GetCurrentProcessId()) + L"-" + std::to_wstring(tick);
    const auto path = output.parent_path() / name;
    if (!fs::create_directory(path)) {
        throw std::runtime_error("Could not reserve a video staging folder next to the output.");
    }
    return path;
}

int rotation(const AVCodecParameters* info) {
    const auto* side = av_packet_side_data_get(info->coded_side_data, info->nb_coded_side_data,
                                              AV_PKT_DATA_DISPLAYMATRIX);
    if (!side || side->size < 9 * sizeof(int32_t)) return 0;
    const auto angle = av_display_rotation_get(reinterpret_cast<const int32_t*>(side->data));
    return std::isfinite(angle) ? (int(std::round(angle / 90)) % 4 + 4) % 4 : 0;
}

cv::Mat imageFromFrame(const AVFrame& frame, Scaler& scaler, int orientation) {
    scaler.context = sws_getCachedContext(scaler.context,frame.width,frame.height,
        static_cast<AVPixelFormat>(frame.format),frame.width,frame.height,AV_PIX_FMT_BGR24,
        SWS_BILINEAR,nullptr,nullptr,nullptr);
    if(!scaler.context) throw std::runtime_error("Cannot convert video pixels to BGR.");
    cv::Mat pixels(frame.height,frame.width,CV_8UC3);
    uint8_t* planes[4]{pixels.data};
    int strides[4]{static_cast<int>(pixels.step)};
    const auto rows=sws_scale(scaler.context,frame.data,frame.linesize,0,frame.height,planes,strides);
    check(rows,"Video pixel conversion failed.");
    if(rows!=frame.height) throw std::runtime_error("视频画面像素转换不完整。");
    if(orientation) {
        constexpr std::array<int,4> rotations{0,cv::ROTATE_90_COUNTERCLOCKWISE,cv::ROTATE_180,cv::ROTATE_90_CLOCKWISE};
        cv::rotate(pixels,pixels,rotations[orientation]);
    }
    return pixels;
}

void openVideo(Input& reader,const fs::path& source,AVIOInterruptCB callbacks) {
    reader.context=avformat_alloc_context();
    if(!reader.context) throw std::bad_alloc();
    reader.context->interrupt_callback=callbacks;
    AVDictionary* options{};
    av_dict_set(&options,"protocol_whitelist","file",0);
    av_dict_set(&options,"format_whitelist","mov,matroska,webm,avi,mpegts,flv,asf,mpeg,ogg",0);
    const auto opened=avformat_open_input(&reader.context,utf8(source).c_str(),nullptr,&options);
    av_dict_free(&options);
    check(opened,"Cannot open the local video.");
    check(avformat_find_stream_info(reader.context,nullptr),"Cannot read the video tracks.");
}

int64_t videoStart(const AVFormatContext& input,const AVStream& track) {
    if(track.start_time!=AV_NOPTS_VALUE) return track.start_time;
    if(input.start_time!=AV_NOPTS_VALUE) return av_rescale_q(input.start_time,AV_TIME_BASE_Q,track.time_base);
    return 0;
}

void render(const fs::path& source,const fs::path& temporary,const FrameProcessor& transform,
            VideoProgress& progress,bool archive,std::optional<std::pair<double,double>> clip={}) {
    Input reader;
    openVideo(reader,source,{interrupt,&progress});
    const AVCodec* decoder{};
    const auto videoIndex = av_find_best_stream(reader.context, AVMEDIA_TYPE_VIDEO, -1, -1, &decoder, 0);
    check(videoIndex, "No decodable video stream.");
    auto* video = reader.context->streams[videoIndex];
    const auto sourceStart=videoStart(*reader.context,*video);
    Codec decoding{avcodec_alloc_context3(decoder)};
    if (!decoding.context) throw std::bad_alloc();
    check(avcodec_parameters_to_context(decoding.context, video->codecpar), "Invalid video properties.");
    decoding.context->thread_count = 2;
    decoding.context->pkt_timebase = video->time_base;
    check(avcodec_open2(decoding.context, decoder, nullptr), "Could not initialize the video decoder.");
    const auto* encoder=avcodec_find_encoder(archive?AV_CODEC_ID_FFV1:AV_CODEC_ID_MPEG4);
    if(!encoder) throw std::runtime_error("The packaged FFmpeg encoder is missing.");
    Codec encoding{avcodec_alloc_context3(encoder)};
    if (!encoding.context) throw std::bad_alloc();
    const auto orientation = rotation(video->codecpar);
    const int width = orientation % 2 ? decoding.context->height : decoding.context->width;
    const int height = orientation % 2 ? decoding.context->width : decoding.context->height;
    if (width <= 0 || height <= 0) throw std::runtime_error("The source video has invalid dimensions.");
    encoding.context->width = archive?width:width+width%2;
    encoding.context->height = archive?height:height+height%2;
    encoding.context->pix_fmt = archive?AV_PIX_FMT_BGR0:AV_PIX_FMT_YUV420P;
    encoding.context->time_base = video->time_base;
    const auto rate = av_guess_frame_rate(reader.context, video, nullptr);
    encoding.context->framerate = rate.num && rate.den ? rate : AVRational{30, 1};
    encoding.context->thread_count = 2;
    if(!archive) {
        encoding.context->time_base=AVRational{1,10000};
        encoding.context->bit_rate=6000000;
    }

    Output writer;
    const auto filename = utf8(temporary);
    check(avformat_alloc_output_context2(&writer.context, nullptr, archive?"matroska":"mp4", filename.c_str()),
          "Cannot initialize the output container.");
    if (writer.context->oformat->flags & AVFMT_GLOBALHEADER) {
        encoding.context->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;
    }
    check(avcodec_open2(encoding.context, encoder, nullptr), "Could not initialize the video encoder.");
    auto* rendered = avformat_new_stream(writer.context, nullptr);
    if (!rendered) throw std::bad_alloc();
    check(avcodec_parameters_from_context(rendered->codecpar, encoding.context), "Cannot add the output video track.");
    rendered->time_base = encoding.context->time_base;
    rendered->avg_frame_rate = encoding.context->framerate;
    std::vector<AVStream*> audio(reader.context->nb_streams);
    for (unsigned index = 0; index < reader.context->nb_streams; ++index) {
        auto* track = reader.context->streams[index];
        if (archive && track->codecpar->codec_type == AVMEDIA_TYPE_AUDIO) {
            auto* copy = avformat_new_stream(writer.context, nullptr);
            if (!copy) throw std::bad_alloc();
            check(avcodec_parameters_copy(copy->codecpar, track->codecpar), "Cannot copy an audio track.");
            copy->codecpar->codec_tag = 0;
            copy->time_base = track->time_base;
            av_dict_copy(&copy->metadata, track->metadata, 0);
            copy->disposition = track->disposition;
            audio[index] = copy;
        }
    }
    check(avio_open(&writer.context->pb, filename.c_str(), AVIO_FLAG_WRITE), "Cannot open the staged MKV.");
    check(avformat_write_header(writer.context, nullptr), "Cannot write the output stream information.");

    Packet input, encoded;
    Frame frame, output;
    Scaler scaler, outputScaler;
    if (!input.data || !encoded.data || !frame.data || !output.data) throw std::bad_alloc();
    output.data->format = encoding.context->pix_fmt;
    output.data->width = encoding.context->width;
    output.data->height = encoding.context->height;
    check(av_frame_get_buffer(output.data, 32), "Cannot allocate the output frame.");
    bool clipDone=false;
    std::optional<int64_t> firstTimestamp;
    const auto sendFrame = [&](const AVFrame* pixels) {
        check(avcodec_send_frame(encoding.context, pixels), "Could not encode an output video frame.");
        for (;;) {
            const auto status = avcodec_receive_packet(encoding.context, encoded.data);
            if (status == AVERROR(EAGAIN) || status == AVERROR_EOF) return;
            check(status, "Could not read encoded video packet.");
            av_packet_rescale_ts(encoded.data, encoding.context->time_base, rendered->time_base);
            encoded.data->stream_index = rendered->index;
            check(av_interleaved_write_frame(writer.context, encoded.data), "Cannot save the video packet.");
        }
    };
    const auto receiveFrames = [&] {
        while(!clipDone) {
            requireRunning(progress);
            const auto status = avcodec_receive_frame(decoding.context, frame.data);
            if (status == AVERROR(EAGAIN) || status == AVERROR_EOF) return;
            check(status, "Could not decode a source frame.");
            if (frame.data->width != (orientation % 2 ? height : width) ||
                frame.data->height != (orientation % 2 ? width : height)) {
                throw std::runtime_error("Video dimensions changed mid-stream.");
            }
            if (frame.data->best_effort_timestamp == AV_NOPTS_VALUE) {
                throw std::runtime_error("A source frame is missing its presentation timestamp.");
            }
            if(clip) {
                const auto time=(frame.data->best_effort_timestamp-sourceStart)*av_q2d(video->time_base);
                if(time<clip->first) {av_frame_unref(frame.data);continue;}
                if(time>clip->second) {clipDone=true;return;}
                if(!firstTimestamp) firstTimestamp=frame.data->best_effort_timestamp;
            }
            const auto pixels=imageFromFrame(*frame.data,scaler,orientation);
            const auto [restored, count] = transform(pixels);
            requireRunning(progress);
            if(restored.empty() || restored.size()!=pixels.size()) throw std::runtime_error("Video processor changed frame dimensions.");
            check(av_frame_make_writable(output.data), "The output frame is busy.");
            if(archive) {
                cv::Mat destination(height,width,CV_8UC4,output.data->data[0],output.data->linesize[0]);
                cv::cvtColor(restored,destination,cv::COLOR_BGR2BGRA);
            } else {
                cv::Mat padded;
                cv::copyMakeBorder(restored,padded,0,height%2,0,width%2,cv::BORDER_CONSTANT);
                outputScaler.context=sws_getCachedContext(outputScaler.context,padded.cols,padded.rows,AV_PIX_FMT_BGR24,
                    padded.cols,padded.rows,AV_PIX_FMT_YUV420P,SWS_BILINEAR,nullptr,nullptr,nullptr);
                if(!outputScaler.context) throw std::runtime_error("Cannot prepare MP4 color conversion.");
                const uint8_t* planes[4]{padded.data};
                int strikes[4]{static_cast<int>(padded.step)};
                check(sws_scale(outputScaler.context,planes,strikes,0,padded.rows,output.data->data,output.data->linesize),
                    "Cannot convert the MP4 frame.");
            }
            output.data->pts=av_rescale_q(frame.data->best_effort_timestamp-firstTimestamp.value_or(archive?0:sourceStart),
                                         video->time_base,encoding.context->time_base);
            output.data->duration=av_rescale_q(frame.data->duration,video->time_base,encoding.context->time_base);
            sendFrame(output.data);
            ++progress.frames;
            progress.faces += count;
            av_frame_unref(frame.data);
        }
    };
    int status{};
    while (!clipDone && (status = av_read_frame(reader.context, input.data)) >= 0) {
        requireRunning(progress);
        const auto index = input.data->stream_index;
        if (index == videoIndex) {
            check(avcodec_send_packet(decoding.context, input.data), "Could not send a packet to video decoder.");
            receiveFrames();
        } else if (auto* copy = audio[index]; copy && input.data->dts != AV_NOPTS_VALUE) {
            av_packet_rescale_ts(input.data, reader.context->streams[index]->time_base, copy->time_base);
            input.data->stream_index = copy->index;
            input.data->pos = -1;
            check(av_interleaved_write_frame(writer.context, input.data), "Cannot save an audio packet.");
        }
        av_packet_unref(input.data);
    }
    requireRunning(progress);
    if(!clipDone) {
        if (status != AVERROR_EOF) check(status, "Could not finish reading the source.");
        check(avcodec_send_packet(decoding.context, nullptr), "Could not flush the decoder.");
        receiveFrames();
    }
    if(!progress.frames || (archive&&!progress.faces)) throw std::runtime_error("No usable output frames.");
    sendFrame(nullptr);
    check(av_write_trailer(writer.context), "Could not finalize the MKV.");
    check(avio_closep(&writer.context->pb), "Could not finish writing the staged video.");
}

}

int visitVideo(const fs::path& path,const FrameVisitor& visitor,int interval,const std::atomic<bool>& cancel,
               int firstFrame,const FrameSelector& select,std::atomic<int>* totalFrames) {
    if(interval<1) throw std::invalid_argument("抽帧间隔必须大于 0。");
    const auto source=localPath(path);
    if(!fs::is_regular_file(source)) throw std::invalid_argument("请选择本机视频。");
    Input reader;
    auto cancelled=[](void* state){return static_cast<std::atomic<bool>*>(state)->load()?1:0;};
    openVideo(reader,source,{cancelled,const_cast<std::atomic<bool>*>(&cancel)});
    const AVCodec* codec{};
    const auto videoIndex=av_find_best_stream(reader.context,AVMEDIA_TYPE_VIDEO,-1,-1,&codec,0);
    check(videoIndex,"无法找到视频轨。");
    const auto* track=reader.context->streams[videoIndex];
    const auto sourceStart=videoStart(*reader.context,*track);
    Codec decoder{avcodec_alloc_context3(codec)};
    if(!decoder.context) throw std::bad_alloc();
    check(avcodec_parameters_to_context(decoder.context,track->codecpar),"读取视频参数失败。");
    decoder.context->thread_count=2;
    decoder.context->pkt_timebase=track->time_base;
    check(avcodec_open2(decoder.context,codec,nullptr),"打开视频解码器失败。");
    const auto orientation=rotation(track->codecpar);
    const double frameRate=av_q2d(av_guess_frame_rate(reader.context,const_cast<AVStream*>(track),nullptr));
    if(totalFrames && *totalFrames==0) {
        const double duration=track->duration>0?track->duration*av_q2d(track->time_base):
            reader.context->duration>0?double(reader.context->duration)/AV_TIME_BASE:0.;
        const double estimate=track->nb_frames>0?double(track->nb_frames):std::ceil(duration*frameRate);
        if(std::isfinite(estimate) && estimate>0.) *totalFrames=static_cast<int>(std::min(estimate,double(INT_MAX)));
    }
    Frame frame;
    Packet packet;
    Scaler scaler;
    if(!frame.data || !packet.data) throw std::bad_alloc();
    int frames=0;
    std::optional<double> previous;
    bool stopped=false;
    auto receive=[&]{
        while(!cancel && !stopped) {
            const auto status=avcodec_receive_frame(decoder.context,frame.data);
            if(status==AVERROR(EAGAIN) || status==AVERROR_EOF) return;
            check(status,"读取视频帧失败。");
            const auto index=frames++;
            const auto timestamp=frame.data->best_effort_timestamp;
            auto seconds=timestamp==AV_NOPTS_VALUE ? -1. : (timestamp-sourceStart)*av_q2d(track->time_base);
            if(!std::isfinite(seconds) || seconds<0 || (previous && seconds<=*previous)) {
                if(frameRate>0 && std::isfinite(frameRate)) seconds=previous ? *previous+1./frameRate : 0.;
                else if(index%interval==0 && index>=firstFrame && (!select || select(index))) {
                    throw std::runtime_error("视频缺少有效时间戳。");
                }
            }
            if(seconds>=0 && (!previous || seconds>*previous)) previous=seconds;
            if(index%interval==0 && index>=firstFrame && (!select || select(index))) {
                stopped=!visitor(imageFromFrame(*frame.data,scaler,orientation),index,seconds);
            }
            av_frame_unref(frame.data);
        }
    };
    int status{};
    while(!cancel && !stopped && (status=av_read_frame(reader.context,packet.data))>=0) {
        if(packet.data->stream_index==videoIndex) {
            check(avcodec_send_packet(decoder.context,packet.data),"读取视频包失败。");
            receive();
        }
        av_packet_unref(packet.data);
    }
    if(!cancel && !stopped) {
        if(status!=AVERROR_EOF) check(status,"视频提前结束。");
        check(avcodec_send_packet(decoder.context,nullptr),"结束视频解码失败。");
        receive();
        if(totalFrames && !cancel && !stopped) *totalFrames=frames;
    }
    if(!frames && !cancel) throw std::runtime_error("视频无可读画面。");
    return frames;
}

void restorePortraitVideo(const fs::path& input, const fs::path& output,
                          const fs::path& models, VideoProgress& progress) {
    const auto source = localPath(input);
    const auto target = localPath(output);
    if (!fs::is_regular_file(source)) throw std::runtime_error("The source video file does not exist.");
    if (_wcsicmp(target.extension().c_str(), L".mkv") || fs::exists(target)) {
        throw std::runtime_error("Choose a new output file with the .mkv extension.");
    }
    const StagingArea staging{temporaryFolder(target)};
    const auto temporary = staging.folder / L"video.mkv";
    try {
        VideoPortraitRestorer restorer(localPath(models));
        render(source,temporary,[&](const cv::Mat& pixels){return restorer.restore(pixels);},progress,true);
        requireRunning(progress);
        if (!MoveFileExW(temporary.c_str(), target.c_str(), MOVEFILE_WRITE_THROUGH)) {
            throw std::runtime_error("Could not publish the restored video. The output may already exist.");
        }
    } catch (const Interrupted&) {
        return;
    } catch (...) {
        if (!progress.cancelled()) throw;
    }
}

static void renderMp4(const fs::path& input,const fs::path& output,const FrameProcessor& transform,
                      VideoProgress& progress,std::optional<std::pair<double,double>> clip={}) {
    const auto source=localPath(input);
    const auto target=localPath(output);
    if(!fs::is_regular_file(source) || _wcsicmp(target.extension().c_str(),L".mp4") || fs::exists(target)) {
        throw std::runtime_error("请选择本机视频和新的 MP4 输出路径。");
    }
    const StagingArea staging{temporaryFolder(target)};
    const auto temporary=staging.folder/L"video.mkv";
    try {
        render(source,temporary,transform,progress,false,clip);
        requireRunning(progress);
        if(!MoveFileExW(temporary.c_str(),target.c_str(),MOVEFILE_WRITE_THROUGH)) {
            throw std::runtime_error("写入 MP4 失败，输出可能已存在。");
        }
    } catch(const Interrupted&) {} catch(...) { if(!progress.cancelled()) throw; }
}

void renderObjectVideo(const fs::path& source,const fs::path& target,const FrameProcessor& transform,
                       VideoProgress& progress) {
    renderMp4(source,target,transform,progress);
}

void renderReviewClip(const fs::path& source,const fs::path& target,double from,double until,VideoProgress& progress) {
    if (!std::isfinite(from) || !std::isfinite(until) || until<from) throw std::invalid_argument("视频片段范围无效。");
    renderMp4(source,target,[](const auto& pixels){return std::pair{pixels,0};},progress,std::pair{from,until});
}

}
