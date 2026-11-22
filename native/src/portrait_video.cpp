#include "portrait_video.h"
#include "local_images.h"

#include <array>
#include <cerrno>
#include <chrono>
#include <cmath>
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
    return static_cast<VideoProgress*>(state)->cancel.load() ? 1 : 0;
}

void requireRunning(const VideoProgress& progress) {
    if (progress.cancel.load()) throw Interrupted{};
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

void render(const fs::path& source, const fs::path& temporary, const fs::path& models, VideoProgress& progress) {
    Input reader{avformat_alloc_context()};
    if (!reader.context) throw std::bad_alloc();
    reader.context->interrupt_callback = {interrupt, &progress};
    AVDictionary* options{};
    av_dict_set(&options, "protocol_whitelist", "file", 0);
    av_dict_set(&options, "format_whitelist", "mov,matroska,webm,avi,mpegts,flv,asf,mpeg,ogg", 0);
    const auto opened = avformat_open_input(&reader.context, utf8(source).c_str(), nullptr, &options);
    av_dict_free(&options);
    check(opened, "Cannot open the local video.");
    check(avformat_find_stream_info(reader.context, nullptr), "Cannot read the video tracks.");
    const AVCodec* decoder{};
    const auto videoIndex = av_find_best_stream(reader.context, AVMEDIA_TYPE_VIDEO, -1, -1, &decoder, 0);
    check(videoIndex, "No decodable video stream.");
    auto* video = reader.context->streams[videoIndex];
    Codec decoding{avcodec_alloc_context3(decoder)};
    if (!decoding.context) throw std::bad_alloc();
    check(avcodec_parameters_to_context(decoding.context, video->codecpar), "Invalid video properties.");
    decoding.context->thread_count = 2;
    decoding.context->pkt_timebase = video->time_base;
    check(avcodec_open2(decoding.context, decoder, nullptr), "Could not initialize the video decoder.");
    const auto* ffv1 = avcodec_find_encoder(AV_CODEC_ID_FFV1);
    if (!ffv1) throw std::runtime_error("The packaged FFmpeg SDK is missing its FFV1 encoder.");
    Codec encoding{avcodec_alloc_context3(ffv1)};
    if (!encoding.context) throw std::bad_alloc();
    const auto orientation = rotation(video->codecpar);
    const int width = orientation % 2 ? decoding.context->height : decoding.context->width;
    const int height = orientation % 2 ? decoding.context->width : decoding.context->height;
    if (width <= 0 || height <= 0) throw std::runtime_error("The source video has invalid dimensions.");
    encoding.context->width = width;
    encoding.context->height = height;
    encoding.context->pix_fmt = AV_PIX_FMT_BGR0;
    encoding.context->time_base = video->time_base;
    const auto rate = av_guess_frame_rate(reader.context, video, nullptr);
    encoding.context->framerate = rate.num && rate.den ? rate : AVRational{30, 1};
    encoding.context->thread_count = 2;

    Output writer;
    const auto filename = utf8(temporary);
    check(avformat_alloc_output_context2(&writer.context, nullptr, "matroska", filename.c_str()),
          "Cannot initialize the output container.");
    if (writer.context->oformat->flags & AVFMT_GLOBALHEADER) {
        encoding.context->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;
    }
    check(avcodec_open2(encoding.context, ffv1, nullptr), "Could not initialize the FFV1 encoder.");
    auto* rendered = avformat_new_stream(writer.context, nullptr);
    if (!rendered) throw std::bad_alloc();
    check(avcodec_parameters_from_context(rendered->codecpar, encoding.context), "Cannot add the output video track.");
    rendered->time_base = encoding.context->time_base;
    rendered->avg_frame_rate = encoding.context->framerate;
    std::vector<AVStream*> audio(reader.context->nb_streams);
    for (unsigned index = 0; index < reader.context->nb_streams; ++index) {
        auto* track = reader.context->streams[index];
        if (track->codecpar->codec_type == AVMEDIA_TYPE_AUDIO) {
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

    VideoPortraitRestorer restorer(models);
    Packet input, encoded;
    Frame frame, output;
    Scaler scaler;
    if (!input.data || !encoded.data || !frame.data || !output.data) throw std::bad_alloc();
    output.data->format = AV_PIX_FMT_BGR0;
    output.data->width = width;
    output.data->height = height;
    check(av_frame_get_buffer(output.data, 32), "Cannot allocate the output frame.");
    const auto sendFrame = [&](const AVFrame* pixels) {
        check(avcodec_send_frame(encoding.context, pixels), "Could not encode an output video frame.");
        for (;;) {
            const auto status = avcodec_receive_packet(encoding.context, encoded.data);
            if (status == AVERROR(EAGAIN) || status == AVERROR_EOF) return;
            check(status, "Could not read FFV1 packet.");
            av_packet_rescale_ts(encoded.data, encoding.context->time_base, rendered->time_base);
            encoded.data->stream_index = rendered->index;
            check(av_interleaved_write_frame(writer.context, encoded.data), "Cannot save the video packet.");
        }
    };
    const auto receiveFrames = [&] {
        for (;;) {
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
            scaler.context = sws_getCachedContext(scaler.context, frame.data->width, frame.data->height,
                static_cast<AVPixelFormat>(frame.data->format), frame.data->width, frame.data->height,
                AV_PIX_FMT_BGR24, SWS_BILINEAR, nullptr, nullptr, nullptr);
            if (!scaler.context) throw std::runtime_error("Cannot convert video pixels to BGR.");
            cv::Mat pixels(frame.data->height, frame.data->width, CV_8UC3);
            uint8_t* planes[4]{pixels.data};
            int strides[4]{static_cast<int>(pixels.step)};
            check(sws_scale(scaler.context, frame.data->data, frame.data->linesize, 0, frame.data->height,
                            planes, strides), "Video pixel conversion failed.");
            if (orientation) {
                constexpr std::array<int, 4> rotations{0, cv::ROTATE_90_COUNTERCLOCKWISE, cv::ROTATE_180,
                                                       cv::ROTATE_90_CLOCKWISE};
                cv::rotate(pixels, pixels, rotations[orientation]);
            }
            const auto [restored, faces] = restorer.restore(pixels);
            requireRunning(progress);
            check(av_frame_make_writable(output.data), "The output frame is busy.");
            cv::Mat destination(height, width, CV_8UC4, output.data->data[0], output.data->linesize[0]);
            cv::cvtColor(restored, destination, cv::COLOR_BGR2BGRA);
            output.data->pts = frame.data->best_effort_timestamp;
            output.data->duration = frame.data->duration;
            sendFrame(output.data);
            ++progress.frames;
            progress.faces += faces;
            av_frame_unref(frame.data);
        }
    };
    int status{};
    while ((status = av_read_frame(reader.context, input.data)) >= 0) {
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
    if (status != AVERROR_EOF) check(status, "Could not finish reading the source.");
    check(avcodec_send_packet(decoding.context, nullptr), "Could not flush the decoder.");
    receiveFrames();
    if (!progress.frames || !progress.faces) throw std::runtime_error("No restorable faces in the source video.");
    sendFrame(nullptr);
    check(av_write_trailer(writer.context), "Could not finalize the MKV.");
    check(avio_closep(&writer.context->pb), "Could not finish writing the staged video.");
}

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
        render(source, temporary, localPath(models), progress);
        requireRunning(progress);
        if (!MoveFileExW(temporary.c_str(), target.c_str(), MOVEFILE_WRITE_THROUGH)) {
            throw std::runtime_error("Could not publish the restored video. The output may already exist.");
        }
    } catch (const Interrupted&) {
        return;
    } catch (...) {
        if (!progress.cancel) throw;
    }
}

}
