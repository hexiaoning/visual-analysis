"""Conservative local video face enhancement with the source audio tracks."""
from __future__ import annotations

from pathlib import Path
import os
from tempfile import TemporaryDirectory
from threading import Event
from typing import Callable

import av
import numpy as np

from .config import AppConfig
from .gpu import create_gpu_session
from .image_enhancement import PRESETS, enhance_image
from .models import required_paths
from .vision.detector import SCRFDDetector
from .vision.local_media import local_path, local_video_path


def create_face_detector(config: AppConfig) -> SCRFDDetector:
    model, _ = required_paths(config.model_profile)
    session = create_gpu_session(model, config.gpu_device_id, backend=config.gpu_backend)
    return SCRFDDetector(session, input_size=config.detector_size, threshold=config.detector_threshold,
                         nms_threshold=config.nms_threshold)


def enhance_faces(frame: np.ndarray, detector: SCRFDDetector, minimum_size: int) -> tuple[np.ndarray, int]:
    height, width = frame.shape[:2]
    result = frame.copy()
    count = 0
    for face in detector.detect(frame):
        left, top, right, bottom = (round(value) for value in face.bbox)
        left, top, right, bottom = max(0, left), max(0, top), min(width, right), min(height, bottom)
        w, h = right - left, bottom - top
        if min(w, h) < minimum_size:
            continue
        roi = frame[top:bottom, left:right]
        enhanced = enhance_image(roi, PRESETS["人脸清晰化（保守）"])
        y, x = np.ogrid[-1:1:complex(h), -1:1:complex(w)]
        weight = np.clip((1 - x * x - y * y) * 2.5, 0, 1)[..., None]
        result[top:bottom, left:right] = np.rint(roi + (enhanced.astype(np.float32) - roi) * weight)
        count += 1
    return result, count


def _process(source: Path, target: Path, detector: SCRFDDetector, minimum_size: int,
             stop: Event, progress: Callable[[int, int], None] | None) -> tuple[int, int]:
    frames = faces = 0
    with av.open(str(source), options={"protocol_whitelist": "file"}) as probe:
        if not probe.streams.video:
            raise ValueError("没有可处理的视频轨道")
        first = next(probe.decode(probe.streams.video[0]), None)
        if first is None:
            raise ValueError("无法解码源视频画面")
        width, height = first.width, first.height
        if round(first.rotation / 90) % 2:
            width, height = height, width
    with av.open(str(source), options={"protocol_whitelist": "file"}) as reader:
        if not reader.streams.video:
            raise ValueError("没有可处理的视频轨道")
        video = reader.streams.video[0]
        audio = list(reader.streams.audio)
        with av.open(str(target), "w", format="matroska") as writer:
            # Lossless BGR avoids introducing compression changes outside the face region.
            stream = writer.add_stream("ffv1", rate=video.guessed_rate or video.average_rate or 30)
            stream.time_base = video.time_base
            stream.codec_context.time_base = video.time_base
            stream.pix_fmt = "bgr0"
            stream.width, stream.height = width, height
            mapped_audio = {track.index: writer.add_stream_from_template(track) for track in audio}
            for packet in reader.demux(video, *audio):
                if stop.is_set():
                    return 0, 0
                if packet.stream.index in mapped_audio:
                    if packet.dts is not None:
                        packet.stream = mapped_audio[packet.stream.index]
                        writer.mux(packet)
                    continue
                for frame in packet.decode():
                    if stop.is_set():
                        return 0, 0
                    if frame.pts is None:
                        raise ValueError("源视频缺少帧时间戳")
                    pixels = frame.to_ndarray(format="bgr24")
                    # Apply source orientation to the exported pixels and the face detector.
                    quarter_turns = round(frame.rotation / 90)
                    if quarter_turns:
                        pixels = np.ascontiguousarray(np.rot90(pixels, quarter_turns))
                    if (pixels.shape[1], pixels.shape[0]) != (stream.width, stream.height):
                        raise ValueError("源视频在片段中改变了分辨率或方向，请分段处理")
                    pixels, count = enhance_faces(pixels, detector, minimum_size)
                    output = av.VideoFrame.from_ndarray(pixels, format="bgr24")
                    output.pts, output.time_base = frame.pts, frame.time_base
                    for encoded in stream.encode(output):
                        writer.mux(encoded)
                    frames += 1
                    faces += count
                    if progress and (frames == 1 or frames % 12 == 0):
                        progress(frames, faces)
            if frames:
                for encoded in stream.encode(None):
                    writer.mux(encoded)
    return frames, faces


def enhance_face_video(source: Path, destination: Path, detector: SCRFDDetector,
                       minimum_size: int, stop: Event,
                       progress: Callable[[int, int], None] | None = None) -> tuple[int, int]:
    if stop.is_set():
        return 0, 0
    source = local_video_path(source)
    destination = local_path(destination)
    if destination.suffix.lower() != ".mkv":
        raise ValueError("请使用 MKV 格式保存视频人脸增强")
    if destination.exists():
        raise FileExistsError("输出文件已存在，请另选文件名")
    with TemporaryDirectory(prefix="face-enhancement-", dir=destination.parent) as temporary:
        staged = Path(temporary) / "video.mkv"
        frames, faces = _process(source, staged, detector, minimum_size, stop, progress)
        if stop.is_set():
            return 0, 0
        if not frames:
            raise ValueError("无法解码源视频画面")
        if not faces:
            raise ValueError("未找到符合最小尺寸设置的人脸")
        # Exclusive creation preserves pre-existing destinations, including a concurrent writer.
        with staged.open("rb") as converted, destination.open("xb") as exported:
            try:
                while data := converted.read(4 * 1024 * 1024):
                    if stop.is_set():
                        exported.close()
                        destination.unlink(missing_ok=True)
                        return 0, 0
                    exported.write(data)
                exported.flush()
                os.fsync(exported.fileno())
            except OSError:
                exported.close()
                destination.unlink(missing_ok=True)
                raise
    return frames, faces
