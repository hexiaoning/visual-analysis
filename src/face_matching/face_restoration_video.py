"""Render offline portrait restoration without losing source timestamps or audio."""
from __future__ import annotations

from pathlib import Path
import os
from tempfile import TemporaryDirectory
from threading import Event
from typing import Callable

import av
import cv2
import numpy as np

from .config import AppConfig
from .face_restoration import (
    AlignedFace, DICFACE_FILENAME, GFPGAN_FILENAME, FaceRestorer, aligned_face,
    create_restoration_detector, paste_face, restoration_model_dir,
)
from .vision.local_media import local_path, local_video_path


class VideoPortraitRestorer:
    def __init__(self, config: AppConfig, stop: Event):
        self.stop = stop
        self.detector = create_restoration_detector(config)
        folder = restoration_model_dir()
        dicface = folder / DICFACE_FILENAME
        gfpgan = folder / GFPGAN_FILENAME
        if dicface.is_file():
            self.model = FaceRestorer(dicface, "DicFace", 512, config, frame_count=5)
        elif gfpgan.is_file():
            self.model = FaceRestorer(gfpgan, "GFPGAN v1.4", 512, config)
        else:
            raise FileNotFoundError("缺少本地视频人像还原模型")
        self.previous: list[tuple[AlignedFace, tuple[np.ndarray, ...], np.ndarray]] = []

    def __call__(self, frame: np.ndarray) -> tuple[np.ndarray, int]:
        result = frame.copy()
        current = []
        remaining = self.previous.copy()
        for detection in self.detector.detect(frame):
            if self.stop.is_set():
                break
            face = aligned_face(frame, detection, self.model.size)
            if face is None:
                continue
            tracked = None
            distance = float("inf")
            for index, earlier in enumerate(remaining):
                before = earlier[0].detection
                delta = np.linalg.norm(before.bbox - detection.bbox)
                if delta < distance and delta < min(detection.width, detection.height) * 0.45:
                    tracked, distance = index, delta

            history = ()
            previous = None
            if tracked is not None:
                before, history, past_residual = remaining.pop(tracked)
                crop_change = np.abs(face.crop.astype(np.float32) - before.crop).mean()
                if crop_change <= 24:
                    previous = before.crop, past_residual
                else:
                    history = ()

            if self.model.frame_count == 1:
                restored, = self.model.restore((face.crop,))
                if previous is not None:
                    restored = _stable_image(face.crop, restored, *previous)
            else:
                inputs = (*history, face.crop)
                inputs = (inputs[0],) * (self.model.frame_count - len(inputs)) + inputs
                restored = self.model.restore(inputs)[-1]
            paste_face(result, face, restored)
            current.append((face, (*history, face.crop)[-4:], restored.astype(np.float32) - face.crop))
        self.previous = current
        return result, len(current)


def _stable_image(current: np.ndarray, restored: np.ndarray, previous: np.ndarray,
                  previous_residual: np.ndarray) -> np.ndarray:
    # Align a prior residual to the current source rather than averaging eyes
    # or lips across differently positioned frames.
    gray_now = cv2.cvtColor(current, cv2.COLOR_BGR2GRAY)
    gray_before = cv2.cvtColor(previous, cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(gray_now, gray_before, None, 0.5, 2, 13, 2, 5, 1.1, 0)
    h, w = gray_now.shape
    x, y = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    x, y = x + flow[..., 0], y + flow[..., 1]
    aligned_source = cv2.remap(gray_before, x, y, cv2.INTER_LINEAR)
    aligned_detail = cv2.remap(previous_residual, x, y, cv2.INTER_LINEAR)
    visible = ((np.abs(gray_now.astype(np.float32) - aligned_source) < 9)
               & (np.linalg.norm(flow, axis=2) < 10))
    result = restored.astype(np.float32)
    smoothed = (current + aligned_detail) * 0.2 + result * 0.8
    result[visible] = smoothed[visible]
    return np.rint(np.clip(result, 0, 255)).astype(np.uint8)


def restore_face_video(
    source: Path, destination: Path, config: AppConfig, stop: Event,
    progress: Callable[[int, int], None] | None = None,
) -> tuple[str, int, int]:
    engine = VideoPortraitRestorer(config, stop)
    frames, faces = render_portrait_video(source, destination, engine, stop, progress)
    return engine.model.name, frames, faces


def _process(source: Path, target: Path, restore,
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
                    pixels, count = restore(pixels)
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


def render_portrait_video(source: Path, destination: Path, restore,
                          stop: Event,
                       progress: Callable[[int, int], None] | None = None) -> tuple[int, int]:
    if stop.is_set():
        return 0, 0
    source = local_video_path(source)
    destination = local_path(destination)
    if destination.suffix.lower() != ".mkv":
        raise ValueError("请使用 MKV 格式保存视频人像还原")
    if destination.exists():
        raise FileExistsError("输出文件已存在，请另选文件名")
    with TemporaryDirectory(prefix="face-enhancement-", dir=destination.parent) as temporary:
        staged = Path(temporary) / "video.mkv"
        frames, faces = _process(source, staged, restore, stop, progress)
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
