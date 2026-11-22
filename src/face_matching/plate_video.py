"""Local, non-generative enhancement of a manually selected video plate."""
from __future__ import annotations

import hashlib
import math
from collections.abc import Iterator
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from threading import Event

import cv2
import numpy as np

from .plate_restoration import align_flow, restore_motion, restore_plate_photo
from .restoration_candidates import RestorationCandidate, choose_candidates, score_candidate
from .restoration_io import encode_png, write_restoration_archive
from .vision.local_media import open_local_video


@dataclass(frozen=True)
class PlateFrame:
    index: int
    region: tuple[int, int, int, int]
    pixels: np.ndarray
    full_frame_sha256: str


@dataclass(frozen=True)
class PlateResult:
    source: Path
    anchor_index: int
    anchor_region: tuple[int, int, int, int]
    anchor: np.ndarray
    reference: PlateFrame
    frames: tuple[PlateFrame, ...]
    candidates: tuple[RestorationCandidate, ...]
    tracked_count: int
    tracking_notice: str
    blur_length: int
    blur_angle: int

    @property
    def enhanced(self) -> np.ndarray:
        return self.candidates[0].pixels


def read_frame(capture: cv2.VideoCapture, index: int) -> np.ndarray:
    if not capture.isOpened():
        raise ValueError("无法打开视频，请检查文件和视频编码")
    if index < 0:
        raise ValueError("帧号无效")
    if capture.get(cv2.CAP_PROP_POS_FRAMES) != index and not capture.set(cv2.CAP_PROP_POS_FRAMES, index):
        raise ValueError("此视频无法跳转到指定帧，请先截取需要处理的片段")
    ok, frame = capture.read()
    if not ok or capture.get(cv2.CAP_PROP_POS_FRAMES) != index + 1:
        raise ValueError("无法读取指定帧")
    return frame


def load_plate_preview(source: Path, index: int) -> tuple[np.ndarray, int | None]:
    capture = open_local_video(source)
    try:
        pixels = read_frame(capture, index)
        count = capture.get(cv2.CAP_PROP_FRAME_COUNT)
        return pixels, int(count) if math.isfinite(count) and count > index else None
    finally:
        capture.release()


def _plate_frame(index: int, region: tuple[int, int, int, int], frame: np.ndarray) -> PlateFrame:
    x1, y1, x2, y2 = region
    return PlateFrame(index, region, frame[y1:y2, x1:x2].copy(), hashlib.sha256(frame).hexdigest())


def _gray(pixels: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(pixels, cv2.COLOR_BGR2GRAY)


def _sharpness(frame: PlateFrame) -> float:
    gray = cv2.GaussianBlur(_gray(frame.pixels), (5, 5), 1)
    horizontal = cv2.Sobel(gray, cv2.CV_32F, 1, 0)
    vertical = cv2.Sobel(gray, cv2.CV_32F, 0, 1)
    return float(np.mean(horizontal ** 2 + vertical ** 2))


def _track(capture: cv2.VideoCapture, anchor: PlateFrame, count: int, stop: Event) -> tuple[list[PlateFrame], str]:
    template = _gray(anchor.pixels)
    height, width = template.shape
    if template.std() < 4:
        return [anchor], "起始选区细节太少，无法可靠跟踪"
    radius = max(16, min(192, width))
    reported_count = capture.get(cv2.CAP_PROP_FRAME_COUNT)
    x, y = anchor.region[:2]
    frames = [anchor]
    for index in range(anchor.index + 1, anchor.index + count):
        if stop.is_set():
            break
        ok, frame = capture.read()
        if not ok and 0 < reported_count <= index:
            return frames, "已到视频结尾"
        if not ok or capture.get(cv2.CAP_PROP_POS_FRAMES) != index + 1:
            return frames, f"第 {index + 1} 帧不可读取，已停止跟踪"
        left, top = max(0, x - radius), max(0, y - radius)
        search = frame[top:y + height + radius, left:x + width + radius]
        if search.shape[0] < height or search.shape[1] < width:
            return frames, f"第 {index + 1} 帧的选区移出画面或画面尺寸变化，已停止跟踪"
        scores = cv2.matchTemplate(_gray(search), template, cv2.TM_CCOEFF_NORMED)
        _, score, _, (dx, dy) = cv2.minMaxLoc(scores)
        # Stop at a cut, occlusion, or an ambiguous match instead of switching plates.
        if not math.isfinite(score) or score < 0.70:
            return frames, f"第 {index + 1} 帧匹配不足，已停止跟踪"
        scores[max(0, dy - height // 2):dy + height // 2 + 1,
               max(0, dx - width // 2):dx + width // 2 + 1] = -1
        if scores.max() >= score - 0.07:
            return frames, f"第 {index + 1} 帧有多个相似位置，已停止跟踪"
        x, y = left + dx, top + dy
        frames.append(_plate_frame(index, (x, y, x + width, y + height), frame))
    return frames, ""


def _align(reference_gray: np.ndarray, pixels: np.ndarray) -> np.ndarray | None:
    height, width = reference_gray.shape
    try:
        score, warp = cv2.findTransformECC(
            reference_gray, _gray(pixels), np.eye(2, 3, dtype=np.float32), cv2.MOTION_EUCLIDEAN,
            (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 35, 0.001),
        )
    except cv2.error:
        return None
    if (not np.isfinite(warp).all() or not np.isfinite(score) or score < 0.88
            or abs(warp[0, 1]) > 0.15 or np.linalg.norm(warp[:, 2]) > min(width, height) / 4):
        return None
    aligned = cv2.warpAffine(pixels.astype(np.float32), warp, (width, height),
                             flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=(np.nan, np.nan, np.nan))
    return aligned


def enhance_plate_video(source: Path, index: int, region: tuple[int, int, int, int],
                        anchor: np.ndarray, count: int, blur_length: int, blur_angle: int,
                        stop: Event) -> PlateResult | None:
    x1, y1, x2, y2 = region
    height, width = anchor.shape[:2]
    if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height
            and 16 <= x2 - x1 <= 640 and 8 <= y2 - y1 <= 320):
        raise ValueError("请框选车牌，选区宽度须为 16～640 像素，高度须为 8～320 像素")
    if not (2 <= count <= 60 and (blur_length == 0 or 3 <= blur_length <= 21) and 0 <= blur_angle <= 179):
        raise ValueError("车牌处理参数超出范围")
    capture = open_local_video(source)
    try:
        if not np.array_equal(read_frame(capture, index), anchor):
            raise ValueError("源视频画面已变化，请重新打开并框选")
        first = _plate_frame(index, region, anchor)
        tracked, tracking_notice = _track(capture, first, count, stop)
    finally:
        capture.release()
    if stop.is_set():
        return None

    ranked = sorted(tracked, key=_sharpness, reverse=True)
    best = ranked[0]
    reference_gray = _gray(best.pixels)
    minimum_quality = _sharpness(best) * 0.65
    used = [best]
    original = best.pixels.astype(np.float32)
    total = original.copy()
    weights = np.ones(original.shape[:2], np.float32)
    observations = []
    for frame in ranked[1:]:
        if stop.is_set():
            return None
        if len(used) >= 8 or _sharpness(frame) < minimum_quality:
            break
        aligned = _align(reference_gray, frame.pixels)
        if aligned is None:
            continue
        # Subpixel resampling changes some edge intensities. Fuse only pixels that
        # still agree, with the reference dominant to limit residual double edges.
        aligned, support = align_flow(best.pixels, aligned)
        valid = support & (np.max(np.abs(aligned - original), axis=2) <= 28)
        if valid.mean() < 0.55:
            continue
        weight = 0.5 / np.sqrt(len(used))
        total[valid] += aligned[valid] * weight
        weights[valid] += weight
        used.append(frame)
        observations.append(np.where(valid[..., None], aligned, original).astype(np.uint8))
    fused = np.rint(total / weights[..., None]).astype(np.uint8)
    candidates = []
    if len(used) > 1:
        candidates.append(score_candidate("多帧一致性融合", fused, best.pixels, tuple(observations)))
        if np.abs(fused.astype(np.float32) - original).mean() > 1:
            candidates.append(score_candidate("参考清晰帧", best.pixels, best.pixels, tuple(observations)))
    if blur_length >= 3:
        restored = restore_motion(fused, blur_length, blur_angle)
        candidates.append(score_candidate("约束去拖影", restored, best.pixels, tuple(observations)))
    if not candidates:
        h, w = best.pixels.shape[:2]
        candidates.extend(restore_plate_photo(best.pixels, (0, 0, w, h)))
    if stop.is_set():
        return None
    ranked_candidates = choose_candidates(candidates)
    return PlateResult(source, index, region, anchor, best, tuple(sorted(used, key=lambda frame: frame.index)),
                       ranked_candidates, len(tracked), tracking_notice, blur_length, blur_angle)


def _original_filename(index: int) -> str:
    return f"original/frame-{index + 1:08d}.png"


def _export_images(result: PlateResult) -> Iterator[tuple[str, bytes]]:
    yield _original_filename(result.anchor_index), encode_png(result.anchor)
    yield "enhanced.png", encode_png(result.enhanced)
    for index, candidate in enumerate(result.candidates[1:], 2):
        yield f"alternatives/{index}.png", encode_png(candidate.pixels)
    later = [frame for frame in result.frames if frame.index != result.anchor_index]
    if not later:
        return
    capture = open_local_video(result.source)
    try:
        if not np.array_equal(read_frame(capture, result.anchor_index), result.anchor):
            raise ValueError("源视频画面已变化，请重新打开并处理")
        originals = {frame.index: frame for frame in later}
        for index in range(result.anchor_index + 1, later[-1].index + 1):
            pixels = read_frame(capture, index)
            original = originals.get(index)
            if original is not None:
                if hashlib.sha256(pixels).hexdigest() != original.full_frame_sha256:
                    raise ValueError(f"源视频第 {index + 1} 帧已变化，请重新打开并处理")
                yield _original_filename(index), encode_png(pixels)
    finally:
        capture.release()


def export_plate_result(destination: Path, result: PlateResult) -> None:
    manifest = {
        "schema_version": 1,
        "algorithm": "plate-video-ecc-flow-v2",
        "candidates": [
            {"file": "enhanced.png" if index == 1 else f"alternatives/{index}.png",
             "label": item.label, "score": item.score, "agreement": item.agreement,
             "detail": item.detail, "observations": item.observations}
            for index, item in enumerate(result.candidates, 1)
        ],
        "source_name": result.source.name,
        "anchor_frame_number": result.anchor_index + 1,
        "anchor_file": _original_filename(result.anchor_index),
        "anchor_region_xyxy": result.anchor_region,
        "reference_frame_number": result.reference.index + 1,
        "original_frames": [
            {"file": _original_filename(frame.index), "frame_number": frame.index + 1, "region_xyxy": frame.region}
            for frame in result.frames
        ],
        "tracked_frame_count": result.tracked_count,
        "tracking_notice": result.tracking_notice,
        "single_frame_fallback": len(result.frames) == 1,
        "motion_blur_length_px": result.blur_length,
        "motion_blur_angle_degrees": result.blur_angle,
        "notice": "还原结果供辅助观察。请同时复核原始视频和原始帧。帧号从 1 开始，框坐标从 0 开始，右、下边界不含在框内。",
    }
    with closing(_export_images(result)) as images:
        write_restoration_archive(destination, images, manifest)
