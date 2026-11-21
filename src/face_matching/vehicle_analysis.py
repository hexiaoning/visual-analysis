"""Local vehicle candidate matching and one-shot catalogue verification."""
from __future__ import annotations

import base64
import binascii
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import os
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from threading import Event
from time import monotonic
import unicodedata

import cv2
import numpy as np

from .config import AppConfig
from .errors import FaceMatchingError, GPUUnavailableError
from .models import required_person_vehicle_model
from .paths import app_home, is_frozen_app
from .person_vehicle import PersonVehicleDetector, crop_object, load_person_vehicle_detector
from .semantic_model import SemanticEncoder
from .vehicle_provider import CANDIDATE_FIELDS, SOURCES
from .vision.io import IMAGE_EXTENSIONS, read_image
from .vision.local_media import local_path, open_local_video


@dataclass(frozen=True)
class VehicleCandidate:
    brand: str
    series: str
    model: str = ""
    year: str = ""
    trim: str = ""
    catalogue_id: str = ""


# A small starter vocabulary gives a local initial result before a licensed
# image catalogue is imported. It intentionally contains series, not trims.
_STARTER_SERIES = (
    VehicleCandidate("比亚迪", "海鸥"), VehicleCandidate("比亚迪", "秦 PLUS"),
    VehicleCandidate("比亚迪", "宋 PLUS"), VehicleCandidate("比亚迪", "汉"),
    VehicleCandidate("特斯拉", "Model 3"), VehicleCandidate("特斯拉", "Model Y"),
    VehicleCandidate("小米", "SU7"), VehicleCandidate("小米", "YU7"),
    VehicleCandidate("理想", "L6"), VehicleCandidate("理想", "L9"),
    VehicleCandidate("问界", "M7"), VehicleCandidate("问界", "M9"),
    VehicleCandidate("吉利", "星愿"), VehicleCandidate("五菱", "宏光 MINIEV"),
    VehicleCandidate("丰田", "卡罗拉"), VehicleCandidate("丰田", "凯美瑞"),
    VehicleCandidate("本田", "思域"), VehicleCandidate("大众", "朗逸"),
)


@dataclass(frozen=True)
class CatalogueEntry:
    candidate: VehicleCandidate
    images: tuple[Path, ...]


@dataclass(frozen=True)
class VehicleMatch:
    candidate: VehicleCandidate
    similarity: float | None


def load_catalogue(value: Path) -> list[CatalogueEntry]:
    path = local_path(value)
    raw = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(raw, dict) or raw.get("schema_version") != 1 or not isinstance(raw.get("vehicles"), list):
        raise ValueError("车型参考库须使用 schema_version 1 和 vehicles 数组")
    entries = []
    for item in raw["vehicles"]:
        if not isinstance(item, dict) or not isinstance(item.get("images"), list):
            raise ValueError("车型参考库记录缺少图片列表")
        values = {key: item.get(key, "") for key in CANDIDATE_FIELDS}
        if any(not isinstance(text, str) or len(text) > 200 for text in values.values()):
            raise ValueError("车型参考库的车型字段须为不超过 200 字的文本")
        candidate = VehicleCandidate(**{key: text.strip() for key, text in values.items()})
        if not candidate.brand or not candidate.series:
            raise ValueError("车型参考库记录缺少品牌或车系")
        images = []
        for name in item["images"]:
            if not isinstance(name, str) or Path(name).is_absolute() or Path(name).drive:
                raise ValueError("参考图须使用参考库目录内的相对路径")
            image = local_path(path.parent / name)
            if not image.is_relative_to(path.parent) or image.suffix.lower() not in IMAGE_EXTENSIONS:
                raise ValueError("参考图须使用参考库目录内的图片")
            images.append(image)
        if not images:
            raise ValueError("每条车型参考库记录至少需要一张参考图")
        entries.append(CatalogueEntry(candidate, tuple(images)))
    if not entries:
        raise ValueError("车型参考库没有可用记录")
    return entries


def _overlap(left: tuple, right: tuple) -> float:
    a, b, c, d = left
    x, y, u, v = right
    intersection = max(0, min(c, u) - max(a, x)) * max(0, min(d, v) - max(b, y))
    union = (c - a) * (d - b) + (u - x) * (v - y) - intersection
    return intersection / union if union > 0 else 0.0


def _sample_target(path: Path, detector, encoder, stop: Event | None = None) -> list[np.ndarray]:
    def vehicles(frame):
        return [item for item in detector.detect(frame) if item.class_id in (2, 3, 5, 7)]

    def largest(item):
        x1, y1, x2, y2 = item.bbox
        return (x2 - x1) * (y2 - y1)

    if path.suffix.lower() in IMAGE_EXTENSIONS:
        image = read_image(path)
        if image is None:
            raise ValueError("无法读取车辆图片")
        targets = vehicles(image)
        if not targets:
            raise ValueError("未检测到车辆。请使用能够清楚看到车身的素材。")
        return [encoder.image(crop_object(image, max(targets, key=largest).bbox))]

    cap = open_local_video(path)
    if not cap.isOpened():
        cap.release()
        raise ValueError("无法读取车辆视频")
    views = []
    previous = None
    misses = 0
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        interval = max(1, round(fps / 6)) if np.isfinite(fps) and fps > 0 else 5
        index = 0
        while len(views) < 8 and (stop is None or not stop.is_set()) and cap.grab():
            index += 1
            if (index - 1) % interval:
                continue
            ok, image = cap.retrieve()
            if not ok:
                break
            targets = vehicles(image)
            if previous is None:
                if not targets:
                    continue
                target = max(targets, key=largest)
            else:
                target = max(targets, key=lambda item: _overlap(previous, item.bbox), default=None)
                if target is None or _overlap(previous, target.bbox) < 0.15:
                    misses += 1
                    if misses > 3:
                        break
                    continue
            previous = target.bbox
            misses = 0
            views.append(encoder.image(crop_object(image, previous)))
    finally:
        cap.release()
    if not views and not (stop and stop.is_set()):
        raise ValueError("视频中未找到可用于比对的车辆")
    return views


def _similarity(views: list[np.ndarray], references: list[np.ndarray]) -> float:
    return float(np.clip(np.max(np.asarray(views) @ np.asarray(references).T, axis=1).mean(), -1, 1))


def _local_matches(views: list[np.ndarray], entries: list[CatalogueEntry], encoder,
                   stop: Event | None = None) -> list[VehicleMatch]:
    matches = []
    for entry in entries:
        if stop is not None and stop.is_set():
            return []
        references = []
        for path in entry.images:
            if stop is not None and stop.is_set():
                return []
            image = read_image(path)
            if image is not None:
                references.append(encoder.image(image))
        if references:
            matches.append(VehicleMatch(entry.candidate, _similarity(views, references)))
    matches.sort(key=lambda match: match.similarity, reverse=True)
    return matches[:3]


def _starter_matches(views: list[np.ndarray], encoder, stop: Event | None) -> list[VehicleMatch]:
    matches = []
    for candidate in _STARTER_SERIES:
        if stop is not None and stop.is_set():
            return []
        feature = encoder.text(f"一辆{candidate.brand} {candidate.series}汽车")
        matches.append(VehicleMatch(candidate, _similarity(views, [feature])))
    return sorted(matches, key=lambda match: match.similarity, reverse=True)[:3]


def _unavailable(status: str) -> list[dict]:
    return [{"source": source, "queried_at": "", "status": status, "records": []} for source in SOURCES]


def _run_catalogue_worker(candidates: list[VehicleCandidate], stop: Event | None = None) -> list[dict]:
    if not candidates:
        return _unavailable("no_candidate")
    settings_path = local_path(app_home() / "vehicle-providers.json")
    if not settings_path.is_file():
        return _unavailable("not_configured")
    try:
        settings = json.loads(settings_path.read_text(encoding="utf-8-sig"))
        if not isinstance(settings, dict):
            return _unavailable("not_configured")
    except (OSError, ValueError):
        return _unavailable("not_configured")
    return _query_worker(settings, candidates, stop)


def _query_worker(settings: dict, candidates: list[VehicleCandidate], stop: Event | None = None) -> list[dict]:
    command = ([str(Path(sys.executable).with_name("VehicleCatalogueWorker.exe"))]
               if is_frozen_app() else [sys.executable, "-m", "face_matching.vehicle_provider"])
    started = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with TemporaryDirectory(prefix="vehicle-catalogue-") as directory:
        request = Path(directory) / "request.json"
        response = Path(directory) / "response.json"
        request.write_text(json.dumps({
            "settings": settings, "candidates": [asdict(candidate) for candidate in candidates],
        }, ensure_ascii=False), encoding="utf-8")
        try:
            with subprocess.Popen([*command, str(request), str(response)],
                                  stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                  creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)) as process:
                deadline = monotonic() + 22
                while process.poll() is None:
                    if (stop is not None and stop.is_set()) or monotonic() >= deadline:
                        process.kill()
                        break
                    try:
                        process.wait(timeout=0.1)
                    except subprocess.TimeoutExpired:
                        continue
            if process.returncode == 0 and response.is_file():
                results = json.loads(response.read_text(encoding="utf-8"))
                if isinstance(results, list) and [item.get("source") for item in results] == list(SOURCES):
                    return results
        except (OSError, ValueError, AttributeError):
            pass
    fallbacks = _unavailable("unavailable")
    for source in fallbacks:
        key = source["source"]
        if key in settings and os.environ.get(f"VISUAL_VEHICLE_{key.upper()}_TOKEN"):
            source.update(status="unavailable", queried_at=started)
    return fallbacks


def _normalized(text: str) -> str:
    return "".join(unicodedata.normalize("NFKC", text).casefold().split())


def _record_match(candidate: VehicleCandidate, record: dict) -> str:
    same_id = bool(candidate.catalogue_id and
                   _normalized(candidate.catalogue_id) == _normalized(record["catalogue_id"]))
    same_series = all(_normalized(getattr(candidate, key)) == _normalized(record[key])
                      for key in ("brand", "series"))
    if not (same_id or same_series):
        return ""
    if candidate.model and _normalized(candidate.model) == _normalized(record["model"]):
        if all(not getattr(candidate, key) or
               _normalized(getattr(candidate, key)) == _normalized(record[key]) for key in ("year", "trim")):
            return "车款"
    return "车系"


def _compare_online(results: list[dict], matches: list[VehicleMatch], views: list[np.ndarray], encoder) -> str:
    supported = []
    for source in results:
        strengths = set()
        for record in source["records"]:
            references = []
            for picture in record.pop("image_jpeg_base64", []):
                try:
                    data = base64.b64decode(picture, validate=True)
                    if not data.startswith(b"\xff\xd8"):
                        continue
                    image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_REDUCED_COLOR_2)
                    if image is not None and image.size:
                        references.append(encoder.image(image))
                except (ValueError, binascii.Error, cv2.error, FaceMatchingError):
                    continue
            score = _similarity(views, references) if views and references else None
            record["image_similarity"] = score
            record["candidate_levels"] = {_candidate_key(match.candidate): level for match in matches
                                          if (level := _record_match(match.candidate, record))}
            strengths.update(record["candidate_levels"])
        if strengths:
            supported.append(strengths)
    if len(supported) == len(SOURCES):
        common = set.intersection(*supported)
        if not common:
            return "双源结果存在差异 · 请复核来源"
        differences = []
        for key in sorted(common):
            matched = [[record for record in source["records"] if key in record["candidate_levels"]]
                       for source in results]
            fields = (("brand", "品牌"), ("series", "车系"), ("model", "车型名称"),
                      ("year", "年款"), ("trim", "配置"))
            for field, label in fields:
                left, right = [{record[field] for record in records if record[field]} for records in matched]
                if left and right and not ({_normalized(value) for value in left} &
                                           {_normalized(value) for value in right}):
                    differences.append(f"{key} · {label} · 工信部 {'、'.join(sorted(left))}"
                                       f" · 懂车帝 {'、'.join(sorted(right))}")
        if differences:
            results[0]["differences"] = differences
            return "已查双源 · 车款资料有差异，请复核"
        return "双源资料字段一致 · 请复核候选及参考图"
    if supported:
        return "部分联网核验"
    if any(source["status"] in {"ok", "no_match"} for source in results):
        return "已查在线资料 · 未找到一致车型"
    return "本地初判，未联网核验"


def _candidate_key(candidate: VehicleCandidate) -> str:
    return " / ".join(text for text in (candidate.brand, candidate.series, candidate.model,
                                      candidate.year, candidate.trim, candidate.catalogue_id) if text)


def _cpu_session(model: Path, *args, **kwargs):
    from onnxruntime import InferenceSession
    return InferenceSession(str(model), providers=["CPUExecutionProvider"])


def _local_candidates(path, entries, config, manual, stop, use_cpu=False):
    if use_cpu:
        encoder = SemanticEncoder(session_factory=_cpu_session)
        detector = PersonVehicleDetector(_cpu_session(required_person_vehicle_model()), people=False)
    else:
        encoder = SemanticEncoder(config.gpu_backend, config.gpu_device_id)
        detector = load_person_vehicle_detector(config.gpu_backend, config.gpu_device_id, people=False)
    views = _sample_target(path, detector, encoder, stop)
    if not views:
        return views, [], encoder, ""
    if entries:
        matches = _local_matches(views, entries, encoder, stop)
        local_method = "本地参考图"
    elif manual is None:
        matches = _starter_matches(views, encoder, stop)
        local_method = "内置车系词表"
    else:
        matches = []
        local_method = "人工候选"
    return views, matches, encoder, local_method


def analyze_vehicle(source: Path, catalogue: Path | None, config: AppConfig,
                    manual: VehicleCandidate | None = None, stop: Event | None = None) -> dict:
    path = local_path(source)
    if not path.is_file():
        raise ValueError("请选择本机车辆图片或视频")
    entries = load_catalogue(catalogue) if catalogue is not None else []
    try:
        views, matches, encoder, local_method = _local_candidates(path, entries, config, manual, stop)
        inference_backend = "GPU"
    except GPUUnavailableError:
        if stop is not None and stop.is_set():
            return {}
        views, matches, encoder, local_method = _local_candidates(path, entries, config, manual, stop, use_cpu=True)
        inference_backend = "CPU"
    if not views:
        return {}
    if manual is not None:
        matches = [VehicleMatch(manual, None), *(match for match in matches if match.candidate != manual)][:3]
    if stop is not None and stop.is_set():
        return {}
    results = _run_catalogue_worker([match.candidate for match in matches], stop)
    if stop is not None and stop.is_set():
        return {}
    status = _compare_online(results, matches, views, encoder)
    return {
        "status": status, "source": str(path), "visual_encoder": encoder.model_id,
        "local_method": local_method, "inference_backend": inference_backend,
        "candidates": [asdict(match) for match in matches], "catalogues": results,
    }
