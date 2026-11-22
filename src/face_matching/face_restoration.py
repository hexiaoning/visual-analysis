"""Local model-based facial restoration in an FFHQ-aligned face crop."""
from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from threading import Event

import cv2
import numpy as np

from .config import AppConfig
from .errors import GPUUnavailableError
from .gpu import create_gpu_session
from .models import required_paths
from .paths import models_dir, packaged_models_dir
from .restoration_candidates import RestorationCandidate, choose_candidates, score_candidate
from .vision.alignment import estimate_similarity_transform
from .vision.detector import Detection, SCRFDDetector
from .vision.local_media import local_path


FFHQ_LANDMARKS = np.array(
    [[192.98138, 239.94708], [318.90277, 240.1936], [256.63416, 314.01935],
     [201.26117, 371.41043], [313.08905, 371.15118]],
    np.float32,
)
GFPGAN_FILENAME = "gfpgan-v1.4-fp32.onnx"
PCFLOW_FILENAME = "pcflow-bfr-fp32.onnx"
DICFACE_FILENAME = "dicface-bfr-fp32.onnx"


def restoration_model_dir() -> Path:
    return local_path(packaged_models_dir() or models_dir())


def cpu_session(path: Path):
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    options.inter_op_num_threads = 1
    return ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])


def create_restoration_detector(config: AppConfig) -> SCRFDDetector:
    model, _ = required_paths(config.model_profile)
    try:
        session = create_gpu_session(model, config.gpu_device_id, backend=config.gpu_backend)
    except GPUUnavailableError:
        session = cpu_session(model)
    return SCRFDDetector(session, input_size=config.detector_size, threshold=config.detector_threshold,
                         nms_threshold=config.nms_threshold)


class FaceRestorer:
    def __init__(self, path: Path, name: str, size: int, config: AppConfig, frame_count: int = 1):
        self.path = local_path(path)
        self.name = name
        self.size = size
        self.frame_count = frame_count
        self.session = self._session(config)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def _session(self, config: AppConfig):
        try:
            return create_gpu_session(self.path, config.gpu_device_id, backend=config.gpu_backend)
        except GPUUnavailableError:
            return cpu_session(self.path)

    def restore(self, faces: tuple[np.ndarray, ...]) -> tuple[np.ndarray, ...]:
        if len(faces) != self.frame_count:
            raise ValueError(f"{self.name} 需要 {self.frame_count} 帧对齐人脸")
        normalized = np.stack([face[:, :, ::-1].transpose(2, 0, 1) for face in faces]).astype(np.float32)
        normalized = normalized / 127.5 - 1
        output = np.asarray(self.session.run([self.output_name], {self.input_name: normalized})[0])
        if output.shape != normalized.shape or not np.isfinite(output).all():
            raise ValueError(f"{self.name} 返回了无效的还原图像")
        bgr = np.clip(np.rint((output + 1) * 127.5), 0, 255).astype(np.uint8)
        return tuple(np.ascontiguousarray(face.transpose(1, 2, 0)[:, :, ::-1]) for face in bgr)


@dataclass(frozen=True)
class AlignedFace:
    crop: np.ndarray
    image_to_face: np.ndarray
    detection: Detection


def aligned_face(image: np.ndarray, detection: Detection, size: int) -> AlignedFace | None:
    points = np.asarray(detection.landmarks, dtype=np.float32).reshape(5, 2)
    if (not np.isfinite(points).all() or min(detection.width, detection.height) < 24
            or np.linalg.norm(points[0] - points[1]) < 6):
        return None
    target = FFHQ_LANDMARKS * (size / 512)
    try:
        matrix = estimate_similarity_transform(points, target)
    except (ValueError, cv2.error):
        return None
    deviations = np.linalg.norm(cv2.transform(points[None], matrix)[0] - target, axis=1)
    if np.max(deviations) > size * 0.14:
        return None
    crop = cv2.warpAffine(image, matrix, (size, size), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REFLECT_101)
    return AlignedFace(crop, matrix, detection)


def paste_face(image: np.ndarray, face: AlignedFace, restored: np.ndarray, strength: float = 1.0) -> None:
    h, w = image.shape[:2]
    size = face.crop.shape[0]
    y, x = np.ogrid[-1:1:complex(size), -1:1:complex(size)]
    mask = np.clip((1 - (x / 0.83)**2 - ((y - 0.10) / 0.89)**2) * 3, 0, 1).astype(np.float32)
    inverse = cv2.invertAffineTransform(face.image_to_face)
    corners = np.array([[[0, 0], [size, 0], [size, size], [0, size]]], dtype=np.float32)
    limits = cv2.transform(corners, inverse)[0]
    left, top = np.maximum(np.floor(limits.min(axis=0)).astype(int), 0)
    right, bottom = np.minimum(np.ceil(limits.max(axis=0)).astype(int), (w, h))
    if right <= left or bottom <= top:
        return
    inverse[:, 2] -= (left, top)
    dimensions = right - left, bottom - top
    alpha = cv2.warpAffine(mask, inverse, dimensions)[..., None] * strength
    region = image[top:bottom, left:right]
    warped = cv2.warpAffine(restored, inverse, dimensions, borderMode=cv2.BORDER_REFLECT_101)
    region[:] = np.rint(region * (1 - alpha) + warped * alpha).astype(np.uint8)


def restore_portrait(
    image: np.ndarray, config: AppConfig, stop: Event,
    region: tuple[int, int, int, int] | None = None,
) -> tuple[RestorationCandidate, ...]:
    models = restoration_model_dir()
    options = [
        (PCFLOW_FILENAME, "PCFlow", 512),
        (GFPGAN_FILENAME, "GFPGAN v1.4", 512),
    ]
    detections = create_restoration_detector(config).detect(image)
    if region:
        x1, y1, x2, y2 = region
        detections = [face for face in detections
                      if x1 <= face.bbox[[0, 2]].mean() < x2 and y1 <= face.bbox[[1, 3]].mean() < y2]
    if not detections:
        raise ValueError("选区内未检测到符合条件的人脸")
    candidates = []
    for filename, name, size in options:
        path = models / filename
        if not path.is_file():
            continue
        faces = [face for detection in detections if (face := aligned_face(image, detection, size)) is not None]
        if not faces:
            continue
        restored_image = image.copy()
        model = FaceRestorer(path, name, size, config)
        scores = []
        for face in faces:
            if stop.is_set():
                return ()
            restored, = model.restore((face.crop,))
            scores.append(score_candidate(name, restored, face.crop))
            paste_face(restored_image, face, restored)
        weakest = min(scores, key=lambda score: score.score)
        candidates.append(replace(weakest, pixels=restored_image))
        del model
    if not candidates:
        raise ValueError("没有可用的本地人脸还原模型，或人脸关键点不足以对齐")
    # Expose both model outputs until an agreement rule is validated for photos.
    return choose_candidates(candidates)
