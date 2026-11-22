"""Observation-constrained restoration for a local plate crop."""
from __future__ import annotations

import cv2
import numpy as np

from .restoration_candidates import RestorationCandidate, choose_candidates, score_candidate


def _kernel(length: int, angle: int) -> np.ndarray:
    kernel = np.zeros((length | 1, length | 1), np.float32)
    center = kernel.shape[0] // 2
    radians = np.deg2rad(angle)
    dx = round((length - 1) * np.cos(radians) / 2)
    dy = round((length - 1) * np.sin(radians) / 2)
    cv2.line(kernel, (center - dx, center - dy), (center + dx, center + dy), 1, 1)
    return kernel / kernel.sum()


def restore_motion(image: np.ndarray, length: int, angle: int) -> np.ndarray:
    psf = _kernel(length, angle)
    radius = length + 4
    padded = cv2.copyMakeBorder(image, radius, radius, radius, radius, cv2.BORDER_REFLECT_101)
    kernel = np.zeros(padded.shape[:2], np.float32)
    kernel[:psf.shape[0], :psf.shape[1]] = psf
    kernel = np.roll(kernel, (-psf.shape[0] // 2 + 1, -psf.shape[1] // 2 + 1), axis=(0, 1))
    spectrum = np.fft.rfft2(kernel)
    inverse = spectrum.conj() / (np.abs(spectrum)**2 + 0.04)
    transformed = np.fft.rfft2(padded, axes=(0, 1))
    restored = 1.04 * np.fft.irfft2(transformed * inverse[..., None], s=kernel.shape, axes=(0, 1))
    residual = (restored[radius:-radius, radius:-radius] - image) * 0.65
    return np.rint(np.clip(image + np.clip(residual, -30, 30), 0, 255)).astype(np.uint8)


def align_flow(reference: np.ndarray, aligned: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Give each warped pixel a bidirectional flow check before it can vote."""
    h, w = reference.shape[:2]
    finite = np.isfinite(aligned).all(axis=2)
    donor = np.where(finite[..., None], aligned, reference).astype(np.uint8)
    ref_gray = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
    donor_gray = cv2.cvtColor(donor, cv2.COLOR_BGR2GRAY)
    if min(h, w) < 24:
        return aligned, finite

    # A small pyramid keeps flow stable on narrow plate crops.
    parameters = (None, 0.5, 2, 13, 3, 5, 1.2, 0)
    forward = cv2.calcOpticalFlowFarneback(ref_gray, donor_gray, *parameters)
    backward = cv2.calcOpticalFlowFarneback(donor_gray, ref_gray, *parameters)
    x, y = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    map_x, map_y = x + forward[..., 0], y + forward[..., 1]
    reverse = cv2.remap(backward, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                        borderValue=(100, 100))
    closure = np.linalg.norm(forward + reverse, axis=2)
    displacement = np.linalg.norm(forward, axis=2)
    valid = ((closure <= 0.85) & (displacement <= 2.5)
             & (cv2.remap(finite.astype(np.uint8), map_x, map_y, cv2.INTER_NEAREST) != 0))
    warped = cv2.remap(donor.astype(np.float32), map_x, map_y, cv2.INTER_LINEAR)
    return warped, valid


def restore_plate_photo(image: np.ndarray, region: tuple[int, int, int, int]) -> tuple[RestorationCandidate, ...]:
    x1, y1, x2, y2 = region
    h, w = image.shape[:2]
    if not (0 <= x1 < x2 <= w and 0 <= y1 < y2 <= h
            and 16 <= x2 - x1 <= 1024 and 8 <= y2 - y1 <= 512):
        raise ValueError("请框选车牌，选区宽度须为 16～1024 像素、高度须为 8～512 像素")
    crop = image[y1:y2, x1:x2]
    choices = []
    for angle in (0, 45, 90, 135):
        for length in (3, 5, 9):
            prediction = restore_motion(crop, length, angle)
            score = score_candidate(f"去拖影 {angle}° / {length}px", prediction, crop)
            choices.append(score)
    ranked = choose_candidates(choices)
    results = []
    for entry in ranked:
        if any(np.abs(entry.pixels.astype(np.float32) - other.pixels).mean() < 1.5 for other in results):
            continue
        results.append(entry)
        if len(results) == 3:
            break
    return tuple(results)
