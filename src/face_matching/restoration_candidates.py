"""Score different restorations against the observations that support them."""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class RestorationCandidate:
    label: str
    pixels: np.ndarray
    score: float
    agreement: float
    detail: float
    observations: int


def score_candidate(
    label: str,
    pixels: np.ndarray,
    reference: np.ndarray,
    observations: tuple[np.ndarray, ...] = (),
) -> RestorationCandidate:
    """A local ranking heuristic, expressed on a 0–100 scale, not a probability.

    Compare blurred, same-grid versions to avoid rewarding sharpening noise.
    The detail term credits extra edge strength only at source-supported edges.
    """
    size = reference.shape[1], reference.shape[0]

    def gray(image: np.ndarray) -> np.ndarray:
        image = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255

    base, candidate = gray(reference), gray(pixels)
    bases = (base, *(gray(observation) for observation in observations))
    lowpass = cv2.GaussianBlur(candidate, (0, 0), 1.15)
    errors = [float(np.abs(lowpass - cv2.GaussianBlur(item, (0, 0), 1.15)).mean()) for item in bases]
    agreement = float(np.clip(1 - 3 * np.median(errors), 0, 1))

    def edges(image: np.ndarray) -> np.ndarray:
        gradients = [cv2.Sobel(image, cv2.CV_32F, *axes, ksize=3) for axes in ((1, 0), (0, 1))]
        return cv2.magnitude(*gradients)

    source_edges = edges(cv2.GaussianBlur(base, (3, 3), 0.6))
    restored_edges = edges(candidate)
    support = source_edges > max(0.08, float(np.quantile(source_edges, 0.7)))
    if support.any():
        source_strength = float(source_edges[support].mean())
        restored_strength = float(restored_edges[support].mean())
        detail = float(np.clip((restored_strength / max(source_strength, 0.01) - 1) / 1.2, 0, 1))
    else:
        detail = 0.0
    score = round(100 * (0.8 * agreement + 0.2 * detail), 2)
    return RestorationCandidate(label, pixels, score, round(agreement * 100, 2), round(detail * 100, 2),
                                len(bases))


def choose_candidates(
    candidates: list[RestorationCandidate],
    *,
    validated_margin: float | None = None,
    validated_agreement: float | None = None,
) -> tuple[RestorationCandidate, ...]:
    """Keep plausible alternatives unless a validated rule can select a winner."""
    ranked = sorted(candidates, key=lambda item: (-item.score, -item.agreement, item.label))
    if len(ranked) < 2 or validated_margin is None or validated_agreement is None:
        return tuple(ranked)
    first, second = ranked[:2]
    if (first.observations >= 2 and first.agreement >= validated_agreement
            and first.score - second.score > validated_margin):
        return (first,)
    return tuple(ranked)
