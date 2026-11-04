from dataclasses import dataclass

import numpy as np

from .database import FaceDatabase
from .vision.recognizer import l2_normalize


MAX_CANDIDATES = 50


@dataclass(frozen=True, slots=True)
class PhotoCandidate:
    person_id: str
    name: str
    id_card: str
    image_path: str
    score: float


def photo_similarity(first: np.ndarray, second: np.ndarray) -> float:
    return float(np.clip(l2_normalize(first) @ l2_normalize(second), -1.0, 1.0))


def rank_photo_candidates(database: FaceDatabase, model_id: str, library_id: str,
                          embedding: np.ndarray) -> list[PhotoCandidate]:
    query = l2_normalize(embedding)
    best_by_person: dict[str, PhotoCandidate] = {}
    for sample in database.list_gallery(model_id, library_id=library_id):
        if sample.embedding.size != query.size:
            continue
        try:
            portrait = l2_normalize(sample.embedding)
        except ValueError:
            continue
        score = float(np.clip(query @ portrait, -1.0, 1.0))
        previous = best_by_person.get(sample.person_id)
        if previous is None or score > previous.score:
            best_by_person[sample.person_id] = PhotoCandidate(
                sample.person_id, sample.name, sample.id_card, sample.image_path, score,
            )
    return sorted(best_by_person.values(), key=lambda candidate: candidate.score, reverse=True)[:MAX_CANDIDATES]
