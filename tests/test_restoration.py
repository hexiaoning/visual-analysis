from PySide6.QtWidgets import QApplication
from types import SimpleNamespace

import numpy as np

from face_matching.face_restoration import FFHQ_LANDMARKS, aligned_face, paste_face
from face_matching.gui.restoration import RestorationWidget
from face_matching.restoration_candidates import RestorationCandidate, choose_candidates
from face_matching.vision.detector import Detection


def candidate(label, score, agreement=97, observations=4):
    return RestorationCandidate(label, np.zeros((8, 8, 3), dtype=np.uint8), score, agreement, 40, observations)


def test_ranking_keeps_close_and_unvalidated_alternatives():
    a, b = candidate("A", 95), candidate("B", 94)
    assert choose_candidates([b, a]) == (a, b)
    rule = {"validated_margin": 3, "validated_agreement": 90}
    assert choose_candidates([b, a], **rule) == (a, b)
    assert choose_candidates([candidate("C", 88), a], **rule) == (a,)
    weak = candidate("D", 99, observations=1)
    assert choose_candidates([a, weak], **rule) == (weak, a)


def test_aligned_paste_preserves_pixels_outside_the_face():
    image = np.full((512, 512, 3), 40, np.uint8)
    detection = Detection(np.array([140, 180, 360, 440]), 1.0, FFHQ_LANDMARKS.copy())
    face = aligned_face(image, detection, 512)
    assert face is not None
    paste_face(image, face, np.full_like(image, 100))
    assert tuple(image[255, 255]) == (100, 100, 100)
    assert tuple(image[0, 0]) == (40, 40, 40)


def test_widget_lists_each_scored_candidate():
    application = QApplication.instance() or QApplication([])
    widget = RestorationWidget(None, SimpleNamespace())
    widget._show_candidates((candidate("A", 95), candidate("B", 94)))
    assert widget.candidate_row.count() == 2
    widget.close()
