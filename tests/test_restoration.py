from PySide6.QtWidgets import QApplication
from types import SimpleNamespace

import numpy as np

from face_matching.face_restoration import FFHQ_LANDMARKS, _photo_faces, aligned_face, paste_face
from face_matching.gui.restoration import RestorationWidget
from face_matching.restoration_candidates import RestorationCandidate, choose_candidates
from face_matching.restoration_cases import run_portrait_cases
from face_matching.restoration_io import save_new_png
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


def test_local_case_report_preserves_both_candidates_and_missing_results(tmp_path, monkeypatch):
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    first, second = inputs / "01.png", inputs / "02.png"
    pixels = np.zeros((8, 8, 3), dtype=np.uint8)
    save_new_png(first, pixels)
    save_new_png(second, pixels)
    options = (candidate("A", 95), candidate("B", 94))

    def restore(items, config, stop):
        assert len(items) == 2
        return (options, ())

    monkeypatch.setattr("face_matching.restoration_cases.restore_portraits", restore)
    report = run_portrait_cases(inputs, tmp_path / "outputs", SimpleNamespace())
    assert not report["ok"]
    case, missing = report["cases"]
    assert case["source"] == first.name
    assert len(case["source_sha256"]) == 64
    assert [(item["model"], item["score"], item["confidence"]) for item in case["candidates"]] == [
        ("A", 95, "待复核"), ("B", 94, "待复核"),
    ]
    assert missing["source"] == second.name
    assert missing["candidates"] == []


def test_photo_faces_retries_with_texture_suppression_at_same_threshold():
    yy, xx = np.indices((512, 512))
    grid = np.uint8((xx + yy) % 2) * 255
    original = np.repeat(grid[..., None], 3, axis=2)
    face = Detection(np.array([140, 180, 360, 440]), 0.8, FFHQ_LANDMARKS.copy())
    inputs = []

    def detect(pixels):
        inputs.append(pixels)
        return [] if pixels is original else [face]

    result = _photo_faces(original, SimpleNamespace(detect=detect), (140, 180, 360, 440))
    assert len(inputs) == 2
    assert inputs[1].std() < original.std()
    assert len(result) == 1
    assert result[0].crop.std() < original.std()
