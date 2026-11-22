from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QPushButton

from face_matching.config import AppConfig
from face_matching.database import FaceSampleInput
from face_matching.gui.main_window import MainWindow
from face_matching.restoration_io import save_new_png


def _photo(path: Path, value: int) -> Path:
    save_new_png(path, np.full((60, 80, 3), value, dtype=np.uint8))
    return path


def _wait(widget, app):
    deadline = time.monotonic() + 5
    while widget.is_busy() and time.monotonic() < deadline:
        app.processEvents()
        QTest.qWait(5)
    assert not widget.is_busy(), widget.status.text()


def test_face_search_page_has_separate_photo_flows_and_exact_library_photo(qt_runtime, tmp_path, monkeypatch):
    monkeypatch.setenv("FACE_MATCHING_HOME", str(tmp_path / "home"))
    window = MainWindow(AppConfig())
    query = _photo(tmp_path / "原始照片.png", 20)
    comparison = _photo(tmp_path / "另一张照片.png", 50)
    strongest = _photo(tmp_path / "候选匹配照片.png", 90)
    weaker = _photo(tmp_path / "候选其他照片.png", 120)
    library = window.database.create_library("目标库")
    other = window.database.create_library("其他库")

    def sample(path, embedding, quality=.8, model_id="current"):
        return FaceSampleInput(str(path), np.array(embedding, np.float32), model_id, quality)

    person_id = window.database.add_person("甲", "ID-1", [
        sample(weaker, [.6, .8], .95), sample(strongest, [.8, .6]),
        sample(weaker, [1, 0], model_id="old"),
    ], library_ids=[library])
    window.database.add_person("乙", "ID-2", [sample(weaker, [.7, .714])], library_ids=[library])
    window.database.add_person("库外人员", "ID-3", [sample(weaker, [1, 0])], library_ids=[other])

    def feature(pixels):
        return SimpleNamespace(embedding=np.array([2, 0] if pixels[0, 0, 0] == 20 else [3, 4], np.float32))

    window.engine = SimpleNamespace(database=window.database, model_id="current", photo_feature=feature)
    try:
        assert window.tabs.count() == 9
        buttons = {button.text(): button for button in window.tabs.widget(0).findChildren(QPushButton)}
        buttons["一对一比对"].click()
        compare = window.photo_compare
        assert window.face_pages.currentWidget() is compare
        assert compare.load_photo(0, query)
        assert not compare.start_button.isEnabled()
        assert compare.load_photo(1, comparison)
        compare.start_button.click()
        _wait(compare, qt_runtime)
        assert compare.score.text().endswith("0.600")

        buttons["身份核查"].click()
        lookup = window.identity_check
        assert window.face_pages.currentWidget() is lookup
        assert lookup.load_photo(0, query)
        assert not lookup.start_button.isEnabled()
        lookup.library.setCurrentIndex(lookup.library.findData(library))
        lookup.start_button.click()
        _wait(lookup, qt_runtime)
        assert [candidate.name for candidate in lookup.candidates] == ["甲", "乙"]
        assert lookup.candidates[0].person_id == person_id
        assert lookup.right.image.isNull()
        lookup.table.selectRow(0)
        assert lookup.score.text().endswith("0.800")
        assert lookup.left.image.pixelColor(0, 0).red() == 20
        assert lookup.right.image.pixelColor(0, 0).red() == 90
        lookup.table.selectRow(1)
        assert lookup.right.image.pixelColor(0, 0).red() == 120

        lookup.library.setCurrentIndex(lookup.library.findData(other))
        assert not lookup.candidates and lookup.right.image.isNull()
        lookup.start_button.click()
        _wait(lookup, qt_runtime)
        assert [candidate.name for candidate in lookup.candidates] == ["库外人员"]
        buttons["来源检索"].click()
        assert not window.face_pages.currentWidget() in (compare, lookup)
    finally:
        for widget in (window.photo_compare, window.identity_check):
            if widget.worker:
                widget.worker.wait()
                qt_runtime.processEvents()
        window.close()
        window.deleteLater()
