import hashlib
import json
import time
from threading import Event
from zipfile import ZipFile

import cv2
import numpy as np
import pytest
from PySide6.QtCore import QPointF, Qt
from PySide6.QtTest import QTest

from face_matching.config import AppConfig
from face_matching.gui.main_window import MainWindow
from face_matching.image_enhancement import PRESETS, enhance_image
from face_matching.plate_video import enhance_plate_video, export_plate_result, load_plate_preview


def write_video(path, frames):
    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"FFV1"), 15, (width, height))
    assert writer.isOpened()
    try:
        for frame in frames:
            writer.write(frame)
    finally:
        writer.release()


@pytest.fixture
def moving_plate(tmp_path):
    rng = np.random.default_rng(123)
    plate = np.full((32, 96, 3), (120, 85, 38), np.uint8)
    cv2.rectangle(plate, (1, 1), (94, 30), (206, 221, 216), 1)
    cv2.putText(plate, "TEST 53", (5, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                (215, 230, 245), 2, cv2.LINE_AA)
    frames = [np.full((144, 256, 3), 25, np.uint8)]
    regions = []
    for step in range(5):
        x, y = 43 + 7 * step, 49 + 3 * step
        dx = (0, 0.3, -0.4, 0.2, -0.25)[step]
        moved = cv2.warpAffine(plate, np.float32([[1, 0, dx], [0, 1, -dx]]), (96, 32),
                               borderMode=cv2.BORDER_REPLICATE)
        noisy = np.clip(moved + rng.normal(0, 7, moved.shape), 0, 255).astype(np.uint8)
        frame = np.full_like(frames[0], 25)
        frame[y:y + 32, x:x + 96] = noisy
        frames.append(frame)
        regions.append((x, y, x + 96, y + 32))
    frames.extend([np.full_like(frames[0], 50), frames[-1].copy()])
    source = tmp_path / "车牌测试.avi"
    write_video(source, frames)
    return source, frames, regions


def test_tracks_moving_plate_and_exports_verified_full_original_frames(moving_plate, tmp_path):
    source, frames, regions = moving_plate
    payload = source.read_bytes()
    anchor, count = load_plate_preview(source, 1)
    np.testing.assert_array_equal(anchor, frames[1])
    assert count == len(frames)
    result = enhance_plate_video(source, 1, regions[0], anchor, 10, 0, 0, Event())
    assert result.tracked_count == 5
    assert "第 7 帧匹配不足" in result.tracking_notice
    assert len(result.frames) >= 3
    assert [frame.region for frame in result.frames] == [regions[frame.index - 1] for frame in result.frames]
    single = enhance_image(result.reference.pixels, PRESETS["车牌清晰化（保守）"])
    assert result.enhanced.shape == single.shape
    assert not np.array_equal(result.enhanced, single)

    destination = tmp_path / "结果.zip"
    export_plate_result(destination, result)
    with ZipFile(destination) as archive:
        manifest = json.loads(archive.read("manifest.json"))
        assert manifest["anchor_frame_number"] == 2
        assert manifest["reference_frame_number"] == result.reference.index + 1
        assert manifest["single_frame_fallback"] is False
        assert [item["frame_number"] for item in manifest["original_frames"]] == [
            frame.index + 1 for frame in result.frames
        ]
        original_files = {item["file"]: item["frame_number"] for item in manifest["original_frames"]}
        original_files[manifest["anchor_file"]] = 2
        assert set(archive.namelist()) == {*original_files, "enhanced.png", "manifest.json"}
        for filename, number in original_files.items():
            pixels = cv2.imdecode(np.frombuffer(archive.read(filename), np.uint8), cv2.IMREAD_COLOR)
            np.testing.assert_array_equal(pixels, frames[number - 1])
        for filename, digest in manifest["sha256"].items():
            assert hashlib.sha256(archive.read(filename)).hexdigest() == digest
    with pytest.raises(FileExistsError):
        export_plate_result(source, result)
    assert source.read_bytes() == payload


def test_refuses_ambiguous_match_and_falls_back_to_anchor(tmp_path):
    rng = np.random.default_rng(5)
    anchor = np.full((140, 200, 3), 40, np.uint8)
    target = rng.integers(20, 230, (24, 44, 3), np.uint8)
    anchor[50:74, 50:94] = target
    ambiguous = anchor.copy()
    ambiguous[93:117, 70:114] = target
    source = tmp_path / "ambiguous.avi"
    write_video(source, [anchor, ambiguous, anchor])
    result = enhance_plate_video(source, 0, (50, 50, 94, 74), anchor, 3, 0, 0, Event())
    assert result.tracked_count == 1
    assert len(result.frames) == 1
    assert "多个相似位置" in result.tracking_notice
    np.testing.assert_array_equal(result.reference.pixels, target)


def test_changed_original_blocks_export_and_cleans_partial_archive(moving_plate, tmp_path):
    source, frames, regions = moving_plate
    anchor = frames[1]
    result = enhance_plate_video(source, 1, regions[0], anchor, 10, 0, 0, Event())
    changed_index = result.frames[-1].index
    assert changed_index > 1
    frames[changed_index][0, 0] = 230
    write_video(source, frames)
    destination = tmp_path / "changed.zip"
    with pytest.raises(ValueError, match="帧已变化"):
        export_plate_result(destination, result)
    assert not destination.exists()


def wait_for_task(app, dialog):
    deadline = time.monotonic() + 5
    while dialog.worker is not None and time.monotonic() < deadline:
        app.processEvents()
        QTest.qWait(5)
    assert dialog.worker is None


def test_main_window_can_choose_frame_draw_box_process_and_export(qt_runtime, moving_plate, tmp_path, monkeypatch):
    source, _, regions = moving_plate
    monkeypatch.setenv("FACE_MATCHING_HOME", str(tmp_path))
    destination = tmp_path / "ui.zip"
    monkeypatch.setattr("face_matching.gui.plate_video.QFileDialog.getOpenFileName",
                        lambda *args: (str(source), ""))
    monkeypatch.setattr("face_matching.gui.plate_video.QFileDialog.getSaveFileName",
                        lambda *args: (str(destination), ""))
    window = MainWindow(AppConfig())
    window.show()
    tools = window.image_tools
    window.tabs.setCurrentWidget(tools)
    QTest.mouseClick(tools.open_plate_video, Qt.LeftButton)
    dialog = tools.plate_video_dialog
    try:
        assert dialog is not None and dialog.isVisible()
        QTest.mouseClick(dialog.open_button, Qt.LeftButton)
        wait_for_task(qt_runtime, dialog)
        assert dialog.frame_number.value() == 1
        dialog.slider.setValue(2)
        wait_for_task(qt_runtime, dialog)
        assert dialog.frame_number.value() == 2
        assert not dialog.run_button.isEnabled()
        x1, y1, x2, y2 = regions[0]
        view = dialog.original
        start = view.mapFromScene(QPointF(x1, y1))
        end = view.mapFromScene(QPointF(x2, y2))
        QTest.mousePress(view.viewport(), Qt.LeftButton, pos=start)
        QTest.mouseMove(view.viewport(), end)
        QTest.mouseRelease(view.viewport(), Qt.LeftButton, pos=end)
        assert dialog.run_button.isEnabled()
        QTest.mouseClick(dialog.run_button, Qt.LeftButton)
        assert not dialog.export_button.isEnabled()
        QTest.keyClick(dialog, Qt.Key_Escape)
        assert dialog.isVisible()
        wait_for_task(qt_runtime, dialog)
        assert "融合" in dialog.status.text()
        assert not dialog.enhanced.image.isNull()
        QTest.mouseClick(dialog.export_button, Qt.LeftButton)
        wait_for_task(qt_runtime, dialog)
        assert destination.is_file()
        assert "已导出" in dialog.status.text()
        dialog.frame_number.setValue(7)
        wait_for_task(qt_runtime, dialog)
        assert not dialog.export_button.isEnabled()
        assert dialog.enhanced.image.isNull()
        assert view.region is None
    finally:
        if dialog is not None:
            wait_for_task(qt_runtime, dialog)
            dialog.close()
        window.close()
    assert tools.plate_video_dialog is None
