from __future__ import annotations

from functools import partial
from pathlib import Path
from threading import Event

import numpy as np
from PySide6.QtCore import QThread, Signal
from PySide6.QtWidgets import (
    QComboBox, QFileDialog, QHBoxLayout, QLabel, QPushButton, QScrollArea, QVBoxLayout, QWidget,
)

from ..config import AppConfig
from ..face_restoration import restore_portrait
from ..face_restoration_video import restore_face_video
from ..plate_restoration import restore_plate_photo
from ..restoration_candidates import RestorationCandidate
from ..restoration_io import save_new_png
from ..vision.io import IMAGE_EXTENSIONS, read_image
from ..vision.local_media import VIDEO_EXTENSIONS, local_path
from .image_pane import ImagePane


class RestorationWorker(QThread):
    result = Signal(object)
    failed = Signal(str)

    def __init__(self, operation, parent):
        super().__init__(parent)
        self.operation = operation

    def run(self) -> None:
        try:
            self.result.emit(self.operation())
        except Exception as error:
            self.failed.emit(str(error))


class RestorationWidget(QWidget):
    def __init__(self, parent: QWidget, config: AppConfig) -> None:
        super().__init__(parent)
        self.config = config
        self.worker: RestorationWorker | None = None
        self.cancel = Event()
        self.media: np.ndarray | None = None
        self.source: Path | None = None
        self.selection: tuple[int, int, int, int] | None = None
        self.plate_dialog = None

        self.open_button = QPushButton("打开图片…")
        self.open_face_video = QPushButton("还原视频人像…")
        self.open_plate_video = QPushButton("还原视频车牌…")
        self.mode = QComboBox()
        self.mode.addItems(["人像还原", "车牌还原"])
        self.run_button = QPushButton("还原图片")
        self.clear_button = QPushButton("清除框选")
        self.stop_button = QPushButton("停止")
        self.status = QLabel("打开本机图片或视频。")
        self.status.setWordWrap(True)

        top = QHBoxLayout()
        for control in (
            self.open_button, self.open_face_video, self.open_plate_video, self.mode, self.run_button,
            self.clear_button, self.stop_button,
        ):
            top.addWidget(control)
        top.addStretch()
        self.original = ImagePane(selectable=True)
        self.candidate_row = QHBoxLayout()
        self.candidate_row.setContentsMargins(0, 0, 0, 0)
        results = QWidget()
        results.setLayout(self.candidate_row)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(results)
        panes = QHBoxLayout()
        source_box = QVBoxLayout()
        source_box.addWidget(QLabel("原图"))
        source_box.addWidget(self.original, 1)
        panes.addLayout(source_box, 2)
        panes.addWidget(scroll, 3)
        layout = QVBoxLayout(self)
        layout.addLayout(top)
        layout.addLayout(panes, 1)
        layout.addWidget(self.status)

        self.open_button.clicked.connect(self._open_image)
        self.open_face_video.clicked.connect(self._open_face_video)
        self.open_plate_video.clicked.connect(self._open_plate_video)
        self.original.region_selected.connect(self.select_region)
        self.clear_button.clicked.connect(lambda: self.select_region(None))
        self.mode.currentIndexChanged.connect(self._mode_changed)
        self.run_button.clicked.connect(self._restore_image)
        self.stop_button.clicked.connect(self.stop)
        self._controls()

    def is_busy(self) -> bool:
        return self.worker is not None or self.plate_dialog is not None

    def stop(self) -> None:
        self.cancel.set()
        if self.plate_dialog is not None:
            self.plate_dialog.close()
        if self.worker is not None:
            self.status.setText("正在停止还原。")

    def _controls(self) -> None:
        busy = self.is_busy()
        self.open_button.setEnabled(not busy)
        self.open_face_video.setEnabled(not busy)
        self.open_plate_video.setEnabled(not busy)
        self.mode.setEnabled(not busy)
        self.original.setEnabled(not busy)
        self.clear_button.setEnabled(not busy and self.selection is not None)
        self.stop_button.setEnabled(self.worker is not None)
        self.run_button.setEnabled(not busy and self.media is not None and
                                   (self.mode.currentIndex() == 0 or self.selection is not None))

    def _clear_results(self) -> None:
        while item := self.candidate_row.takeAt(0):
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

    def _mode_changed(self) -> None:
        self._clear_results()
        if self.media is not None:
            prompt = "可框选人脸，也可还原整图里的可对齐人脸。" if self.mode.currentIndex() == 0 else "请拖动框选车牌。"
            self.status.setText(prompt)
        self._controls()

    def _open_image(self) -> None:
        patterns = " ".join(f"*{suffix}" for suffix in sorted(IMAGE_EXTENSIONS))
        filename, _ = QFileDialog.getOpenFileName(self, "打开本机图片", "", f"图片 ({patterns})")
        if filename:
            self.load_image(Path(filename))

    def load_image(self, path: Path) -> bool:
        if self.is_busy():
            return False
        try:
            path = local_path(path)
            image = read_image(path)
        except (ValueError, OSError) as error:
            self.status.setText(str(error))
            return False
        if image is None:
            self.status.setText("无法读取该图片。")
            return False
        self.source, self.media, self.selection = path, image, None
        self.original.set_region(None)
        self.original.set_image(image)
        self._mode_changed()
        return True

    def select_region(self, region) -> None:
        if not self.is_busy():
            self.selection = None if region is None else tuple(round(float(value)) for value in region)
            self.original.set_region(self.selection)
            self.original.fit_image()
            self._clear_results()
            self._controls()

    def _start(self, operation, callback) -> None:
        self._clear_results()
        self.status.setText("正在离线还原。")
        self.worker = RestorationWorker(operation, self)
        self.worker.result.connect(callback)
        self.worker.failed.connect(self.status.setText)
        self.worker.finished.connect(self._finished)
        self._controls()
        self.worker.start()

    def _finished(self) -> None:
        worker, self.worker = self.worker, None
        if worker is not None:
            worker.deleteLater()
        if self.cancel.is_set():
            self.status.setText("已停止还原。")
        self._controls()

    def _restore_image(self) -> None:
        if self.media is None or self.is_busy():
            return
        self.cancel.clear()
        if self.mode.currentIndex() == 0:
            operation = partial(restore_portrait, self.media, self.config, self.cancel, self.selection)
        elif self.selection is not None:
            operation = partial(restore_plate_photo, self.media, self.selection)
        else:
            return
        self._start(operation, self._show_candidates)

    def _show_candidates(self, candidates: tuple[RestorationCandidate, ...]) -> None:
        if self.cancel.is_set():
            return
        for item in candidates:
            card = QWidget()
            column = QVBoxLayout(card)
            label = QLabel(f"{item.label} · {item.score:.2f} 分 · 置信度 {item.confidence}\n"
                           f"来源一致性 {item.agreement:.2f} · 边缘细节 {item.detail:.2f}")
            label.setWordWrap(True)
            pane = ImagePane()
            pane.set_image(item.pixels)
            save = QPushButton("保存 PNG…")
            save.clicked.connect(partial(self._save_candidate, item))
            column.addWidget(label)
            column.addWidget(pane, 1)
            column.addWidget(save)
            self.candidate_row.addWidget(card, 1)
        self.status.setText(f"{len(candidates)} 个结果。分数用于同源候选排序，置信度依据原图与候选的一致性。")

    def _save_candidate(self, candidate: RestorationCandidate) -> None:
        stem = self.source.stem if self.source else "图片"
        name, _ = QFileDialog.getSaveFileName(self, "保存还原结果", f"{stem}-还原.png", "PNG (*.png)")
        if name:
            try:
                save_new_png(Path(name).with_suffix(".png"), candidate.pixels)
                self.status.setText(f"已保存 {name}")
            except (OSError, ValueError) as error:
                self.status.setText(str(error))

    def _open_face_video(self) -> None:
        if self.is_busy():
            return
        patterns = " ".join(f"*{suffix}" for suffix in sorted(VIDEO_EXTENSIONS))
        source, _ = QFileDialog.getOpenFileName(self, "打开本机视频", "", f"视频 ({patterns})")
        if not source:
            return
        output, _ = QFileDialog.getSaveFileName(self, "保存人像还原视频",
                                                f"{Path(source).stem}-人像还原.mkv", "MKV (*.mkv)")
        if not output:
            return
        self.cancel.clear()
        target = Path(output).with_suffix(".mkv")
        operation = partial(restore_face_video, Path(source), target, self.config, self.cancel)
        self._start(operation, lambda report: self._video_saved(target, report))

    def _video_saved(self, path: Path, report: tuple[str, int, int]) -> None:
        if not self.cancel.is_set():
            model, frames, faces = report
            self.status.setText(f"{model} 已处理 {frames} 帧、{faces} 个人脸区域。已保存 {path}。")

    def _open_plate_video(self) -> None:
        if self.is_busy():
            return
        from .plate_video import PlateVideoDialog

        self.plate_dialog = PlateVideoDialog(self)
        self.plate_dialog.finished.connect(self._plate_closed)
        self._controls()
        self.plate_dialog.open()

    def _plate_closed(self) -> None:
        dialog, self.plate_dialog = self.plate_dialog, None
        if dialog is not None:
            dialog.deleteLater()
        self._controls()
