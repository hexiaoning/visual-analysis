from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
from PySide6.QtCore import QSignalBlocker, Qt, QThread, Signal
from PySide6.QtGui import QImage
from PySide6.QtWidgets import (
    QComboBox, QFileDialog, QHBoxLayout, QHeaderView, QLabel, QPushButton,
    QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)

from ..database import FaceDatabase
from ..engine import FaceEngine
from ..errors import EnrollmentError
from ..photo_identity import MAX_CANDIDATES, PhotoCandidate, photo_similarity, rank_photo_candidates
from ..vision.io import IMAGE_EXTENSIONS, read_image
from .annotation_compare import ImagePane
from .workspace_style import workspace_panel


class PhotoIdentityWorker(QThread):
    completed = Signal(object)
    failed = Signal(str)

    def __init__(self, engine: FaceEngine, photos: tuple[np.ndarray, ...],
                 library_id: str | None, parent: QWidget) -> None:
        super().__init__(parent)
        self.engine = engine
        self.photos = photos
        self.library_id = library_id

    def run(self) -> None:
        try:
            embeddings = []
            for number, photo in enumerate(self.photos, 1):
                try:
                    embeddings.append(self.engine.photo_feature(photo).embedding)
                except EnrollmentError as exc:
                    raise EnrollmentError(f"照片 {number}：{exc}") from exc
            if self.library_id is None:
                result = photo_similarity(embeddings[0], embeddings[1])
            else:
                result = rank_photo_candidates(
                    self.engine.database, self.engine.model_id, self.library_id, embeddings[0],
                )
            self.completed.emit(result)
        except Exception as exc:
            self.failed.emit(str(exc))


def _preview_image(pixels: np.ndarray) -> QImage:
    height, width = pixels.shape[:2]
    return QImage(pixels.data, width, height, pixels.strides[0], QImage.Format_BGR888).copy()


class PhotoIdentityWidget(QWidget):
    def __init__(self, prepare_engine: Callable[[], FaceEngine | None],
                 database: FaceDatabase | None = None) -> None:
        super().__init__()
        self.prepare_engine = prepare_engine
        self.database = database
        self.worker: PhotoIdentityWorker | None = None
        self.photos: list[np.ndarray | None] = [None] if database is not None else [None, None]
        self.candidates: list[PhotoCandidate] = []
        self.library: QComboBox | None = None
        self.table: QTableWidget | None = None
        self.left = ImagePane("请选择原始照片" if database is not None else "请选择第一张照片")
        self.right = ImagePane("点击右侧候选查看库中照片" if database is not None else "请选择第二张照片")
        for pane in (self.left, self.right):
            pane.setMinimumSize(160, 230)
        self.score = QLabel("相似度\n—")
        self.score.setAlignment(Qt.AlignCenter)
        self.score.setProperty("workspaceScore", True)
        self.score.setMinimumWidth(88)

        self.open_buttons = []
        controls = QHBoxLayout()
        for slot, title in enumerate(("选择原始照片…",) if database is not None
                                     else ("选择第一张照片…", "选择第二张照片…")):
            button = QPushButton(title)
            button.clicked.connect(lambda checked=False, index=slot: self.choose_photo(index))
            controls.addWidget(button)
            self.open_buttons.append(button)
        if database is not None:
            self.library = QComboBox()
            self.library.currentIndexChanged.connect(self._library_changed)
            controls.addWidget(QLabel("目标人脸库"))
            controls.addWidget(self.library, 1)
        else:
            controls.addStretch()
        self.start_button = QPushButton("查找候选" if database is not None else "开始比对")
        self.start_button.setProperty("workspacePrimary", True)
        self.start_button.clicked.connect(self.start)
        controls.addWidget(self.start_button)

        photos = QHBoxLayout()
        photos.addWidget(workspace_panel("原始照片" if database is not None else "第一张照片", self.left), 1)
        photos.addWidget(self.score, 0, Qt.AlignVCenter)
        photos.addWidget(workspace_panel("库中匹配照片" if database is not None else "第二张照片", self.right), 1)
        body = QHBoxLayout()
        body.addLayout(photos, 3)
        if database is not None:
            self.table = QTableWidget(0, 3)
            self.table.setHorizontalHeaderLabels(["姓名", "身份证号", "相似度"])
            self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
            self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
            self.table.setSelectionBehavior(QTableWidget.SelectRows)
            self.table.setSelectionMode(QTableWidget.SingleSelection)
            self.table.setEditTriggers(QTableWidget.NoEditTriggers)
            self.table.setAlternatingRowColors(True)
            self.table.itemSelectionChanged.connect(self.show_candidate)
            body.addWidget(workspace_panel("身份候选", self.table), 2)
        self.hint = (
            f"选择一张原照和目标人脸库，按相似度列出最多 {MAX_CANDIDATES} 位待核实候选。点击候选查看库照。"
            if database is not None else
            "分别选择两张清晰的单人照片。相似度范围为 −1 至 1，分值越高表示人脸特征越接近。"
        )
        self.status = QLabel(self.hint)
        self.status.setWordWrap(True)
        self.status.setProperty("workspaceNote", True)
        layout = QVBoxLayout(self)
        layout.addLayout(controls)
        layout.addLayout(body, 1)
        layout.addWidget(self.status)
        self.refresh_libraries()
        self.update_buttons()

    def is_busy(self) -> bool:
        return self.worker is not None

    def refresh_libraries(self) -> None:
        if self.library is None or self.is_busy():
            return
        selected = self.library.currentData()
        with QSignalBlocker(self.library):
            self.library.clear()
            self.library.addItem("请选择人脸库", None)
            for identifier, name in self.database.list_libraries():
                self.library.addItem(name, identifier)
            self.library.setCurrentIndex(max(0, self.library.findData(selected)))
        if selected != self.library.currentData():
            self._library_changed()

    def _library_changed(self) -> None:
        self.clear_results()
        self.update_buttons()

    def clear_results(self) -> None:
        self.score.setText("相似度\n—")
        self.status.setText(self.hint)
        if self.table is not None:
            self.candidates.clear()
            self.table.setRowCount(0)
            self.right.empty_text = "点击右侧候选查看库中照片"
            self.right.set_image(QImage())

    def update_buttons(self) -> None:
        idle = not self.is_busy()
        for button in self.open_buttons:
            button.setEnabled(idle)
        if self.library is not None:
            self.library.setEnabled(idle)
        scope_ready = self.library is None or self.library.currentData() is not None
        self.start_button.setEnabled(idle and scope_ready and all(photo is not None for photo in self.photos))

    def choose_photo(self, slot: int) -> None:
        extensions = " ".join(f"*{extension}" for extension in sorted(IMAGE_EXTENSIONS))
        path, _ = QFileDialog.getOpenFileName(self, "选择人脸照片", "", f"图片 ({extensions})")
        if path:
            self.load_photo(slot, Path(path))

    def load_photo(self, slot: int, path: Path) -> bool:
        if self.is_busy():
            return False
        pixels = read_image(path)
        if pixels is None or pixels.shape[0] * pixels.shape[1] > 40_000_000:
            self.status.setText(f"无法打开照片或超过 4000 万像素：{path.name}")
            return False
        self.photos[slot] = pixels
        pane = (self.left, self.right)[slot]
        pane.set_image(_preview_image(pixels))
        pane.setToolTip(str(path))
        self.clear_results()
        self.update_buttons()
        return True

    def start(self) -> None:
        if not self.start_button.isEnabled():
            return
        self.refresh_libraries()
        if not self.start_button.isEnabled():
            return
        engine = self.prepare_engine()
        if engine is None:
            return
        self.clear_results()
        library_id = self.library.currentData() if self.library is not None else None
        photos = tuple(photo for photo in self.photos if photo is not None)
        self.worker = PhotoIdentityWorker(engine, photos, library_id, self)
        self.worker.completed.connect(self.show_result)
        self.worker.failed.connect(self.status.setText)
        self.worker.finished.connect(self._finished)
        self.status.setText("正在使用 GPU 提取人脸特征…")
        self.update_buttons()
        self.worker.start()

    def show_result(self, result: float | list[PhotoCandidate]) -> None:
        if self.table is None:
            self.score.setText(f"相似度\n{result:.3f}")
            self.status.setText("比对完成 · 分值反映人脸特征的接近程度，请结合照片核实。")
            return
        self.candidates = result
        self.table.setRowCount(len(result))
        for row, candidate in enumerate(result):
            for col, text in enumerate((candidate.name, candidate.id_card, f"{candidate.score:.3f}")):
                item = QTableWidgetItem(text)
                item.setToolTip(text)
                self.table.setItem(row, col, item)
        self.status.setText(f"找到 {len(result)} 位候选 · 点击一行查看对应库照。"
                            if result else "所选库没有与当前模型兼容的可用特征。请录入照片或重建特征。")

    def show_candidate(self) -> None:
        selected = self.table.selectedItems()
        if not selected:
            self.score.setText("相似度\n—")
            self.right.set_image(QImage())
            return
        candidate = self.candidates[selected[0].row()]
        self.score.setText(f"相似度\n{candidate.score:.3f}")
        pixels = read_image(candidate.image_path)
        self.right.empty_text = "库中照片无法读取"
        self.right.setToolTip(candidate.image_path)
        self.right.set_image(_preview_image(pixels) if pixels is not None else QImage())

    def _finished(self) -> None:
        worker, self.worker = self.worker, None
        worker.deleteLater()
        self.update_buttons()
