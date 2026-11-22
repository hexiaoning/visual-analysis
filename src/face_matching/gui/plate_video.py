from __future__ import annotations

from collections.abc import Callable
from functools import partial
from pathlib import Path
from threading import Event

import numpy as np
from PySide6.QtCore import QSignalBlocker, Qt, QThread, Signal
from PySide6.QtWidgets import (
    QCheckBox, QDialog, QFileDialog, QHBoxLayout, QLabel, QPushButton, QSlider,
    QSpinBox, QVBoxLayout, QWidget,
)

from ..plate_video import PlateResult, enhance_plate_video, export_plate_result, load_plate_preview
from ..vision.local_media import VIDEO_EXTENSIONS
from .image_pane import ImagePane


class PlateVideoWorker(QThread):
    completed = Signal(object)
    failed = Signal(str)

    def __init__(self, operation: Callable[[], object], parent: QWidget) -> None:
        super().__init__(parent)
        self.operation = operation

    def run(self) -> None:
        try:
            self.completed.emit(self.operation())
        except FileExistsError:
            self.failed.emit("文件已存在，请换一个新文件名")
        except Exception as exc:
            self.failed.emit(str(exc))


class PlateVideoDialog(QDialog):
    def __init__(self, parent: QWidget) -> None:
        super().__init__(parent)
        self.setWindowTitle("视频车牌还原")
        self.resize(1180, 680)
        self.source: Path | None = None
        self.frame: np.ndarray | None = None
        self.frame_index = 0
        self.frame_count: int | None = None
        self.result: PlateResult | None = None
        self.worker: PlateVideoWorker | None = None
        self.stop: Event | None = None

        self.open_button = QPushButton("打开视频…")
        self.open_button.clicked.connect(self._open)
        self.source_label = QLabel("请选择本机视频")
        self.source_label.setWordWrap(True)
        self.frame_number = QSpinBox()
        self.frame_number.setRange(1, 1)
        self.frame_number.setPrefix("第 ")
        self.frame_number.setSuffix(" 帧")
        self.frame_number.setKeyboardTracking(False)
        self.frame_number.valueChanged.connect(self._seek)
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(1, 1)
        self.slider.setTracking(False)
        self.slider.valueChanged.connect(self._seek)
        navigation = QHBoxLayout()
        navigation.addWidget(self.open_button)
        navigation.addWidget(self.frame_number)
        navigation.addWidget(self.slider, 1)

        self.original = ImagePane(selectable=True)
        self.original.region_selected.connect(self._select_region)
        self.reference = ImagePane()
        self.enhanced = ImagePane()
        self.result_label = QLabel("还原结果")
        self.alternatives = QHBoxLayout()
        self.reference_label = QLabel("参考原帧 · 车牌区域")
        panes = QHBoxLayout()
        for label, view, stretch in (
            (QLabel("起始原帧 · 拖动框选车牌"), self.original, 2),
            (self.reference_label, self.reference, 1),
            (self.result_label, self.enhanced, 1),
        ):
            column = QVBoxLayout()
            column.addWidget(label)
            column.addWidget(view, 1)
            panes.addLayout(column, stretch)
        panes.addLayout(self.alternatives, 1)

        self.zoom_button = QPushButton("起始帧 100%")
        self.zoom_button.clicked.connect(self.original.actual_size)
        self.clear_button = QPushButton("清除框选 / 查看全帧")
        self.clear_button.clicked.connect(self._clear_region)
        self.scan_count = QSpinBox()
        self.scan_count.setRange(2, 60)
        self.scan_count.setValue(30)
        self.scan_count.setToolTip("检查的最多帧数，含起始帧。最多融合 8 张有效帧。")
        self.scan_count.valueChanged.connect(self._invalidate)
        self.deblur = QCheckBox("手动去拖影")
        self.deblur.toggled.connect(self._invalidate)
        self.blur_length = QSpinBox()
        self.blur_length.setRange(3, 21)
        self.blur_length.setValue(5)
        self.blur_length.setSuffix(" 像素")
        self.blur_length.setToolTip("拖影长度，以原帧像素计")
        self.blur_length.valueChanged.connect(self._invalidate)
        self.blur_angle = QSpinBox()
        self.blur_angle.setRange(0, 179)
        self.blur_angle.setSuffix("°")
        self.blur_angle.setToolTip("拖影角度，0° 为水平，90° 为竖直")
        self.blur_angle.valueChanged.connect(self._invalidate)
        settings = QHBoxLayout()
        for widget in (self.zoom_button, self.clear_button, QLabel("检查帧数上限"), self.scan_count,
                       self.deblur, self.blur_length, self.blur_angle):
            settings.addWidget(widget)
        settings.addStretch()

        self.run_button = QPushButton("跟踪并增强")
        self.run_button.clicked.connect(self._enhance)
        self.stop_button = QPushButton("停止处理")
        self.stop_button.clicked.connect(self._cancel)
        self.export_button = QPushButton("导出结果与原帧 ZIP…")
        self.export_button.clicked.connect(self._export)
        self.close_button = QPushButton("关闭")
        self.close_button.clicked.connect(self.reject)
        actions = QHBoxLayout()
        actions.addWidget(self.run_button)
        actions.addWidget(self.stop_button)
        actions.addWidget(self.export_button)
        actions.addStretch()
        actions.addWidget(self.close_button)

        self.status = QLabel("选择车牌清楚一些的起始帧，再沿车牌边缘框选。")
        self.status.setWordWrap(True)
        note = QLabel("增强结果用于辅助观察。请对照原帧复核车牌字符。")
        layout = QVBoxLayout(self)
        layout.addLayout(navigation)
        layout.addWidget(self.source_label)
        layout.addLayout(panes, 1)
        layout.addLayout(settings)
        layout.addLayout(actions)
        layout.addWidget(self.status)
        layout.addWidget(note)
        for button in self.findChildren(QPushButton):
            button.setAutoDefault(False)
        self._update_controls()

    def _update_controls(self) -> None:
        idle = self.worker is None
        ready = idle and self.frame is not None
        self.open_button.setEnabled(idle)
        for widget in (self.frame_number, self.original, self.zoom_button, self.clear_button,
                       self.scan_count, self.deblur):
            widget.setEnabled(ready)
        self.slider.setEnabled(ready and self.frame_count is not None and self.frame_count > 1)
        self.blur_length.setEnabled(ready and self.deblur.isChecked())
        self.blur_angle.setEnabled(ready and self.deblur.isChecked())
        self.run_button.setEnabled(ready and self.original.region is not None)
        self.stop_button.setEnabled(not idle and self.stop is not None and not self.stop.is_set())
        self.export_button.setEnabled(idle and self.result is not None)
        self.close_button.setEnabled(idle)

    def _start(self, operation: Callable[[], object], completed: Callable, message: str) -> None:
        self.worker = PlateVideoWorker(operation, self)
        self.worker.completed.connect(completed)
        self.worker.failed.connect(self._failed)
        self.worker.finished.connect(self._finished)
        self.status.setText(message)
        self._update_controls()
        self.worker.start()

    def _failed(self, message: str) -> None:
        self.status.setText(f"操作失败 · {message}")
        self._sync_navigation()

    def _finished(self) -> None:
        worker, self.worker = self.worker, None
        self.stop = None
        self._update_controls()
        worker.deleteLater()

    def _open(self) -> None:
        suffixes = " ".join(f"*{suffix}" for suffix in sorted(VIDEO_EXTENSIONS))
        path, _ = QFileDialog.getOpenFileName(self, "打开本机视频", "", f"视频 ({suffixes})")
        if path:
            self._preview(Path(path), 0)

    def _seek(self, number: int) -> None:
        if self.source is not None and number != self.frame_index + 1:
            self._preview(self.source, number - 1)

    def _preview(self, source: Path, index: int) -> None:
        if self.worker is not None:
            return

        def preview() -> tuple[Path, int, np.ndarray, int | None]:
            return source, index, *load_plate_preview(source, index)

        self._start(preview, self._show_frame, f"正在读取第 {index + 1} 帧…")

    def _sync_navigation(self) -> None:
        maximum = min(self.frame_count or 2_147_483_647, 2_147_483_647)
        with QSignalBlocker(self.frame_number), QSignalBlocker(self.slider):
            self.frame_number.setMaximum(maximum if self.frame is not None else 1)
            self.slider.setMaximum(maximum if self.frame_count is not None else 1)
            self.frame_number.setValue(self.frame_index + 1)
            self.slider.setValue(self.frame_index + 1)

    def _show_frame(self, preview: tuple[Path, int, np.ndarray, int | None]) -> None:
        self.source, self.frame_index, self.frame, self.frame_count = preview
        self.source_label.setText(f"{self.source.name} · 共 {self.frame_count} 帧" if self.frame_count
                                  else self.source.name)
        self._sync_navigation()
        self.original.set_region(None)
        self.original.set_image(self.frame)
        self._invalidate()

    def _select_region(self, region: tuple[int, int, int, int]) -> None:
        self.original.set_region(region)
        self._invalidate()

    def _clear_region(self) -> None:
        self.original.set_region(None)
        self.original.fit_image()
        self._invalidate()

    def _invalidate(self) -> None:
        self.result = None
        self.reference.clear_image()
        self.enhanced.clear_image()
        self.result_label.setText("还原结果")
        while item := self.alternatives.takeAt(0):
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        self.reference_label.setText("参考原帧 · 车牌区域")
        self._update_controls()
        if self.original.region is None:
            self.status.setText("拖动框选车牌，宽度 16～640 像素，高度 8～320 像素。")
        else:
            self.status.setText("点击“跟踪并增强”，从当前帧开始向后查找同一车牌。")

    def _enhance(self) -> None:
        region = self.original.region
        if self.worker is not None or self.source is None or self.frame is None or region is None:
            return
        self._invalidate()
        self.stop = Event()
        length = self.blur_length.value() if self.deblur.isChecked() else 0
        angle = self.blur_angle.value() if length else 0
        operation = partial(enhance_plate_video, self.source, self.frame_index, region, self.frame,
                            self.scan_count.value(), length, angle, self.stop)
        self._start(operation, self._show_result, "正在跟踪车牌、对齐并融合有效帧…")

    def _cancel(self) -> None:
        if self.stop is not None:
            self.stop.set()
            self._update_controls()
            self.status.setText("正在停止处理…")

    def _show_result(self, result: PlateResult | None) -> None:
        if result is None or (self.stop is not None and self.stop.is_set()):
            self.status.setText("已停止处理。")
            return
        self.result = result
        self.reference.set_image(result.reference.pixels)
        self.reference_label.setText(f"参考原帧 · 第 {result.reference.index + 1} 帧的车牌")
        self.enhanced.set_image(result.enhanced)
        first, *alternatives = result.candidates
        self.result_label.setText(f"{first.label} · {first.score:.2f}")
        for candidate in alternatives:
            card = QWidget()
            column = QVBoxLayout(card)
            column.addWidget(QLabel(f"{candidate.label} · {candidate.score:.2f}"))
            preview = ImagePane()
            preview.setMinimumWidth(180)
            preview.set_image(candidate.pixels)
            column.addWidget(preview)
            self.alternatives.addWidget(card)
        if len(result.frames) == 1:
            message = f"可用帧不足，已回退为单帧还原（第 {result.reference.index + 1} 帧）。"
        else:
            numbers = "、".join(str(frame.index + 1) for frame in result.frames)
            message = f"已跟踪 {result.tracked_count} 帧（含起始帧），融合第 {numbers} 帧。"
        self.status.setText(f"{message} {result.tracking_notice}".rstrip())

    def _export(self) -> None:
        result = self.result
        if self.worker is not None or result is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "导出车牌还原结果", f"{result.source.stem}-第{result.anchor_index + 1}帧-车牌.zip", "ZIP (*.zip)",
        )
        if not path:
            return
        destination = Path(path)
        if destination.suffix.lower() != ".zip":
            destination = destination.with_name(destination.name + ".zip")

        def export() -> Path:
            export_plate_result(destination, result)
            return destination

        self._start(export, self._exported, "正在导出增强图和用于复核的原帧…")

    def _exported(self, destination: Path) -> None:
        self.status.setText(f"已导出 · {destination}")

    def done(self, result: int) -> None:
        if self.worker is not None:
            self.status.setText("请等待当前操作结束后再关闭。")
            return
        super().done(result)
