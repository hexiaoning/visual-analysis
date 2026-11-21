"""Vehicle workspace in the unified application."""
from __future__ import annotations

from dataclasses import replace
from datetime import datetime
import json
from pathlib import Path
from threading import Event

from PySide6.QtCore import QThread, Qt, Signal
from PySide6.QtWidgets import (
    QFileDialog, QFormLayout, QHBoxLayout, QHeaderView, QLabel, QLineEdit,
    QPlainTextEdit, QPushButton, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)

from ..paths import app_home
from ..vehicle_analysis import VehicleCandidate, analyze_vehicle
from ..vehicle_provider import SOURCE_NAMES
from ..vision.local_media import local_path
from .workspace_style import workspace_panel


_STATUS = {
    "ok": "已查询",
    "no_match": "查询完成，未返回匹配资料",
    "not_configured": "授权接口或凭据未配置",
    "unauthorized": "接口未授权",
    "invalid_response": "接口返回格式不正确",
    "unavailable": "网络或接口不可用",
    "no_candidate": "缺少可查询的候选",
}


def _local_time(value: str) -> str:
    return datetime.fromisoformat(value).astimezone().strftime("%Y-%m-%d %H:%M:%S %z") if value else "未查询"


class VehicleWorker(QThread):
    complete = Signal(object)
    failed = Signal(str)

    def __init__(self, source, catalogue, config, manual, parent):
        super().__init__(parent)
        self.source, self.catalogue, self.config, self.manual = source, catalogue, config, manual
        self.stopped = Event()

    def run(self):
        try:
            result = analyze_vehicle(self.source, self.catalogue, self.config, self.manual, self.stopped)
            if not self.stopped.is_set():
                self.complete.emit(result)
        except Exception as exc:
            if not self.stopped.is_set():
                self.failed.emit(str(exc))


class VehicleAnalysisWidget(QWidget):
    def __init__(self, config, parent=None):
        super().__init__(parent)
        self.config = config
        self.worker: VehicleWorker | None = None
        self.report: dict | None = None

        self.source = QLineEdit()
        self.catalogue = QLineEdit()
        for field in (self.source, self.catalogue):
            field.setReadOnly(True)
        default_catalogue = app_home() / "vehicle-catalogue.json"
        if default_catalogue.is_file():
            self.catalogue.setText(str(default_catalogue))
        self.brand = QLineEdit()
        self.series = QLineEdit()
        self.brand.setMaxLength(100)
        self.series.setMaxLength(100)
        self.brand.setPlaceholderText("可选")
        self.series.setPlaceholderText("可选")

        media_button = QPushButton("选择图片或视频")
        media_button.clicked.connect(self._choose_media)
        catalogue_button = QPushButton("选择参考库")
        catalogue_button.clicked.connect(self._choose_catalogue)
        form = QFormLayout()
        form.addRow("本机素材", self._picker(self.source, media_button))
        form.addRow("本地车型参考库", self._picker(self.catalogue, catalogue_button))
        form.addRow("人工候选品牌", self.brand)
        form.addRow("人工候选车系", self.series)
        self.start_button = QPushButton("识别并联网核验")
        self.start_button.setProperty('workspacePrimary', True)
        self.start_button.clicked.connect(self.start)
        self.stop_button = QPushButton("停止")
        self.stop_button.clicked.connect(self.stop)
        self.stop_button.setEnabled(False)
        self.export_button = QPushButton("导出识别记录")
        self.export_button.clicked.connect(self.export)
        self.export_button.setEnabled(False)
        actions = QHBoxLayout()
        for button in (self.start_button, self.stop_button, self.export_button):
            actions.addWidget(button)
        actions.addStretch()
        form.addRow(actions)
        inputs = QWidget()
        inputs.setLayout(form)

        self.status = QLabel("选择本机素材。可导入参考库提高车型覆盖，或填写人工候选。")
        self.status.setTextFormat(Qt.PlainText)
        self.status.setWordWrap(True)
        self.sources = QTableWidget(0, 3)
        self.sources.setHorizontalHeaderLabels(["来源", "查询时间", "查询状态"])
        self.sources.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.sources.setEditTriggers(QTableWidget.NoEditTriggers)
        self.sources.verticalHeader().hide()
        self.sources.setMaximumHeight(105)
        self.details = QPlainTextEdit()
        self.details.setReadOnly(True)

        layout = QVBoxLayout(self)
        layout.addWidget(workspace_panel("车型识别", inputs))
        layout.addWidget(self.status)
        layout.addWidget(self.sources)
        layout.addWidget(workspace_panel("本地候选和联网资料", self.details), 1)

    @staticmethod
    def _picker(field, button):
        row = QHBoxLayout()
        row.addWidget(field, 1)
        row.addWidget(button)
        return row

    def _choose_media(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "选择车辆图片或视频", "",
            "图片或视频 (*.jpg *.jpeg *.png *.bmp *.webp *.tif *.tiff *.mp4 *.mov *.m4v *.avi *.mkv *.webm *.wmv)",
        )
        if path:
            self.source.setText(path)

    def _choose_catalogue(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择车型参考库", "", "JSON (*.json)")
        if path:
            self.catalogue.setText(path)

    def start(self):
        if self.is_busy():
            return
        if not self.source.text():
            self.status.setText("请选择本机车辆照片或视频。")
            return
        brand, series = self.brand.text().strip(), self.series.text().strip()
        if bool(brand) != bool(series):
            self.status.setText("人工候选请同时填写品牌和车系。")
            return
        manual = VehicleCandidate(brand, series) if series else None
        catalogue = Path(self.catalogue.text()) if self.catalogue.text() else None
        self.report = None
        self.details.clear()
        self.sources.setRowCount(0)
        self.export_button.setEnabled(False)
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.status.setText("正在提取本地车型候选并查阅两家来源。")
        self.worker = VehicleWorker(Path(self.source.text()), catalogue, replace(self.config), manual, self)
        self.worker.complete.connect(self._show_result)
        self.worker.failed.connect(self.status.setText)
        self.worker.finished.connect(self._finished)
        self.worker.start()

    def is_busy(self):
        return self.worker is not None and self.worker.isRunning()

    def stop(self):
        if self.worker is not None and self.worker.isRunning():
            self.worker.stopped.set()
            self.stop_button.setEnabled(False)
            self.status.setText("正在停止车型识别。")

    def _finished(self):
        worker, self.worker = self.worker, None
        if worker.stopped.is_set():
            self.status.setText("已停止车型识别。")
        worker.deleteLater()
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)

    def _show_result(self, report):
        self.report = report
        self.status.setText(report["status"])
        lines = [f"本地初判方式 · {report['local_method']} · {report['inference_backend']}"]
        for match in report["candidates"]:
            candidate = match["candidate"]
            title = " ".join(candidate[key] for key in ("brand", "series", "model", "year", "trim") if candidate[key])
            similarity = match["similarity"]
            lines.append(f"{title} · {'人工候选' if similarity is None else f'外观相似度 {similarity:.3f}'}")
        self.sources.setRowCount(len(report["catalogues"]))
        for row, source in enumerate(report["catalogues"]):
            name = SOURCE_NAMES[source["source"]]
            values = (name, _local_time(source["queried_at"]), _STATUS[source["status"]])
            for column, value in enumerate(values):
                self.sources.setItem(row, column, QTableWidgetItem(value))
            if source.get("differences"):
                lines.extend(("", "双源资料差异", *source["differences"]))
            for record in source["records"]:
                title = " ".join(record[key] for key in ("brand", "series", "model", "year", "trim") if record[key])
                lines.extend(("", f"{name} · {title}"))
                if record["record_id"]:
                    lines.append(f"资料编号 {record['record_id']}")
                if record["source_url"]:
                    lines.append(record["source_url"])
                if record["image_similarity"] is not None:
                    lines.append(f"在线参考图相似度 {record['image_similarity']:.3f}")
                for candidate, level in record["candidate_levels"].items():
                    lines.append(f"资料支持至{level}：{candidate}")
                lines.extend(f"{key}  {value}" for key, value in record["parameters"].items())
        self.details.setPlainText("\n".join(lines))
        self.export_button.setEnabled(True)

    def export(self):
        if self.report is None:
            return
        path, _ = QFileDialog.getSaveFileName(self, "导出识别记录", "车型识别.json", "JSON (*.json)")
        if not path:
            return
        try:
            local_path(path).write_text(json.dumps(self.report, ensure_ascii=False, indent=2), encoding="utf-8")
            self.status.setText(f"{self.report['status']} · 已导出识别记录")
        except (OSError, ValueError) as exc:
            self.status.setText(str(exc))
