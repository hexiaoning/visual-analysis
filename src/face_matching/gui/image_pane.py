from __future__ import annotations

import numpy as np
from PySide6.QtCore import QRectF, Qt, Signal
from PySide6.QtGui import QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import QGraphicsScene, QGraphicsView


class ImagePane(QGraphicsView):
    region_selected = Signal(object)

    def __init__(self, selectable: bool = False) -> None:
        super().__init__()
        self.setScene(QGraphicsScene(self))
        self.setMinimumSize(280, 280)
        self.setBackgroundBrush(Qt.black)
        self.setRenderHint(QPainter.SmoothPixmapTransform)
        self.setDragMode(QGraphicsView.RubberBandDrag if selectable else QGraphicsView.NoDrag)
        self.selectable = selectable
        self.image = QImage()
        self.region: tuple[int, int, int, int] | None = None
        self.region_item = None
        self.origin = None
        self.fit_mode = True

    def set_image(self, pixels: np.ndarray) -> None:
        height, width = pixels.shape[:2]
        self.image = QImage(pixels.data, width, height, pixels.strides[0], QImage.Format_BGR888).copy()
        self.scene().clear()
        self.region_item = None
        self.scene().addPixmap(QPixmap.fromImage(self.image))
        self.scene().setSceneRect(QRectF(0, 0, width, height))
        self.set_region(self.region)
        self.fit_image()

    def clear_image(self) -> None:
        self.set_region(None)
        self.scene().clear()
        self.scene().setSceneRect(QRectF())
        self.image = QImage()
        self.origin = None

    def set_region(self, region: tuple[int, int, int, int] | None) -> None:
        self.region = region
        if self.region_item is not None:
            self.scene().removeItem(self.region_item)
            self.region_item = None
        if region is not None:
            x1, y1, x2, y2 = region
            pen = QPen(Qt.cyan, 2)
            pen.setCosmetic(True)
            self.region_item = self.scene().addRect(QRectF(x1, y1, x2 - x1, y2 - y1), pen)

    def fit_image(self) -> None:
        self.fit_mode = True
        if not self.image.isNull():
            rect = QRectF(*self.region[:2], self.region[2] - self.region[0],
                          self.region[3] - self.region[1]) if self.region else self.sceneRect()
            self.fitInView(rect, Qt.KeepAspectRatio)

    def actual_size(self) -> None:
        self.fit_mode = False
        self.resetTransform()
        if self.region:
            self.centerOn((self.region[0] + self.region[2]) / 2,
                          (self.region[1] + self.region[3]) / 2)
        else:
            self.centerOn(self.sceneRect().center())

    def resizeEvent(self, event: object) -> None:
        super().resizeEvent(event)
        if self.fit_mode:
            self.fit_image()

    def mousePressEvent(self, event: object) -> None:
        if self.selectable and event.button() == Qt.LeftButton and not self.image.isNull():
            self.origin = self.mapToScene(event.position().toPoint())
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event: object) -> None:
        super().mouseReleaseEvent(event)
        if self.origin is None or event.button() != Qt.LeftButton:
            return
        point = self.mapToScene(event.position().toPoint())
        rect = QRectF(self.origin, point).normalized().intersected(self.sceneRect())
        self.origin = None
        x1, y1 = max(0, int(rect.left())), max(0, int(rect.top()))
        x2, y2 = min(self.image.width(), int(rect.right())), min(self.image.height(), int(rect.bottom()))
        if x2 - x1 >= 4 and y2 - y1 >= 4:
            self.region_selected.emit((x1, y1, x2, y2))
