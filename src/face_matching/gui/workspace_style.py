from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QVBoxLayout, QWidget


WORKSPACE_STYLE = """
QWidget#workspace {
    background: #edf2f9;
}
QFrame#workspaceHeader {
    background: #164fb9;
    border-bottom: 1px solid #103c8e;
}
QLabel#workspaceBrand {
    color: white;
    font-size: 23px;
    font-weight: bold;
}
QLabel#workspaceVersion {
    color: white;
    background: #235fc8;
    border: 1px solid #6695e5;
    border-radius: 4px;
    padding: 4px 10px;
}
QTabWidget#workspaceTabs::pane {
    background: #edf2f9;
    border: 1px solid #cbd8e9;
}
QTabWidget#workspaceTabs QTabBar::tab {
    background: #e4ebf6;
    color: #334763;
    border: 1px solid #cbd8e9;
    border-bottom: 3px solid #e4ebf6;
    padding: 7px 17px;
    margin-right: 2px;
}
QTabWidget#workspaceTabs QTabBar::tab:hover {
    background: #f6f9fe;
}
QTabWidget#workspaceTabs QTabBar::tab:selected {
    background: white;
    color: #164fb9;
    border-bottom-color: #2061d2;
    font-weight: bold;
}
QFrame[workspacePanel="true"] {
    background: white;
    border: 1px solid #cbd8e9;
    border-radius: 4px;
}
QLabel[workspaceHeading="true"] {
    color: #204a84;
    background: #e7f0fc;
    border-bottom: 1px solid #cbd8e9;
    padding: 6px 10px;
    font-weight: bold;
}
QLabel[workspaceNote="true"] {
    color: #435b78;
}
QPushButton {
    color: #253b56;
    background: white;
    border: 1px solid #bbcbe0;
    border-radius: 4px;
    padding: 4px 10px;
}
QPushButton:hover {
    background: #eaf2fd;
    border-color: #759bd8;
}
QPushButton:pressed, QPushButton:checked {
    background: #dbe9fd;
}
QPushButton:focus, QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus {
    border: 1px solid #2061d2;
}
QPushButton[workspacePrimary="true"] {
    color: white;
    background: #1857c5;
    border-color: #1857c5;
    font-weight: bold;
}
QPushButton[workspacePrimary="true"]:hover {
    background: #1046a6;
}
QPushButton[workspacePrimary="true"]:pressed {
    background: #0d3989;
}
QPushButton:disabled, QPushButton[workspacePrimary="true"]:disabled {
    color: #6f8096;
    background: #e8eef6;
    border-color: #d0dbe9;
}
QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {
    background: white;
    border: 1px solid #c5d3e6;
    border-radius: 3px;
    min-height: 23px;
}
QLineEdit {
    padding: 0 6px;
}
QTableWidget, QListWidget {
    background: white;
    alternate-background-color: #f5f8fd;
    border: 1px solid #d4deec;
    gridline-color: #e3ebf5;
    selection-background-color: #d7e7ff;
    selection-color: #17365e;
}
QHeaderView::section {
    background: #f0f5fc;
    color: #334d6b;
    border: 0;
    border-right: 1px solid #dce6f4;
    border-bottom: 1px solid #d0ddef;
    padding: 4px;
    font-weight: bold;
}
QStatusBar {
    background: #e4ebf6;
    color: #334d6b;
}
"""


def workspace_header(version: str) -> QFrame:
    header = QFrame()
    header.setObjectName("workspaceHeader")
    row = QHBoxLayout(header)
    row.setContentsMargins(18, 10, 18, 10)
    brand = QLabel("Face Matching · 离线综合分析")
    brand.setObjectName("workspaceBrand")
    badge = QLabel(f"离线运行  /  v{version}")
    badge.setObjectName("workspaceVersion")
    row.addWidget(brand)
    row.addStretch()
    row.addWidget(badge)
    return header


def workspace_panel(title: str, content: QWidget) -> QFrame:
    panel = QFrame()
    panel.setProperty("workspacePanel", True)
    layout = QVBoxLayout(panel)
    layout.setContentsMargins(1, 1, 1, 1)
    layout.setSpacing(0)
    heading = QLabel(title)
    heading.setProperty("workspaceHeading", True)
    layout.addWidget(heading)
    layout.addWidget(content, 1)
    return panel
