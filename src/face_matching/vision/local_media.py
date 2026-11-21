"""Local-only media entry points shared by the visual workspaces."""
from __future__ import annotations

import os
from pathlib import Path
import re
import stat

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import cv2


VIDEO_EXTENSIONS = {".mp4", ".avi", ".mkv", ".mov", ".m4v", ".wmv", ".webm"}
_DRIVE_URI = re.compile(r"^[A-Za-z]:[\\/]")
_URI = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*:")
_ASF = bytes.fromhex("3026b2758e66cf11a6d900aa0062ce6c")


def local_path(value: str | Path) -> Path:
    """Reject network shares, device paths and redirections before opening them."""
    source = os.fspath(value)
    if source.startswith(("\\\\", "//")) or (_URI.match(source) and not _DRIVE_URI.match(source)):
        raise ValueError("只支持本机文件和文件夹")
    path = Path(os.path.abspath(source))
    if os.name == "nt":
        import ctypes

        if not _DRIVE_URI.match(str(path)) or ctypes.windll.kernel32.GetDriveTypeW(str(path.anchor)) == 4:
            raise ValueError("无法使用网络或设备路径")
    for part in (*reversed(path.parents), path):
        try:
            info = part.lstat()
        except FileNotFoundError:
            break
        if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT:
            raise ValueError("无法使用链接、重定向目录或云占位文件")
    return path


def local_video_path(value: str | Path) -> Path:
    path = local_path(value)
    if path.suffix.lower() not in VIDEO_EXTENSIONS:
        raise ValueError("只支持本机 MP4、MOV、AVI、MKV、WebM、WMV 或 M4V 视频")
    with path.open("rb") as source:
        header = source.read(32)
    if not (
        header[4:8] in {b"ftyp", b"moov", b"mdat", b"wide", b"free", b"skip"}
        or (header.startswith(b"RIFF") and header[8:12] == b"AVI ")
        or header.startswith((b"\x1a\x45\xdf\xa3", _ASF))
    ):
        raise ValueError("视频容器无效。请先保存为本机视频文件。")
    return path


def open_local_video(value: str | Path) -> cv2.VideoCapture:
    import cv2

    path = local_video_path(value)
    # Apply for every OpenCV FFmpeg input. The explicit backend keeps Qt and OS
    # URL handlers out of local file decoding.
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "protocol_whitelist;file"
    return cv2.VideoCapture(str(path), cv2.CAP_FFMPEG)
