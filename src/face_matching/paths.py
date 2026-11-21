from __future__ import annotations

import os
import sys
from pathlib import Path

from .vision.local_media import local_path


def app_home() -> Path:
    override = os.environ.get("FACE_MATCHING_HOME")
    if override:
        return local_path(Path(override).expanduser())
    base = os.environ.get("LOCALAPPDATA")
    if base:
        return local_path(Path(base) / "FaceMatching")
    return local_path(Path.home() / ".face-matching")


def ensure_app_dirs() -> Path:
    root = app_home()
    for child in ("models", "data", "data/face_images", "logs"):
        (root / child).mkdir(parents=True, exist_ok=True)
    return root


def models_dir() -> Path:
    return ensure_app_dirs() / "models"


def is_frozen_app() -> bool:
    return bool(getattr(sys, "frozen", False))


def packaged_models_dir() -> Path | None:
    """Return the read-only model directory embedded in a portable build."""
    override = os.environ.get("FACE_MATCHING_BUNDLED_MODELS")
    if override:
        return local_path(Path(override).expanduser())
    bundle_root = getattr(sys, "_MEIPASS", None)
    if bundle_root:
        return Path(bundle_root) / "models"
    return None


def data_dir() -> Path:
    return ensure_app_dirs() / "data"


def database_path() -> Path:
    return data_dir() / "faces.db"


def config_path() -> Path:
    return ensure_app_dirs() / "config.json"
