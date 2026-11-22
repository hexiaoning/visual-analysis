from __future__ import annotations

from collections.abc import Iterable
import hashlib
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import cv2
import numpy as np

from .vision.local_media import local_path


def encode_png(image: np.ndarray) -> bytes:
    ok, payload = cv2.imencode(".png", image)
    if not ok:
        raise OSError("PNG 编码失败")
    return payload.tobytes()


def save_new_png(destination: Path, image: np.ndarray) -> None:
    payload = encode_png(image)
    destination = local_path(destination)
    with destination.open("xb") as stream:
        try:
            stream.write(payload)
        except BaseException:
            stream.close()
            destination.unlink(missing_ok=True)
            raise


def write_restoration_archive(destination: Path, files: Iterable[tuple[str, bytes]], manifest: dict) -> None:
    hashes = {}
    # Opening in exclusive mode also protects against a save-dialog overwrite.
    destination = local_path(destination)
    stream = destination.open("xb")
    try:
        with stream, ZipFile(stream, "w", compression=ZIP_DEFLATED) as archive:
            for name, payload in files:
                archive.writestr(name, payload)
                hashes[name] = hashlib.sha256(payload).hexdigest()
            archive.writestr("manifest.json", json.dumps(dict(manifest, sha256=hashes), ensure_ascii=False, indent=2))
    except BaseException:
        destination.unlink(missing_ok=True)
        raise
