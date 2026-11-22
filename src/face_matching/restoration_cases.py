"""Run local portrait cases through the same engine used by the GUI."""
from __future__ import annotations

from pathlib import Path
from tempfile import mkdtemp
from threading import Event

from .config import AppConfig
from .face_restoration import restore_portraits
from .models import file_sha256
from .restoration_io import save_new_png
from .vision.io import IMAGE_EXTENSIONS, read_image
from .vision.local_media import local_path


def run_portrait_cases(inputs: Path, outputs: Path, config: AppConfig) -> dict:
    inputs, outputs = local_path(inputs), local_path(outputs)
    files = sorted(path for path in inputs.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)
    if not files:
        raise ValueError("案例目录中没有可读的图片")
    outputs.mkdir(parents=True, exist_ok=True)
    run = Path(mkdtemp(prefix="run-", dir=outputs))
    reports = []
    for start in range(0, len(files), 3):
        selection = files[start:start + 3]
        decoded = [read_image(path) for path in selection]
        if any(image is None for image in decoded):
            bad = next(path.name for path, image in zip(selection, decoded, strict=True) if image is None)
            raise ValueError(f"无法解码案例：{bad}")
        candidates = restore_portraits(tuple((image, None) for image in decoded), config, Event())
        for source, options in zip(selection, candidates, strict=True):
            directory = run / f"{len(reports) + 1:03}-{source.stem}"
            directory.mkdir()
            entries = []
            for index, item in enumerate(options, 1):
                destination = directory / f"candidate-{index}.png"
                save_new_png(destination, item.pixels)
                entries.append({
                    "model": item.label, "score": item.score, "confidence": item.confidence,
                    "agreement": item.agreement, "detail": item.detail,
                    "file": str(destination),
                })
            reports.append({"source": source.name, "source_sha256": file_sha256(source), "candidates": entries})
    return {"ok": all(case["candidates"] for case in reports), "cases": reports, "output_directory": str(run)}
