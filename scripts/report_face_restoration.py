"""Summarize benchmark outputs using the exact same decoded and aligned samples."""
from __future__ import annotations

import argparse
from html import escape
import json
from pathlib import Path
from statistics import fmean
from urllib.parse import quote

import cv2
import numpy as np

from face_matching.face_restoration import FFHQ_LANDMARKS, cpu_session
from face_matching.restoration_candidates import score_candidate
from face_matching.vision.alignment import ARCFACE_TEMPLATE, estimate_similarity_transform
from face_matching.vision.recognizer import FaceEmbedder


OPTIONS = ("CodeFormer", "PCFlow", "GFPGAN", "DicFace")
MATRIX = estimate_similarity_transform(FFHQ_LANDMARKS, ARCFACE_TEMPLATE)


def load(path: Path) -> np.ndarray:
    image = cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Missing image: {path}")
    return image


def crop(image: np.ndarray) -> np.ndarray:
    return cv2.warpAffine(image, MATRIX, (112, 112))


def temporal_change(sources: list[np.ndarray], results: list[np.ndarray]) -> float | None:
    if len(sources) < 2:
        return None
    originals = np.asarray([crop(frame) for frame in sources], dtype=np.float32)
    restored = np.asarray([crop(frame) for frame in results], dtype=np.float32)
    source_change = np.diff(originals, axis=0)
    residual_change = np.diff(restored - originals, axis=0)
    stable = np.max(np.abs(source_change), axis=3) < 8
    return float(np.abs(residual_change[stable]).mean()) if stable.any() else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("face_embedding_model", type=Path)
    args = parser.parse_args()
    root = args.results
    manifest = json.loads((root / "inputs.json").read_text(encoding="utf-8"))
    timing = {
        method: {row["id"]: row["inference_seconds"]
                 for row in json.loads((root / f"{method}.json").read_text(encoding="utf-8"))}
        for method in OPTIONS if (root / f"{method}.json").exists()
    }
    embedder = FaceEmbedder(cpu_session(args.face_embedding_model))
    table, gallery = [], []
    for sample in manifest:
        case = sample["id"]
        paths = sorted((root / "inputs" / case).glob("*.png"))
        sources = [load(path) for path in paths]
        unprocessed = [embedder.embed(crop(image), mirror_augmentation=False) for image in sources]
        images = [("原始对齐图", paths[len(paths)//2])]
        for model in OPTIONS:
            output = sorted((root / model / case).glob("*.png"))
            if len(output) != len(sources):
                continue
            results = [load(path) for path in output]
            embeddings = [embedder.embed(crop(image), mirror_augmentation=False) for image in results]
            distance = fmean(float(old @ new) for old, new in zip(unprocessed, embeddings, strict=True))
            scores = [score_candidate(model, result, source) for source, result in zip(sources, results, strict=True)]
            table.append({
                "case": case, "type": sample["group"], "model": model, "frames": len(output),
                "source_embedding_cosine": distance, "source_agreement": fmean(item.agreement for item in scores),
                "supported_edge_detail": fmean(item.detail for item in scores),
                "temporal_residual_mae": temporal_change(sources, results),
                "seconds": timing.get(model, {}).get(case),
            })
            images.append((model, output[len(output)//2]))
        tiles = "".join(f'<figure><img src="{quote(path.relative_to(root).as_posix())}" alt="{escape(name)}">'
                        f'<figcaption>{escape(name)}</figcaption></figure>' for name, path in images)
        gallery.append(f'<section><h2>{escape(case)}</h2><div class="images">{tiles}</div></section>')
    (root / "comparison.json").write_text(json.dumps(table, ensure_ascii=False, indent=2), encoding="utf-8")
    grouped = []
    for category in ("photo", "clip"):
        for model in OPTIONS:
            items = [row for row in table if row["type"] == category and row["model"] == model]
            if not items:
                continue
            metrics = {}
            for key in ("source_embedding_cosine", "source_agreement", "supported_edge_detail", "temporal_residual_mae"):
                values = [item[key] for item in items if item[key] is not None]
                metrics[key] = fmean(values) if values else None
            durations = [item["seconds"] / item["frames"] for item in items if item["seconds"] is not None]
            grouped.append({"category": category, "model": model, "samples": len(items),
                            **metrics, "seconds_per_frame": fmean(durations) if durations else None})
    (root / "summary.json").write_text(json.dumps(grouped, ensure_ascii=False, indent=2), encoding="utf-8")

    def display(value) -> str:
        return f"{value:.3f}" if isinstance(value, float) else "—" if value is None else str(value)

    columns = ["素材", "模型", "完整样本", "原图特征余弦 ↑", "原图一致性 ↑", "受支持边缘细节 ↑",
               "时序残差变化 ↓", "秒 / 帧"]
    rows = "".join("<tr>" + "".join(f"<td>{escape(display(value))}</td>" for value in item.values()) + "</tr>"
                   for item in grouped)
    page = """<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<title>Visual Analysis 人像还原对比</title><style>
body{font:16px system-ui,sans-serif;margin:2rem;max-width:1400px;color:#172136;background:#fafbfd}
table{border-collapse:collapse;width:100%;background:white}th,td{border:1px solid #dbe1ec;padding:.55rem;text-align:left}
.images{display:flex;gap:16px;flex-wrap:wrap}figure{margin:0}img{width:190px;height:190px;object-fit:contain}
section{margin-top:2rem}figcaption{margin:.35rem 0}
</style><h1>Visual Analysis 人像还原对比</h1>
<p>同一批作者公开示例，照片取自 CodeFormer，视频取自 KEEP 的 real_*。各模型使用同一批 512 像素对齐人脸。
特征余弦比较的是原始低质图，一致性与时序误差用于对照。原始素材没有同帧高清真值。
本机 Windows 11、Core i3‑8100，推理期间存在共享 CPU 负载，耗时仅供参考。</p>
""" + "<table><thead><tr>" + "".join(f"<th>{item}</th>" for item in columns)
    page += "</tr></thead><tbody>" + rows + "</tbody></table>" + "".join(gallery) + "</html>"
    (root / "comparison.html").write_text(page, encoding="utf-8")
    print(json.dumps(grouped, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
