"""Prepare shared, deterministic photo and clip crops and run reference models offline."""
from __future__ import annotations

import argparse
import hashlib
from importlib import import_module
import json
import logging
from pathlib import Path
import sys
from time import perf_counter

import av
import cv2
import numpy as np


def prepare(args) -> None:
    from face_matching.face_restoration import aligned_face, cpu_session
    from face_matching.vision.detector import SCRFDDetector

    detector = SCRFDDetector(cpu_session(args.detector), input_size=640, threshold=0.35)

    def crop(pixels):
        faces = sorted(detector.detect(pixels), key=lambda item: item.width * item.height, reverse=True)
        return next((face.crop for item in faces
                     if (face := aligned_face(pixels, item, 512)) is not None), None)

    assets = []
    for photo in sorted(args.photos.glob("*.jpg")):
        pixels = cv2.imdecode(np.fromfile(photo, np.uint8), cv2.IMREAD_COLOR)
        face = crop(pixels)
        if face is not None:
            assets.append((photo, [face], "photo", [0]))
    for clip in sorted(args.clips.glob("real_*.mp4")):
        faces, times = [], []
        with av.open(str(clip), options={"protocol_whitelist": "file"}) as vid:
            for frame in vid.decode(video=0):
                face = crop(frame.to_ndarray(format="bgr24"))
                if face is None:
                    faces.clear()
                    times.clear()
                    continue
                faces.append(face)
                times.append(float(frame.pts * frame.time_base) if frame.pts is not None else None)
                if len(faces) == 5:
                    break
        if len(faces) == 5:
            assets.append((clip, faces, "clip", times))
    manifest = []
    for source, faces, group, times in assets:
        name = f"{group}-{source.stem}"
        target = args.output / "inputs" / name
        target.mkdir(parents=True, exist_ok=True)
        for index, pixels in enumerate(faces):
            cv2.imencode(".png", pixels)[1].tofile(target / f"{index:03}.png")
        with source.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        manifest.append({"id": name, "source_file": source.name, "sha256": digest, "group": group,
                         "pts_seconds": times})
    (args.output / "inputs.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Photo/clip benchmark samples: {[(entry['id'], len(entry['pts_seconds'])) for entry in manifest]}", flush=True)


def reference_loader(args):
    from export_gfpgan_onnx import package
    import torch

    torch.set_num_threads(2)
    torch.manual_seed(1024)
    basic = args.upstream / "basicsr"
    package("basicsr", basic)
    package("basicsr.archs", basic / "archs")
    package("basicsr.utils", basic / "utils")
    sys.modules["basicsr.utils"].get_root_logger = lambda *a, **k: logging.getLogger("basicsr")
    if args.model == "CodeFormer":
        architecture = import_module("basicsr.archs.codeformer_arch").CodeFormer
        network = architecture(dim_embd=512, codebook_size=1024, n_head=8, n_layers=9,
                               connect_list=["32", "64", "128", "256"])
        infer = lambda batch: torch.cat(
            [network(face[None], w=0.7, adain=True)[0] for face in batch], dim=0,
        )
    else:
        arch = import_module("basicsr.archs.dir_dist_codeformer_multiscale_arch")
        network = arch.TemporalCodeFormerDirDistMultiScale(
            dim_embed=512, n_head=8, n_layers=9, codebook_size=1024,
            connect_list=["32", "64", "128", "256"], frame_length=5,
        )
        infer = lambda batch: network(batch, w=0.5)[0]
    checkpoint = torch.load(args.weights, map_location="cpu", weights_only=True)
    network.load_state_dict(checkpoint["params_ema"], strict=True)
    network.eval()

    def run(data):
        with torch.inference_mode():
            return infer(torch.from_numpy(data)).numpy()

    return run


def infer(args) -> None:
    # Install the local media-only guard so benchmark adapters cannot fetch
    # optional weights or send the selected test input to a service.
    from face_matching.__main__ import _reject_outbound

    sys.addaudithook(_reject_outbound)
    if args.model in {"CodeFormer", "DicFace"}:
        run = reference_loader(args)
    else:
        from face_matching.face_restoration import cpu_session

        sess = cpu_session(args.weights)
        input_name = sess.get_inputs()[0].name
        run = lambda data: np.concatenate([sess.run(None, {input_name: item[None]})[0] for item in data])
    rows = []
    for clip in sorted((args.output / "inputs").iterdir()):
        if args.model == "DicFace" and not clip.name.startswith("clip-"):
            continue
        frames = [cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_COLOR) for path in sorted(clip.glob("*.png"))]
        tensor = np.stack([frame[:, :, ::-1].transpose(2, 0, 1) for frame in frames]).astype(np.float32) / 127.5 - 1
        started = perf_counter()
        result = run(tensor)
        elapsed = perf_counter() - started
        if result.shape != tensor.shape or not np.isfinite(result).all():
            raise ValueError(f"{args.model} returned invalid output for {clip.name}")
        target = args.output / args.model / clip.name
        target.mkdir(parents=True, exist_ok=True)
        for index, pixels in enumerate(np.clip(np.rint((result + 1) * 127.5), 0, 255).astype(np.uint8)):
            cv2.imencode(".png", pixels.transpose(1, 2, 0)[:, :, ::-1])[1].tofile(target / f"{index:03}.png")
        rows.append({"id": clip.name, "frames": len(frames), "inference_seconds": elapsed})
        print(f"{args.model} {clip.name}: {len(frames)} frames, {elapsed:.3f}s", flush=True)
    (args.output / f"{args.model}.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="method", required=True)
    inputs = sub.add_parser("prepare")
    inputs.add_argument("output", type=Path)
    inputs.add_argument("photos", type=Path)
    inputs.add_argument("clips", type=Path)
    inputs.add_argument("detector", type=Path)
    models = sub.add_parser("infer")
    models.add_argument("output", type=Path)
    models.add_argument("model", choices=["PCFlow", "DicFace", "GFPGAN", "CodeFormer"])
    models.add_argument("weights", type=Path)
    models.add_argument("--upstream", type=Path)
    args = parser.parse_args()
    if args.method == "prepare":
        prepare(args)
    else:
        infer(args)


if __name__ == "__main__":
    main()
