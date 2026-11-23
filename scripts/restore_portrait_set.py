"""Generate clear face hypotheses from one photograph with the offline ONNX models.

  python scripts/restore_portrait_set.py photo.jpg output --models path/to/models

Requires numpy, OpenCV and onnxruntime. Use --aligned for an already FFHQ-aligned
square crop. Outputs are 512-pixel face crops. The manifest lists every distinct
generated image, including hypotheses with poor low-frequency agreement.
"""

import argparse
import hashlib
import json
import math
import tempfile
from pathlib import Path
from fractions import Fraction

import cv2
import numpy as np
import onnxruntime as ort


ANCHORS = np.float32([[192.98, 239.95], [318.90, 240.19], [256.63, 314.02],
                      [201.26, 371.41], [313.09, 371.15]])
MODELS = {"gfpgan": "gfpgan-v1.4-fp32.onnx", "pcflow": "pcflow-bfr-fp32.onnx"}


def session(path):
    options = ort.SessionOptions()
    options.intra_op_num_threads = 3
    options.inter_op_num_threads = 1
    options.enable_cpu_mem_arena = False
    return ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])


def session_run(net, tensor):
    return net.run(None, {net.get_inputs()[0].name: tensor})


def align(image, models):
    height, width = image.shape[:2]
    scale = 640 / max(width, height)
    size = (max(1, round(width * scale)), max(1, round(height * scale)))
    canvas = np.zeros((640, 640, 3), np.uint8)
    canvas[:size[1], :size[0]] = cv2.resize(image, size)
    blob = cv2.dnn.blobFromImage(canvas, 1 / 128, mean=(127.5,) * 3, swapRB=True)
    outputs = session_run(session(models / "scrfd_10g_bnkps.onnx"), blob)
    if len(outputs) != 9:
        raise ValueError("Expected three SCRFD score, box and landmark levels.")
    detections = []
    for level, stride in enumerate((8, 16, 32)):
        scores = outputs[level].reshape(-1)
        points = outputs[6 + level].reshape(-1, 5, 2)
        for index in np.flatnonzero(scores >= .32):
            grid = 640 // stride
            xy = np.float32([(index // 2) % grid, (index // 2) // grid])
            observed = (xy + points[index]) * stride / scale
            if np.linalg.norm(observed[0] - observed[1]) < 3:
                continue
            detections.append((float(scores[index]), observed))
    for score, observed in sorted(detections, key=lambda row: row[0], reverse=True):
        matrix, _ = cv2.estimateAffinePartial2D(observed, ANCHORS, method=cv2.LMEDS)
        if matrix is None or not np.isfinite(matrix).all():
            continue
        fitted = cv2.transform(observed[None], matrix)[0]
        if np.linalg.norm(fitted - ANCHORS, axis=1).max() <= 512 * .14:
            crop = cv2.warpAffine(image, matrix, (512, 512), borderMode=cv2.BORDER_REFLECT_101)
            return crop, {"score": score, "source_to_face": matrix.tolist()}
    raise ValueError("No face could be aligned. For a pre-aligned square portrait use --aligned.")


def restore(net, crop):
    blob = cv2.dnn.blobFromImage(crop, 1 / 127.5, mean=(127.5,) * 3, swapRB=True)
    predicted = session_run(net, blob)[0]
    if predicted.shape != (1, 3, 512, 512) or not np.isfinite(predicted).all():
        raise ValueError("Restorer returned invalid face pixels.")
    return np.clip(np.rint((predicted[0, ::-1].transpose(1, 2, 0) + 1) * 127.5),
                   0, 255).astype(np.uint8)


def flow(first, second):
    def structure(image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return cv2.GaussianBlur(gray, (0, 0), 1.8)

    return cv2.calcOpticalFlowFarneback(structure(first), structure(second),
                                        None, .5, 4, 39, 4, 7, 1.5, 0)


def correspondence_field(first, second):
    # At coordinates of the first image, pull detail from first towards or
    # away from the correspondence in the second. Downweight inconsistent
    # correspondences and keep the background and weakly supported edges fixed.
    forward, backward = flow(first, second), flow(second, first)
    y, x = np.indices(first.shape[:2], dtype=np.float32)
    grid = np.stack([x, y], axis=-1)
    reverse_at_match = cv2.remap(backward, grid + forward, None, cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_REPLICATE)
    error = np.linalg.norm(forward + reverse_at_match, axis=2)
    radius = ((x - 256) / 190) ** 2 + ((y - 296) / 191) ** 2
    support = np.exp(-.5 * (error / 2.5) ** 2) * np.clip((1.15 - radius) / .35, 0, 1)
    smooth = cv2.GaussianBlur(forward * support[:, :, None], (0, 0), 4)
    magnitude = np.linalg.norm(smooth, axis=2, keepdims=True)
    smooth *= np.minimum(1, 12 / np.maximum(magnitude, 1e-6))
    return grid, smooth


def shape_parameters(field, radius=1.):
    # The displacement maps are affine in t. With the unwarped image at t=0,
    # a half-step times the peak flow bounds the distance to the nearest map.
    peak = float(np.max(np.linalg.norm(field, axis=2)))
    intervals = max(1, math.ceil(.65 * peak / (2 * radius)))
    for tag, end in (("towards", -.65), ("away", .65)):
        yield tag, end
        for index in range(1, intervals):
            yield f"{tag}-{index}-of-{intervals}", end * index / intervals


def shape_branches(first, second):
    grid, field = correspondence_field(first, second)
    for tag, amount in shape_parameters(field):
        yield tag, cv2.remap(first, grid + amount * field, None, cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_REFLECT_101)


def exposure_hypothesis(crop):
    # A dark capture can arise from a brighter original followed by attenuation.
    # Retain the native exposure alongside this alternative, including dark skin.
    luma = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    black = min(12., float(np.percentile(luma, 1)))
    white = float(np.percentile(luma[108:445, 100:412], 95))
    if white >= 160 or white - black < 10:
        return None
    gain = min(3., (210. - black) / (white - black))
    return np.clip(np.rint((crop.astype(np.float32) - black) * gain + black),
                   0, 255).astype(np.uint8)


def generate(crop, models):
    priors = {}
    bright = exposure_hypothesis(crop)
    for name, filename in MODELS.items():
        net = session(models / filename)
        priors[name] = restore(net, crop)
        yield name, priors[name]
        yield name + "-mirror", cv2.flip(restore(net, cv2.flip(crop, 1)), 1)
        if bright is not None:
            yield name + "-exposure", restore(net, bright)
        del net
    for name, other in (("gfpgan", "pcflow"), ("pcflow", "gfpgan")):
        for direction, pixels in shape_branches(priors[name], priors[other]):
            yield name + "-" + direction, pixels


def save_set(photo, out, models, aligned=False, sparse_assets=None, sensor_camera=None):
    if out.exists():
        raise ValueError(f"Choose a new output directory: {out}")
    raw = photo.read_bytes()
    image = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Cannot decode image: {photo}")
    if aligned:
        if image.shape[0] != image.shape[1]:
            raise ValueError("--aligned requires a square FFHQ-aligned face crop.")
        crop, alignment = cv2.resize(image, (512, 512)), {"provided_aligned": True}
    else:
        crop, alignment = align(image, models)
    out.parent.mkdir(parents=True, exist_ok=True)
    report = {"input_sha256": hashlib.sha256(raw).hexdigest(), "alignment": alignment,
              "geometry_search": {"scope": "two fitted optical-flow paths before resampling",
                                  "parameter_interval": [-.65, .65],
                                  "nearest_map_displacement_bound_px": 1.},
              "candidates": [], "open_world_coverage": "unverified"}
    coarse = cv2.resize(crop, (16, 16), interpolation=cv2.INTER_AREA).astype(float)
    seen = {}
    with tempfile.TemporaryDirectory(dir=out.parent) as tmp:
        stage = Path(tmp) / "set"
        stage.mkdir()
        sensor = None
        if sensor_camera is not None:
            from explore_portrait_preimage import CameraPreviews
            sensor = CameraPreviews(stage / "sensor-preimage", sensor_camera, image, alignment)
        def branches():
            yield from generate(crop, models)
            if sparse_assets is not None:
                # Release the ONNX sessions before loading the optional weights.
                import torch
                from portrait_sparse_prior import SparsePortraitPrior
                torch.set_num_threads(3)
                yield from SparsePortraitPrior(*sparse_assets).generate(crop).items()

        for branch, pixels in branches():
            if sensor is not None:
                sensor.add(branch, pixels)
            key = hashlib.sha256(pixels.tobytes()).hexdigest()
            if key in seen:
                seen[key]["branches"].append(branch)
                continue
            small = cv2.resize(pixels, (16, 16), interpolation=cv2.INTER_AREA)
            ok, encoded = cv2.imencode(".png", pixels)
            if not ok:
                raise ValueError(f"Could not encode {branch}.")
            path = stage / (branch + ".png")
            encoded.tofile(str(path))
            row = {"file": path.name, "branches": [branch],
                   "sha256": hashlib.sha256(encoded).hexdigest(),
                   "coarse_rgb_rmse_255": float(np.sqrt(np.mean((small - coarse) ** 2)))}
            report["candidates"].append(row)
            seen[key] = row
        if sensor is not None:
            sensor.finish()
            report["sensor_preimage"] = "sensor-preimage/set.json"
            report["sensor_preview_count"] = len(sensor.seen)
        (stage / "manifest.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        stage.rename(out)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("photo", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--models", type=Path, required=True,
                        help="Directory containing the offline SCRFD, PCFlow and GFPGAN ONNX models.")
    parser.add_argument("--aligned", action="store_true", help="Input is already an aligned square face.")
    parser.add_argument("--sensor-scale", type=int, help="Also explore an integer block-average camera at this upscale factor.")
    parser.add_argument("--sensor-gain", type=Fraction, help="Rational exposure gain of the block-average camera (default 1).")
    parser.add_argument("--sensor-offset", type=int, help="Integer channel offset of this camera (default 0).")
    parser.add_argument("--sparse-prior", type=Path, nargs=3,
                        metavar=("SWINIR_DIR", "DICFACE_DIR", "DICFACE_WEIGHTS"),
                        help="Include DiffBIR SwinIR plus full and sparse DicFace detail codes.")
    args = parser.parse_args()
    camera = None
    if args.sensor_scale is not None:
        from portrait_preimage import BlockMeanCamera
        camera = BlockMeanCamera(args.sensor_scale, args.sensor_gain if args.sensor_gain is not None else 1,
                                 args.sensor_offset or 0)
    elif args.sensor_gain is not None or args.sensor_offset is not None:
        parser.error("Exposure parameters require --sensor-scale.")
    report = save_set(args.photo, args.out, args.models, args.aligned, args.sparse_prior, camera)
    print(f"{len(report['candidates'])} distinct portraits saved to {args.out}")
    if camera is not None:
        print(f"{report['sensor_preview_count']} camera-consistent previews; set description at "
              f"{args.out / report['sensor_preimage']}")
