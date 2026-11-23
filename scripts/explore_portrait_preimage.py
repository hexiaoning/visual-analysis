"""Inspect the complete digital preimage of a declared block-average camera.

The set contains every compatible RGB8 raster. Enumerated rasters need not
depict natural faces. The portrait generator supplies face-guided proposals.
"""
import argparse
from fractions import Fraction
from hashlib import sha256
from itertools import islice
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import cv2
import numpy as np

from portrait_preimage import BlockMeanCamera


def save_png(path, pixels):
    ok, data = cv2.imencode(".png", pixels)
    if not ok:
        raise ValueError(f"PNG encoding failed for {path}")
    data.tofile(str(path))
    return sha256(data).hexdigest()


def read_raster(path):
    image = cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_UNCHANGED)
    if image is None or image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Expected an 8-bit, three-channel raster: {path}")
    return image


def raster_proposal(face, observation, camera, alignment):
    h, w = observation.shape[:2]
    size = (w * camera.scale, h * camera.scale)
    if alignment.get("provided_aligned"):
        return cv2.resize(face, size, interpolation=cv2.INTER_CUBIC)
    # Map high-resolution pixel centres through the original face alignment.
    # Beyond the aligned crop, retain the upsampled observed scene.
    to_crop = np.asarray(alignment["source_to_face"], dtype=np.float64)
    transform = to_crop.copy()
    transform[:, :2] /= camera.scale
    transform[:, 2] += to_crop[:, :2].sum(axis=1) * (.5 / camera.scale - .5)
    backdrop = cv2.resize(observation, size, interpolation=cv2.INTER_CUBIC)
    return cv2.warpAffine(face, transform, size, dst=backdrop,
                          flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_TRANSPARENT)


def eye_mouth_proposals(face):
    # Positions refer to the generator's FFHQ-aligned 512-pixel face.
    # Move the two eyes or the corners of the mouth symmetrically. Using
    # separate directions lets one vary while the other remains anchored.
    y, x = np.indices((512, 512), dtype=np.float32)
    def pair(left, right, cy, width, height):
        vertical = np.exp(-.5 * ((y - cy) / height) ** 2)
        return vertical * (np.exp(-.5 * ((x - left) / width) ** 2)
                           - np.exp(-.5 * ((x - right) / width) ** 2))
    eyes = pair(193, 319, 240, 30, 20)
    mouth = pair(201, 313, 371, 23, 16)
    for eye_shift in (-1, 0, 1):
        for mouth_shift in (-1, 0, 1):
            if eye_shift or mouth_shift:
                offset = 7 * eye_shift * eyes + 6 * mouth_shift * mouth
                yield (f"eye{eye_shift:+d}-mouth{mouth_shift:+d}",
                       cv2.remap(face, x + offset, y, cv2.INTER_CUBIC,
                                 borderMode=cv2.BORDER_REFLECT_101))


class CameraPreviews:
    def __init__(self, folder, camera, observation, alignment):
        lower, upper = camera.bounds(observation)
        self.empty = bool(np.any(lower > upper))
        folder.mkdir()
        self.folder, self.camera = folder, camera
        self.observation, self.alignment = observation, alignment
        self.seen = {}
        self.anchor = None
        self.report = {
            "model": "rgb8-mean-half-up-v1", "scale": camera.scale,
            "gain": str(camera.gain), "offset": camera.offset,
            "observation": "observation.png",
            "observation_sha256": save_png(folder / "observation.png", observation),
            "scope": "all compatible RGB8 rasters under this camera; previews are selected members",
            "empty": self.empty,
            "candidates": [],
        }

    def add(self, branch, pixels):
        if self.empty:
            return
        if branch == "gfpgan-exposure" or (branch == "gfpgan" and self.anchor is None):
            self.anchor = (branch, pixels)
        proposal = raster_proposal(pixels, self.observation, self.camera, self.alignment)
        result = self.camera.project(self.observation, proposal)
        if not np.array_equal(self.camera.forward(result), self.observation):
            raise ArithmeticError("Projected portrait failed the decoded-pixel replay.")
        key = sha256(result.tobytes()).hexdigest()
        if key in self.seen:
            self.seen[key]["branches"].append(branch)
            return
        filename = branch + ".png"
        record = {"file": filename, "branches": [branch],
                  "sha256": save_png(self.folder / filename, result)}
        self.seen[key] = record
        self.report["candidates"].append(record)

    def finish(self):
        if self.anchor is not None:
            branch, pixels = self.anchor
            for kind, shape in eye_mouth_proposals(pixels):
                self.add(branch + "-" + kind, shape)
        (self.folder / "set.json").write_text(json.dumps(self.report, indent=2), encoding="utf-8")


def load_set(descriptor):
    info = json.loads(descriptor.read_text(encoding="utf-8"))
    if info["model"] != "rgb8-mean-half-up-v1":
        raise ValueError("Unexpected camera model.")
    path = descriptor.parent / info["observation"]
    raw = path.read_bytes()
    if sha256(raw).hexdigest() != info["observation_sha256"]:
        raise ValueError("Observation does not match the declared set.")
    camera = BlockMeanCamera(info["scale"], Fraction(info["gain"]), info["offset"])
    image = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Observation cannot be decoded.")
    return camera, image


def raster_budget_bound(camera, observed, radius):
    """Packing lower bound for covering all compatible rasters in pixel L-infinity."""
    if not 0 <= radius <= 255:
        raise ValueError("Use an integer pixel radius in [0, 255].")
    lower, upper = camera.bounds(observed)
    if np.any(lower > upper):
        return {"scope": "empty camera preimage", "max_channel_error": radius,
                "minimum_representatives_at_least": "0", "independent_swaps": 0}
    gap = 2 * radius + 1
    n = camera.scale ** 2
    independent = 0
    if n >= 4 and gap <= 255:
        # Two pixel values (0,gap) or (255-gap,255) can trade places.
        # The other n-2 bytes can realize any residual integer sum. The two
        # attainable sum intervals overlap for n >= 4. Swaps in different
        # block channels are independent and give pairwise distance > 2r.
        independent = int(np.count_nonzero((upper >= gap) & (lower <= 255*n-gap)))
    return {"scope": "all compatible RGB8 rasters under the stated camera",
            "max_channel_error": radius, "minimum_representatives_at_least": f"2^{independent}",
            "independent_swaps": independent}


def enumerate_page(camera, observed, output, count, start=None, discard_bits=0):
    from portrait_preimage_iterator import cover_blocks
    if (count is not None and count < 1) or output.exists():
        raise ValueError("Use a positive count and a new output directory.")
    lower, upper = camera.bounds(observed)
    first = None
    if start is not None:
        if not camera.contains(observed, start):
            raise ValueError("Starting raster lies outside the stated preimage.")
        first = camera.blocks(start)
    if not 0 <= discard_bits <= 8:
        raise ValueError("discard_bits must be between 0 and 8.")
    stream = (iter(()) if np.any(lower > upper) else
              cover_blocks(lower.ravel(), upper.ravel(), camera.scale ** 2,
                           discard_bits=discard_bits, start=first))
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent) as tmp:
        page = Path(tmp) / "page"
        page.mkdir()
        written = 0
        selected = stream if count is None else islice(stream, count)
        for blocks in selected:
            raster = camera.image(blocks, observed.shape)
            if not np.array_equal(camera.forward(raster), observed):
                raise ArithmeticError("Enumeration produced an invalid raster.")
            save_png(page / f"{written:05d}.png", raster)
            written += 1
        exhausted = next(stream, None) is None
        report = {"rasters_written": written, "all_feasible_cells_written": exhausted,
                  "per_channel_cover_radius_if_exhausted": 2 ** discard_bits - 1}
        (page / "page.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        page.rename(output)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Observation image for define; set.json for all other commands.")
    commands = parser.add_subparsers(dest="command", required=True)
    define = commands.add_parser("define", help="Define the complete preimage directly from a photo.")
    define.add_argument("output", type=Path)
    define.add_argument("--scale", type=int, required=True)
    define.add_argument("--gain", type=Fraction, default=Fraction(1))
    define.add_argument("--offset", type=int, default=0)
    check = commands.add_parser("check", help="Test exact membership of an RGB8 raster.")
    check.add_argument("candidate", type=Path)
    budget = commands.add_parser("bound", help="Lower bound on the number of representatives for all rasters.")
    budget.add_argument("--radius", type=int, required=True, help="Maximum error of any RGB8 channel.")
    window = commands.add_parser("enumerate", help="Write a prefix of the complete finite raster enumeration.")
    window.add_argument("output", type=Path)
    limit = window.add_mutually_exclusive_group(required=True)
    limit.add_argument("--count", type=int, help="Write at most this many representatives.")
    limit.add_argument("--all", action="store_true", help="Materialize the entire finite cover; practical for small sets.")
    window.add_argument("--discard-bits", type=int, choices=range(9), default=0,
                        help="Take one compatible representative per feasible quantization cell.")
    window.add_argument("--start", type=Path, help="Start with this compatible raster and wrap once.")
    args = parser.parse_args()
    if args.command == "define":
        if args.output.exists():
            parser.error("Choose a new output directory.")
        raw = args.source.read_bytes()
        observed = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        if observed is None:
            parser.error("Cannot decode the observation.")
        camera = BlockMeanCamera(args.scale, args.gain, args.offset)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(dir=args.output.parent) as tmp:
            stage = Path(tmp) / "set"
            description = CameraPreviews(stage, camera, observed, {})
            description.report["input_sha256"] = sha256(raw).hexdigest()
            description.finish()
            stage.rename(args.output)
        print(args.output / "set.json")
        raise SystemExit(0)
    camera, observed = load_set(args.source)
    if args.command == "check":
        match = camera.contains(observed, read_raster(args.candidate))
        print(json.dumps({"member": match}))
        raise SystemExit(0 if match else 1)
    if args.command == "bound":
        print(json.dumps(raster_budget_bound(camera, observed, args.radius)))
        raise SystemExit(0)
    seed = read_raster(args.start) if args.start else None
    print(json.dumps(enumerate_page(camera, observed, args.output, args.count, seed, args.discard_bits)))
