"""Evaluate human judgments against multiple known original portraits per observation.

Manifest paths are relative to the manifest (absolute paths also work). Example:
{"groups": {"case1": {
  "observation": "native.jpg",
  "references": {"original": "original.png", "alternative": "alternative.png"},
  "candidates": {"a": "candidate-a.png", "b": "candidate-b.png"},
  "replay": {"kind": "motion-jpeg65-v1", "seed": 20322422, "kernel": "psf.npy"}
}}}

Run with --review-sheet to produce a hash-bound, editable copy on stdout. Keep it
beside the manifest if using relative paths. For each reference and candidate,
replace "unreviewed" with "same" or "different" using the agreed visual criterion.
Run again on the edited sheet for the report. Missing judgments are unreviewed.
Reports describe the listed references only. Optional replay checks require
numpy and opencv-python. The motion recipe uses an original native JPEG and
the frozen motion kernel. For block-average fixtures, use a native PNG and
{"kind": "block-mean-rgb8-v1", "scale": 16}. Both references and candidates
must match every decoded observation pixel under that recipe.
"""
import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys


def fingerprint(data):
    return sha256(data).hexdigest()


def unique_mapping(pairs):
    result = dict(pairs)
    if len(result) != len(pairs):
        raise ValueError("Duplicate key in JSON")
    return result


def image_files(root, mapping, minimum, name):
    if not isinstance(mapping, dict) or len(mapping) < minimum:
        raise ValueError(f"Expected at least {minimum} {name}")
    files = {}
    for key, value in mapping.items():
        if not key.strip() or not isinstance(value, str) or not value.strip():
            raise ValueError(f"Expected an ID and path for each of {name}")
        files[key] = root / value
    return files


def verify_motion_replay(recipe, root, references, observed):
    if not isinstance(recipe, dict) or set(recipe) != {"kind", "seed", "kernel"}:
        raise ValueError("Replay requires kind, seed and frozen kernel")
    if recipe["kind"] != "motion-jpeg65-v1" or type(recipe["seed"]) is not int or recipe["seed"] < 0:
        raise ValueError("Expected motion-jpeg65-v1 with a nonnegative integer seed")
    import cv2
    import numpy as np

    kernel = np.load(root / recipe["kernel"], allow_pickle=False)
    if (not isinstance(kernel, np.ndarray) or kernel.dtype != np.float32 or kernel.ndim != 2
            or any(s % 2 != 1 or s > 512 for s in kernel.shape)
            or not np.all(kernel >= 0) or not np.isclose(kernel.sum(), 1)):
        raise ValueError("Expected an odd-sized, normalized float32 motion kernel")
    rng = np.random.default_rng(recipe["seed"])
    # The capture drew motion length and direction before saving the kernel.
    rng.integers(11, 25)
    rng.uniform(.15, np.pi - .15)
    size = rng.integers(58, 93)
    gain, floor = rng.uniform(.34, .63), rng.uniform(.003, .045)
    tint = rng.uniform(.78, 1.14, (1, 1, 3)).astype(np.float32)
    noise = rng.normal(0, .004, (size, size, 3))
    seen_pixels = set()
    for key, path in references.items():
        sharp = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        if sharp is None or sharp.shape != (512, 512, 3) or sharp.dtype not in (np.uint8, np.uint16):
            raise ValueError(f"{key}: expected a 512x512 8- or 16-bit BGR reference")
        values = sharp.astype(np.float32) / np.iinfo(sharp.dtype).max
        pixels = fingerprint(values.tobytes())
        if pixels in seen_pixels:
            raise ValueError(f"{key}: repeated reference pixels")
        seen_pixels.add(pixels)
        blurred = cv2.filter2D(values, -1, kernel, borderType=cv2.BORDER_REFLECT_101)
        reduced = cv2.resize(blurred, (size, size), interpolation=cv2.INTER_AREA)
        sensor = reduced * gain * tint + floor + noise
        quantized = np.clip(sensor * 255 + .5, 0, 255).astype(np.uint8)
        ok, jpeg = cv2.imencode(".jpg", quantized, [cv2.IMWRITE_JPEG_QUALITY, 65])
        if not ok or jpeg.tobytes() != observed:
            raise ValueError(f"{key}: replay from the clear reference differs from the observation JPEG bytes")


def verify_block_replay(recipe, references, candidates, observed):
    if set(recipe) != {"kind", "scale"}:
        raise ValueError("Block replay requires kind and scale")
    import cv2
    import numpy as np
    from portrait_preimage import BlockMeanCamera

    camera = BlockMeanCamera(recipe["scale"])

    def decode(pixels, label):
        image = cv2.imdecode(np.frombuffer(pixels, np.uint8), cv2.IMREAD_UNCHANGED)
        if image is None or image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"{label}: expected an 8-bit three-channel image")
        return image

    observation = decode(observed, "observation")
    for role, files in (("reference", references), ("candidate", candidates)):
        seen = set()
        for key, path in files.items():
            sharp = decode(path.read_bytes(), f"{role} {key}")
            if not camera.contains(observation, sharp):
                raise ValueError(f"{role} {key}: block replay differs from the observation pixels")
            if role == "reference":
                digest = fingerprint(sharp.tobytes())
                if digest in seen:
                    raise ValueError(f"reference {key}: repeated reference pixels")
                seen.add(digest)


def evaluate_group(group, root, require_replay, make_sheet):
    observation = (root / group["observation"]).read_bytes()
    refs = image_files(root, group["references"], 2, "references")
    candidates = image_files(root, group["candidates"], 1, "candidates")
    hashes = {role: {k: fingerprint(p.read_bytes()) for k, p in files.items()}
              for role, files in (("references", refs), ("candidates", candidates))}
    hashes["observation"] = fingerprint(observation)
    if len(set(hashes["references"].values())) != len(refs):
        raise ValueError("Repeated reference files")
    recipe = group.get("replay")
    if recipe is not None and not isinstance(recipe, dict):
        raise ValueError("Expected a replay recipe object")
    recipe_kind = recipe.get("kind") if recipe is not None else None
    if recipe_kind == "block-mean-rgb8-v1":
        hashes["block_replay_recipe"] = fingerprint(json.dumps(recipe, sort_keys=True).encode())
    binding = fingerprint(json.dumps(hashes, sort_keys=True).encode())
    previous = group.get("review_set_sha256")
    if previous is not None and previous != binding:
        raise ValueError("Reviewed files or IDs changed")
    if recipe_kind == "block-mean-rgb8-v1":
        verify_block_replay(recipe, refs, candidates, observation)
    elif recipe is not None:
        verify_motion_replay(recipe, root, refs, observation)
    elif require_replay:
        raise ValueError("A replay recipe is required")

    reviews = group.get("reviews", {})
    if not isinstance(reviews, dict) or set(reviews) - refs.keys():
        raise ValueError("Invalid review matrix or unknown reference ID")
    matrix, by_status = {}, {"covered": [], "missed": [], "unreviewed": []}
    for ref in refs:
        row = reviews.get(ref, {})
        if not isinstance(row, dict) or set(row) - candidates.keys():
            raise ValueError(f"{ref}: invalid review row or unknown candidate ID")
        cells = {k: row.get(k, "unreviewed") for k in candidates}
        if any(v not in ("same", "different", "unreviewed") for v in cells.values()):
            raise ValueError(f"{ref}: invalid human judgment")
        if previous is None and any(v != "unreviewed" for v in cells.values()):
            raise ValueError("Record judgments on a hash-bound --review-sheet")
        status = ("covered" if "same" in cells.values() else
                  "unreviewed" if "unreviewed" in cells.values() else "missed")
        by_status[status].append(ref)
        matrix[ref] = cells
    if make_sheet:
        return {**group, "review_set_sha256": binding, "reviews": matrix}
    return {"jpeg_replay": "verified_from_all_references" if recipe_kind == "motion-jpeg65-v1" else "unchecked",
            "block_replay": "verified_from_all_references_and_candidates" if recipe_kind == "block-mean-rgb8-v1" else "unchecked",
            "all_known_references": ("missed" if by_status["missed"] else
                                     "unreviewed" if by_status["unreviewed"] else "covered"),
            **by_status, "reviews": matrix, "files_sha256": hashes}


def evaluate(manifest, root, require_replay=False, make_sheet=False):
    groups = manifest["groups"]
    if not isinstance(groups, dict) or not groups:
        raise ValueError("Expected at least one observation group")
    results = {}
    for key, group in groups.items():
        try:
            results[key] = evaluate_group(group, root, require_replay, make_sheet)
        except (KeyError, TypeError, ValueError, OSError) as exc:
            raise ValueError(f"{key}: {exc}") from exc
    if make_sheet:
        return {**manifest, "groups": results}
    return {"scope": "Listed references and candidates only", "groups": results,
            "group_counts": {s: sum(r["all_known_references"] == s for r in results.values())
                             for s in ("covered", "missed", "unreviewed")}}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--review-sheet", action="store_true", help="emit an editable judgment matrix")
    parser.add_argument("--require-replay", action="store_true", help="require exact replay under each group's recipe")
    args = parser.parse_args()
    try:
        data = json.loads(args.manifest.read_text(encoding="utf-8-sig"), object_pairs_hook=unique_mapping)
        report = evaluate(data, args.manifest.parent, args.require_replay, args.review_sheet)
    except (ImportError, KeyError, TypeError, ValueError, OSError) as exc:
        parser.exit(2, f"{exc}\n")
    print(json.dumps(report, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
