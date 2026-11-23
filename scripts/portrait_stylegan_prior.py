"""Sample the official NVIDIA StyleGAN2 FFHQ 512 network on CPU.

Supply a trusted NVlabs/stylegan3 source checkout (tested commit
c233a919a6faee6e36a316ddd4eddababad1adf9) and NVIDIA NGC's version 1
stylegan2-ffhq-512x512.pkl. The checksum below pins that exact binary.
Code and model are loaded locally. The caller controls PyTorch's CPU threads.
"""

from collections.abc import Iterator
from hashlib import sha256
import importlib
from numbers import Integral
from pathlib import Path
import sys

import numpy as np


STYLEGAN_WEIGHTS_SHA256 = "d2b1c92f41ce8a64c55f7a75ef06c4c0eef9e17b1eb29aae8c10fb37b3e60478"
OFFICIAL_WEIGHTS_URL = "https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan2/versions/1/files/stylegan2-ffhq-512x512.pkl"


def _load_generator(source: Path, weights: Path):
    source = source.resolve()
    for relative in ("legacy.py", "torch_utils/persistence.py", "dnnlib/__init__.py"):
        if not (source / relative).is_file():
            raise ValueError(f"Missing {relative} in the NVlabs/stylegan3 source directory")
    # The NVIDIA loader and pickle use absolute Python imports. Reusing another
    # checkout's project modules would silently mix two implementations.
    for name in ("legacy", "dnnlib", "torch_utils"):
        module = sys.modules.get(name)
        if module is not None and not Path(getattr(module, "__file__", "") or "").resolve().is_relative_to(source):
            raise ValueError(f"Already loaded {name} outside the requested NVIDIA source directory")
    with weights.open("rb") as handle:
        digest = sha256()
        for chunk in iter(lambda: handle.read(2**20), b""):
            digest.update(chunk)
        if digest.hexdigest() != STYLEGAN_WEIGHTS_SHA256:
            raise ValueError("Unexpected SHA-256 for the official StyleGAN2 FFHQ 512 checkpoint")
        handle.seek(0)
        sys.path.insert(0, str(source))
        try:
            loader = importlib.import_module("legacy")
            model = loader.load_network_pkl(handle)["G_ema"].eval().requires_grad_(False).cpu()
        finally:
            sys.path.remove(str(source))
    if model.c_dim != 0 or model.img_resolution != 512:
        raise ValueError("Expected the unconditional 512px FFHQ generator")
    return model


def generate_stylegan_faces(source: Path, weights: Path, first_seed: int, count: int) -> Iterator[tuple[int, np.ndarray]]:
    """Yield (seed, contiguous BGR uint8 image) using fixed noise and truncation 0.75."""
    if any(isinstance(n, bool) or not isinstance(n, Integral) for n in (first_seed, count)):
        raise ValueError("first_seed and count must be integers")
    first_seed, count = int(first_seed), int(count)
    if not 0 <= first_seed < 2**32 or count < 0 or first_seed + count > 2**32:
        raise ValueError("Requested seeds must be in [0, 2**32) and count must be nonnegative")
    if count == 0:
        return
    import torch

    model = _load_generator(Path(source), Path(weights))
    labels = torch.zeros((1, 0))
    for seed in range(first_seed, first_seed + count):
        with torch.inference_mode():
            z = np.random.RandomState(seed).randn(1, model.z_dim).astype(np.float32)
            rgb = model(torch.from_numpy(z), labels, truncation_psi=.75, noise_mode="const", force_fp32=True)
            if rgb.shape != (1, 3, 512, 512) or not torch.isfinite(rgb).all():
                raise ValueError(f"Invalid StyleGAN2 output for seed {seed}")
            pixels = (rgb[0].permute(1, 2, 0) * 127.5 + 128).clamp(0, 255).to(torch.uint8).numpy()
            bgr = np.ascontiguousarray(pixels[:, :, ::-1])
        yield seed, bgr
