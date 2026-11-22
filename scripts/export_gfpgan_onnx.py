"""Export the official GFPGAN v1.4 clean network and check ONNX parity.

Run on the build machine using the pinned upstream GFPGAN checkout and BasicSR
installation. The exported file is the only runtime artifact from PyTorch.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.util
from pathlib import Path
import sys
from types import ModuleType

import numpy as np
import onnxruntime as ort
import torch


WEIGHTS_SHA256 = "e2cd4703ab14f4d01fd1383a8a8b266f9a5833dacee8e6a79d3bf21a1b6be5ad"


def package(name: str, directory: Path) -> None:
    """Use the upstream architecture modules without discovery of training code."""
    module = ModuleType(name)
    module.__path__ = [str(directory)]
    sys.modules[name] = module


def load_model(source: Path, weights: Path) -> torch.nn.Module:
    with weights.open("rb") as stream:
        if hashlib.file_digest(stream, "sha256").hexdigest() != WEIGHTS_SHA256:
            raise ValueError("GFPGAN v1.4 source checkpoint failed SHA-256 verification")
    spec = importlib.util.find_spec("basicsr")
    if spec is None or spec.origin is None:
        raise RuntimeError("Install BasicSR in the export environment")
    basic_root = Path(spec.origin).parent
    package("basicsr", basic_root)
    package("basicsr.archs", basic_root / "archs")
    package("basicsr.utils", basic_root / "utils")
    sys.modules["basicsr.utils"].get_root_logger = importlib.import_module("basicsr.utils.logger").get_root_logger
    package("gfpgan", source / "gfpgan")
    package("gfpgan.archs", source / "gfpgan" / "archs")
    arch = importlib.import_module("gfpgan.archs.gfpganv1_clean_arch")
    network = arch.GFPGANv1Clean(
        out_size=512, num_style_feat=512, channel_multiplier=2, decoder_load_path=None,
        fix_decoder=False, num_mlp=8, input_is_latent=True, different_w=True, narrow=1,
        sft_half=True,
    )
    checkpoint = torch.load(weights, map_location="cpu", weights_only=True)
    network.load_state_dict(checkpoint["params_ema"], strict=True)
    return network.eval()


class DeterministicDecoder(torch.nn.Module):
    def __init__(self, network: torch.nn.Module):
        super().__init__()
        self.network = network

    def forward(self, face):
        return self.network(face, return_rgb=False, randomize_noise=False)[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("upstream", type=Path)
    parser.add_argument("weights", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(2)
    network = DeterministicDecoder(load_model(args.upstream, args.weights)).eval()
    sample = torch.rand(1, 3, 512, 512, generator=torch.Generator().manual_seed(42)) * 2 - 1
    with torch.inference_mode():
        expected = network(sample).numpy()
        torch.onnx.export(
            network, sample, args.output, opset_version=17, input_names=["face"],
            output_names=["restored"], dynamo=False,
        )
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    result = ort.InferenceSession(str(args.output), options, providers=["CPUExecutionProvider"]).run(
        ["restored"], {"face": sample.numpy()},
    )[0]
    mean_error = float(np.abs(expected - result).mean())
    if mean_error > 0.001:
        raise ValueError(f"GFPGAN ONNX parity failed, mean absolute error={mean_error:.6f}")
    print(f"GFPGAN ONNX/PyTorch mean absolute error: {mean_error:.8f}", flush=True)
    with args.output.open("rb") as stream:
        print(f"{hashlib.file_digest(stream, 'sha256').hexdigest()}  {args.output.stat().st_size}", flush=True)


if __name__ == "__main__":
    main()
