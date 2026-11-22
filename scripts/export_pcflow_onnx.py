"""Export PCFlow's official 512-pixel blind face model with five Euler steps.

Conversion uses the exact author checkpoints and TAESD3 decoder. The exported
input and output match the other restorers' RGB float32 NCHW range of [-1, 1].
"""
from __future__ import annotations

import argparse
import importlib
from pathlib import Path

import numpy as np
import onnxruntime as ort
from safetensors.torch import load_file
import torch

from export_gfpgan_onnx import package


class PCFlowEuler(torch.nn.Module):
    def __init__(self, encoder, flow, decoder):
        super().__init__()
        self.encoder = encoder
        self.flow = flow
        self.decoder = decoder
        random = torch.Generator().manual_seed(1024)
        self.register_buffer("initial_noise", torch.randn(1, 16, 64, 64, generator=random))
        self.register_buffer("times", torch.arange(5, dtype=torch.float32) / 5)

    def forward(self, image):
        condition = self.encoder((image + 1) * 0.5)
        latent = self.initial_noise
        for index in range(5):
            velocity = self.flow(self.times[index:index + 1], torch.cat([latent, condition], dim=1))
            latent = latent + 0.2 * velocity
        return self.decoder(latent).clamp(0, 1) * 2 - 1


def load_pcflow(source: Path, directory: Path):
    package("models", source / "models")
    lunet = importlib.import_module("models.lunet")
    taesd = importlib.import_module("models.taesd")
    flow = lunet.LUnetAdapter(in_channels=32, out_channels=16, hid_channels=128, ch_mult=[1, 2, 1, 2], n_mid_blocks=3)
    flow.load_state_dict(torch.load(directory / "pcflow-model-500.pth", map_location="cpu", weights_only=True))
    encoder = taesd.Encoder(16)
    parameters = torch.load(directory / "pcflow-lq-encoder.pth", map_location="cpu", weights_only=True)
    prefix = "model.encoder."
    encoder.load_state_dict({key.removeprefix(prefix): value
                             for key, value in parameters.items() if key.startswith(prefix)})
    decoder = taesd.Decoder(16)
    prefix = "decoder."
    decoder.load_state_dict({key.removeprefix(prefix): value
                             for key, value in load_file(directory / "taesd3.safetensors").items()
                             if key.startswith(prefix)})
    return PCFlowEuler(encoder, flow, decoder).eval()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("weights_directory", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(2)
    model = load_pcflow(args.source, args.weights_directory)
    example = torch.rand(1, 3, 512, 512, generator=torch.Generator().manual_seed(19)) * 2 - 1
    with torch.inference_mode():
        expected = model(example).numpy()
        torch.onnx.export(model, example, args.output, opset_version=17, input_names=["face"],
                          output_names=["restored"], dynamo=False, do_constant_folding=False)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    actual = ort.InferenceSession(str(args.output), options, providers=["CPUExecutionProvider"]).run(
        ["restored"], {"face": example.numpy()},
    )[0]
    mean_error = float(np.abs(expected - actual).mean())
    if mean_error > 0.002:
        raise ValueError(f"PCFlow ONNX parity failed, mean absolute error={mean_error:.6f}")
    print(f"PCFlow ONNX/PyTorch mean absolute error: {mean_error:.8f}", flush=True)


if __name__ == "__main__":
    main()
