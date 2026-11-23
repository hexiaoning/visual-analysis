"""Generate three portrait hypotheses from an aligned 512px BGR crop on CPU.

SwinIR uses the DiffBIR face checkpoint, followed by the DicFace BFR-100k
decoder. Supply the trusted local source trees and checkpoints explicitly.
Requires PyTorch 2.3+, OpenCV, NumPy, einops and timm. The SwinIR source
directory may have its own deps/ directory containing timm.
"""
import builtins
from contextlib import redirect_stdout
from hashlib import sha256
import importlib.util
import logging
from pathlib import Path
import sys
from types import SimpleNamespace

import cv2
import numpy as np
import torch


SWINIR_SHA256 = "62fb6605702e6ad35712299a23cc19ea7cd2fb4bbd50a8080fcefa8ad2620c71"
DICFACE_SHA256 = "fb14b9d74f6b629d0b3f1f824cc7213eaa68a4759fbf2aa6a559a58cd7e297af"
ROUTES = {"diffbir-face-dicface": 0., "mixed-35": .35, "mixed-60": .60}


def _weights(path, expected):
    digest = sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(2 ** 20), b""):
            digest.update(chunk)
    if digest.hexdigest() != expected:
        raise ValueError(f"Unexpected checkpoint SHA-256: {path}")
    return torch.load(path, map_location="cpu", weights_only=True, mmap=True)


def _source(path, aliases=None):
    spec = importlib.util.spec_from_file_location("portrait_prior_" + path.stem, path)
    module = importlib.util.module_from_spec(spec)
    if aliases:
        # DicFace imports its vendored BasicSR by absolute name. Bind only
        # these imports, leaving an already loaded BasicSR installation alone.
        def resolve(name, globals=None, locals=None, fromlist=(), level=0):
            if name in aliases and level == 0:
                return aliases[name]
            return builtins.__import__(name, globals, locals, fromlist, level)
        module.__dict__["__builtins__"] = {**vars(builtins), "__import__": resolve}
    spec.loader.exec_module(module)
    return module


def _swinir(directory, checkpoint):
    deps = str(directory / "deps")
    sys.path.insert(0, deps)
    try:
        architecture = _source(directory / "swinir.py")
    finally:
        sys.path.remove(deps)
    model = architecture.SwinIR(
        img_size=64, patch_size=1, in_chans=3, embed_dim=180, depths=[6] * 8,
        num_heads=[6] * 8, window_size=8, mlp_ratio=2, sf=8, img_range=1.,
        upsampler="nearest+conv", resi_connection="1conv", unshuffle=True, unshuffle_scale=8)
    payload = _weights(checkpoint, SWINIR_SHA256)
    state = {k.removeprefix("module."): v for k, v in payload.get("state_dict", payload).items()}
    model.load_state_dict(state, strict=True, assign=True)
    return model.eval().requires_grad_(False)


def _dicface(directory, checkpoint):
    base = directory / "basicsr"
    imports = {"basicsr.utils": SimpleNamespace(get_root_logger=lambda: logging.getLogger("dicface")),
               "basicsr.utils.registry": _source(base / "utils" / "registry.py")}
    imports["basicsr.archs.vqgan_arch"] = _source(base / "archs" / "vqgan_arch.py", imports)
    architecture = _source(base / "archs" / "dir_dist_codeformer_multiscale_arch.py", imports)
    # The upstream constructor prints training-freeze messages to stdout.
    with redirect_stdout(sys.stderr), torch.device("meta"):
        model = architecture.TemporalCodeFormerDirDistMultiScale(
            dim_embed=512, n_head=8, n_layers=9, codebook_size=1024,
            connect_list=["32", "64", "128", "256"], frame_length=5)
    model.load_state_dict(_weights(checkpoint, DICFACE_SHA256)["params_ema"], strict=True, assign=True)
    torch.nn.utils.convert_conv2d_weight_memory_format(model, torch.channels_last)
    return model.eval().requires_grad_(False)


def _distribution(model, stage):
    blob = cv2.dnn.blobFromImage(stage, 1 / 127.5, (512, 512), (127.5,) * 3, swapRB=True)
    value = torch.from_numpy(blob).to(memory_format=torch.channels_last)
    skips = {}
    cuts = {model.fuse_encoder_block[k]: k for k in model.connect_list}
    for index, block in enumerate(model.encoder.blocks):
        value = block(value)
        if index in cuts:
            skips[cuts[index]] = value
    combined = model.multiscale(skips["64"], skips["32"], value)
    tokens = model.feat_emb(combined.flatten(2).permute(2, 0, 1)).contiguous()
    position = model.position_emb[:, None, :]
    temporal = model.position_emb_temporal[2:3, None, :].repeat(1, tokens.shape[0], 1)
    for spatial_layer, temporal_layer in zip(model.ft_layers, model.dir_dist_layers):
        tokens = spatial_layer(tokens, query_pos=position)
        tokens = temporal_layer(tokens, query_pos=temporal, frame_length=1, batch_size=1)
    alpha = model.softplus_layer(model.idx_pred_layer(tokens).permute(1, 0, 2)) + .01
    return skips, alpha / alpha.sum(dim=-1, keepdim=True)


def _decode(model, skips, weights):
    value = (weights @ model.quantize.embedding.weight).reshape(1, 16, 16, 256).permute(0, 3, 1, 2)
    cuts = {model.fuse_generator_block[k]: k for k in model.connect_list}
    for index, block in enumerate(model.generator.blocks):
        value = block(value)
        if index in cuts:
            level = cuts[index]
            value = model.fuse_convs_dict[level](skips[level], value, 1.)
    if value.shape != (1, 3, 512, 512) or not torch.isfinite(value).all():
        raise ValueError("Invalid DicFace output")
    bgr = value[0, [2, 1, 0]].permute(1, 2, 0).numpy()
    return np.rint((np.clip(bgr, -1, 1) + 1) * 127.5).astype(np.uint8)


class SparsePortraitPrior:
    def __init__(self, swinir_dir, dicface_dir, dicface_weights, swinir_weights=None):
        directory = Path(swinir_dir)
        self.swinir = _swinir(directory, swinir_weights or directory / "face-swinir-v1.ckpt")
        self.dicface = _dicface(Path(dicface_dir), dicface_weights)

    @torch.inference_mode()
    def generate(self, crop):
        """Return full-code, 35% and 60% top-four mixtures in that order."""
        if crop.shape != (512, 512, 3) or crop.dtype != np.uint8:
            raise ValueError("Expected an aligned 512x512 8-bit BGR crop")
        blob = cv2.dnn.blobFromImage(crop, 1 / 255., swapRB=True)
        prediction = self.swinir(torch.from_numpy(blob))
        if prediction.shape != (1, 3, 512, 512) or not torch.isfinite(prediction).all():
            raise ValueError("Invalid SwinIR output")
        bgr = prediction[0, [2, 1, 0]].permute(1, 2, 0).numpy()
        stage = np.rint(np.clip(bgr, 0, 1) * 255).astype(np.uint8)
        skips, weights = _distribution(self.dicface, stage)
        values, indices = weights.topk(4, dim=-1)
        top_four = torch.zeros_like(weights).scatter_(-1, indices, values / values.sum(dim=-1, keepdim=True))
        return {name: _decode(self.dicface, skips, torch.lerp(weights, top_four, amount))
                for name, amount in ROUTES.items()}
