"""Exact preimages of an RGB8 block-mean, exposure, and quantization operator."""

from dataclasses import dataclass, field
from fractions import Fraction
import math
from numbers import Integral, Rational

import numpy as np


def _shape(shape):
    if (len(shape) != 3 or any(isinstance(v, (bool, np.bool_)) or not isinstance(v, Integral)
                               for v in shape) or min(shape[:2]) < 1 or shape[2] != 3):
        raise ValueError("Expected a nonempty (height, width, 3) image shape")
    return int(shape[0]), int(shape[1])


def _rgb8(image):
    image = np.asarray(image)
    _shape(image.shape)
    if image.dtype != np.uint8:
        raise ValueError("Expected uint8 RGB pixels")
    return image


@dataclass(frozen=True)
class BlockMeanCamera:
    """Round gain * channel mean + offset half-up, then clip to [0, 255]."""

    scale: int
    gain: Fraction = Fraction(1)
    offset: int = 0
    _cuts: np.ndarray = field(init=False, repr=False, compare=False)
    _lower: np.ndarray = field(init=False, repr=False, compare=False)
    _upper: np.ndarray = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        if isinstance(self.scale, bool) or not isinstance(self.scale, Integral) or self.scale < 1:
            raise ValueError("scale must be a positive integer")
        if isinstance(self.offset, bool) or not isinstance(self.offset, Integral):
            raise ValueError("offset must be an integer in RGB8 intensity units")
        if isinstance(self.gain, bool) or not isinstance(self.gain, Rational):
            raise ValueError("gain must be an exact rational, e.g. Fraction('0.75')")
        scale, offset = int(self.scale), int(self.offset)
        gain = Fraction(int(self.gain.numerator), int(self.gain.denominator))
        object.__setattr__(self, "scale", scale)
        object.__setattr__(self, "offset", offset)
        object.__setattr__(self, "gain", gain)
        n = scale ** 2
        last = n * 255
        if last + 1 > np.iinfo(np.int64).max:
            raise ValueError("Block sum exceeds the int64 range")

        # cuts[y] is the first sum producing y or higher. Negative gain uses
        # the complementary sum (255*n - sum) so the cuts remain ascending.
        intercept = offset + (255 * gain if gain < 0 else 0)
        cuts = [0]
        for y in range(1, 256):
            if gain == 0:
                cut = 0 if y <= intercept else last + 1
            else:
                cut = math.ceil((y - Fraction(1, 2) - intercept) * n / abs(gain))
            cuts.append(min(last + 1, max(0, cut)))
        cuts.append(last + 1)
        cuts = np.array(cuts, dtype=np.int64)
        lower, upper = cuts[:-1], cuts[1:] - 1
        if gain < 0:
            lower, upper = last - upper, last - lower
        object.__setattr__(self, "_cuts", cuts[1:-1])
        object.__setattr__(self, "_lower", lower)
        object.__setattr__(self, "_upper", upper)

    def bounds(self, observation):
        """Inclusive sum bounds per observed channel. Lower > upper means infeasible."""
        obs = _rgb8(observation)
        return self._lower[obs], self._upper[obs]

    def blocks(self, sharp):
        """Rows follow observation (y, x, channel); columns follow in-block (y, x)."""
        sharp = _rgb8(sharp)
        h, w, _ = sharp.shape
        s = self.scale
        if h % s or w % s:
            raise ValueError("Sharp height and width must be multiples of scale")
        return sharp.reshape(h // s, s, w // s, s, 3).transpose(0, 2, 4, 1, 3).reshape(-1, s * s)

    def image(self, blocks, observation_shape):
        """Invert the block layout, checking integral RGB8 intensity values."""
        h, w = _shape(observation_shape)
        s = self.scale
        rows = np.asarray(blocks)
        if rows.shape != (h * w * 3, s * s) or rows.dtype.kind not in "iu":
            raise ValueError("Expected one integer block per observation channel")
        if rows.dtype != np.uint8 and (rows.min() < 0 or rows.max() > 255):
            raise ValueError("Block values must be in [0, 255]")
        return rows.astype(np.uint8, copy=False).reshape(h, w, 3, s, s).transpose(0, 3, 1, 4, 2).reshape(h*s, w*s, 3)

    def forward(self, sharp):
        sharp = _rgb8(sharp)
        sums = self.blocks(sharp).sum(axis=1, dtype=np.int64)
        if self.gain < 0:
            sums = self.scale ** 2 * 255 - sums
        levels = np.searchsorted(self._cuts, sums, side="right")
        return levels.astype(np.uint8).reshape(sharp.shape[0] // self.scale, sharp.shape[1] // self.scale, 3)

    def contains(self, observation, sharp):
        obs = _rgb8(observation)
        shape = (obs.shape[0] * self.scale, obs.shape[1] * self.scale, 3)
        sharp = np.asarray(sharp)
        return sharp.shape == shape and sharp.dtype == np.uint8 and np.array_equal(self.forward(sharp), obs)

    def project(self, observation, proposal):
        """Minimize the sum of squared RGB8 pixel changes among all exact preimages."""
        obs = _rgb8(observation)
        proposal = _rgb8(proposal)
        if proposal.shape != (obs.shape[0] * self.scale, obs.shape[1] * self.scale, 3):
            raise ValueError("Proposal dimensions do not match observation and scale")
        lower, upper = self.bounds(obs)
        if np.any(lower > upper):
            raise ValueError("Observation has no RGB8 preimage under this camera")
        rows = self.blocks(proposal).copy()
        sums = rows.sum(axis=1, dtype=np.int64)
        targets = np.clip(sums, lower.ravel(), upper.ravel())
        changed = np.flatnonzero(sums != targets)
        # An extra unit on a pixel already changed by k costs 2*k+1. Allocate
        # whole levels to every unsaturated pixel, then break final ties in raster order.
        batch = max(1, 1_000_000 // rows.shape[1])
        for start in range(0, len(changed), batch):
            ids = changed[start:start + batch]
            values = rows[ids].astype(np.int16)
            delta = targets[ids] - sums[ids]
            direction = np.where(delta > 0, 1, -1).astype(np.int16)[:, None]
            capacities = np.where(direction > 0, 255 - values, values)
            budget = np.abs(delta)
            level = np.zeros((len(ids), 1), dtype=np.int16)
            for step in (128, 64, 32, 16, 8, 4, 2, 1):
                trial = level + step
                fits = np.minimum(capacities, trial).sum(axis=1, dtype=np.int64) <= budget
                level[fits] = trial[fits]
            moves = np.minimum(capacities, level)
            remaining = budget - moves.sum(axis=1, dtype=np.int64)
            available = capacities > level
            moves += available & (np.cumsum(available, axis=1, dtype=np.int64) <= remaining[:, None])
            rows[ids] = values + direction * moves
        return self.image(rows, obs.shape)
