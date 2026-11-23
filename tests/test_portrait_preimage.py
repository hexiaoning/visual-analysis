from fractions import Fraction
import itertools
import math
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from portrait_preimage import BlockMeanCamera


class BlockMeanCameraTest(unittest.TestCase):
    def test_all_sums_and_output_codes_against_fraction_oracle(self):
        sums = np.arange(4 * 255 + 1, dtype=np.int64)
        # Each row realizes the corresponding 2x2 block sum exactly.
        rows = np.clip(sums[:, None] - 255 * np.arange(4), 0, 255).astype(np.uint8)
        codes = np.arange(256, dtype=np.uint8).reshape(-1, 1, 1).repeat(3, axis=2)
        for gain, offset in [(1, 0), (Fraction(5, 3), -25), (Fraction(-7, 2), 282),
                             (0, 148), (0, -100), (700, 3), (Fraction(10**60 + 1, 10**60), -(10**80))]:
            with self.subTest(gain=gain, offset=offset):
                camera = BlockMeanCamera(2, gain, offset)
                sharp = camera.image(rows.repeat(3, axis=0), (len(sums), 1, 3))
                truth = np.array([min(255, max(0, math.floor(Fraction(gain) * int(s) / 4
                                                                  + offset + Fraction(1, 2))))
                                  for s in sums])
                np.testing.assert_array_equal(camera.forward(sharp)[:, 0, 0], truth)
                lower, upper = camera.bounds(codes)
                self.assertEqual(lower.dtype, np.int64)
                np.testing.assert_array_equal((lower[:, 0, :1] <= sums) & (sums <= upper[:, 0, :1]),
                                              np.arange(256)[:, None] == truth)

    def test_invisible_rational_differences_at_rounding_boundary(self):
        below = BlockMeanCamera(1, Fraction(10**85 - 1, 2 * 10**85))
        above = BlockMeanCamera(1, Fraction(10**85 + 1, 2 * 10**85))
        sharp = np.ones((1, 1, 3), dtype=np.uint8)
        self.assertTrue((below.forward(sharp) == 0).all())
        self.assertTrue((above.forward(sharp) == 1).all())

    def test_layout_membership_and_idempotence(self):
        rng = np.random.default_rng(431)
        camera = BlockMeanCamera(3, Fraction(3, 2), -18)
        raw = rng.integers(0, 256, (6, 12, 3), dtype=np.uint8)
        obs = camera.forward(raw)
        blocks = camera.blocks(raw)
        np.testing.assert_array_equal(blocks[3], raw[:3, 3:6, 0].ravel())
        np.testing.assert_array_equal(camera.image(blocks, obs.shape), raw)
        alternative = camera.image(blocks[:, ::-1], obs.shape)
        self.assertTrue(camera.contains(obs, alternative))
        self.assertFalse(np.array_equal(raw, alternative))
        proposal = rng.integers(0, 256, raw.shape, dtype=np.uint8)
        projected = camera.project(obs, proposal)
        self.assertTrue(camera.contains(obs, projected))
        np.testing.assert_array_equal(camera.project(obs, projected), projected)
        np.testing.assert_array_equal(camera.project(obs, raw), raw)

    def test_least_squares_against_all_feasible_integer_blocks(self):
        # For the tested low observation codes the upper sum is <= 5, so every
        # feasible pixel value is <= 5; enumeration covers the full RGB8 preimage.
        all_rows = np.array(list(itertools.product(range(6), repeat=4)), dtype=np.int64)
        sums = all_rows.sum(axis=1)
        proposals = np.array(list(itertools.product(range(3), repeat=4)), dtype=np.uint8)
        shape = (len(proposals), 1, 3)
        for gain, codes in [(4, range(6)), (1, range(2))]:
            camera = BlockMeanCamera(2, gain)
            sharp = camera.image(np.repeat(proposals, 3, axis=0), shape)
            for code in codes:
                with self.subTest(gain=gain, code=code):
                    obs = np.full(shape, code, np.uint8)
                    result = camera.blocks(camera.project(obs, sharp))[::3]
                    feasible = all_rows[(2 * gain * sums + 4) // 8 == code]
                    squared = (feasible[None] - proposals[:, None].astype(np.int64)) ** 2
                    optimum = squared.sum(axis=2).min(axis=1)
                    np.testing.assert_array_equal(((result.astype(np.int64) - proposals)**2).sum(axis=1), optimum)
                    self.assertTrue(camera.contains(obs, camera.project(obs, sharp)))

    def test_saturation_balancing_and_integer_limited_camera(self):
        camera = BlockMeanCamera(2, Fraction(4), -765)
        # A required sum of 1017 from [0,255,253,251] saturates the last
        # three before the remaining deficit can be assigned to the first.
        proposal = np.tile(np.array([[0, 255, 253, 251]], dtype=np.uint8), (3, 1))
        target = np.full((1, 1, 3), 252, dtype=np.uint8)
        result = camera.blocks(camera.project(target, camera.image(proposal, target.shape)))
        np.testing.assert_array_equal(result, np.tile([252, 255, 255, 255], (3, 1)))
        camera = BlockMeanCamera(1, Fraction(2))
        impossible = np.ones((1, 1, 3), dtype=np.uint8)
        lower, upper = camera.bounds(impossible)
        self.assertTrue(np.all(lower > upper))
        self.assertFalse(camera.contains(impossible, impossible))
        with self.assertRaisesRegex(ValueError, "no RGB8 preimage"):
            camera.project(impossible, impossible)

    def test_invalid_dimensions_types_and_overflow(self):
        for args in [(0,), (True,), (10**15,), (1, .7), (1, 1, .5)]:
            with self.subTest(args=args), self.assertRaises(ValueError):
                BlockMeanCamera(*args)
        cam = BlockMeanCamera(2)
        for bad in [np.zeros((2, 3), dtype=np.uint8), np.zeros((3, 3, 3), dtype=np.uint8),
                    np.zeros((2, 2, 3), dtype=float)]:
            with self.assertRaises(ValueError):
                cam.forward(bad)
        obs = np.zeros((1, 1, 3), dtype=np.uint8)
        self.assertFalse(cam.contains(obs, obs))
        with self.assertRaises(ValueError):
            cam.project(obs, obs)
        with self.assertRaises(ValueError):
            cam.image(np.full((3, 4), 300), obs.shape)


if __name__ == "__main__":
    unittest.main()
