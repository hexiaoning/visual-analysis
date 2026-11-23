from fractions import Fraction
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from portrait_preimage import BlockMeanCamera


class PortraitSmoothProjectionTest(unittest.TestCase):
    def test_feasible_noncontiguous_image_is_preserved_without_aliasing(self):
        camera = BlockMeanCamera(2, Fraction(3, 4), -9)
        sharp = np.arange(6 * 8 * 3, dtype=np.uint8).reshape(6, 8, 3)[:, :, ::-1]
        observation = camera.forward(sharp)
        out = camera.project_smooth(observation, sharp)
        np.testing.assert_array_equal(out, sharp)
        self.assertFalse(np.shares_memory(out, sharp))

    def test_smooth_projection_reduces_quadrant_seam_and_replays_exactly(self):
        camera = BlockMeanCamera(32)
        proposal = np.full((64, 64, 3), 115, dtype=np.uint8)
        observation = np.uint8([[[80, 87, 94], [150, 156, 120]],
                                [[112, 91, 122], [180, 143, 162]]])
        direct = camera.project(observation, proposal)
        smooth = camera.project_smooth(observation, proposal)
        np.testing.assert_array_equal(camera.forward(smooth), observation)

        def seam_excess(result):
            correction = result.astype(np.int16) - proposal
            excess = []
            for axis in (0, 1):
                normal = np.abs(np.diff(correction, axis=axis))
                boundary = np.take(normal, 31, axis).mean()
                neighbors = np.take(normal, [30, 32], axis).mean()
                excess.append(abs(boundary - neighbors))
            return np.mean(excess)

        self.assertGreater(seam_excess(direct), 20)
        self.assertLess(seam_excess(smooth), 1.5)

    def test_negative_gain_saturation_and_irregular_grid(self):
        rng = np.random.default_rng(19)
        sharp = rng.integers(0, 256, (6, 12, 3), dtype=np.uint8)
        sharp[:3, :3] = (0, 255, 90)
        proposal = 255 - sharp
        camera = BlockMeanCamera(3, Fraction(-2), 280)
        observation = camera.forward(sharp)
        self.assertEqual(observation[0, 0].tolist(), [255, 0, 100])
        projected = camera.project_smooth(observation, proposal)
        self.assertEqual(projected.dtype, np.uint8)
        np.testing.assert_array_equal(camera.forward(projected), observation)
        np.testing.assert_array_equal(proposal, 255 - sharp)

    def test_invalid_observation_or_proposal(self):
        camera = BlockMeanCamera(2, Fraction(0), 17)
        proposal = np.zeros((2, 2, 3), dtype=np.uint8)
        obs = np.full((1, 1, 3), 17, dtype=np.uint8)
        np.testing.assert_array_equal(camera.project_smooth(obs, proposal), proposal)
        with self.assertRaisesRegex(ValueError, "no RGB8 preimage"):
            camera.project_smooth(obs + 1, proposal)
        with self.assertRaisesRegex(ValueError, "dimensions"):
            camera.project_smooth(obs, proposal[:1])
        with self.assertRaisesRegex(ValueError, "uint8"):
            camera.project_smooth(obs, proposal.astype(np.float32))


if __name__ == "__main__":
    unittest.main()
