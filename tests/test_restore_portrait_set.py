import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import restore_portrait_set as portrait


class RestorePortraitSetTest(unittest.TestCase):
    def test_continuous_flow_path_sampling_bound(self):
        for peak in (0., 1., 3.2, 7.9, 12.):
            with self.subTest(peak=peak):
                field = np.float32([[[0., peak]]])
                samples = sorted([0., *(t for _, t in portrait.shape_parameters(field))])
                self.assertEqual((samples[0], samples[-1]), (-.65, .65))
                self.assertLessEqual(peak * np.max(np.diff(samples)) / 2, 1.)

    def test_exposure_preserves_original_and_skips_flat_inputs(self):
        dark = np.repeat(np.arange(128, dtype=np.uint8), 4)
        dark = np.broadcast_to(dark[None, :, None], (512, 512, 3)).copy()
        saved = dark.copy()
        bright = portrait.exposure_hypothesis(dark)
        self.assertGreater(np.mean(bright), np.mean(dark))
        np.testing.assert_array_equal(dark, saved)
        self.assertIsNone(portrait.exposure_hypothesis(np.zeros_like(dark)))
        self.assertIsNone(portrait.exposure_hypothesis(np.full_like(dark, 230)))

    def test_geometry_directions_and_identical_inputs(self):
        image = np.zeros((512, 512, 3), np.uint8)
        cv2.circle(image, (220, 240), 20, (255, 255, 255), -1)
        other = cv2.warpAffine(image, np.float32([[1, 0, 4], [0, 1, 0]]), (512, 512))
        images = dict(portrait.shape_branches(image, other))
        def center(pixels):
            moments = cv2.moments(pixels[:, :, 0])
            return moments["m10"] / moments["m00"]
        self.assertGreater(center(images["towards"]), center(image) + 1)
        self.assertLess(center(images["away"]), center(image) - 1)
        for _, result in portrait.shape_branches(image, image):
            np.testing.assert_array_equal(result, image)

    def test_delivery_preserves_all_distinct_proposals_and_failed_runs_are_atomic(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            photo = root / "portrait.png"
            first = np.full((512, 512, 3), 72, np.uint8)
            second = 255 - first
            cv2.imencode(".png", first)[1].tofile(str(photo))

            def generate(crop, _):
                np.testing.assert_array_equal(crop, first)
                yield "one", first
                yield "duplicate", first.copy()
                yield "two", second

            with patch.object(portrait, "generate", generate):
                out = root / "complete"
                result = portrait.save_set(photo, out, root, aligned=True)
                self.assertEqual(result["open_world_coverage"], "unverified")
                self.assertEqual(json.loads((out / "manifest.json").read_text()), result)
                rows = result["candidates"]
                self.assertEqual([r["branches"] for r in rows], [["one", "duplicate"], ["two"]])
                for row in rows:
                    self.assertEqual(hashlib.sha256((out / row["file"]).read_bytes()).hexdigest(),
                                     row["sha256"])
                with self.assertRaises(ValueError):
                    portrait.save_set(photo, out, root, aligned=True)

            def interrupted(crop, _):
                yield "one", crop
                raise RuntimeError("model failed")

            with patch.object(portrait, "generate", interrupted):
                with self.assertRaisesRegex(RuntimeError, "model failed"):
                    portrait.save_set(photo, root / "incomplete", root, aligned=True)
            self.assertEqual({p.name for p in root.iterdir()}, {"portrait.png", "complete"})
