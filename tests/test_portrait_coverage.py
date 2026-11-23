from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from evaluate_portrait_coverage import evaluate


class PortraitCoverageTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.group = {"observation": "native.jpg", "references": {"r": "r.png", "s": "s.png"},
                      "candidates": {"a": "a.png", "b": "b.png"}}
        self.data = {"groups": {"test": self.group}}
        for name in ["native.jpg", "r.png", "s.png", "a.png", "b.png"]:
            (self.root / name).write_bytes(name.encode())

    def test_partial_judgments_and_confirmed_miss(self):
        sheet = evaluate(self.data, self.root, make_sheet=True)
        reviews = sheet["groups"]["test"]["reviews"]
        reviews["r"] = {"a": "same"}
        reviews["s"] = {"a": "different"}
        result = evaluate(sheet, self.root)["groups"]["test"]
        self.assertEqual((result["covered"], result["unreviewed"], result["missed"]), (["r"], ["s"], []))
        self.assertEqual(result["jpeg_replay"], "unchecked")
        self.assertEqual(result["all_known_references"], "unreviewed")
        reviews["s"]["b"] = "different"
        result = evaluate(sheet, self.root)["groups"]["test"]
        self.assertEqual((result["all_known_references"], result["missed"]), ("missed", ["s"]))
        reviews["s"]["b"] = "same"
        self.assertEqual(evaluate(sheet, self.root)["group_counts"]["covered"], 1)
        (self.root / "b.png").write_bytes(b"modified")
        with self.assertRaisesRegex(ValueError, "files or IDs changed"):
            evaluate(sheet, self.root)

    def test_invalid_or_unbound_reviews(self):
        for matrix in [{"r": {"c": "same"}}, {"t": {"a": "same"}},
                       {"r": {"a": "maybe"}}, {"r": {"a": "same"}}]:
            with self.subTest(matrix=matrix):
                self.group["reviews"] = matrix
                with self.assertRaises(ValueError):
                    evaluate(self.data, self.root)

    def test_receipt_is_not_a_replay(self):
        self.group["receipt"] = {"same_jpeg_bytes": True}
        self.assertEqual(evaluate(self.data, self.root)["groups"]["test"]["jpeg_replay"], "unchecked")
        with self.assertRaisesRegex(ValueError, "recipe is required"):
            evaluate(self.data, self.root, require_replay=True)

    def test_exact_replay_from_distinct_originals(self):
        try:
            import cv2
            import numpy as np
        except ImportError:
            self.skipTest("motion JPEG replay requires numpy and OpenCV")
        kernel = np.ones((1, 1), np.float32)
        np.save(self.root / "psf.npy", kernel)
        # A black input has zero response before sensor offset and noise.
        rng = np.random.default_rng(7)
        rng.integers(11, 25)
        rng.uniform(.15, np.pi - .15)
        side = rng.integers(58, 93)
        rng.uniform(.34, .63)
        offset = rng.uniform(.003, .045)
        rng.uniform(.78, 1.14, 3)
        sensor = offset + rng.normal(0, .004, (side, side, 3))
        pixels = np.clip(sensor * 255 + .5, 0, 255).astype(np.uint8)
        ok, jpeg = cv2.imencode(".jpg", pixels, [cv2.IMWRITE_JPEG_QUALITY, 65])
        self.assertTrue(ok)
        (self.root / "native.jpg").write_bytes(jpeg.tobytes())
        sharp = np.zeros((512, 512, 3), np.uint16)
        def write_ref(name, image):
            ok, data = cv2.imencode(".png", image)
            self.assertTrue(ok)
            (self.root / name).write_bytes(data.tobytes())
        write_ref("r.png", sharp)
        sharp[0, 0, 0] = 1
        write_ref("s.png", sharp)
        self.group["replay"] = {"kind": "motion-jpeg65-v1", "seed": 7, "kernel": "psf.npy"}
        report = evaluate(self.data, self.root, require_replay=True)["groups"]["test"]
        self.assertEqual(report["jpeg_replay"], "verified_from_all_references")
        self.assertEqual(report["all_known_references"], "unreviewed")
        write_ref("s.png", np.full_like(sharp, 65535))
        with self.assertRaisesRegex(ValueError, "s: replay from the clear reference differs"):
            evaluate(self.data, self.root)


if __name__ == "__main__":
    unittest.main()
