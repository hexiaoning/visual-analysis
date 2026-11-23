import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from explore_portrait_preimage import enumerate_page, load_set, raster_budget_bound, read_raster, save_png
from portrait_preimage import BlockMeanCamera
from restore_portrait_set import save_set


class ExplorePortraitPreimageTest(unittest.TestCase):
    def test_single_photo_to_inspectable_set_and_exhausted_cover(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            observed = np.array([[[28, 79, 211]]], np.uint8)
            photo = root / "photo.png"
            save_png(photo, observed)
            face = np.full((512, 512, 3), 120, np.uint8)
            camera = BlockMeanCamera(2)
            with patch("restore_portrait_set.generate", return_value=iter([("prior", face)])):
                report = save_set(photo, root / "out", root, aligned=True, sensor_camera=camera)
            descriptor = root / "out" / report["sensor_preimage"]
            loaded, sensor = load_set(descriptor)
            np.testing.assert_array_equal(sensor, observed)
            preview = read_raster(descriptor.parent / "prior.png")
            self.assertTrue(loaded.contains(observed, preview))
            self.assertEqual(loaded, camera)
            self.assertEqual(report["open_world_coverage"], "unverified")
            self.assertEqual(raster_budget_bound(camera, observed, 8)["independent_swaps"], 3)
            page = enumerate_page(loaded, sensor, root / "page", 2, preview, discard_bits=8)
            self.assertTrue(page["all_feasible_cells_written"])
            self.assertEqual(page["rasters_written"], 1)
            np.testing.assert_array_equal(read_raster(root / "page" / "00000.png"), preview)
            page = enumerate_page(loaded, sensor, root / "partial", 2, preview)
            self.assertFalse(page["all_feasible_cells_written"])
            self.assertTrue(loaded.contains(sensor, read_raster(root / "partial" / "00001.png")))
            (descriptor.parent / "observation.png").write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "Observation"):
                load_set(descriptor)

    def test_alignment_maps_cropped_faces_back_to_sensor_grid(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            y, x = np.indices((21, 35))
            observed = np.stack((4 * x, 8 * y, 2 * (x + y)), -1).astype(np.uint8)
            photo = root / "frame.png"
            save_png(photo, observed)
            face = np.tile(np.arange(512, dtype=np.uint16)[None, :, None] // 2,
                           (512, 1, 3)).astype(np.uint8)
            transform = {"score": .9, "source_to_face": [[15, 0, -30], [0, 15, 0]]}
            camera = BlockMeanCamera(2)
            with patch("restore_portrait_set.align", return_value=(face, transform)), \
                    patch("restore_portrait_set.generate", return_value=iter([("gfpgan", face)])):
                report = save_set(photo, root / "out", root, sensor_camera=camera)
            desc = root / "out" / report["sensor_preimage"]
            saved = json.loads(desc.read_text())
            branches = [b for candidate in saved["candidates"] for b in candidate["branches"]]
            self.assertEqual(len(branches), 9)
            for row in saved["candidates"]:
                result = read_raster(desc.parent / row["file"])
                self.assertEqual(result.shape, (42, 70, 3))
                np.testing.assert_array_equal(camera.forward(result), observed)

    def test_packing_threshold_and_single_pixel_camera(self):
        observed = np.array([[[0, 128, 255]]], np.uint8)
        self.assertEqual(raster_budget_bound(BlockMeanCamera(1), observed, 0)["independent_swaps"], 0)
        self.assertEqual(raster_budget_bound(BlockMeanCamera(2), observed, 1)["independent_swaps"], 1)
        self.assertEqual(raster_budget_bound(BlockMeanCamera(2), observed, 128)["independent_swaps"], 0)

    def test_empty_declared_preimage_preserves_unrestricted_portrait_outputs(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            photo = root / "source.png"
            observed = np.full((5, 5, 3), 65, np.uint8)
            save_png(photo, observed)
            static = BlockMeanCamera(2, 0, 19)
            with patch("restore_portrait_set.generate", return_value=iter([("prior", observed)])):
                report = save_set(photo, root / "out", root, aligned=True, sensor_camera=static)
            info = json.loads((root / "out" / report["sensor_preimage"]).read_text())
            self.assertTrue(info["empty"])
            self.assertEqual(info["candidates"], [])
            self.assertEqual(len(report["candidates"]), 1)
            page = enumerate_page(static, observed, root / "empty", 1)
            self.assertEqual(page["rasters_written"], 0)
            self.assertTrue(page["all_feasible_cells_written"])
