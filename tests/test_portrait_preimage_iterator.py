import itertools
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from portrait_preimage_iterator import compatible_blocks, cover_blocks


def bounds(values):
    return np.array(values, dtype=np.int64)


class PortraitPreimageIteratorTest(unittest.TestCase):
    def test_multiple_rows_against_lexicographic_brute_force_and_full_rotation(self):
        lower, upper = bounds([1, 0]), bounds([2, 1])
        expected = [
            values for values in itertools.product(range(3), repeat=4)
            if 1 <= sum(values[:2]) <= 2 and sum(values[2:]) <= 1
        ]
        for pivot in [None, 0, 1, len(expected)//2, len(expected)-1]:
            with self.subTest(pivot=pivot):
                start = None if pivot is None else np.array(expected[pivot], np.uint8).reshape(2, 2)
                result = list(compatible_blocks(lower, upper, 2, start))
                position = pivot or 0
                wanted = expected[position:] + expected[:position]
                self.assertEqual([tuple(row.ravel()) for row in result], wanted)
                self.assertTrue(all(row.dtype == np.uint8 and row.shape == (2, 2) for row in result))

    def test_high_sums_fixed_rows_and_empty_product(self):
        lower, upper = bounds([0, 508, 510]), bounds([0, 510, 510])
        expected = [pair for pair in itertools.product(range(253, 256), repeat=2) if sum(pair) >= 508]
        matrices = list(compatible_blocks(lower, upper, 2))
        self.assertEqual([tuple(m[1]) for m in matrices], expected)
        for matrix in matrices:
            np.testing.assert_array_equal(matrix[[0, 2]], [[0, 0], [255, 255]])
        for empty_start in (None, np.empty((0, 2), dtype=np.uint8)):
            empty = list(compatible_blocks(bounds([]), bounds([]), 2, empty_start))
            self.assertEqual([m.shape for m in empty], [(0, 2)])
        fixed = np.array([[31], [255], [0]], dtype=np.uint8)
        only = list(compatible_blocks(bounds(fixed[:, 0]), bounds(fixed[:, 0]), 1, fixed))
        self.assertEqual(len(only), 1)
        np.testing.assert_array_equal(only[0], fixed)

    def test_early_yields_on_a_real_image_sized_start_and_independent_storage(self):
        start = np.full((3 * 128 * 128, 16), 128, np.uint8)
        lower = np.full(start.shape[0], 2040, np.int64)
        upper = np.full(start.shape[0], 2055, np.int64)
        iterator = compatible_blocks(lower, upper, 16, start)
        first = next(iterator)
        np.testing.assert_array_equal(first, start)
        first[:] = 0
        start[:] = 255
        lower[:] = upper[:] = 0
        second = next(iterator)
        self.assertEqual(int(second[-1, -1]), 129)
        self.assertEqual(int(second[0, 0]), 128)
        self.assertEqual(int(next(iterator)[-1, -1]), 130)
        self.assertEqual(int(second[-1, -1]), 129)

    def test_long_block_crosses_lexicographic_boundary_without_recursion(self):
        n = 4096
        start = np.zeros((1, n), np.uint8)
        start[0, 0] = 1
        it = compatible_blocks(bounds([1]), bounds([1]), n, start)
        np.testing.assert_array_equal(next(it), start)
        self.assertEqual(np.flatnonzero(next(it)).tolist(), [n - 1])
        self.assertEqual(np.flatnonzero(next(it)).tolist(), [n - 2])

    def test_invalid_bounds_shapes_dtypes_and_initial_points(self):
        setups = [
            (bounds([-1]), bounds([0]), 1, None),
            (bounds([2]), bounds([1]), 1, None),
            (bounds([0]), bounds([256]), 1, None),
            (bounds([0]), bounds([1, 2]), 1, None),
            (bounds([[0]]), bounds([[1]]), 1, None),
            (np.array([0.]), bounds([0]), 1, None),
            (bounds([0]), bounds([0]), 0, None),
            (bounds([0]), bounds([0]), 1.5, None),
            (bounds([0]), bounds([0]), True, None),
            (bounds([0]), bounds([0]), 2**62, None),
            (bounds([0]), bounds([1]), 2, np.zeros((2, 1), np.uint8)),
            (bounds([0]), bounds([1]), 2, np.zeros((1, 2), np.int64)),
            (bounds([0]), bounds([1]), 2, np.ones((1, 2), np.uint8)),
        ]
        for lower, upper, length, start in setups:
            with self.subTest(lower=lower, upper=upper, length=length, start=start):
                with self.assertRaises(ValueError):
                    next(compatible_blocks(lower, upper, length, start))

    def test_coverage_of_all_nonempty_bins_against_original_pixel_domain(self):
        pairs = np.indices((256, 256), dtype=np.int16).reshape(2, -1).T
        sums = pairs.sum(axis=1)
        for low, high, bits in [(7, 7, 2), (0, 3, 1), (191, 209, 5), (505, 510, 3), (1, 510, 8)]:
            with self.subTest(low=low, high=high, bits=bits):
                original = pairs[(sums >= low) & (sums <= high)]
                cells = np.unique(original >> bits, axis=0)
                output = list(cover_blocks(bounds([low]), bounds([high]), 2, bits))
                representatives = np.stack(output)[:, 0, :]
                np.testing.assert_array_equal(representatives >> bits, cells)
                totals = representatives.sum(axis=1)
                self.assertTrue(np.all((totals >= low) & (totals <= high)))
                lookup = {tuple(rep >> bits): rep for rep in representatives}
                actual = np.array([lookup[tuple(cell)] for cell in original >> bits])
                self.assertLessEqual(int(np.abs(original - actual.astype(np.int16)).max()), (1 << bits) - 1)

    def test_bin_enumeration_has_complete_cartesian_product_and_starts_with_supplied_image(self):
        lower, upper = bounds([0, 249]), bounds([7, 260])
        bits = 6
        pairs = list(itertools.product(range(4), repeat=2))
        # Every code pair spans an inclusive range of raw sums.
        allowed = [[p for p in pairs if 64 * sum(p) <= hi and 64 * sum(p) + 126 >= lo]
                   for lo, hi in zip(lower, upper)]
        ordered = list(itertools.product(*allowed))
        start = np.uint8([[6, 1], [180, 70]])
        start_codes = tuple(map(tuple, start >> bits))
        pos = ordered.index(start_codes)
        it = cover_blocks(lower, upper, 2, bits, start)
        first = next(it)
        np.testing.assert_array_equal(first, start)
        start[:] = first[:] = 0
        result = [start_codes]
        for rep in it:
            result.append(tuple(map(tuple, rep >> bits)))
            self.assertTrue(np.all((rep.sum(axis=1) >= lower) & (rep.sum(axis=1) <= upper)))
        self.assertEqual(result, ordered[pos:] + ordered[:pos])

    def test_zero_bits_matches_exact_stream_and_eight_bits_strictly_one_cell(self):
        lower, upper = bounds([2, 508]), bounds([3, 510])
        start = np.uint8([[1, 2], [254, 255]])
        for first in (None, start):
            exact = list(compatible_blocks(lower, upper, 2, first))
            reduced = list(cover_blocks(lower, upper, 2, 0, first))
            np.testing.assert_array_equal(exact, reduced)
            single = list(cover_blocks(lower, upper, 2, 8, first))
            self.assertEqual(len(single), 1)
            self.assertTrue(np.all((single[0].sum(axis=1) >= lower) & (single[0].sum(axis=1) <= upper)))
            if first is not None:
                np.testing.assert_array_equal(single[0], first)
        self.assertEqual([m.shape for m in cover_blocks(bounds([]), bounds([]), 2, 3)], [(0, 2)])
        for bad in (-1, 9, 2.5, True):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                next(cover_blocks(lower, upper, 2, bad))
        with self.assertRaises(ValueError):
            next(cover_blocks(lower, upper, 2, 8, np.zeros((2, 2), dtype=np.uint8)))


if __name__ == "__main__":
    unittest.main()
