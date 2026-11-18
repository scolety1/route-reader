import unittest

from route_reader.geo import resample_polyline


class ResampleBoundaryTests(unittest.TestCase):
    def test_empty_polyline_is_explicitly_rejected(self):
        for points in ([], ()):
            with self.subTest(points=points):
                with self.assertRaisesRegex(ValueError, "empty polyline"):
                    resample_polyline(points)

    def test_single_point_preserves_stationary_trace(self):
        self.assertEqual(resample_polyline([(2.0, 3.0)], 3), ((2.0, 3.0),) * 3)

    def test_repeated_points_preserve_stationary_trace(self):
        self.assertEqual(resample_polyline([(2.0, 3.0)] * 4, 3), ((2.0, 3.0),) * 3)

    def test_invalid_sample_count_keeps_existing_contract(self):
        with self.assertRaisesRegex(ValueError, "count must be at least two"):
            resample_polyline([(2.0, 3.0)], 1)
