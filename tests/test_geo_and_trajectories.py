from __future__ import annotations

import random
import unittest

from route_reader.geo import haversine_m, resample_polyline, xy_to_latlon
from route_reader.trajectories import discrete_frechet_m, trajectory_distance_m


class GeoAndTrajectoryTests(unittest.TestCase):
    def test_haversine_remains_accurate_with_small_gps_noise(self) -> None:
        origin = (39.7392, -104.9903)
        target = xy_to_latlon(1000.0, 0.0, *origin)
        rng = random.Random(7)
        estimates = []
        for _ in range(30):
            noisy = xy_to_latlon(1000.0 + rng.gauss(0, 6), rng.gauss(0, 6), *origin)
            estimates.append(haversine_m(*origin, *noisy))
        self.assertAlmostEqual(sum(estimates) / len(estimates), 1000.0, delta=4.0)

    def test_frechet_and_blended_distance_separate_corridors(self) -> None:
        base = resample_polyline(((0, 0), (1000, 50), (2000, 0)), 24)
        noisy_same = tuple((x, y + 12) for x, y in base)
        alternate = resample_polyline(((0, 0), (1000, 600), (2000, 0)), 24)
        self.assertLess(discrete_frechet_m(base, noisy_same), 20)
        self.assertLess(trajectory_distance_m(base, noisy_same), 20)
        self.assertGreater(trajectory_distance_m(base, alternate), 300)


if __name__ == "__main__":
    unittest.main()
