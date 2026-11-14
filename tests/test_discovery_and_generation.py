from __future__ import annotations

import unittest

from route_reader.discovery import discover_recurring_journeys
from route_reader.ingestion import segment_trips
from route_reader.synthetic import generate_canonical_dataset


class DiscoveryAndGenerationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.dataset = generate_canonical_dataset()
        cls.trips = segment_trips(cls.dataset.raw_points)

    def test_generation_is_deterministic_and_observations_have_no_labels(self) -> None:
        repeat = generate_canonical_dataset()
        self.assertEqual(self.dataset.raw_points, repeat.raw_points)
        self.assertEqual(self.dataset.truths, repeat.truths)
        self.assertEqual(len(self.dataset.truths), 50)
        self.assertEqual(
            set(self.dataset.raw_points[0].__dataclass_fields__),
            {"timestamp", "latitude", "longitude", "accuracy_m"},
        )

    def test_trip_segmentation_survives_stops_drops_noise_and_outliers(self) -> None:
        self.assertEqual(len(self.trips), 50)
        self.assertTrue(all(trip.duration_seconds > 400 for trip in self.trips))
        self.assertTrue(all(len(trip.points) > 35 for trip in self.trips))

    def test_recurring_places_and_od_are_discovered(self) -> None:
        discovery = discover_recurring_journeys(self.trips[:40])
        self.assertEqual(len(discovery.places), 2)
        self.assertEqual(len(discovery.od_pairs), 1)
        self.assertEqual(len(discovery.od_pairs[0].trip_ids), 40)


if __name__ == "__main__":
    unittest.main()
