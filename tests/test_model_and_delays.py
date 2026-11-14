from __future__ import annotations

import unittest
from datetime import timedelta

from route_reader.model import TemporalRouteModel
from route_reader.pipeline import learn_from_raw_points
from route_reader.synthetic import generate_canonical_dataset


class ModelAndDelayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.dataset = generate_canonical_dataset()
        cls.learned = learn_from_raw_points(cls.dataset.raw_points)

    def test_route_clustering_recovers_four_noise_tolerant_variants(self) -> None:
        self.assertEqual(len(self.learned.route_clusters), 4)
        self.assertEqual(sorted(len(cluster.trip_ids) for cluster in self.learned.route_clusters), [8, 8, 8, 16])

    def test_temporal_prediction_changes_preferred_route(self) -> None:
        date = self.learned.test_trips[0].departure + timedelta(days=14)
        early = date.replace(hour=15, minute=12, second=0, microsecond=0)
        peak = date.replace(hour=15, minute=41, second=0, microsecond=0)
        early_best = self.learned.model.predict_all(early).best.route_cluster_id
        peak_best = self.learned.model.predict_all(peak).best.route_cluster_id
        self.assertNotEqual(early_best, peak_best)

    def test_predictions_expose_confidence_and_evidence(self) -> None:
        query = self.learned.test_trips[0].departure
        prediction = self.learned.model.predict_all(query).best
        self.assertIn(prediction.confidence, {"low", "medium", "high"})
        self.assertGreater(prediction.interval_high_seconds, prediction.interval_low_seconds)
        self.assertGreater(prediction.effective_sample_size, 0)
        self.assertTrue(prediction.delay_evidence)

    def test_sparse_history_uses_low_confidence_fallback(self) -> None:
        trips = self.learned.train_trips[:2]
        assignments = {trip.trip_id: "sparse" for trip in trips}
        model = TemporalRouteModel.fit(trips, assignments, ())
        prediction = model.predict_all(trips[-1].departure).best
        self.assertEqual(prediction.confidence, "low")
        self.assertIn("fallback", prediction.delay_evidence[0])

    def test_delay_zone_detection_finds_a_strong_temporal_pattern(self) -> None:
        self.assertGreaterEqual(len(self.learned.delay_zones), 4)
        strongest = max(self.learned.delay_zones, key=lambda zone: zone.peak_median_delay_seconds)
        self.assertGreaterEqual(strongest.affected_trips, 3)
        self.assertGreater(strongest.peak_median_delay_seconds, 100)
        self.assertGreater(strongest.peak_median_delay_seconds, strongest.median_delay_seconds * 2)


if __name__ == "__main__":
    unittest.main()
