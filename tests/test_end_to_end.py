from __future__ import annotations

import inspect
import unittest

import route_reader.delays
import route_reader.discovery
import route_reader.ingestion
import route_reader.model
import route_reader.pipeline
import route_reader.trajectories
from route_reader.evaluation import evaluate
from route_reader.pipeline import chronological_split, learn_from_raw_points
from route_reader.synthetic import generate_canonical_dataset


class EndToEndFixtureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.dataset = generate_canonical_dataset()
        cls.learned = learn_from_raw_points(cls.dataset.raw_points)
        cls.result = evaluate(cls.learned, cls.dataset)

    def test_split_is_strictly_chronological(self) -> None:
        train, test = chronological_split(self.learned.all_trips)
        self.assertEqual((len(train), len(test)), (40, 10))
        self.assertLess(max(trip.departure for trip in train), min(trip.departure for trip in test))

    def test_learning_modules_cannot_import_hidden_fixture_truth(self) -> None:
        modules = (
            route_reader.ingestion,
            route_reader.discovery,
            route_reader.trajectories,
            route_reader.delays,
            route_reader.model,
            route_reader.pipeline,
        )
        for module in modules:
            source = inspect.getsource(module)
            self.assertNotIn("TripTruth", source, module.__name__)
            self.assertNotIn("SyntheticDataset", source, module.__name__)
            self.assertNotIn("counterfactual_duration_seconds", source, module.__name__)

    def test_canonical_quantitative_acceptance_fixture(self) -> None:
        metrics = self.result.metrics
        self.assertEqual(metrics.segmented_trips, 50)
        self.assertEqual(metrics.recurring_places_found, 2)
        self.assertEqual(metrics.discovered_route_variants, 4)
        self.assertGreaterEqual(metrics.train_route_ari, 0.90)
        self.assertGreaterEqual(metrics.held_out_route_assignment_accuracy, 0.90)
        self.assertLessEqual(metrics.travel_time_mae_seconds, 30.0)
        self.assertGreaterEqual(metrics.route_ranking_accuracy, 0.80)
        self.assertLess(metrics.mean_recommendation_regret_seconds, metrics.common_route_baseline_regret_seconds)
        self.assertLess(metrics.mean_recommendation_regret_seconds, metrics.overall_median_baseline_regret_seconds)
        self.assertGreater(metrics.delay_feature_mae_improvement_seconds, 1.0)
        self.assertGreaterEqual(metrics.confidence_interval_coverage, 0.70)


if __name__ == "__main__":
    unittest.main()
