from __future__ import annotations

import inspect
import math
import unittest
from datetime import timedelta

import route_reader.robust_delays
import route_reader.robust_discovery
import route_reader.robust_ingestion
import route_reader.robust_model
import route_reader.robust_pipeline
import route_reader.route_families
from route_reader.adversarial import generate_life_history, observe_life_history
from route_reader.benchmark import ScenarioSpec, run_scenario
from route_reader.robust_evaluation import evaluate_robust_lab
from route_reader.robust_pipeline import learn_robust_lab


class RobustnessLabTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.history = generate_life_history()
        cls.normal_observed = observe_life_history(cls.history, "normal")
        cls.normal_learned = learn_robust_lab(cls.normal_observed.raw_points)
        cls.normal = evaluate_robust_lab(cls.normal_learned, cls.history, cls.normal_observed)

    def test_normal_regime_degrades_gracefully_without_becoming_perfect(self) -> None:
        metrics = self.normal.metrics
        self.assertGreaterEqual(metrics.segmentation_f1, 0.90)
        self.assertGreaterEqual(metrics.place_recall, 0.70)
        self.assertLess(metrics.place_precision, 0.90)
        self.assertGreaterEqual(metrics.route_variant_ari, 0.70)
        self.assertGreaterEqual(metrics.route_ranking_accuracy, 0.70)

    def test_chronological_boundary_is_strict(self) -> None:
        self.assertLess(
            max(item.departure for item in self.normal_learned.train_trips),
            min(item.departure for item in self.normal_learned.test_trips),
        )

    def test_runtime_modules_have_no_truth_or_counterfactual_imports(self) -> None:
        modules = (
            route_reader.robust_ingestion,
            route_reader.robust_discovery,
            route_reader.route_families,
            route_reader.robust_delays,
            route_reader.robust_model,
            route_reader.robust_pipeline,
        )
        for module in modules:
            source = inspect.getsource(module)
            self.assertNotIn("TrueJourney", source, module.__name__)
            self.assertNotIn("LifeHistory", source, module.__name__)
            self.assertNotIn("counterfactual_duration_seconds", source, module.__name__)
            self.assertNotIn("from .adversarial", source, module.__name__)

    def test_cold_start_status_and_width_do_not_manufacture_confidence(self) -> None:
        cold = self.normal.cold_start
        self.assertEqual(cold[0].stage, "insufficient history")
        self.assertEqual(cold[1].stage, "tentative route hypothesis")
        self.assertGreater(cold[0].mean_interval_width_seconds, cold[-1].mean_interval_width_seconds)

    def test_exploration_never_masquerades_as_fastest(self) -> None:
        od = self.normal_learned.od_learning[self.normal.target_od_id]
        query = self.normal_learned.test_trips[0].departure
        for offset in range(0, 70, 5):
            departure = query.replace(hour=15, minute=0, second=0, microsecond=0) + timedelta(minutes=offset)
            recommendation = od.model.predict_all(departure)
            candidate = recommendation.exploration_candidate
            if candidate:
                self.assertNotEqual(candidate.variant_id, recommendation.fastest_expected.variant_id)
                self.assertGreaterEqual(candidate.expected_slowdown_seconds, 0)

    def test_static_strong_signal_beats_bucket_and_delay_ablation(self) -> None:
        result, _ = run_scenario(
            ScenarioSpec("test-static-strong", "normal", behavior_drift=False, temporal_signal="strong")
        )
        metrics = result.metrics
        self.assertLess(metrics["recommendation_regret_seconds"], metrics["time_bucket_regret_seconds"])
        self.assertGreater(metrics["delay_regret_improvement_seconds"], 0)
        self.assertGreater(metrics["delay_precision"], 0)

    def test_random_and_behavioral_dwell_are_not_accepted_without_signal(self) -> None:
        result, _ = run_scenario(
            ScenarioSpec("test-static-none", "normal", behavior_drift=False, temporal_signal="none")
        )
        self.assertEqual(result.metrics["active_delay_zones"], 0)

    def test_sparse_and_hostile_regimes_widen_or_abstain(self) -> None:
        good, _ = run_scenario(ScenarioSpec("test-good", "good"))
        sparse, _ = run_scenario(ScenarioSpec("test-sparse", "sparse"))
        hostile, _ = run_scenario(ScenarioSpec("test-hostile", "hostile"))
        self.assertGreater(sparse.metrics["mean_interval_width_seconds"], good.metrics["mean_interval_width_seconds"])
        self.assertLess(hostile.metrics["segmentation_f1"], good.metrics["segmentation_f1"])
        self.assertTrue(
            math.isnan(hostile.metrics["mean_interval_width_seconds"])
            or hostile.metrics["mean_interval_width_seconds"] > good.metrics["mean_interval_width_seconds"]
        )

    def test_route_family_is_not_forced_to_equal_variant_count(self) -> None:
        od = self.normal_learned.od_learning[self.normal.target_od_id]
        self.assertLess(len(od.hierarchy.families), len(od.hierarchy.variants))
        self.assertGreaterEqual(self.normal.metrics.route_family_ari, 0.30)


if __name__ == "__main__":
    unittest.main()
