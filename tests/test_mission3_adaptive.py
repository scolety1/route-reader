from __future__ import annotations

import inspect
import unittest
from datetime import datetime, timedelta

import route_reader.adaptive_model
import route_reader.place_regions
import route_reader.stable_routes
from route_reader.adaptive_model import AdaptiveJourneyModel, AdaptiveRouteModel
from route_reader.adversarial import LifeConfig, generate_life_history, observe_life_history
from route_reader.place_regions import discover_place_regions
from route_reader.robust_evaluation import align_trips_to_truth
from route_reader.robust_ingestion import RobustTrip, segment_trips_robust
from route_reader.stable_routes import StableRouteRegistry
from route_reader.types import ObservablePoint


def _trip(identifier: str, day: int, minute: int, duration: float, quality: float = 0.9) -> RobustTrip:
    departure = datetime(2026, 1, 1) + timedelta(days=day, minutes=minute)
    points = tuple(
        ObservablePoint(departure + timedelta(seconds=index * duration / 5), 40.0, -105.0 + index * 0.001, 5.0)
        for index in range(6)
    )
    return RobustTrip(identifier, points, departure, departure + timedelta(seconds=duration), 40.0, -105.0, 40.0, -104.995, quality, 5.0, duration / 5, True, True)


class Mission3AdaptiveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.history = generate_life_history()
        cls.normal_observed = observe_life_history(cls.history, "normal")
        cls.normal_trips = segment_trips_robust(cls.normal_observed.raw_points)

    def test_runtime_modules_do_not_import_simulator_truth(self) -> None:
        for module in (route_reader.adaptive_model, route_reader.place_regions, route_reader.stable_routes):
            source = inspect.getsource(module)
            self.assertNotIn("from .adversarial", source)
            self.assertNotIn("TrueJourney", source)
            self.assertNotIn("counterfactual_duration_seconds", source)

    def test_partial_endpoint_clusters_are_not_promoted(self) -> None:
        partials = tuple(
            RobustTrip(
                f"P{i}",
                _trip(f"X{i}", i, 480, 500).points,
                datetime(2026, 1, 1) + timedelta(days=i),
                datetime(2026, 1, 1) + timedelta(days=i, seconds=500),
                40.0,
                -105.0,
                40.01,
                -104.99,
                0.5,
                12.0,
                40.0,
                False,
                False,
            )
            for i in range(8)
        )
        discovery = discover_place_regions(partials)
        self.assertFalse(any(region.status in {"recurring", "strong_recurring"} for region in discovery.regions))
        self.assertGreaterEqual(discovery.suppressed_truncated_clusters, 1)

    def test_normal_places_are_regions_with_higher_precision(self) -> None:
        train = tuple(sorted(self.normal_trips, key=lambda item: item.departure)[: int(len(self.normal_trips) * 0.8)])
        discovery = discover_place_regions(train)
        confirmed = [region for region in discovery.regions if region.status != "candidate"]
        self.assertLessEqual(len(confirmed), 7)
        self.assertGreaterEqual(len(confirmed), 5)
        self.assertTrue(any(region.radius_m > 120 for region in confirmed))

    def test_stable_ids_survive_history_growth(self) -> None:
        aligned = align_trips_to_truth(self.normal_trips, self.history, self.normal_observed.recorded_journey_ids)
        target = tuple(trip for trip in self.normal_trips if trip.trip_id in aligned and aligned[trip.trip_id].od_id == "School->Work")
        registry = StableRouteRegistry()
        snapshots = [registry.update(target[:count]) for count in (5, 10, 20, min(40, len(target)))]
        self.assertTrue(set(route.route_id for route in snapshots[0].routes) & set(route.route_id for route in snapshots[-1].routes))
        self.assertGreaterEqual(snapshots[-1].retained_identity_fraction, 0.75)
        self.assertTrue(any(snapshot.revisions for snapshot in snapshots))

    def test_walk_forward_residuals_end_before_deployment(self) -> None:
        trips = tuple(_trip(f"T{i}", i, 480 + i % 3 * 20, 600 + i % 4 * 10) for i in range(15))
        model = AdaptiveRouteModel("R", trips)
        prediction = model.predict(trips[-1].departure + timedelta(days=1))
        self.assertLessEqual(prediction.calibration_cutoff, trips[-1].departure)
        self.assertGreaterEqual(prediction.calibration_residuals, 4)

    def test_weak_signal_backs_off_to_overall_median(self) -> None:
        trips = tuple(_trip(f"W{i}", i, 430 + (i * 37) % 180, 600) for i in range(18))
        model = AdaptiveRouteModel("R", trips)
        self.assertEqual(model.selected_model, "overall_median")

    def test_strong_signal_can_graduate(self) -> None:
        trips = tuple(
            _trip(f"S{i}", i, 450 if i % 2 == 0 else 960, 520 if i % 2 == 0 else 760)
            for i in range(24)
        )
        model = AdaptiveRouteModel("R", trips)
        self.assertIn(model.selected_model, {"time_bucket", "temporal_kernel"})

    def test_no_signal_does_not_invent_temporal_advantage(self) -> None:
        trips = tuple(_trip(f"N{i}", i, 400 + (i * 53) % 300, 610) for i in range(20))
        model = AdaptiveRouteModel("R", trips)
        self.assertEqual(model.selected_model, "overall_median")

    def test_delay_candidate_requires_positive_validation_gain(self) -> None:
        trips = tuple(_trip(f"D{i}", i, 480, 600 + i % 2) for i in range(20))
        model = AdaptiveRouteModel("R", trips)
        delay = next(item for item in model.evidence if item.model == "temporal_delay")
        self.assertFalse(delay.eligible)
        self.assertFalse(model.predict(trips[-1].departure + timedelta(days=1)).delay_feature_enabled)

    def test_hostile_quality_abstains(self) -> None:
        trips = tuple(_trip(f"H{i}", i, 480, 600, quality=0.10) for i in range(8))
        recommendation = AdaptiveJourneyModel((AdaptiveRouteModel("R", trips),)).predict_all(trips[-1].departure + timedelta(days=1))
        self.assertEqual(recommendation.abstention_reason, "current GPS history too degraded")

    def test_exploration_is_never_the_fastest(self) -> None:
        first = AdaptiveRouteModel("A", tuple(_trip(f"A{i}", i, 480, 590) for i in range(12)))
        second = AdaptiveRouteModel("B", tuple(_trip(f"B{i}", i, 480, 610 + i % 3 * 12) for i in range(7)))
        recommendation = AdaptiveJourneyModel((first, second)).predict_all(datetime(2026, 3, 1, 8))
        if recommendation.exploration_candidate and recommendation.fastest_expected:
            self.assertNotEqual(recommendation.exploration_candidate.variant_id, recommendation.fastest_expected.variant_id)
            self.assertGreaterEqual(recommendation.exploration_candidate.expected_slowdown_seconds, 0)


if __name__ == "__main__":
    unittest.main()
