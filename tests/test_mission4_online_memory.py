from __future__ import annotations

import inspect
import unittest
from datetime import datetime, timedelta

import route_reader.online_corridors
from route_reader.adaptive_model import AdaptiveRouteModel
from route_reader.adversarial import LifeConfig, generate_life_history, observe_life_history
from route_reader.geo import xy_to_latlon
from route_reader.mission4_evaluation import strict_walk_forward
from route_reader.online_corridors import OnlineCorridorMemory, extract_corridor_samples
from route_reader.robust_evaluation import align_trips_to_truth
from route_reader.robust_ingestion import RobustTrip, segment_trips_robust
from route_reader.synthetic import ORIGIN_LAT, ORIGIN_LON
from route_reader.types import ObservablePoint


def _route_trip(identifier: str, day: int, path: tuple[tuple[float, float], ...], *, dwell: tuple[bool, bool] = (True, True), quality: float = 0.9, minute: int = 480) -> RobustTrip:
    departure = datetime(2026, 1, 1) + timedelta(days=day, minutes=minute)
    points = tuple(
        ObservablePoint(departure + timedelta(seconds=index * 35), *xy_to_latlon(x, y, ORIGIN_LAT, ORIGIN_LON), 6.0)
        for index, (x, y) in enumerate(path)
    )
    return RobustTrip(identifier, points, departure, points[-1].timestamp, points[0].latitude, points[0].longitude, points[-1].latitude, points[-1].longitude, quality, 6.0, 35.0, dwell[0], dwell[1])


BASE = ((0, 0), (500, 0), (1000, 0), (1500, 0), (2000, 0), (2500, 0))
LATE_NORTH = ((0, 0), (500, 0), (1000, 0), (1500, 0), (1900, 260), (2500, 300))
DETOUR = ((0, 0), (500, 0), (1000, 0), (1350, 520), (1900, 500), (2500, 0))


class OnlineCorridorMemoryTests(unittest.TestCase):
    def test_corridor_extraction_uses_physical_spacing(self) -> None:
        short = extract_corridor_samples(_route_trip("S", 0, BASE))
        dense_path = tuple((index * 100, 0) for index in range(26))
        dense = extract_corridor_samples(_route_trip("D", 1, dense_path))
        self.assertLessEqual(abs(len(short) - len(dense)), 2)
        self.assertGreater(len(short), 20)

    def test_shared_corridor_and_late_divergence_are_preserved(self) -> None:
        memory = OnlineCorridorMemory()
        for index in range(6):
            memory.observe(_route_trip(f"A{index}", index * 2, BASE))
            memory.observe(_route_trip(f"B{index}", index * 2 + 1, LATE_NORTH))
        self.assertEqual(len(memory.established_variants), 2)
        self.assertGreater(memory.snapshot().shared_corridor_nodes, 0)

    def test_partial_shared_route_does_not_force_branch_identity(self) -> None:
        memory = OnlineCorridorMemory()
        for index in range(6):
            memory.observe(_route_trip(f"A{index}", index * 2, BASE))
            memory.observe(_route_trip(f"B{index}", index * 2 + 1, LATE_NORTH))
        partial = _route_trip("P", 20, BASE[:4], dwell=(True, False))
        match = memory.observe(partial)
        self.assertEqual(match.state, "shared_corridor_only")
        self.assertIsNone(match.variant_id)
        self.assertLess(match.identity_confidence, 0.60)

    def test_one_off_detour_remains_candidate(self) -> None:
        memory = OnlineCorridorMemory()
        for index in range(6):
            memory.observe(_route_trip(f"A{index}", index, BASE))
        match = memory.observe(_route_trip("detour", 8, DETOUR))
        self.assertEqual(memory.variants[match.variant_id].status, "candidate")

    def test_genuine_new_route_graduates_after_recurrence(self) -> None:
        memory = OnlineCorridorMemory()
        for index in range(6):
            memory.observe(_route_trip(f"A{index}", index, BASE))
        states = []
        for index in range(5):
            match = memory.observe(_route_trip(f"N{index}", 10 + index, LATE_NORTH))
            states.append(match.state)
        self.assertEqual(states[0], "candidate")
        self.assertEqual(states[-1], "established")

    def test_small_route_evolution_reuses_identity(self) -> None:
        memory = OnlineCorridorMemory()
        for index in range(6):
            memory.observe(_route_trip(f"A{index}", index, BASE))
        original = memory.established_variants[0].variant_id
        evolved = tuple((x, y + (60 if 500 <= x <= 2000 else 0)) for x, y in BASE)
        match = memory.observe(_route_trip("evolved", 8, evolved))
        self.assertEqual(match.variant_id, original)

    def test_append_only_assignments_never_reinterpret_history(self) -> None:
        memory = OnlineCorridorMemory()
        for index in range(5):
            memory.observe(_route_trip(f"A{index}", index, BASE))
        original = dict(memory.assignments)
        for index in range(5):
            memory.observe(_route_trip(f"B{index}", 10 + index, LATE_NORTH))
        self.assertEqual({key: memory.assignments[key] for key in original}, original)

    def test_prediction_aggregation_only_inherits_high_overlap_parent(self) -> None:
        memory = OnlineCorridorMemory()
        for index in range(6):
            memory.observe(_route_trip(f"A{index}", index, BASE))
        for index in range(3):
            match = memory.observe(_route_trip(f"B{index}", 10 + index, LATE_NORTH))
        history = memory.prediction_history(match.variant_id)
        self.assertLessEqual(len(history), 9)
        self.assertTrue(all(item.trip_id.startswith(("A", "B")) for item in history))

    def test_online_runtime_cannot_access_simulator_truth(self) -> None:
        source = inspect.getsource(route_reader.online_corridors)
        self.assertNotIn("from .adversarial", source)
        self.assertNotIn("TrueJourney", source)
        self.assertNotIn("counterfactual_duration_seconds", source)

    def test_strict_walk_forward_has_no_future_leakage(self) -> None:
        history = generate_life_history(LifeConfig(behavior_drift=False, temporal_signal="weak"))
        observed = observe_life_history(history, "normal")
        trips = segment_trips_robust(observed.raw_points)
        aligned = align_trips_to_truth(trips, history, observed.recorded_journey_ids)
        target = tuple(trip for trip in trips if trip.trip_id in aligned and aligned[trip.trip_id].od_id == "School->Work")
        result = strict_walk_forward(target, {trip.trip_id: aligned[trip.trip_id] for trip in target})
        self.assertEqual(result.leakage_violations, 0)
        for decision in result.decisions:
            self.assertLess(decision.history_cutoff, decision.departure)
            self.assertTrue(all(cutoff is None or cutoff < decision.departure for cutoff in decision.calibration_cutoffs))

    def test_weak_signal_remains_near_simple_baseline(self) -> None:
        history = generate_life_history(LifeConfig(behavior_drift=False, temporal_signal="weak"))
        observed = observe_life_history(history, "normal")
        trips = segment_trips_robust(observed.raw_points)
        aligned = align_trips_to_truth(trips, history, observed.recorded_journey_ids)
        target = tuple(trip for trip in trips if trip.trip_id in aligned and aligned[trip.trip_id].od_id == "School->Work")
        result = strict_walk_forward(target, {trip.trip_id: aligned[trip.trip_id] for trip in target})
        policies = {item.policy: item for item in result.policies}
        self.assertLessEqual(policies["adaptive"].mean_regret_seconds, policies["overall_median"].mean_regret_seconds + 2.0)

    def test_strong_history_can_graduate_a_route_model(self) -> None:
        history = generate_life_history(LifeConfig(behavior_drift=False, temporal_signal="strong"))
        observed = observe_life_history(history, "normal")
        trips = segment_trips_robust(observed.raw_points)
        aligned = align_trips_to_truth(trips, history, observed.recorded_journey_ids)
        target = tuple(trip for trip in trips if trip.trip_id in aligned and aligned[trip.trip_id].od_id == "School->Work")
        memory = OnlineCorridorMemory()
        for trip in target:
            memory.observe(trip)
        selected = [AdaptiveRouteModel(item.variant_id, memory.prediction_history(item.variant_id)).selected_model for item in memory.established_variants]
        self.assertTrue(any(item in {"time_bucket", "recency", "temporal_kernel"} for item in selected))

    def test_good_and_normal_have_comparable_durable_counts(self) -> None:
        counts = []
        history = generate_life_history()
        for regime in ("good", "normal"):
            observed = observe_life_history(history, regime)
            trips = segment_trips_robust(observed.raw_points)
            aligned = align_trips_to_truth(trips, history, observed.recorded_journey_ids)
            memory = OnlineCorridorMemory()
            for trip in trips:
                if trip.trip_id in aligned and aligned[trip.trip_id].od_id == "School->Work":
                    memory.observe(trip)
            counts.append(len(memory.established_variants))
        self.assertLessEqual(abs(counts[0] - counts[1]), 1)
        self.assertGreaterEqual(min(counts), 3)

    def test_degraded_quality_does_not_manufacture_identity_confidence(self) -> None:
        memory = OnlineCorridorMemory()
        matches = [memory.observe(_route_trip(f"H{i}", i, BASE, quality=0.12, dwell=(False, False))) for i in range(4)]
        self.assertLess(max(item.identity_confidence for item in matches), 0.25)

    def test_shared_nodes_support_corridor_level_delay_attribution(self) -> None:
        memory = OnlineCorridorMemory()
        for index in range(6):
            memory.observe(_route_trip(f"A{index}", index * 2, BASE))
            memory.observe(_route_trip(f"B{index}", index * 2 + 1, LATE_NORTH))
        shared = [node for node in memory.nodes.values() if len(node.variant_ids) > 1]
        self.assertTrue(shared)
        self.assertTrue(all(len(node.support_trips) >= 2 for node in shared[:3]))


if __name__ == "__main__":
    unittest.main()
