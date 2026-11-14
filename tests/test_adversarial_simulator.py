from __future__ import annotations

import unittest

from route_reader.adversarial import OBSERVATION_REGIMES, LifeConfig, generate_life_history, observe_life_history


class AdversarialSimulatorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.history = generate_life_history()

    def test_five_month_life_history_is_deterministic_and_multi_place(self) -> None:
        repeated = generate_life_history()
        self.assertEqual(self.history, repeated)
        self.assertEqual(len(self.history.places_xy), 7)
        self.assertEqual(len(self.history.journeys), 413)
        self.assertEqual(sum(item.od_id == "School->Work" for item in self.history.journeys), 60)
        self.assertGreaterEqual(len({item.od_id for item in self.history.journeys}), 10)

    def test_fixture_contains_requested_segmentation_complications(self) -> None:
        complications = {item.complication for item in self.history.journeys}
        self.assertTrue(
            {
                "campus-walk",
                "parking-wait",
                "gas-stop",
                "return-toward-origin",
                "partial-start",
                "partial-end",
                "destination-walk",
                "stationary-intermediate",
            }.issubset(complications)
        )

    def test_same_truth_replays_through_degrading_observation_regimes(self) -> None:
        observed = {name: observe_life_history(self.history, name) for name in OBSERVATION_REGIMES}
        self.assertGreater(len(observed["good"].raw_points), len(observed["normal"].raw_points))
        self.assertGreater(len(observed["normal"].raw_points), len(observed["sparse"].raw_points))
        self.assertGreater(len(observed["sparse"].raw_points), len(observed["hostile"].raw_points))
        self.assertEqual(
            set(observed["normal"].raw_points[0].__dataclass_fields__),
            {"timestamp", "latitude", "longitude", "accuracy_m"},
        )

    def test_static_scenario_has_time_windows_but_no_new_route(self) -> None:
        history = generate_life_history(LifeConfig(behavior_drift=False))
        target = [item for item in history.journeys if item.od_id == "School->Work"]
        self.assertGreaterEqual(len({round(item.departure.hour * 60 + item.departure.minute, -1) for item in target}), 4)
        self.assertFalse(any("N" in item.counterfactual_duration_seconds for item in target))


if __name__ == "__main__":
    unittest.main()
