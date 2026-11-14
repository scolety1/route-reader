from __future__ import annotations

import inspect
import unittest
from dataclasses import fields
from datetime import datetime, timedelta

import route_reader.off_policy
import route_reader.partial_feedback
import route_reader.ranking_policy
import route_reader.user_behavior
from route_reader.mission5_evaluation import build_partial_feedback_fixture, strict_partial_feedback_replay
from route_reader.off_policy import evaluate_logged_policy
from route_reader.partial_feedback import (
    ActionEstimate,
    DecisionContext,
    FrozenDecisionLog,
    ObservedTripOutcome,
    PartialFeedbackState,
    RankingDecision,
)
from route_reader.ranking_policy import ExplorationConfig, RegretRankingPolicy
from route_reader.user_behavior import choose_user_action


BASE = datetime(2026, 1, 5, 15, 0)


def _observe(state: PartialFeedbackState, identifier: str, departure: datetime, action: str, duration: float, *, resolved: bool = True, quality: float = 1.0) -> None:
    context = DecisionContext(identifier, departure, departure.weekday(), departure.hour * 60 + departure.minute, quality, 0.9)
    outcome = ObservedTripOutcome(identifier, action, departure + timedelta(seconds=duration), duration, quality, resolved)
    state.observe(context, outcome)


def _ranking_fixture() -> tuple[PartialFeedbackState, DecisionContext]:
    state = PartialFeedbackState()
    sequence = 0
    # Overall B is faster, but A consistently wins in the Monday decision
    # context. A decision-aligned selector should prefer A on the next Monday.
    for week in range(6):
        monday = BASE + timedelta(weeks=week)
        tuesday = monday + timedelta(days=1)
        for departure, action, duration in (
            (monday, "A", 100.0),
            (monday + timedelta(minutes=2), "B", 120.0),
            (tuesday, "A", 140.0),
            (tuesday + timedelta(minutes=2), "B", 90.0),
        ):
            sequence += 1
            _observe(state, f"D{sequence}", departure, action, duration)
    query = BASE + timedelta(weeks=7)
    return state, DecisionContext("query", query, query.weekday(), 15 * 60, 0.9, 0.9)


def _decision(context: DecisionContext, *, selected: str = "A", policy_actions: dict[str, str] | None = None) -> RankingDecision:
    return RankingDecision(
        context,
        ("A", "B"),
        "A",
        "A",
        selected,
        None,
        {"A": 1.0, "B": 0.0},
        policy_actions or {"target": "A"},
        (
            ActionEstimate("A", 100.0, 20.0, 10, 10, "established", "test", 0.7),
            ActionEstimate("B", 120.0, 20.0, 10, 10, "established", "test", 0.3),
        ),
        "test-v1",
        "A",
        0.0,
        0.8,
        False,
        None,
    )


class PartialFeedbackBoundaryTests(unittest.TestCase):
    def test_chosen_action_only_updates_runtime_state(self) -> None:
        state = PartialFeedbackState()
        _observe(state, "one", BASE, "A", 100.0)
        self.assertEqual(state.action_counts(), {"A": 1})
        self.assertEqual(state.action_history("B"), ())

    def test_observed_outcome_has_no_counterfactual_container(self) -> None:
        names = {field.name for field in fields(ObservedTripOutcome)}
        self.assertEqual(names, {"decision_id", "action_id", "completed_at", "duration_seconds", "observation_quality", "resolved_variant"})
        with self.assertRaises(TypeError):
            ObservedTripOutcome("x", "A", BASE, 100.0, 1.0, True, {"B": 90.0})

    def test_runtime_modules_cannot_import_simulator_truth(self) -> None:
        for module in (route_reader.partial_feedback, route_reader.ranking_policy, route_reader.user_behavior, route_reader.off_policy):
            source = inspect.getsource(module)
            self.assertNotIn("from .adversarial", source)
            self.assertNotIn("TrueJourney", source)
            self.assertNotIn("SimulationEvaluationTruth", source)

    def test_propensity_is_frozen_before_outcome(self) -> None:
        context = DecisionContext("x", BASE, 0, 900.0, 0.9, 0.9)
        decision = _decision(context)
        log = FrozenDecisionLog(decision, "mixed", {"A": 0.6, "B": 0.4}, "A", 0.6, BASE)
        outcome = ObservedTripOutcome("x", "A", BASE + timedelta(seconds=100), 100.0, 1.0)
        completed = log.complete(outcome)
        self.assertIsNone(log.observed_outcome)
        self.assertEqual(completed.actual_action_propensity, 0.6)
        self.assertLess(completed.frozen_at, completed.observed_outcome.completed_at)

    def test_inconsistent_actual_propensity_is_rejected(self) -> None:
        context = DecisionContext("x", BASE, 0, 900.0, 0.9, 0.9)
        with self.assertRaises(ValueError):
            FrozenDecisionLog(_decision(context), "mixed", {"A": 0.6, "B": 0.4}, "A", 0.5, BASE)

    def test_partial_route_outcome_does_not_update_variant_evidence(self) -> None:
        state = PartialFeedbackState()
        _observe(state, "partial", BASE, "A", 100.0, resolved=False)
        self.assertEqual(len(state.outcomes), 1)
        self.assertEqual(state.action_counts(), {})
        self.assertEqual(state.action_history("A"), ())


class RankingAndExplorationTests(unittest.TestCase):
    def test_pairwise_objective_can_beat_overall_duration_model(self) -> None:
        state, context = _ranking_fixture()
        decision = RegretRankingPolicy(state).decide(context, ("A", "B"))
        self.assertEqual(decision.fallback_action, "B")
        self.assertEqual(decision.recommended_action, "A")
        self.assertGreater(next(item for item in decision.estimates if item.action_id == "A").probability_best, 0.56)

    def test_repeated_route_can_only_be_bounded_exploration_candidate(self) -> None:
        state = PartialFeedbackState()
        for index in range(6):
            _observe(state, f"A{index}", BASE + timedelta(days=index * 2), "A", 100.0)
            if index < 3:
                _observe(state, f"B{index}", BASE + timedelta(days=index * 2, minutes=2), "B", 125.0)
        context = DecisionContext("query", BASE + timedelta(days=20), 0, 900.0, 0.9, 0.9)
        decision = RegretRankingPolicy(state).decide(context, ("A", "B"))
        self.assertEqual(decision.recommended_action, "A")
        self.assertIsNotNone(decision.exploration)
        self.assertEqual(decision.exploration.action_id, "B")
        self.assertLessEqual(decision.exploration.probability, 0.15)
        self.assertLessEqual(decision.exploration.expected_cost_seconds, 45.0)

    def test_materially_worse_route_is_not_explored(self) -> None:
        state = PartialFeedbackState()
        for index in range(6):
            _observe(state, f"A{index}", BASE + timedelta(days=index * 2), "A", 100.0)
            if index < 3:
                _observe(state, f"B{index}", BASE + timedelta(days=index * 2, minutes=2), "B", 200.0)
        context = DecisionContext("query", BASE + timedelta(days=20), 0, 900.0, 0.9, 0.9)
        decision = RegretRankingPolicy(state).decide(context, ("A", "B"))
        self.assertIsNone(decision.exploration)

    def test_low_quality_context_disables_exploration(self) -> None:
        state, context = _ranking_fixture()
        poor = DecisionContext("poor", context.departure, context.weekday, context.departure_minute, 0.1, 0.9)
        decision = RegretRankingPolicy(state).decide(poor, ("A", "B"))
        self.assertIsNone(decision.exploration)

    def test_no_signal_mature_evidence_stops_exploration(self) -> None:
        state = PartialFeedbackState()
        for index in range(20):
            departure = BASE + timedelta(days=index * 2)
            _observe(state, f"A{index}", departure, "A", 100.0 + index % 3)
            _observe(state, f"B{index}", departure + timedelta(minutes=2), "B", 101.0 + index % 3)
        context = DecisionContext("query", BASE + timedelta(days=45), 0, 900.0, 0.9, 0.9)
        decision = RegretRankingPolicy(state).decide(context, ("A", "B"))
        self.assertIsNone(decision.exploration)

    def test_recent_route_degradation_is_detected_without_counterfactuals(self) -> None:
        state = PartialFeedbackState()
        for index, duration in enumerate((100, 101, 99, 102, 100, 101, 170, 175, 168, 172)):
            departure = BASE + timedelta(days=index * 2)
            _observe(state, f"A{index}", departure, "A", float(duration))
            _observe(state, f"B{index}", departure + timedelta(minutes=2), "B", 120.0)
        context = DecisionContext("query", BASE + timedelta(days=25), 0, 900.0, 0.9, 0.9)
        decision = RegretRankingPolicy(state).decide(context, ("A", "B"))
        self.assertTrue(decision.detected_drift)

    def test_ranking_retains_trusted_fallback_when_evidence_is_weak(self) -> None:
        state = PartialFeedbackState()
        for index in range(6):
            departure = BASE + timedelta(days=index * 2)
            _observe(state, f"A{index}", departure, "A", 110.0 + (-1) ** index * 8)
            _observe(state, f"B{index}", departure + timedelta(minutes=2), "B", 111.0 + (-1) ** (index + 1) * 8)
        context = DecisionContext("query", BASE + timedelta(days=20), 0, 900.0, 0.9, 0.9)
        decision = RegretRankingPolicy(state).decide(context, ("A", "B"))
        self.assertEqual(decision.recommended_action, decision.fallback_action)


class UserChoiceAndOffPolicyTests(unittest.TestCase):
    def test_recommendation_and_actual_action_are_separate(self) -> None:
        context = DecisionContext("x", BASE, 0, 900.0, 0.9, 0.9)
        choice = choose_user_action(_decision(context), "habitual", random_draw=0.99, habitual_action="A", planned_action="B")
        self.assertEqual(choice.actual_action, "B")
        self.assertNotEqual(choice.actual_action, "A")
        self.assertAlmostEqual(sum(choice.logging_propensities.values()), 1.0)
        self.assertEqual(choice.actual_action_propensity, choice.logging_propensities["B"])

    def _logs(self, *, poor_support: bool = False) -> tuple[FrozenDecisionLog, ...]:
        logs = []
        for index in range(40):
            departure = BASE + timedelta(days=index)
            context = DecisionContext(f"L{index}", departure, departure.weekday(), 900.0, 0.9, 0.9)
            decision = _decision(context)
            if poor_support:
                actual, probabilities = "B", {"A": 0.001, "B": 0.999}
            else:
                actual, probabilities = ("A", {"A": 0.5, "B": 0.5}) if index % 2 == 0 else ("B", {"A": 0.5, "B": 0.5})
            duration = 100.0 if actual == "A" else 120.0
            log = FrozenDecisionLog(decision, "test", probabilities, actual, probabilities[actual], departure)
            outcome = ObservedTripOutcome(context.decision_id, actual, departure + timedelta(seconds=duration), duration, 1.0)
            logs.append(log.complete(outcome))
        return tuple(logs)

    def test_snips_and_dr_recover_value_under_adequate_support(self) -> None:
        estimates = {item.estimator: item for item in evaluate_logged_policy(self._logs(), "target")}
        self.assertTrue(estimates["snips_clipped"].reliable)
        self.assertAlmostEqual(estimates["snips_clipped"].estimated_mean_seconds, 100.0)
        self.assertAlmostEqual(estimates["doubly_robust_clipped"].estimated_mean_seconds, 100.0)
        self.assertGreaterEqual(estimates["snips_clipped"].effective_sample_size, 19.0)

    def test_poor_support_is_reported_as_unreliable(self) -> None:
        estimates = evaluate_logged_policy(self._logs(poor_support=True), "target")
        self.assertTrue(all(not item.reliable for item in estimates))
        self.assertTrue(all(item.support_coverage == 0.0 for item in estimates))
        self.assertTrue(all(item.unsupported_decisions == 40 for item in estimates))

    def test_importance_weights_are_clipped(self) -> None:
        departure = BASE
        context = DecisionContext("clip", departure, 0, 900.0, 0.9, 0.9)
        log = FrozenDecisionLog(_decision(context), "test", {"A": 0.001, "B": 0.999}, "A", 0.001, departure)
        outcome = ObservedTripOutcome("clip", "A", departure + timedelta(seconds=100), 100.0, 1.0)
        estimates = evaluate_logged_policy((log.complete(outcome),), "target", clip_weight=10.0)
        self.assertTrue(all(item.maximum_importance_weight == 10.0 for item in estimates))

    def test_estimator_agrees_with_separate_simulator_truth_when_supported(self) -> None:
        trips, truth = build_partial_feedback_fixture(seed=20260812, temporal_signal="strong", behavior_drift=False)
        replay = strict_partial_feedback_replay(trips, truth, seed=20260812, temporal_signal="strong", behavior_drift=False)
        validation = next(item for item in replay.ope if item.target_policy == "mission5_ranking")
        estimate = next(item for item in validation.estimates if item.estimator == "doubly_robust_clipped")
        self.assertTrue(estimate.reliable)
        self.assertLess(abs(estimate.estimated_mean_seconds - validation.simulator_true_mean_seconds), 12.0)
        self.assertEqual(replay.leakage_violations, 0)
        self.assertTrue(replay.propensity_frozen_before_outcome)


if __name__ == "__main__":
    unittest.main()
