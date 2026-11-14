"""Runtime-only types for learning from the route that was actually taken.

There is deliberately no simulator import and no field capable of carrying a
counterfactual outcome.  A policy sees opaque action IDs, departure context, and
one completed action outcome at a time.
"""

from __future__ import annotations

import math
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from datetime import datetime


@dataclass(frozen=True)
class DecisionContext:
    decision_id: str
    departure: datetime
    weekday: int
    departure_minute: float
    observation_quality: float
    route_identity_confidence: float


@dataclass(frozen=True)
class ObservedTripOutcome:
    decision_id: str
    action_id: str
    completed_at: datetime
    duration_seconds: float
    observation_quality: float
    resolved_variant: bool = True

    def __post_init__(self) -> None:
        if self.duration_seconds <= 0:
            raise ValueError("observed duration must be positive")


@dataclass(frozen=True)
class ActionEstimate:
    action_id: str
    expected_seconds: float
    uncertainty_seconds: float
    evidence_count: int
    recent_count: int
    maturity: str
    model: str
    probability_best: float


@dataclass(frozen=True)
class ExplorationEvidence:
    action_id: str
    expected_cost_seconds: float
    probability: float
    information_score: float
    reason: str


@dataclass(frozen=True)
class RankingDecision:
    context: DecisionContext
    eligible_actions: tuple[str, ...]
    fastest_expected_action: str
    recommended_action: str
    policy_selected_action: str
    exploration: ExplorationEvidence | None
    policy_propensities: dict[str, float]
    policy_actions: dict[str, str]
    estimates: tuple[ActionEstimate, ...]
    policy_version: str
    fallback_action: str
    expected_advantage_seconds: float
    decision_confidence: float
    detected_drift: bool
    abstention_reason: str | None

    def __post_init__(self) -> None:
        if set(self.policy_propensities) != set(self.eligible_actions):
            raise ValueError("policy propensities must cover exactly the eligible actions")
        if any(probability < 0 for probability in self.policy_propensities.values()):
            raise ValueError("policy propensities cannot be negative")
        if not math.isclose(sum(self.policy_propensities.values()), 1.0, abs_tol=1e-9):
            raise ValueError("policy propensities must sum to one")
        if self.policy_propensities.get(self.policy_selected_action, 0.0) <= 0:
            raise ValueError("selected action must have positive policy propensity")


@dataclass(frozen=True)
class FrozenDecisionLog:
    decision: RankingDecision
    user_model: str
    behavior_propensities: dict[str, float]
    actual_action: str
    actual_action_propensity: float
    frozen_at: datetime
    observed_outcome: ObservedTripOutcome | None = None

    def __post_init__(self) -> None:
        if any(probability < 0 for probability in self.behavior_propensities.values()):
            raise ValueError("behavior propensities cannot be negative")
        if not math.isclose(sum(self.behavior_propensities.values()), 1.0, abs_tol=1e-9):
            raise ValueError("behavior propensities must sum to one")
        logged = self.behavior_propensities.get(self.actual_action, 0.0)
        if logged <= 0 or not math.isclose(logged, self.actual_action_propensity, abs_tol=1e-9):
            raise ValueError("actual-action propensity must be frozen exactly at decision time")
        if self.frozen_at < self.decision.context.departure:
            raise ValueError("propensity log cannot predate its decision context")

    def complete(self, outcome: ObservedTripOutcome) -> "FrozenDecisionLog":
        if self.observed_outcome is not None:
            raise ValueError("decision outcome is already complete")
        if outcome.decision_id != self.decision.context.decision_id:
            raise ValueError("outcome decision does not match frozen decision")
        if outcome.action_id != self.actual_action:
            raise ValueError("only the actual logged action may complete a decision")
        if outcome.completed_at <= self.frozen_at:
            raise ValueError("outcome must occur after propensity logging")
        return replace(self, observed_outcome=outcome)


def _circular_minutes(left: float, right: float) -> float:
    distance = abs(left - right)
    return min(distance, 1440.0 - distance)


def _weighted_median(values: list[float], weights: list[float]) -> float:
    ordered = sorted(zip(values, weights))
    midpoint = sum(weights) / 2.0
    running = 0.0
    for value, weight in ordered:
        running += weight
        if running >= midpoint:
            return value
    return ordered[-1][0]


class PartialFeedbackState:
    """Chronological action-outcome history with chosen-action-only updates."""

    def __init__(self) -> None:
        self._outcomes: list[tuple[DecisionContext, ObservedTripOutcome]] = []
        self._decision_ids: set[str] = set()

    @property
    def outcomes(self) -> tuple[tuple[DecisionContext, ObservedTripOutcome], ...]:
        return tuple(self._outcomes)

    def observe(self, context: DecisionContext, outcome: ObservedTripOutcome) -> None:
        if context.decision_id != outcome.decision_id:
            raise ValueError("context and outcome decision IDs differ")
        if context.decision_id in self._decision_ids:
            raise ValueError("decision outcome already observed")
        if self._outcomes and context.departure <= self._outcomes[-1][0].departure:
            raise ValueError("partial-feedback outcomes must arrive chronologically")
        self._decision_ids.add(context.decision_id)
        self._outcomes.append((context, outcome))

    def action_history(self, action_id: str) -> tuple[tuple[DecisionContext, ObservedTripOutcome], ...]:
        return tuple(item for item in self._outcomes if item[1].action_id == action_id and item[1].resolved_variant)

    def action_counts(self) -> Counter[str]:
        return Counter(outcome.action_id for _, outcome in self._outcomes if outcome.resolved_variant)

    def known_actions(self, minimum_evidence: int = 1) -> tuple[str, ...]:
        counts = self.action_counts()
        return tuple(sorted(action for action, count in counts.items() if count >= minimum_evidence))

    def predict(self, action_id: str, context: DecisionContext, model: str) -> tuple[float, float, int, int]:
        history = self.action_history(action_id)
        if not history:
            raise ValueError(f"no observed outcome for {action_id}")
        values = [item.duration_seconds for _, item in history]
        if model == "overall_median":
            estimate = statistics.median(values)
            relevant = len(values)
        elif model == "time_bucket":
            nearby = [outcome.duration_seconds for prior, outcome in history if _circular_minutes(prior.departure_minute, context.departure_minute) <= 45]
            estimate = statistics.median(nearby) if len(nearby) >= 2 else statistics.median(values)
            relevant = len(nearby)
        else:
            weights = []
            for prior, outcome in history:
                age = max(0.0, (context.departure - prior.departure).total_seconds() / 86400.0)
                recency = 0.5 ** (age / 42.0)
                if model == "recency":
                    temporal = 1.0
                else:
                    temporal = math.exp(-0.5 * (_circular_minutes(prior.departure_minute, context.departure_minute) / 38.0) ** 2)
                weekday = 1.10 if prior.weekday == context.weekday else 1.0
                weights.append(max(0.02, recency * temporal * weekday * outcome.observation_quality))
            estimate = _weighted_median(values, weights)
            relevant = sum(weight >= 0.18 for weight in weights)
        deviations = [abs(value - estimate) for value in values]
        robust_sigma = 1.4826 * statistics.median(deviations) if deviations else 0.0
        uncertainty = max(16.0, robust_sigma + 65.0 / math.sqrt(max(1, relevant)))
        recent = sum((context.departure - prior.departure).days <= 42 for prior, _ in history)
        return estimate, uncertainty, len(values), recent

    def pairwise_probability(self, left: str, right: str, context: DecisionContext) -> tuple[float, int]:
        """Context-matched empirical probability that left is faster than right."""

        # Weekday is a legitimate departure-time context and prevents adjacent
        # but behaviorally distinct commute windows from being blurred together.
        left_history = tuple(item for item in self.action_history(left) if item[0].weekday == context.weekday)
        right_history = tuple(item for item in self.action_history(right) if item[0].weekday == context.weekday)
        comparisons = []
        for left_context, left_outcome in left_history:
            candidates = [
                (abs((left_context.departure - right_context.departure).days), right_context, right_outcome)
                for right_context, right_outcome in right_history
                if _circular_minutes(left_context.departure_minute, right_context.departure_minute) <= 24
                and abs((left_context.departure - right_context.departure).days) <= 56
            ]
            if candidates:
                _, right_context, right_outcome = min(candidates, key=lambda item: item[0])
                query_weight = math.exp(-_circular_minutes(left_context.departure_minute, context.departure_minute) / 75.0)
                comparisons.append((left_outcome.duration_seconds < right_outcome.duration_seconds, query_weight))
        if not comparisons:
            return 0.5, 0
        wins = sum(weight * float(won) for won, weight in comparisons)
        total = sum(weight for _, weight in comparisons)
        # Beta(2,2) shrinkage prevents a handful of noisy comparisons dominating.
        return (wins + 2.0) / (total + 4.0), len(comparisons)

    def policy_predictions(self, actions: tuple[str, ...], context: DecisionContext) -> dict[str, dict[str, float]]:
        result: dict[str, dict[str, float]] = defaultdict(dict)
        for model in ("overall_median", "time_bucket", "recency", "temporal_kernel"):
            for action in actions:
                result[model][action] = self.predict(action, context, model)[0]
        return dict(result)
