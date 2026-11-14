"""Conservative route ranking and exploration over partial-feedback state.

The policy optimizes route ordering directly through context-matched pairwise
wins. Duration models remain evidence inputs, but chronological prediction MAE
does not choose the Mission 5 action. All inputs are runtime-observable.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass

from .partial_feedback import (
    ActionEstimate,
    DecisionContext,
    ExplorationEvidence,
    PartialFeedbackState,
    RankingDecision,
    _circular_minutes,
)


@dataclass(frozen=True)
class ExplorationConfig:
    """Inspectable personal-product exploration limits, expressed in seconds."""

    maximum_expected_cost_seconds: float = 45.0
    maximum_probability: float = 0.15
    minimum_uncertainty_seconds: float = 38.0
    cooldown_decisions: int = 4
    minimum_route_identity_confidence: float = 0.55
    minimum_observation_quality: float = 0.30
    minimum_established_evidence: int = 5
    minimum_pairwise_comparisons: int = 4
    safe_improvement_probability: float = 0.56
    maximum_supported_ranking_disagreement_seconds: float = 20.0


class RegretRankingPolicy:
    """A safe-improvement ranking policy with bounded information gathering."""

    VERSION = "mission5-pairwise-safe-v1"

    def __init__(self, state: PartialFeedbackState, config: ExplorationConfig = ExplorationConfig()):
        self.state = state
        self.config = config
        self._decision_number = 0
        self._last_exploration_decision = -10_000

    @staticmethod
    def _choose(predictions: dict[str, float]) -> str:
        return min(predictions, key=lambda action: (predictions[action], action))

    def _mission4_mae_action(self, actions: tuple[str, ...], context: DecisionContext) -> str:
        """Partial-log analogue of Mission 4's MAE-oriented complexity gate."""

        selected_predictions: dict[str, float] = {}
        for action in actions:
            history = self.state.action_history(action)
            overall_errors: list[float] = []
            bucket_errors: list[float] = []
            for index in range(5, len(history)):
                prior = history[:index]
                actual_context, actual = history[index]
                values = [item.duration_seconds for _, item in prior]
                overall = statistics.median(values)
                nearby = [
                    item.duration_seconds
                    for prior_context, item in prior
                    if _circular_minutes(prior_context.departure_minute, actual_context.departure_minute) <= 45
                ]
                bucket = statistics.median(nearby) if len(nearby) >= 2 else overall
                overall_errors.append(abs(overall - actual.duration_seconds))
                bucket_errors.append(abs(bucket - actual.duration_seconds))
            model = "overall_median"
            if len(bucket_errors) >= 4 and statistics.mean(bucket_errors) + 2.0 < statistics.mean(overall_errors):
                model = "time_bucket"
            selected_predictions[action] = self.state.predict(action, context, model)[0]
        return self._choose(selected_predictions)

    def _drift_detected(self, action: str) -> bool:
        history = self.state.action_history(action)
        if len(history) < 9:
            return False
        values = [outcome.duration_seconds for _, outcome in history]
        historical = statistics.median(values[:-4])
        recent = statistics.median(values[-4:])
        deviations = [abs(value - historical) for value in values[:-4]]
        scale = 1.4826 * statistics.median(deviations) if deviations else 0.0
        return abs(recent - historical) >= max(30.0, 1.5 * scale)

    def decide(
        self,
        context: DecisionContext,
        eligible_actions: tuple[str, ...],
        *,
        random_draw: float = 1.0,
    ) -> RankingDecision:
        actions = tuple(sorted(set(eligible_actions)))
        if not actions:
            raise ValueError("ranking requires at least one known feasible action")
        counts = self.state.action_counts()
        for action in actions:
            if counts[action] < 1:
                raise ValueError(f"eligible action {action} has no observed outcome")

        deployable = tuple(
            action
            for action in actions
            if counts[action] >= self.config.minimum_established_evidence
        )
        if not deployable:
            raise ValueError("ranking requires at least one established fallback action")

        predictions = self.state.policy_predictions(actions, context)
        policy_actions = {
            "most_common": max(deployable, key=lambda action: (counts[action], action)),
            "overall_median": self._choose({action: predictions["overall_median"][action] for action in deployable}),
            "time_bucket": self._choose({action: predictions["time_bucket"][action] for action in deployable}),
            "recency": self._choose({action: predictions["recency"][action] for action in deployable}),
            "temporal_kernel": self._choose({action: predictions["temporal_kernel"][action] for action in deployable}),
        }
        # Localized-delay influence stays behind its historical contribution gate.
        # ObservedTripOutcome carries no localized feature, so the gate is closed
        # and this candidate conservatively inherits the temporal-only choice.
        policy_actions["temporal_delay"] = policy_actions["temporal_kernel"]
        policy_actions["mission4_adaptive"] = self._mission4_mae_action(deployable, context)

        ranking_scores: dict[str, float] = {}
        comparison_counts: dict[str, int] = {}
        for action in actions:
            probabilities = []
            comparisons = 0
            for other in actions:
                if other == action:
                    continue
                probability, count = self.state.pairwise_probability(action, other, context)
                probabilities.append(probability)
                comparisons += count
            ranking_scores[action] = statistics.mean(probabilities) if probabilities else 0.5
            comparison_counts[action] = comparisons

        fallback = policy_actions["overall_median"]
        ranking_action = max(deployable, key=lambda action: (ranking_scores[action], -predictions["temporal_kernel"][action], action))
        pair_probability, pair_count = self.state.pairwise_probability(ranking_action, fallback, context)
        expected_disagreement = predictions["temporal_kernel"][ranking_action] - predictions["temporal_kernel"][fallback]
        enough_ranking_evidence = (
            ranking_action == fallback
            or (
                pair_count >= self.config.minimum_pairwise_comparisons
                and pair_probability >= self.config.safe_improvement_probability
                and expected_disagreement <= self.config.maximum_supported_ranking_disagreement_seconds
            )
        )
        recommended = ranking_action if enough_ranking_evidence else fallback
        abstention_reason = None
        if ranking_action != fallback and not enough_ranking_evidence:
            abstention_reason = "ranking evidence did not clear the safe-improvement gate; trusted overall median retained"
        policy_actions["mission5_ranking"] = recommended

        estimates = []
        for action in actions:
            expected, uncertainty, evidence, recent = self.state.predict(action, context, "temporal_kernel")
            maturity = "candidate" if evidence < 5 else "repeated" if evidence < 8 else "established"
            estimates.append(
                ActionEstimate(
                    action,
                    expected,
                    uncertainty,
                    evidence,
                    recent,
                    maturity,
                    "pairwise-ranking+temporal-kernel-outcome",
                    ranking_scores[action],
                )
            )
        estimates_tuple = tuple(sorted(estimates, key=lambda item: item.action_id))
        estimate_by_action = {item.action_id: item for item in estimates_tuple}
        fastest_expected = min(
            (item for item in estimates_tuple if item.action_id in deployable),
            key=lambda item: (item.expected_seconds, item.action_id),
        ).action_id
        detected_drift = any(self._drift_detected(action) for action in actions)

        exploration = self._exploration_candidate(
            context,
            actions,
            recommended,
            estimate_by_action,
            ranking_scores,
            detected_drift,
        )
        propensities = {action: 0.0 for action in actions}
        if exploration is None:
            propensities[recommended] = 1.0
            selected = recommended
        else:
            propensities[recommended] = 1.0 - exploration.probability
            propensities[exploration.action_id] += exploration.probability
            selected = exploration.action_id if random_draw < exploration.probability else recommended
            if selected == exploration.action_id:
                self._last_exploration_decision = self._decision_number

        if not math.isclose(sum(propensities.values()), 1.0, abs_tol=1e-9):
            raise AssertionError("decision propensities must sum to one")
        advantage = estimate_by_action[fallback].expected_seconds - estimate_by_action[recommended].expected_seconds
        if ranking_action == fallback:
            confidence = min(1.0, comparison_counts[ranking_action] / 12.0) * max(0.0, (ranking_scores[ranking_action] - 0.5) * 2.0)
        else:
            confidence = min(1.0, pair_count / 10.0) * max(0.0, (pair_probability - 0.5) * 2.0)
        decision = RankingDecision(
            context,
            actions,
            fastest_expected,
            recommended,
            selected,
            exploration,
            propensities,
            policy_actions,
            estimates_tuple,
            self.VERSION,
            fallback,
            advantage,
            confidence,
            detected_drift,
            abstention_reason,
        )
        self._decision_number += 1
        return decision

    def _exploration_candidate(
        self,
        context: DecisionContext,
        actions: tuple[str, ...],
        recommended: str,
        estimates: dict[str, ActionEstimate],
        ranking_scores: dict[str, float],
        detected_drift: bool,
    ) -> ExplorationEvidence | None:
        config = self.config
        if len(actions) < 2:
            return None
        if context.route_identity_confidence < config.minimum_route_identity_confidence:
            return None
        if context.observation_quality < config.minimum_observation_quality:
            return None
        if self._decision_number - self._last_exploration_decision <= config.cooldown_decisions:
            return None
        best_expected = estimates[recommended].expected_seconds
        candidates: list[tuple[float, str, float, str]] = []
        for action in actions:
            if action == recommended:
                continue
            estimate = estimates[action]
            expected_cost = estimate.expected_seconds - best_expected
            if expected_cost > config.maximum_expected_cost_seconds:
                continue
            stale = estimate.recent_count <= max(1, estimate.evidence_count // 3)
            immature = estimate.evidence_count < 8
            uncertain = estimate.uncertainty_seconds >= config.minimum_uncertainty_seconds
            if not (uncertain or stale or immature or detected_drift):
                continue
            information = (
                estimate.uncertainty_seconds / max(1.0, config.minimum_uncertainty_seconds)
                + float(stale)
                + 0.7 * float(immature)
                + 0.8 * float(detected_drift)
                + max(0.0, 0.55 - ranking_scores[action])
            )
            reason_bits = []
            if immature:
                reason_bits.append("newly maturing route")
            if stale:
                reason_bits.append("stale evidence")
            if uncertain:
                reason_bits.append("material uncertainty")
            if detected_drift:
                reason_bits.append("recent drift")
            candidates.append((information, action, max(0.0, expected_cost), ", ".join(reason_bits)))
        if not candidates:
            return None
        information, action, expected_cost, reason = max(candidates, key=lambda item: (item[0], -item[2], item[1]))
        estimate = estimates[action]
        uncertainty_factor = min(1.0, max(0.0, (estimate.uncertainty_seconds - 16.0) / 65.0))
        evidence_decay = 1.0 / (1.0 + max(0, estimate.evidence_count - 4) / 10.0)
        cost_factor = max(0.15, 1.0 - expected_cost / max(1.0, config.maximum_expected_cost_seconds))
        drift_factor = 1.0 if detected_drift else 0.85
        probability = config.maximum_probability * uncertainty_factor * evidence_decay * cost_factor * drift_factor
        if estimate.evidence_count < config.minimum_established_evidence:
            # A repeated-but-not-established alternative gets a short, still
            # budget-capped opportunity to collect the two observations needed
            # for normal lifecycle graduation.
            probability = max(probability, config.maximum_probability)
        probability = min(config.maximum_probability, max(0.01, probability))
        return ExplorationEvidence(action, expected_cost, probability, information, reason)
