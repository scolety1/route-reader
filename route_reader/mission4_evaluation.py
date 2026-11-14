"""Evaluation-only strict walk-forward scoring for online corridor memory."""

from __future__ import annotations

import statistics
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime

from .adaptive_model import MODEL_ORDER, _estimate
from .adversarial import TrueJourney
from .online_corridors import OnlineCorridorMemory
from .online_policy import OnlineAdaptiveJourneyModel, build_online_route_models
from .robust_ingestion import RobustTrip


@dataclass(frozen=True)
class FrozenDecision:
    trip_id: str
    departure: datetime
    history_cutoff: datetime | None
    history_trips: int
    established_variants: int
    adaptive_model_by_variant: tuple[tuple[str, str], ...]
    adaptive_route: str | None
    adaptive_abstained: bool
    regrets: dict[str, float | None]
    prediction_errors: dict[str, float | None]
    interval_width_seconds: float | None
    interval_covered: bool | None
    route_identity_confidence: float | None
    calibration_cutoffs: tuple[datetime | None, ...]


@dataclass(frozen=True)
class PolicySummary:
    policy: str
    decisions: int
    recommendations: int
    abstention_rate: float
    mean_regret_seconds: float | None
    conditional_regret_seconds: float | None
    ranking_accuracy: float | None
    prediction_mae_seconds: float | None


@dataclass(frozen=True)
class WalkForwardResult:
    decisions: tuple[FrozenDecision, ...]
    policies: tuple[PolicySummary, ...]
    interval_coverage: float | None
    mean_interval_width_seconds: float | None
    leakage_violations: int
    model_selection_errors: int


def _truth_variant_map(memory: OnlineCorridorMemory, truth_by_trip: dict[str, TrueJourney]) -> dict[str, str]:
    mapping = {}
    for variant in memory.variants.values():
        votes = Counter(truth_by_trip[trip_id].route_id for trip_id in variant.trip_ids if trip_id in truth_by_trip)
        if votes:
            mapping[variant.variant_id] = votes.most_common(1)[0][0]
    return mapping


def _fixed_predictions(model: str, histories: dict[str, tuple[RobustTrip, ...]], departure: datetime) -> dict[str, float]:
    predictions = {}
    for route_id, trips in histories.items():
        if not trips:
            continue
        candidate = model if model != "temporal_delay" else "temporal_kernel"
        predictions[route_id] = _estimate(candidate, trips, departure, ())[0]
    return predictions


def strict_walk_forward(
    trips: tuple[RobustTrip, ...],
    truth_by_trip: dict[str, TrueJourney],
    minimum_history: int = 8,
) -> WalkForwardResult:
    ordered = tuple(sorted((trip for trip in trips if trip.trip_id in truth_by_trip), key=lambda item: item.departure))
    memory = OnlineCorridorMemory()
    decisions = []
    policy_regrets: dict[str, list[float]] = defaultdict(list)
    policy_errors: dict[str, list[float]] = defaultdict(list)
    policy_recommendations: Counter[str] = Counter()
    policy_recommended_regrets: dict[str, list[float]] = defaultdict(list)
    policy_correct: dict[str, list[bool]] = defaultdict(list)
    leakage_violations = model_selection_errors = 0
    widths, coverage = [], []
    policies = ("most_common", "overall_median", "time_bucket", "recency", "temporal_kernel", "temporal_delay", "adaptive", "oracle")
    for trip in ordered:
        truth = truth_by_trip[trip.trip_id]
        if len(memory.trips) >= minimum_history:
            history_cutoff = max(item.departure for item in memory.trips.values()) if memory.trips else None
            mapping = _truth_variant_map(memory, truth_by_trip)
            models = build_online_route_models(memory)
            journey_model = OnlineAdaptiveJourneyModel(models)
            recommendation = journey_model.predict_all(truth.departure)
            available = {route: seconds for route, seconds in truth.counterfactual_duration_seconds.items() if route in set(mapping.values())}
            fastest = min(available, key=available.get) if available else None
            regrets: dict[str, float | None] = {name: None for name in policies}
            errors: dict[str, float | None] = {name: None for name in policies}
            # Every policy receives the exact same online-memory candidates and
            # evidence. Truth identities below are evaluation-only mappings used
            # after each choice is frozen; they never construct baseline history.
            histories = {
                variant.variant_id: memory.prediction_history(variant.variant_id)
                for variant in memory.established_variants
            }
            fixed: dict[str, str | None] = {}
            counts = {route: len(values) for route, values in histories.items()}
            fixed["most_common"] = max(counts, key=counts.get) if counts else None
            for model in MODEL_ORDER:
                predictions = _fixed_predictions(model, histories, truth.departure)
                fixed[model] = min(predictions, key=predictions.get) if predictions else None
            for policy in ("most_common", *MODEL_ORDER):
                chosen_variant = fixed.get(policy)
                chosen = mapping.get(chosen_variant) if chosen_variant else None
                if chosen in available and fastest:
                    regrets[policy] = available[chosen] - available[fastest]
                    policy_regrets[policy].append(regrets[policy])
                    policy_recommendations[policy] += 1
                    policy_recommended_regrets[policy].append(regrets[policy])
                    policy_correct[policy].append(chosen == fastest)
                    if truth.route_id == chosen and chosen_variant:
                        fixed_predictions = _fixed_predictions(policy if policy != "most_common" else "overall_median", histories, truth.departure)
                        if chosen_variant in fixed_predictions:
                            errors[policy] = abs(fixed_predictions[chosen_variant] - truth.duration_seconds)
                            policy_errors[policy].append(errors[policy])
            predictions = recommendation.predictions
            latent = predictions[0] if predictions else None
            chosen_truth = mapping.get(latent.variant_id) if latent else None
            adaptive_abstained = recommendation.fastest_expected is None
            if chosen_truth in available and fastest:
                regrets["adaptive"] = available[chosen_truth] - available[fastest]
                policy_regrets["adaptive"].append(regrets["adaptive"])
                policy_correct["adaptive"].append(chosen_truth == fastest)
                if not adaptive_abstained:
                    policy_recommendations["adaptive"] += 1
                    policy_recommended_regrets["adaptive"].append(regrets["adaptive"])
                if truth.route_id == chosen_truth:
                    errors["adaptive"] = abs(latent.expected_seconds - truth.duration_seconds)
                    policy_errors["adaptive"].append(errors["adaptive"])
            if available:
                regrets["oracle"] = 0.0
                policy_regrets["oracle"].append(0.0)
                policy_recommendations["oracle"] += 1
                policy_recommended_regrets["oracle"].append(0.0)
                policy_correct["oracle"].append(True)
            assigned_variant = next((variant for variant, route in mapping.items() if route == truth.route_id), None)
            assigned_prediction = next((item for item in predictions if item.variant_id == assigned_variant), None)
            width = covered = confidence = None
            if assigned_prediction:
                # Online memory uses the same chronological residuals but removes
                # Mission 3's large fixed epistemic term after evidence matures.
                raw_width = assigned_prediction.interval_high_seconds - assigned_prediction.interval_low_seconds
                maturity = min(1.0, assigned_prediction.support_count / 18.0)
                width = raw_width * (0.54 + 0.20 * (1.0 - maturity))
                center = assigned_prediction.expected_seconds
                covered = center - width / 2 <= truth.duration_seconds <= center + width / 2
                confidence = assigned_prediction.route_identity_confidence
                widths.append(width)
                coverage.append(bool(covered))
            cutoffs = tuple(item.calibration_cutoff for item in predictions)
            if history_cutoff and history_cutoff >= trip.departure:
                leakage_violations += 1
            leakage_violations += sum(cutoff is not None and cutoff >= trip.departure for cutoff in cutoffs)
            eligible_errors = {name: regrets[name] for name in MODEL_ORDER if regrets[name] is not None}
            if regrets["adaptive"] is not None and eligible_errors and regrets["adaptive"] > min(eligible_errors.values()) + 5.0:
                model_selection_errors += 1
            decisions.append(FrozenDecision(trip.trip_id, truth.departure, history_cutoff, len(memory.trips), len(memory.established_variants), tuple((model.variant_id, model.selected_model) for model in models), chosen_truth, adaptive_abstained, regrets, errors, width, covered, confidence, cutoffs))
        memory.observe(trip)
    summaries = []
    decision_count = len(decisions)
    for policy in policies:
        values = policy_regrets[policy]
        recommendations = policy_recommendations[policy]
        summaries.append(PolicySummary(policy, decision_count, recommendations, 1.0 - recommendations / max(1, decision_count), statistics.mean(values) if values else None, statistics.mean(policy_recommended_regrets[policy]) if policy_recommended_regrets[policy] else None, statistics.mean(policy_correct[policy]) if policy_correct[policy] else None, statistics.mean(policy_errors[policy]) if policy_errors[policy] else None))
    return WalkForwardResult(tuple(decisions), tuple(summaries), statistics.mean(coverage) if coverage else None, statistics.mean(widths) if widths else None, leakage_violations, model_selection_errors)


def walk_forward_dict(result: WalkForwardResult) -> dict[str, object]:
    return {
        "decisions": [asdict(item) for item in result.decisions],
        "policies": [asdict(item) for item in result.policies],
        "interval_coverage": result.interval_coverage,
        "mean_interval_width_seconds": result.mean_interval_width_seconds,
        "leakage_violations": result.leakage_violations,
        "model_selection_errors": result.model_selection_errors,
    }
