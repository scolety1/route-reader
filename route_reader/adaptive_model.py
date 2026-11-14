"""Chronological model selection and empirical calibration for personal routes.

This runtime module deliberately knows nothing about simulator identities.  Every
candidate is evaluated by rolling origin: a prediction at time *t* can only use
observations before *t*.  Complexity has to earn deployment through lower recent
error, rather than being enabled merely because enough rows exist.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from datetime import datetime

from .robust_delays import RobustDelayZone
from .robust_ingestion import RobustTrip


MODEL_ORDER = ("overall_median", "time_bucket", "recency", "temporal_kernel", "temporal_delay")
MODEL_COMPLEXITY = {name: index for index, name in enumerate(MODEL_ORDER)}
MODEL_MIN_RESIDUALS = {
    "overall_median": 4,
    "time_bucket": 6,
    "recency": 5,
    "temporal_kernel": 8,
    "temporal_delay": 10,
}


@dataclass(frozen=True)
class ModelEvidence:
    model: str
    walk_forward_mae_seconds: float | None
    validation_predictions: int
    eligible: bool
    penalized_score_seconds: float | None
    reason: str


@dataclass(frozen=True)
class AdaptivePrediction:
    variant_id: str
    model_used: str
    expected_seconds: float
    interval_low_seconds: float
    interval_high_seconds: float
    support_count: int
    calibration_residuals: int
    calibration_cutoff: datetime | None
    model_selection_confidence: float
    route_identity_confidence: float
    observation_quality_confidence: float
    departure_familiarity: float
    expected_delay_seconds: float
    delay_feature_enabled: bool
    selection_reason: str
    evidence: tuple[ModelEvidence, ...]


@dataclass(frozen=True)
class AdaptiveExplorationCandidate:
    variant_id: str
    information_score: float
    expected_slowdown_seconds: float
    reason: str


@dataclass(frozen=True)
class AdaptiveRecommendation:
    departure: datetime
    predictions: tuple[AdaptivePrediction, ...]
    fastest_expected: AdaptivePrediction | None
    exploration_candidate: AdaptiveExplorationCandidate | None
    abstention_reason: str | None


def _weighted_median(values: list[float], weights: list[float]) -> float:
    pairs = sorted(zip(values, weights))
    target = sum(weights) / 2.0
    cumulative = 0.0
    for value, weight in pairs:
        cumulative += weight
        if cumulative >= target:
            return value
    return pairs[-1][0]


def _weighted_mean(values: list[float], weights: list[float]) -> float:
    return sum(value * weight for value, weight in zip(values, weights)) / sum(weights)


def _circular_minutes(left: float, right: float) -> float:
    distance = abs(left - right)
    return min(distance, 1440.0 - distance)


def _quantile(values: list[float], probability: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = int(position)
    upper = min(len(ordered) - 1, lower + 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _zone_delay(trip: RobustTrip, zones: tuple[RobustDelayZone, ...]) -> float:
    return sum(zone.trip_delays.get(trip.trip_id, 0.0) for zone in zones if zone.active)


def _estimate(
    model: str,
    history: tuple[RobustTrip, ...],
    departure: datetime,
    zones: tuple[RobustDelayZone, ...],
) -> tuple[float, float]:
    durations = [trip.duration_seconds for trip in history]
    if not durations:
        raise ValueError("prediction history cannot be empty")
    minute = departure.hour * 60 + departure.minute + departure.second / 60.0
    if model == "overall_median":
        return statistics.median(durations), 0.0
    if model == "time_bucket":
        nearby = [trip.duration_seconds for trip in history if _circular_minutes(trip.departure_minute, minute) <= 45]
        return (statistics.median(nearby) if len(nearby) >= 2 else statistics.median(durations)), 0.0
    ages = [max(0.0, (departure - trip.departure).total_seconds() / 86400.0) for trip in history]
    if model == "recency":
        weights = [0.5 ** (age / 35.0) * max(0.15, trip.observation_quality) for age, trip in zip(ages, history)]
        return _weighted_median(durations, weights), 0.0
    bandwidth = 35.0
    weights = [
        math.exp(-0.5 * (_circular_minutes(trip.departure_minute, minute) / bandwidth) ** 2)
        * 0.5 ** (age / 50.0)
        * (1.12 if trip.departure.weekday() == departure.weekday() else 1.0)
        * max(0.15, trip.observation_quality)
        for age, trip in zip(ages, history)
    ]
    if sum(weights) < 0.35:
        weights = [0.5 ** (age / 35.0) * max(0.15, trip.observation_quality) for age, trip in zip(ages, history)]
    if model == "temporal_kernel":
        return _weighted_median(durations, weights), 0.0
    baseline = [trip.duration_seconds - _zone_delay(trip, zones) for trip in history]
    expected_delay = _weighted_median([_zone_delay(trip, zones) for trip in history], weights)
    return _weighted_mean(baseline, weights) + expected_delay, expected_delay


class AdaptiveRouteModel:
    """Per-route selector with walk-forward scoring and chronological residuals."""

    def __init__(
        self,
        variant_id: str,
        trips: tuple[RobustTrip, ...],
        zones: tuple[RobustDelayZone, ...] = (),
        route_identity_confidence: float = 1.0,
    ):
        self.variant_id = variant_id
        self.trips = tuple(sorted(trips, key=lambda item: item.departure))
        self.zones = tuple(zone for zone in zones if zone.variant_id == variant_id and zone.active)
        self.route_identity_confidence = max(0.0, min(1.0, route_identity_confidence))
        self._residuals = self._walk_forward()
        self.evidence, self.selected_model = self._select()

    def _walk_forward(self) -> dict[str, list[tuple[datetime, float]]]:
        result = {model: [] for model in MODEL_ORDER}
        for index in range(3, len(self.trips)):
            query = self.trips[index]
            prior = self.trips[:index]
            for model in MODEL_ORDER:
                if model == "temporal_delay" and not self.zones:
                    continue
                estimate, _ = _estimate(model, prior, query.departure, self.zones)
                result[model].append((query.departure, abs(estimate - query.duration_seconds)))
        return result

    def _select(self) -> tuple[tuple[ModelEvidence, ...], str]:
        evidence = []
        scores: dict[str, float] = {}
        for model in MODEL_ORDER:
            residuals = self._residuals[model]
            minimum = MODEL_MIN_RESIDUALS[model]
            eligible = len(residuals) >= minimum and (model != "temporal_delay" or bool(self.zones))
            if residuals:
                latest = residuals[-1][0]
                weights = [0.5 ** (max(0.0, (latest - stamp).total_seconds() / 86400.0) / 55.0) for stamp, _ in residuals]
                mae = sum(weight * error for weight, (_, error) in zip(weights, residuals)) / sum(weights)
            else:
                mae = None
            # Penalty fades with evidence but remains a real tie-breaker.  It is
            # deliberately expressed in seconds so selection stays auditable.
            penalty = MODEL_COMPLEXITY[model] * 1.0 / math.sqrt(max(1, len(residuals)))
            score = None if mae is None else mae + penalty
            if eligible and score is not None:
                scores[model] = score
            if model == "temporal_delay" and self.zones and eligible:
                kernel = next((item.walk_forward_mae_seconds for item in evidence if item.model == "temporal_kernel"), None)
                if kernel is None or mae is None or mae >= kernel - 2.0:
                    eligible = False
                    scores.pop(model, None)
                    reason = "delay feature failed the positive walk-forward contribution gate"
                else:
                    reason = f"delay feature improved temporal validation by {kernel - mae:.1f}s"
            elif eligible:
                reason = "minimum chronological evidence met"
            else:
                reason = f"requires {minimum} walk-forward residuals" if model != "temporal_delay" or not self.zones else "no accepted active delay zone"
            evidence.append(ModelEvidence(model, mae, len(residuals), eligible, score if eligible else None, reason))
        selected = min(scores, key=lambda name: (scores[name], MODEL_COMPLEXITY[name])) if scores else "overall_median"
        # One-standard-error-inspired simplicity rule: a richer model must beat
        # the best simpler eligible model by a material quarter-second margin.
        for simpler in MODEL_ORDER[: MODEL_ORDER.index(selected)]:
            if simpler in scores and scores[simpler] <= scores[selected] + 0.25:
                selected = simpler
                break
        return tuple(evidence), selected

    def predict(self, departure: datetime) -> AdaptivePrediction:
        estimate, expected_delay = _estimate(self.selected_model, self.trips, departure, self.zones)
        residual_pairs = self._residuals[self.selected_model]
        residuals = [value for _, value in residual_pairs]
        empirical = _quantile(residuals, 0.90) if residuals else 150.0
        quality = statistics.mean(trip.observation_quality for trip in self.trips)
        minute = departure.hour * 60 + departure.minute + departure.second / 60.0
        nearest = min((_circular_minutes(trip.departure_minute, minute) for trip in self.trips), default=180.0)
        familiarity = math.exp(-nearest / 75.0)
        limited = 75.0 / math.sqrt(max(1, len(residuals)))
        disagreement = max(_estimate(model, self.trips, departure, self.zones)[0] for model in MODEL_ORDER[:-1]) - min(
            _estimate(model, self.trips, departure, self.zones)[0] for model in MODEL_ORDER[:-1]
        )
        half_width = max(28.0, empirical + limited + 0.18 * disagreement) * (1.0 + 0.45 * (1.0 - quality))
        eligible = [item for item in self.evidence if item.eligible and item.penalized_score_seconds is not None]
        scores = sorted(item.penalized_score_seconds for item in eligible if item.penalized_score_seconds is not None)
        margin = (scores[1] - scores[0]) if len(scores) > 1 else 0.0
        model_confidence = min(1.0, len(residuals) / 20.0) * min(1.0, margin / 12.0 + 0.35)
        selected_evidence = next(item for item in self.evidence if item.model == self.selected_model)
        reason = (
            f"{self.selected_model} selected from {selected_evidence.validation_predictions} chronological predictions; "
            f"walk-forward MAE {selected_evidence.walk_forward_mae_seconds:.1f}s"
            if selected_evidence.walk_forward_mae_seconds is not None
            else "overall median fallback: insufficient chronological validation history"
        )
        return AdaptivePrediction(
            self.variant_id,
            self.selected_model,
            estimate,
            max(0.0, estimate - half_width),
            estimate + half_width,
            len(self.trips),
            len(residuals),
            residual_pairs[-1][0] if residual_pairs else None,
            model_confidence,
            self.route_identity_confidence,
            quality,
            familiarity,
            expected_delay,
            self.selected_model == "temporal_delay",
            reason,
            self.evidence,
        )


class AdaptiveJourneyModel:
    def __init__(self, route_models: tuple[AdaptiveRouteModel, ...]):
        self.route_models = route_models

    def predict_all(self, departure: datetime) -> AdaptiveRecommendation:
        if not self.route_models:
            return AdaptiveRecommendation(departure, (), None, None, "insufficient history")
        predictions = tuple(sorted((model.predict(departure) for model in self.route_models), key=lambda item: item.expected_seconds))
        if max(item.observation_quality_confidence for item in predictions) < 0.16:
            return AdaptiveRecommendation(departure, predictions, None, None, "current GPS history too degraded")
        if max(item.route_identity_confidence for item in predictions) < 0.35:
            return AdaptiveRecommendation(departure, predictions, None, None, "route identity uncertain")
        mature = [item for item in predictions if item.support_count >= 3]
        if not mature:
            return AdaptiveRecommendation(departure, predictions, None, None, "insufficient history")
        fastest = mature[0]
        candidate = None
        alternatives = [item for item in mature[1:] if item.expected_seconds - fastest.expected_seconds <= 90]
        if alternatives:
            exploratory = max(alternatives, key=lambda item: (1.0 - item.model_selection_confidence) / math.sqrt(item.support_count))
            score = 100.0 * (1.0 - exploratory.model_selection_confidence) / math.sqrt(exploratory.support_count)
            if score >= 8.0:
                candidate = AdaptiveExplorationCandidate(
                    exploratory.variant_id,
                    score,
                    exploratory.expected_seconds - fastest.expected_seconds,
                    "near-competitive route with comparatively uncertain evidence",
                )
        intervals_overlap = len(mature) > 1 and mature[1].interval_low_seconds <= fastest.interval_high_seconds
        abstention = "no reliable advantage between alternatives" if intervals_overlap and fastest.model_selection_confidence < 0.28 else None
        return AdaptiveRecommendation(departure, predictions, None if abstention else fastest, candidate, abstention)
