"""Drift-aware temporal prediction, calibrated uncertainty, and exploration value."""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from datetime import datetime, timedelta

from .robust_delays import RobustDelayZone
from .robust_ingestion import RobustTrip


@dataclass(frozen=True)
class RobustPrediction:
    variant_id: str
    predicted_seconds: float
    interval_low_seconds: float
    interval_high_seconds: float
    confidence: str
    total_trips: int
    relevant_trips: int
    recent_trips: int
    effective_sample_size: float
    mean_observation_quality: float
    evidence_age_days: float
    recent_vs_historical_seconds: float
    expected_delay_seconds: float
    exploration_score: float


@dataclass(frozen=True)
class ExplorationCandidate:
    variant_id: str
    expected_slowdown_seconds: float
    exploration_score: float
    reason: str


@dataclass(frozen=True)
class RobustRecommendation:
    departure: datetime
    fastest_expected: RobustPrediction
    predictions: tuple[RobustPrediction, ...]
    exploration_candidate: ExplorationCandidate | None


@dataclass(frozen=True)
class _History:
    variant_id: str
    trips: tuple[RobustTrip, ...]
    zones: tuple[RobustDelayZone, ...]


def _weighted_mean(values: list[float], weights: list[float]) -> float:
    total = sum(weights)
    return statistics.median(values) if total <= 0 else sum(value * weight for value, weight in zip(values, weights)) / total


def _weighted_std(values: list[float], weights: list[float], center: float) -> float:
    total = sum(weights)
    return 0.0 if total <= 0 else math.sqrt(sum(weight * (value - center) ** 2 for value, weight in zip(values, weights)) / total)


def _effective_n(weights: list[float]) -> float:
    denominator = sum(weight * weight for weight in weights)
    return 0.0 if denominator == 0 else sum(weights) ** 2 / denominator


class RobustTemporalModel:
    def __init__(self, histories: tuple[_History, ...], recency_half_life_days: float | None = 72.0):
        self.histories = histories
        self.recency_half_life_days = recency_half_life_days

    @classmethod
    def fit(
        cls,
        trips: tuple[RobustTrip, ...],
        assignments: dict[str, str],
        zones: tuple[RobustDelayZone, ...],
        recency_half_life_days: float | None = 72.0,
    ) -> "RobustTemporalModel":
        histories = []
        for variant_id in sorted(set(assignments.values())):
            route_trips = tuple(trip for trip in trips if assignments.get(trip.trip_id) == variant_id)
            route_zones = tuple(zone for zone in zones if zone.variant_id == variant_id and zone.active)
            histories.append(_History(variant_id, route_trips, route_zones))
        return cls(tuple(histories), recency_half_life_days)

    def _weights(self, trips: tuple[RobustTrip, ...], departure: datetime, bandwidth: float) -> list[float]:
        query_minute = departure.hour * 60 + departure.minute + departure.second / 60.0
        weights = []
        for trip in trips:
            temporal = math.exp(-0.5 * ((trip.departure_minute - query_minute) / bandwidth) ** 2)
            weekday = 1.15 if trip.departure.weekday() == departure.weekday() else 1.0
            age_days = max(0.0, (departure - trip.departure).total_seconds() / 86400.0)
            recency = 1.0 if self.recency_half_life_days is None else 0.5 ** (age_days / self.recency_half_life_days)
            weights.append(temporal * weekday * recency * max(0.12, trip.observation_quality))
        return weights

    def _predict(self, history: _History, departure: datetime, use_delay_zones: bool) -> RobustPrediction:
        trips = history.trips
        durations = [trip.duration_seconds for trip in trips]
        query_minute = departure.hour * 60 + departure.minute + departure.second / 60.0
        weights = self._weights(trips, departure, 22.0)
        effective = _effective_n(weights)
        recent_cutoff = departure - timedelta(days=55)
        recent = [trip for trip in trips if trip.departure >= recent_cutoff]
        relevant = sum(abs(trip.departure_minute - query_minute) <= 25 for trip in trips)
        expected_delay = 0.0
        if use_delay_zones and history.zones:
            baseline_values = [duration - sum(zone.trip_delays.get(trip.trip_id, 0.0) for zone in history.zones) for duration, trip in zip(durations, trips)]
            estimate = _weighted_mean(baseline_values, weights)
            zone_weights = self._weights(trips, departure, 13.0)
            for zone in history.zones:
                zone_delay = _weighted_mean([zone.trip_delays.get(trip.trip_id, 0.0) for trip in trips], zone_weights)
                expected_delay += zone_delay
            estimate += expected_delay
        else:
            estimate = _weighted_mean(durations, weights)
        historical = statistics.median(durations)
        recent_mean = statistics.mean(trip.duration_seconds for trip in recent) if recent else historical
        disagreement = recent_mean - historical
        center = _weighted_mean(durations, weights)
        reliability = _weighted_std(durations, weights, center)
        mean_quality = _weighted_mean([trip.observation_quality for trip in trips], weights)
        nearest_time = min((abs(trip.departure_minute - query_minute) for trip in trips), default=90.0)
        unfamiliarity = 1.0 + min(1.2, nearest_time / 70.0)
        quality_penalty = 1.0 + (1.0 - mean_quality) * 1.3
        epistemic = 70.0 / math.sqrt(max(0.7, effective))
        half_width = max(28.0, 1.10 * math.sqrt(reliability**2 + epistemic**2 + (0.35 * disagreement) ** 2) * unfamiliarity * quality_penalty)
        evidence_age = max(0.0, (departure - max(trip.departure for trip in trips)).total_seconds() / 86400.0)
        if len(trips) < 3:
            confidence = "insufficient"
            half_width = max(half_width, 165.0)
        elif len(trips) < 5 or relevant < 2:
            confidence = "tentative"
        elif len(trips) >= 12 and len(recent) >= 4 and mean_quality >= 0.55 and half_width <= 125:
            confidence = "strong"
        else:
            confidence = "useful"
        return RobustPrediction(history.variant_id, estimate, max(0.0, estimate - half_width), estimate + half_width, confidence, len(trips), relevant, len(recent), effective, mean_quality, evidence_age, disagreement, expected_delay, 0.0)

    def predict_all(self, departure: datetime, use_delay_zones: bool = True) -> RobustRecommendation:
        predictions = [self._predict(history, departure, use_delay_zones) for history in self.histories]
        predictions.sort(key=lambda item: item.predicted_seconds)
        fastest = predictions[0]
        scored = []
        for prediction in predictions[1:]:
            gap = prediction.predicted_seconds - fastest.predicted_seconds
            proximity = max(0.0, 1.0 - gap / 80.0)
            uncertainty = min(1.0, (prediction.interval_high_seconds - prediction.interval_low_seconds) / 300.0)
            stale = min(1.0, prediction.evidence_age_days / 90.0 + (0.35 if prediction.recent_trips < 3 else 0.0))
            sample_gap = min(1.0, 2.2 / math.sqrt(max(1, prediction.total_trips)))
            score = 100.0 * proximity * (0.42 * uncertainty + 0.33 * stale + 0.25 * sample_gap)
            scored.append((score, gap, prediction))
        candidate = None
        if scored:
            score, gap, prediction = max(scored, key=lambda item: item[0])
            if gap >= 0 and gap <= 80 and score >= 24:
                candidate = ExplorationCandidate(
                    prediction.variant_id,
                    gap,
                    score,
                    f"predicted {gap:.0f}s slower than {fastest.variant_id}, but evidence is sparse, stale, or uncertain",
                )
        return RobustRecommendation(departure, fastest, tuple(predictions), candidate)
