"""Interpretable, uncertainty-aware personal temporal route model."""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from datetime import datetime

from .geo import quantile
from .types import DelayZone, Recommendation, RoutePrediction, Trip


@dataclass(frozen=True)
class _RouteHistory:
    cluster_id: str
    trips: tuple[Trip, ...]
    zones: tuple[DelayZone, ...]


def _weighted_mean(values: list[float], weights: list[float]) -> float:
    total = sum(weights)
    if total <= 0:
        return statistics.median(values)
    return sum(value * weight for value, weight in zip(values, weights)) / total


def _weighted_std(values: list[float], weights: list[float], center: float) -> float:
    total = sum(weights)
    if total <= 0:
        return statistics.pstdev(values) if len(values) > 1 else 0.0
    variance = sum(
        weight * (value - center) ** 2 for value, weight in zip(values, weights)
    ) / total
    return math.sqrt(max(0.0, variance))


def _effective_n(weights: list[float]) -> float:
    denominator = sum(weight * weight for weight in weights)
    return 0.0 if denominator == 0 else sum(weights) ** 2 / denominator


class TemporalRouteModel:
    """Kernel smoother decomposing stable motion from inferred local delays."""

    def __init__(self, histories: tuple[_RouteHistory, ...], trained_at: datetime):
        self._histories = histories
        self.trained_at = trained_at

    @classmethod
    def fit(
        cls,
        trips: tuple[Trip, ...],
        route_assignments: dict[str, str],
        zones: tuple[DelayZone, ...],
    ) -> "TemporalRouteModel":
        histories = []
        cluster_ids = sorted(set(route_assignments.values()))
        for cluster_id in cluster_ids:
            route_trips = tuple(
                trip for trip in trips if route_assignments[trip.trip_id] == cluster_id
            )
            route_zones = tuple(
                zone for zone in zones if zone.route_cluster_id == cluster_id
            )
            histories.append(_RouteHistory(cluster_id, route_trips, route_zones))
        trained_at = max(trip.arrival for trip in trips)
        return cls(tuple(histories), trained_at)

    @staticmethod
    def _weights(
        trips: tuple[Trip, ...], departure: datetime, bandwidth_minutes: float
    ) -> list[float]:
        query_minute = departure.hour * 60 + departure.minute + departure.second / 60.0
        weights = []
        for trip in trips:
            time_distance = abs(trip.departure_minute - query_minute)
            time_weight = math.exp(-0.5 * (time_distance / bandwidth_minutes) ** 2)
            weekday_weight = 1.18 if trip.departure.weekday() == departure.weekday() else 1.0
            age_days = max(0.0, (departure - trip.departure).total_seconds() / 86_400.0)
            recency_weight = math.exp(-age_days / 240.0)
            weights.append(time_weight * weekday_weight * recency_weight)
        return weights

    def _predict_history(
        self, history: _RouteHistory, departure: datetime, use_delay_zones: bool
    ) -> RoutePrediction:
        trips = history.trips
        durations = [trip.duration_seconds for trip in trips]
        query_minute = departure.hour * 60 + departure.minute + departure.second / 60.0
        if len(trips) < 3:
            estimate = statistics.median(durations)
            spread = max(90.0, statistics.pstdev(durations) if len(durations) > 1 else 150.0)
            return RoutePrediction(
                history.cluster_id,
                estimate,
                max(0.0, estimate - 1.64 * spread),
                estimate + 1.64 * spread,
                "low",
                len(trips),
                float(len(trips)),
                statistics.median(abs(trip.departure_minute - query_minute) for trip in trips),
                max(0.0, (departure - max(trip.departure for trip in trips)).total_seconds() / 86_400.0),
                spread,
                0.0,
                ("sparse-history fallback to route median",),
            )

        broad_weights = self._weights(trips, departure, 29.0)
        expected_delay = 0.0
        delay_evidence: list[str] = []
        if use_delay_zones and history.zones:
            trip_delay_totals = [
                sum(zone.trip_delays.get(trip.trip_id, 0.0) for zone in history.zones)
                for trip in trips
            ]
            baseline_values = [
                duration - delay
                for duration, delay in zip(durations, trip_delay_totals)
            ]
            baseline = _weighted_mean(baseline_values, broad_weights)
            for zone in history.zones:
                zone_values = [zone.trip_delays.get(trip.trip_id, 0.0) for trip in trips]
                has_sharp_temporal_peak = zone.peak_median_delay_seconds >= max(
                    60.0, zone.median_delay_seconds * 2.0
                )
                zone_weights = self._weights(
                    trips, departure, 8.5 if has_sharp_temporal_peak else 20.0
                )
                zone_delay = _weighted_mean(zone_values, zone_weights)
                expected_delay += zone_delay
                if zone_delay >= 7.0:
                    delay_evidence.append(
                        f"{zone.zone_id}: ~{zone_delay:.0f}s localized dwell expected"
                    )
            estimate = baseline + expected_delay
        else:
            # Ablation: broad time smoothing without localized dwell features.
            estimate = _weighted_mean(durations, broad_weights)

        reliability_center = _weighted_mean(durations, broad_weights)
        reliability = _weighted_std(durations, broad_weights, reliability_center)
        effective = _effective_n(broad_weights)
        half_interval = 1.28 * math.sqrt(reliability**2 + 55.0**2 / max(1.0, effective))
        relevant = sum(
            1 for trip in trips if abs(trip.departure_minute - query_minute) <= 22.0
        )
        time_distances = sorted(abs(trip.departure_minute - query_minute) for trip in trips)
        nearby = time_distances[: max(1, min(5, len(time_distances)))]
        if effective >= 5.0 and relevant >= 4 and half_interval <= 105.0:
            confidence = "high"
        elif effective >= 3.0 and relevant >= 2 and half_interval <= 165.0:
            confidence = "medium"
        else:
            confidence = "low"
        if not delay_evidence:
            delay_evidence.append("no material recurring localized delay expected")
        return RoutePrediction(
            route_cluster_id=history.cluster_id,
            predicted_seconds=estimate,
            interval_low_seconds=max(0.0, estimate - half_interval),
            interval_high_seconds=estimate + half_interval,
            confidence=confidence,
            relevant_trips=relevant,
            effective_sample_size=effective,
            median_time_distance_minutes=statistics.median(nearby),
            evidence_age_days=max(
                0.0,
                (departure - max(trip.departure for trip in trips)).total_seconds() / 86_400.0,
            ),
            reliability_std_seconds=reliability,
            expected_delay_seconds=expected_delay,
            delay_evidence=tuple(delay_evidence),
        )

    def predict_all(
        self, departure: datetime, use_delay_zones: bool = True
    ) -> Recommendation:
        predictions = [
            self._predict_history(history, departure, use_delay_zones)
            for history in self._histories
        ]
        predictions.sort(key=lambda item: item.predicted_seconds)
        exploration_note = None
        best = predictions[0]
        history_counts = {history.cluster_id: len(history.trips) for history in self._histories}
        candidates = [
            prediction
            for prediction in predictions[1:]
            if prediction.predicted_seconds - best.predicted_seconds <= 55.0
            and history_counts[prediction.route_cluster_id]
            < history_counts[best.route_cluster_id]
        ]
        if candidates:
            candidate = min(candidates, key=lambda item: history_counts[item.route_cluster_id])
            exploration_note = (
                f"{candidate.route_cluster_id} is less-tested and predicted within "
                f"{candidate.predicted_seconds - best.predicted_seconds:.0f}s of {best.route_cluster_id}; "
                "choosing it would add evidence, but exploration is not the default."
            )
        return Recommendation(departure, tuple(predictions), exploration_note)

    def route_trip_counts(self) -> dict[str, int]:
        return {history.cluster_id: len(history.trips) for history in self._histories}

    def historical_medians(self) -> dict[str, float]:
        return {
            history.cluster_id: quantile(
                [trip.duration_seconds for trip in history.trips], 0.5
            )
            for history in self._histories
        }
