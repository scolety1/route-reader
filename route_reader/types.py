"""Shared observable and learned data structures.

Simulator truth intentionally lives in ``synthetic.py`` instead of these model-facing
types.  The learning pipeline accepts only ``ObservablePoint`` objects.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


@dataclass(frozen=True)
class ObservablePoint:
    timestamp: datetime
    latitude: float
    longitude: float
    accuracy_m: float


@dataclass(frozen=True)
class Trip:
    trip_id: str
    points: tuple[ObservablePoint, ...]
    departure: datetime
    arrival: datetime
    origin_latitude: float
    origin_longitude: float
    destination_latitude: float
    destination_longitude: float

    @property
    def duration_seconds(self) -> float:
        return (self.arrival - self.departure).total_seconds()

    @property
    def departure_minute(self) -> float:
        return (
            self.departure.hour * 60
            + self.departure.minute
            + self.departure.second / 60.0
        )


@dataclass(frozen=True)
class Place:
    place_id: str
    latitude: float
    longitude: float
    observations: int


@dataclass(frozen=True)
class ODPair:
    od_id: str
    origin_place_id: str
    destination_place_id: str
    trip_ids: tuple[str, ...]


@dataclass(frozen=True)
class RouteCluster:
    cluster_id: str
    trip_ids: tuple[str, ...]
    prototype_xy: tuple[tuple[float, float], ...]


@dataclass(frozen=True)
class DelayEvent:
    trip_id: str
    route_cluster_id: str
    x_m: float
    y_m: float
    delay_seconds: float
    departure_minute: float


@dataclass(frozen=True)
class DelayZone:
    zone_id: str
    route_cluster_id: str
    center_x_m: float
    center_y_m: float
    observations: int
    affected_trips: int
    median_delay_seconds: float
    peak_start_minute: float
    peak_end_minute: float
    peak_median_delay_seconds: float
    trip_delays: dict[str, float] = field(repr=False)


@dataclass(frozen=True)
class RoutePrediction:
    route_cluster_id: str
    predicted_seconds: float
    interval_low_seconds: float
    interval_high_seconds: float
    confidence: str
    relevant_trips: int
    effective_sample_size: float
    median_time_distance_minutes: float
    evidence_age_days: float
    reliability_std_seconds: float
    expected_delay_seconds: float
    delay_evidence: tuple[str, ...]
    exploratory: bool = False


@dataclass(frozen=True)
class Recommendation:
    departure: datetime
    predictions: tuple[RoutePrediction, ...]
    exploration_note: str | None

    @property
    def best(self) -> RoutePrediction:
        return self.predictions[0]
