"""Conservative recurring-delay inference with noise rejection and expiry."""

from __future__ import annotations

import statistics
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta

from .geo import euclidean, median_smooth_xy
from .robust_ingestion import RobustTrip


@dataclass(frozen=True)
class RobustDelayEvent:
    trip_id: str
    variant_id: str
    departure: datetime
    departure_minute: float
    x_m: float
    y_m: float
    seconds: float
    observation_quality: float


@dataclass(frozen=True)
class RobustDelayZone:
    zone_id: str
    variant_id: str
    center_x_m: float
    center_y_m: float
    affected_trips: int
    passing_trips: int
    recurrence_fraction: float
    temporal_lift: float
    median_seconds: float
    peak_window_start: int
    recent_support_fraction: float
    active: bool
    confidence: str
    trip_delays: dict[str, float]


def extract_robust_delay_events(
    trips: tuple[RobustTrip, ...], assignments: dict[str, str]
) -> tuple[RobustDelayEvent, ...]:
    events = []
    for trip in trips:
        if trip.trip_id not in assignments or trip.observation_quality < 0.10:
            continue
        xy = median_smooth_xy(trip.points, window=3)
        slow = []
        for index in range(len(trip.points) - 1):
            dt = (trip.points[index + 1].timestamp - trip.points[index].timestamp).total_seconds()
            if dt <= 0 or dt > 95:
                continue
            speed = euclidean(xy[index], xy[index + 1]) / dt
            if speed < 3.1 and euclidean(xy[index], xy[0]) > 210 and euclidean(xy[index], xy[-1]) > 210:
                slow.append(index)
        if not slow:
            continue
        groups: list[list[int]] = [[slow[0]]]
        for index in slow[1:]:
            prior = groups[-1][-1]
            gap = (trip.points[index].timestamp - trip.points[prior + 1].timestamp).total_seconds()
            if index - prior <= 3 and gap <= 70 and euclidean(xy[index], xy[prior]) <= 130:
                groups[-1].append(index)
            else:
                groups.append([index])
        for group in groups:
            start, end = group[0], group[-1] + 1
            seconds = (trip.points[end].timestamp - trip.points[start].timestamp).total_seconds()
            if seconds < 14:
                continue
            points = xy[start : end + 1]
            events.append(
                RobustDelayEvent(
                    trip.trip_id,
                    assignments[trip.trip_id],
                    trip.departure,
                    trip.departure_minute,
                    statistics.median(point[0] for point in points),
                    statistics.median(point[1] for point in points),
                    seconds,
                    trip.observation_quality,
                )
            )
    return tuple(events)


def discover_robust_delay_zones(
    trips: tuple[RobustTrip, ...], assignments: dict[str, str], events: tuple[RobustDelayEvent, ...]
) -> tuple[RobustDelayZone, ...]:
    by_variant: dict[str, list[RobustDelayEvent]] = defaultdict(list)
    for event in events:
        by_variant[event.variant_id].append(event)
    trip_by_id = {trip.trip_id: trip for trip in trips}
    latest = max((trip.departure for trip in trips), default=datetime.min)
    candidates = []
    for variant_id, variant_events in by_variant.items():
        radius = 185.0 + (1.0 - statistics.mean(event.observation_quality for event in variant_events)) * 85.0
        parents = list(range(len(variant_events)))

        def find(index: int) -> int:
            while parents[index] != index:
                parents[index] = parents[parents[index]]
                index = parents[index]
            return index

        for left in range(len(variant_events)):
            for right in range(left + 1, len(variant_events)):
                if euclidean((variant_events[left].x_m, variant_events[left].y_m), (variant_events[right].x_m, variant_events[right].y_m)) <= radius:
                    a, b = find(left), find(right)
                    if a != b:
                        parents[b] = a
        components: dict[int, list[RobustDelayEvent]] = defaultdict(list)
        for index, event in enumerate(variant_events):
            components[find(index)].append(event)
        passing = [trip for trip in trips if assignments.get(trip.trip_id) == variant_id]
        for members in components.values():
            trip_delays: dict[str, float] = defaultdict(float)
            for event in members:
                trip_delays[event.trip_id] += event.seconds
            affected = len(trip_delays)
            recurrence = affected / max(1, len(passing))
            bins: dict[int, list[str]] = defaultdict(list)
            for trip in passing:
                bins[int(trip.departure_minute // 30) * 30].append(trip.trip_id)
            peak_window, peak_rate = 0, 0.0
            for window, ids in bins.items():
                rate = sum(trip_id in trip_delays for trip_id in ids) / len(ids)
                if len(ids) >= 2 and rate > peak_rate:
                    peak_window, peak_rate = window, rate
            temporal_lift = max(0.0, peak_rate - recurrence)
            recent_ids = {trip.trip_id for trip in passing if trip.departure >= latest - timedelta(days=55)}
            older_ids = {trip.trip_id for trip in passing if trip.trip_id not in recent_ids}
            recent_rate = sum(item in trip_delays for item in recent_ids) / max(1, len(recent_ids))
            older_rate = sum(item in trip_delays for item in older_ids) / max(1, len(older_ids))
            expired = len(recent_ids) >= 3 and older_rate >= 0.32 and recent_rate <= 0.10
            accepted = affected >= 4 and not expired and (recurrence >= 0.34 or temporal_lift >= 0.28)
            if not accepted:
                continue
            confidence = "high" if affected >= 7 and (recurrence >= 0.45 or temporal_lift >= 0.45) else "medium"
            candidates.append(
                (variant_id, statistics.median(event.x_m for event in members), statistics.median(event.y_m for event in members), affected, len(passing), recurrence, temporal_lift, statistics.median(event.seconds for event in members), peak_window, recent_rate, confidence, dict(trip_delays))
            )
    candidates.sort(key=lambda item: (item[0], -item[3], item[1]))
    return tuple(
        RobustDelayZone(f"RZ{index + 1}", item[0], item[1], item[2], item[3], item[4], item[5], item[6], item[7], item[8], item[9], True, item[10], item[11])
        for index, item in enumerate(candidates)
    )
