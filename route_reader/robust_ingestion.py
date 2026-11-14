"""Segmentation for mixed, partial, low-quality personal location histories."""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from datetime import datetime, timedelta

from .geo import euclidean, median_smooth_xy
from .types import ObservablePoint


@dataclass(frozen=True)
class RobustTrip:
    trip_id: str
    points: tuple[ObservablePoint, ...]
    departure: datetime
    arrival: datetime
    origin_latitude: float
    origin_longitude: float
    destination_latitude: float
    destination_longitude: float
    observation_quality: float
    median_accuracy_m: float
    maximum_gap_seconds: float
    has_origin_dwell: bool
    has_destination_dwell: bool

    @property
    def duration_seconds(self) -> float:
        return (self.arrival - self.departure).total_seconds()

    @property
    def departure_minute(self) -> float:
        return self.departure.hour * 60 + self.departure.minute + self.departure.second / 60.0


def _sessions(points: tuple[ObservablePoint, ...], gap: timedelta = timedelta(minutes=75)) -> list[list[ObservablePoint]]:
    ordered = sorted(points, key=lambda item: item.timestamp)
    sessions: list[list[ObservablePoint]] = []
    for point in ordered:
        if not sessions or point.timestamp - sessions[-1][-1].timestamp > gap:
            sessions.append([point])
        elif point.timestamp > sessions[-1][-1].timestamp:
            sessions[-1].append(point)
    return sessions


def _qualified_runs(session: list[ObservablePoint]) -> tuple[list[tuple[int, int]], tuple[tuple[float, float], ...]]:
    xy = median_smooth_xy(session, window=3)
    moving = []
    for index in range(len(session) - 1):
        dt = (session[index + 1].timestamp - session[index].timestamp).total_seconds()
        if dt <= 0:
            continue
        speed = euclidean(xy[index], xy[index + 1]) / dt
        if 1.9 <= speed <= 48.0:
            moving.append(index)
    if not moving:
        return [], xy
    groups: list[list[int]] = [[moving[0]]]
    for index in moving[1:]:
        prior = groups[-1][-1]
        elapsed = (session[index].timestamp - session[prior + 1].timestamp).total_seconds()
        if index - prior <= 3 and elapsed <= 85.0:
            groups[-1].append(index)
        else:
            groups.append([index])
    runs = []
    for group in groups:
        start, end = group[0], group[-1]
        displacement = euclidean(xy[start], xy[end + 1])
        duration = (session[end + 1].timestamp - session[start].timestamp).total_seconds()
        if len(group) >= 2 and displacement >= 75.0 and duration >= 16.0:
            runs.append((start, end))
    return runs, xy


def _heading(xy: tuple[tuple[float, float], ...], run: tuple[int, int], at_end: bool) -> tuple[float, float]:
    if at_end:
        left, right = max(run[0], run[1] - 2), run[1] + 1
    else:
        left, right = run[0], min(run[1] + 1, run[0] + 3)
    dx, dy = xy[right][0] - xy[left][0], xy[right][1] - xy[left][1]
    length = math.hypot(dx, dy)
    return (0.0, 0.0) if length == 0 else (dx / length, dy / length)


def _should_join(
    session: list[ObservablePoint],
    xy: tuple[tuple[float, float], ...],
    previous: tuple[int, int],
    current: tuple[int, int],
) -> bool:
    gap = (session[current[0]].timestamp - session[previous[1] + 1].timestamp).total_seconds()
    if gap <= 215:
        return True
    if gap >= 620:
        return False
    incoming = _heading(xy, previous, at_end=True)
    outgoing = _heading(xy, current, at_end=False)
    alignment = incoming[0] * outgoing[0] + incoming[1] * outgoing[1]
    stop_displacement = euclidean(xy[previous[1] + 1], xy[current[0]])
    # Continue through a gas/food stop when movement resumes in the same direction;
    # split a short two-trip chain when the person leaves in a new direction.
    return alignment >= 0.38 and stop_displacement <= 190.0


def segment_trips_robust(points: tuple[ObservablePoint, ...]) -> tuple[RobustTrip, ...]:
    trips: list[RobustTrip] = []
    for session in _sessions(points):
        if len(session) < 5:
            continue
        runs, xy = _qualified_runs(session)
        if not runs:
            continue
        groups: list[list[tuple[int, int]]] = [[runs[0]]]
        for run in runs[1:]:
            if _should_join(session, xy, groups[-1][-1], run):
                groups[-1].append(run)
            else:
                groups.append([run])
        for group_index, group in enumerate(groups):
            start, end = group[0][0], group[-1][1] + 1
            duration = (session[end].timestamp - session[start].timestamp).total_seconds()
            displacement = euclidean(xy[start], xy[end])
            if duration < 70 or displacement < 160:
                continue
            previous_boundary = groups[group_index - 1][-1][1] + 1 if group_index > 0 else 0
            next_boundary = groups[group_index + 1][0][0] if group_index + 1 < len(groups) else len(session) - 1
            before = session[previous_boundary : start + 1]
            after = session[end : next_boundary + 1]
            has_origin_dwell = len(before) >= 3 and (before[-1].timestamp - before[0].timestamp).total_seconds() >= 80
            has_destination_dwell = len(after) >= 3 and (after[-1].timestamp - after[0].timestamp).total_seconds() >= 80
            origin_points = before if has_origin_dwell else session[start : min(end + 1, start + 3)]
            destination_points = after if has_destination_dwell else session[max(start, end - 2) : end + 1]
            trip_points = tuple(session[start : end + 1])
            gaps = [
                (right.timestamp - left.timestamp).total_seconds()
                for left, right in zip(trip_points, trip_points[1:])
            ]
            median_accuracy = statistics.median(point.accuracy_m for point in trip_points)
            maximum_gap = max(gaps, default=0.0)
            density = min(1.0, len(trip_points) / max(8.0, duration / 18.0))
            accuracy_score = math.exp(-median_accuracy / 42.0)
            gap_score = math.exp(-max(0.0, maximum_gap - 25.0) / 210.0)
            endpoint_score = 0.7 + 0.15 * has_origin_dwell + 0.15 * has_destination_dwell
            quality = max(0.05, min(1.0, density * accuracy_score * gap_score * endpoint_score))
            trips.append(
                RobustTrip(
                    trip_id=f"RT{len(trips) + 1:04d}",
                    points=trip_points,
                    departure=trip_points[0].timestamp,
                    arrival=trip_points[-1].timestamp,
                    origin_latitude=statistics.median(point.latitude for point in origin_points),
                    origin_longitude=statistics.median(point.longitude for point in origin_points),
                    destination_latitude=statistics.median(point.latitude for point in destination_points),
                    destination_longitude=statistics.median(point.longitude for point in destination_points),
                    observation_quality=quality,
                    median_accuracy_m=median_accuracy,
                    maximum_gap_seconds=maximum_gap,
                    has_origin_dwell=has_origin_dwell,
                    has_destination_dwell=has_destination_dwell,
                )
            )
    return tuple(sorted(trips, key=lambda item: item.departure))
