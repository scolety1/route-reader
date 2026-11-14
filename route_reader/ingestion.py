"""Raw-stream ingestion and V0 trip segmentation."""

from __future__ import annotations

import statistics
from datetime import timedelta

from .geo import euclidean, median_smooth_xy
from .types import ObservablePoint, Trip


def split_sessions(
    points: tuple[ObservablePoint, ...], gap: timedelta = timedelta(hours=2)
) -> list[list[ObservablePoint]]:
    ordered = sorted(points, key=lambda point: point.timestamp)
    if not ordered:
        return []
    sessions: list[list[ObservablePoint]] = [[ordered[0]]]
    for point in ordered[1:]:
        if point.timestamp - sessions[-1][-1].timestamp > gap:
            sessions.append([point])
        else:
            sessions[-1].append(point)
    return sessions


def _candidate_motion_runs(
    session: list[ObservablePoint], speed_threshold_mps: float
) -> list[tuple[int, int]]:
    xy = median_smooth_xy(session, window=3)
    moving_edges: list[int] = []
    for edge_index in range(len(session) - 1):
        dt = (session[edge_index + 1].timestamp - session[edge_index].timestamp).total_seconds()
        if dt <= 0:
            continue
        if euclidean(xy[edge_index], xy[edge_index + 1]) / dt >= speed_threshold_mps:
            moving_edges.append(edge_index)
    if not moving_edges:
        return []

    raw_runs: list[list[int]] = [[moving_edges[0]]]
    for edge_index in moving_edges[1:]:
        prior = raw_runs[-1][-1]
        elapsed = (
            session[edge_index].timestamp - session[prior + 1].timestamp
        ).total_seconds()
        if edge_index - prior <= 3 and elapsed <= 38.0:
            raw_runs[-1].append(edge_index)
        else:
            raw_runs.append([edge_index])

    valid: list[tuple[int, int]] = []
    for run in raw_runs:
        start, end = run[0], run[-1]
        displacement = euclidean(xy[start], xy[end + 1])
        duration = (session[end + 1].timestamp - session[start].timestamp).total_seconds()
        if len(run) >= 3 and displacement >= 90.0 and duration >= 20.0:
            valid.append((start, end))
    return valid


def segment_trips(
    points: tuple[ObservablePoint, ...],
    speed_threshold_mps: float = 2.1,
    max_in_trip_pause_seconds: float = 230.0,
) -> tuple[Trip, ...]:
    """Find journeys without place or route labels.

    Short movement bursts caused by stationary GPS outliers are rejected. Sustained
    movement runs separated by intersection dwell are joined into one trip.
    """

    trips: list[Trip] = []
    for session in split_sessions(points):
        if len(session) < 12:
            continue
        runs = _candidate_motion_runs(session, speed_threshold_mps)
        if not runs:
            continue
        merged: list[list[tuple[int, int]]] = [[runs[0]]]
        for run in runs[1:]:
            previous = merged[-1][-1]
            gap = (
                session[run[0]].timestamp - session[previous[1] + 1].timestamp
            ).total_seconds()
            if gap <= max_in_trip_pause_seconds:
                merged[-1].append(run)
            else:
                merged.append([run])
        group = max(
            merged,
            key=lambda candidate: (
                session[candidate[-1][1] + 1].timestamp
                - session[candidate[0][0]].timestamp
            ).total_seconds(),
        )
        start_index = group[0][0]
        end_index = group[-1][1] + 1
        duration = (
            session[end_index].timestamp - session[start_index].timestamp
        ).total_seconds()
        if duration < 120.0:
            continue

        before = session[: start_index + 1]
        after = session[end_index:]
        origin_lat = statistics.median(point.latitude for point in before)
        origin_lon = statistics.median(point.longitude for point in before)
        destination_lat = statistics.median(point.latitude for point in after)
        destination_lon = statistics.median(point.longitude for point in after)
        trip_points = tuple(session[start_index : end_index + 1])
        trips.append(
            Trip(
                trip_id=f"T{len(trips) + 1:03d}",
                points=trip_points,
                departure=trip_points[0].timestamp,
                arrival=trip_points[-1].timestamp,
                origin_latitude=origin_lat,
                origin_longitude=origin_lon,
                destination_latitude=destination_lat,
                destination_longitude=destination_lon,
            )
        )
    return tuple(trips)
