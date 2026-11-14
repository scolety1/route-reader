"""Inference of recurring localized slowdown/dwell structure."""

from __future__ import annotations

import statistics
from collections import defaultdict

from .geo import euclidean, median_smooth_xy
from .types import DelayEvent, DelayZone, Trip


def extract_delay_events(
    trips: tuple[Trip, ...], route_assignments: dict[str, str]
) -> tuple[DelayEvent, ...]:
    events: list[DelayEvent] = []
    for trip in trips:
        xy = median_smooth_xy(trip.points, window=3)
        slow_edges: list[int] = []
        for index in range(len(trip.points) - 1):
            dt = (
                trip.points[index + 1].timestamp - trip.points[index].timestamp
            ).total_seconds()
            if dt <= 0:
                continue
            speed = euclidean(xy[index], xy[index + 1]) / dt
            if (
                speed < 3.0
                and euclidean(xy[index], xy[0]) > 230.0
                and euclidean(xy[index], xy[-1]) > 230.0
            ):
                slow_edges.append(index)
        if not slow_edges:
            continue
        groups: list[list[int]] = [[slow_edges[0]]]
        for index in slow_edges[1:]:
            previous = groups[-1][-1]
            gap = (
                trip.points[index].timestamp - trip.points[previous + 1].timestamp
            ).total_seconds()
            spatial_gap = euclidean(xy[index], xy[previous])
            if index - previous <= 3 and gap <= 35.0 and spatial_gap <= 95.0:
                groups[-1].append(index)
            else:
                groups.append([index])
        for group in groups:
            start, end = group[0], group[-1] + 1
            duration = (
                trip.points[end].timestamp - trip.points[start].timestamp
            ).total_seconds()
            if duration < 12.0:
                continue
            event_points = xy[start : end + 1]
            events.append(
                DelayEvent(
                    trip_id=trip.trip_id,
                    route_cluster_id=route_assignments[trip.trip_id],
                    x_m=statistics.median(point[0] for point in event_points),
                    y_m=statistics.median(point[1] for point in event_points),
                    delay_seconds=duration,
                    departure_minute=trip.departure_minute,
                )
            )
    return tuple(events)


def _event_components(events: list[DelayEvent], radius_m: float) -> list[list[int]]:
    parents = list(range(len(events)))

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parents[right_root] = left_root

    for left in range(len(events)):
        for right in range(left + 1, len(events)):
            if events[left].route_cluster_id != events[right].route_cluster_id:
                continue
            if euclidean(
                (events[left].x_m, events[left].y_m),
                (events[right].x_m, events[right].y_m),
            ) <= radius_m:
                union(left, right)
    components: dict[int, list[int]] = defaultdict(list)
    for index in range(len(events)):
        components[find(index)].append(index)
    return list(components.values())


def discover_delay_zones(
    events: tuple[DelayEvent, ...],
    radius_m: float = 175.0,
    min_affected_trips: int = 3,
) -> tuple[DelayZone, ...]:
    event_list = list(events)
    candidates = []
    for component in _event_components(event_list, radius_m):
        members = [event_list[index] for index in component]
        affected = {event.trip_id for event in members}
        if len(affected) < min_affected_trips:
            continue
        trip_delays: dict[str, float] = defaultdict(float)
        for event in members:
            trip_delays[event.trip_id] += event.delay_seconds
        by_window: dict[int, list[float]] = defaultdict(list)
        for event in members:
            window = int(event.departure_minute // 15) * 15
            by_window[window].append(event.delay_seconds)
        peak_start, peak_values = max(
            by_window.items(), key=lambda item: statistics.median(item[1])
        )
        candidates.append(
            (
                members[0].route_cluster_id,
                statistics.median(event.x_m for event in members),
                statistics.median(event.y_m for event in members),
                len(members),
                len(affected),
                statistics.median(event.delay_seconds for event in members),
                float(peak_start),
                float(peak_start + 15),
                statistics.median(peak_values),
                dict(trip_delays),
            )
        )
    candidates.sort(key=lambda item: (item[0], -item[4], item[1], item[2]))
    return tuple(
        DelayZone(
            zone_id=f"Z{index + 1}",
            route_cluster_id=item[0],
            center_x_m=item[1],
            center_y_m=item[2],
            observations=item[3],
            affected_trips=item[4],
            median_delay_seconds=item[5],
            peak_start_minute=item[6],
            peak_end_minute=item[7],
            peak_median_delay_seconds=item[8],
            trip_delays=item[9],
        )
        for index, item in enumerate(candidates)
    )
