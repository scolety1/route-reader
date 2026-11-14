"""End-to-end learning pipeline operating on observable points only."""

from __future__ import annotations

from dataclasses import dataclass

from .delays import discover_delay_zones, extract_delay_events
from .discovery import DiscoveredJourneys, discover_recurring_journeys
from .ingestion import segment_trips
from .model import TemporalRouteModel
from .trajectories import assign_to_routes, cluster_routes
from .types import DelayEvent, DelayZone, ObservablePoint, RouteCluster, Trip


@dataclass(frozen=True)
class LearnedRouteReader:
    all_trips: tuple[Trip, ...]
    train_trips: tuple[Trip, ...]
    test_trips: tuple[Trip, ...]
    discovery: DiscoveredJourneys
    route_clusters: tuple[RouteCluster, ...]
    train_route_assignments: dict[str, str]
    test_route_assignments: dict[str, str]
    delay_events: tuple[DelayEvent, ...]
    delay_zones: tuple[DelayZone, ...]
    model: TemporalRouteModel


def chronological_split(
    trips: tuple[Trip, ...], train_fraction: float = 0.8
) -> tuple[tuple[Trip, ...], tuple[Trip, ...]]:
    if not 0.5 <= train_fraction < 1.0:
        raise ValueError("train_fraction must be in [0.5, 1.0)")
    ordered = tuple(sorted(trips, key=lambda trip: trip.departure))
    split_index = max(1, min(len(ordered) - 1, int(len(ordered) * train_fraction)))
    return ordered[:split_index], ordered[split_index:]


def learn_from_raw_points(
    raw_points: tuple[ObservablePoint, ...], train_fraction: float = 0.8
) -> LearnedRouteReader:
    trips = segment_trips(raw_points)
    if len(trips) < 8:
        raise RuntimeError(f"too few trips discovered: {len(trips)}")
    train_trips, test_trips = chronological_split(trips, train_fraction)
    discovery = discover_recurring_journeys(train_trips)
    if not discovery.od_pairs:
        raise RuntimeError("no recurring origin/destination pair discovered")
    primary_od = max(discovery.od_pairs, key=lambda pair: len(pair.trip_ids))
    primary_ids = set(primary_od.trip_ids)
    route_train_trips = tuple(
        trip for trip in train_trips if trip.trip_id in primary_ids
    )
    clusters = cluster_routes(route_train_trips)
    train_assignments = {
        trip_id: cluster.cluster_id
        for cluster in clusters
        for trip_id in cluster.trip_ids
    }
    test_assignments = assign_to_routes(test_trips, clusters)
    events = extract_delay_events(route_train_trips, train_assignments)
    zones = discover_delay_zones(events)
    model = TemporalRouteModel.fit(route_train_trips, train_assignments, zones)
    return LearnedRouteReader(
        all_trips=trips,
        train_trips=train_trips,
        test_trips=test_trips,
        discovery=discovery,
        route_clusters=clusters,
        train_route_assignments=train_assignments,
        test_route_assignments=test_assignments,
        delay_events=events,
        delay_zones=zones,
        model=model,
    )
