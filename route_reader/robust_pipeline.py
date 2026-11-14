"""Truth-free orchestration for the adversarial robustness laboratory."""

from __future__ import annotations

from dataclasses import dataclass

from .geo import haversine_m
from .robust_delays import RobustDelayEvent, RobustDelayZone, discover_robust_delay_zones, extract_robust_delay_events
from .robust_discovery import RobustDiscovery, discover_recurring_journeys_robust
from .robust_ingestion import RobustTrip, segment_trips_robust
from .robust_model import RobustTemporalModel
from .route_families import RouteHierarchy, assign_route_variants, cluster_route_hierarchy, hierarchy_train_assignments
from .types import ObservablePoint, Place


@dataclass(frozen=True)
class RobustODLearning:
    od_id: str
    trip_ids: tuple[str, ...]
    hierarchy: RouteHierarchy
    train_assignments: dict[str, str]
    model_assignments: dict[str, str]
    test_assignments: dict[str, str]
    delay_events: tuple[RobustDelayEvent, ...]
    delay_zones: tuple[RobustDelayZone, ...]
    model: RobustTemporalModel
    uniform_history_model: RobustTemporalModel


@dataclass(frozen=True)
class RobustLearnedLab:
    all_trips: tuple[RobustTrip, ...]
    train_trips: tuple[RobustTrip, ...]
    test_trips: tuple[RobustTrip, ...]
    discovery: RobustDiscovery
    test_trip_places: dict[str, tuple[str, str]]
    od_learning: dict[str, RobustODLearning]


def robust_chronological_split(
    trips: tuple[RobustTrip, ...], train_fraction: float = 0.8
) -> tuple[tuple[RobustTrip, ...], tuple[RobustTrip, ...]]:
    ordered = tuple(sorted(trips, key=lambda item: item.departure))
    if len(ordered) < 2:
        return ordered, ()
    split = max(1, min(len(ordered) - 1, int(len(ordered) * train_fraction)))
    return ordered[:split], ordered[split:]


def _assign_places(trips: tuple[RobustTrip, ...], places: tuple[Place, ...], radius_m: float = 325.0) -> dict[str, tuple[str, str]]:
    def nearest(latitude: float, longitude: float, quality: float) -> str:
        if not places:
            return "unknown"
        distance, place_id = min(
            (haversine_m(latitude, longitude, place.latitude, place.longitude), place.place_id)
            for place in places
        )
        return place_id if distance <= radius_m * (1.0 + 0.35 * (1.0 - quality)) else "unknown"

    return {
        trip.trip_id: (
            nearest(trip.origin_latitude, trip.origin_longitude, trip.observation_quality),
            nearest(trip.destination_latitude, trip.destination_longitude, trip.observation_quality),
        )
        for trip in trips
    }


def learn_robust_lab(raw_points: tuple[ObservablePoint, ...], train_fraction: float = 0.8) -> RobustLearnedLab:
    trips = segment_trips_robust(raw_points)
    train_trips, test_trips = robust_chronological_split(trips, train_fraction)
    discovery = discover_recurring_journeys_robust(train_trips)
    test_places = _assign_places(test_trips, discovery.places)
    train_by_id = {trip.trip_id: trip for trip in train_trips}
    od_learning = {}
    for od in discovery.od_pairs:
        od_trips = tuple(train_by_id[trip_id] for trip_id in od.trip_ids if trip_id in train_by_id)
        if len(od_trips) < 4:
            continue
        hierarchy = cluster_route_hierarchy(od_trips)
        if not hierarchy.variants:
            continue
        train_assignments = hierarchy_train_assignments(hierarchy)
        # Two good recent observations are allowed into the predictor but remain
        # explicitly "insufficient" confidence. Singletons stay structural only.
        eligible = {
            variant.variant_id
            for variant in hierarchy.variants
            if len(variant.trip_ids) >= 2 and variant.mean_observation_quality >= 0.18
        }
        if not eligible:
            eligible = {hierarchy.variants[0].variant_id}
        model_assignments = {
            trip_id: variant_id
            for trip_id, variant_id in train_assignments.items()
            if variant_id in eligible
        }
        model_trips = tuple(trip for trip in od_trips if trip.trip_id in model_assignments)
        candidate_test = tuple(
            trip
            for trip in test_trips
            if test_places.get(trip.trip_id) == (od.origin_place_id, od.destination_place_id)
        )
        test_assignments = assign_route_variants(candidate_test, hierarchy)
        events = extract_robust_delay_events(model_trips, model_assignments)
        zones = discover_robust_delay_zones(model_trips, model_assignments, events)
        model = RobustTemporalModel.fit(model_trips, model_assignments, zones, recency_half_life_days=28.0)
        uniform = RobustTemporalModel.fit(model_trips, model_assignments, zones, recency_half_life_days=None)
        od_learning[od.od_id] = RobustODLearning(
            od.od_id,
            od.trip_ids,
            hierarchy,
            train_assignments,
            model_assignments,
            test_assignments,
            events,
            zones,
            model,
            uniform,
        )
    return RobustLearnedLab(trips, train_trips, test_trips, discovery, test_places, od_learning)
