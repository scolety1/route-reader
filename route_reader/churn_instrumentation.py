"""Observable instrumentation for explaining batch route-structure churn."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass

from .robust_ingestion import RobustTrip
from .stable_routes import StableRouteRegistry
from .trajectories import represent_trip, trajectory_distance_m


@dataclass(frozen=True)
class ChurnCauseEvent:
    history_trips: int
    trip_id: str
    cause: str
    nearest_route_distance_m: float | None
    observation_quality: float
    origin_dwell: bool
    destination_dwell: bool
    structural_variants: int


def _classify(trip: RobustTrip, distance: float | None, threshold: float) -> str:
    if not trip.has_origin_dwell or not trip.has_destination_dwell:
        return "partial_or_truncated_trace"
    if trip.maximum_gap_seconds > 100 or trip.observation_quality < 0.30:
        return "sampling_density_or_degraded_gps"
    if distance is None:
        return "initial_route_hypothesis"
    if distance <= threshold * 1.10:
        return "prototype_drift_or_threshold_reinterpretation"
    if distance <= threshold * 1.55:
        return "lateral_geometry_or_route_evolution"
    return "large_divergence_or_new_route"


def instrument_batch_churn(trips: tuple[RobustTrip, ...]) -> dict[str, object]:
    """Replay Mission 3 batch memory and record why new structure appears."""

    ordered = tuple(sorted(trips, key=lambda item: item.departure))
    registry = StableRouteRegistry()
    events = []
    prior_routes = ()
    previous_ids: set[str] = set()
    total_reassignments = 0
    previous_assignments: dict[str, str] = {}
    for count, trip in enumerate(ordered, 1):
        representation = represent_trip(trip)
        nearest = min((trajectory_distance_m(representation, route.prototype_xy) for route in prior_routes), default=None)
        snapshot = registry.update(ordered[:count])
        current_ids = {route.route_id for route in snapshot.routes}
        created = current_ids - previous_ids
        for route_id in created:
            route = next(route for route in snapshot.routes if route.route_id == route_id)
            if trip.trip_id not in route.trip_ids and count > 1:
                continue
            events.append(ChurnCauseEvent(count, trip.trip_id, _classify(trip, nearest, snapshot.adaptive_threshold_m), nearest, trip.observation_quality, trip.has_origin_dwell, trip.has_destination_dwell, len(snapshot.routes)))
        common = set(previous_assignments) & set(snapshot.assignments)
        total_reassignments += sum(previous_assignments[item] != snapshot.assignments[item] for item in common)
        previous_assignments = snapshot.assignments
        prior_routes = snapshot.routes
        previous_ids = current_ids
    causes = Counter(event.cause for event in events)
    return {
        "trips": len(ordered),
        "structural_creation_events": len(events),
        "causes": dict(sorted(causes.items())),
        "historical_identity_reassignments": total_reassignments,
        "final_structural_variants": len(prior_routes),
        "events": [asdict(event) for event in events],
    }
