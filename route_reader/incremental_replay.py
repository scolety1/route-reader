"""Leakage-safe replay of one recurring journey arriving a trip at a time."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta

from .adaptive_model import AdaptiveJourneyModel, AdaptiveRouteModel
from .place_regions import discover_place_regions
from .robust_ingestion import RobustTrip
from .stable_routes import RouteRevision, StableRouteRegistry


@dataclass(frozen=True)
class ReplayStep:
    history_trips: int
    known_places: int
    candidate_places: int
    known_od_pairs: int
    known_route_variants: int
    retained_route_identity_fraction: float
    selected_models: tuple[tuple[str, str], ...]
    recommended_route: str | None
    exploration_candidate: str | None
    abstention_reason: str | None
    major_revisions: tuple[RouteRevision, ...]


def replay_route_history(trips: tuple[RobustTrip, ...]) -> tuple[ReplayStep, ...]:
    """Replay a pre-scoped recurring journey without consulting future rows.

    Scoping is the same boundary a product would get after a user selects a
    candidate OD.  Every state transition below is computed from the prefix only.
    """

    ordered = tuple(sorted(trips, key=lambda item: item.departure))
    registry = StableRouteRegistry()
    steps = []
    for count in range(1, len(ordered) + 1):
        prefix = ordered[:count]
        places = discover_place_regions(prefix)
        snapshot = registry.update(prefix)
        models = []
        for route in snapshot.routes:
            route_trips = tuple(trip for trip in prefix if snapshot.assignments.get(trip.trip_id) == route.route_id)
            identity = snapshot.retained_identity_fraction * min(1.0, len(route_trips) / 6.0)
            models.append(AdaptiveRouteModel(route.route_id, route_trips, route_identity_confidence=identity))
        query = prefix[-1].departure + timedelta(days=7)
        recommendation = AdaptiveJourneyModel(tuple(models)).predict_all(query)
        confirmed = [region for region in places.regions if region.status in {"recurring", "strong_recurring"}]
        candidates = [region for region in places.regions if region.status == "candidate"]
        assigned_pairs = {
            (
                places.endpoint_assignments.get((trip.trip_id, "origin")),
                places.endpoint_assignments.get((trip.trip_id, "destination")),
            )
            for trip in prefix
        }
        known_ods = sum(a is not None and b is not None and a != b for a, b in assigned_pairs)
        steps.append(
            ReplayStep(
                count,
                len(confirmed),
                len(candidates),
                known_ods,
                len(snapshot.routes),
                snapshot.retained_identity_fraction,
                tuple((model.variant_id, model.selected_model) for model in models),
                recommendation.fastest_expected.variant_id if recommendation.fastest_expected else None,
                recommendation.exploration_candidate.variant_id if recommendation.exploration_candidate else None,
                recommendation.abstention_reason,
                snapshot.revisions,
            )
        )
    return tuple(steps)


def milestone_steps(steps: tuple[ReplayStep, ...]) -> tuple[ReplayStep, ...]:
    requested = {5, 10, 20, 40, len(steps)}
    return tuple(step for step in steps if step.history_trips in requested)
