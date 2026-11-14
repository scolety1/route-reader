"""Observation-normalized route identities with explicit revision lineage."""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass

from .geo import euclidean, quantile, resample_polyline
from .robust_ingestion import RobustTrip
from .route_families import cluster_route_hierarchy
from .trajectories import represent_trip, trajectory_distance_m


@dataclass(frozen=True)
class StableRoute:
    route_id: str
    revision: int
    trip_ids: tuple[str, ...]
    prototype_xy: tuple[tuple[float, float], ...]
    mean_quality: float
    parent_ids: tuple[str, ...]


@dataclass(frozen=True)
class RouteRevision:
    snapshot_size: int
    event: str
    route_id: str
    related_ids: tuple[str, ...]
    prototype_movement_m: float
    explanation: str


@dataclass(frozen=True)
class StableRouteSnapshot:
    history_trips: int
    routes: tuple[StableRoute, ...]
    assignments: dict[str, str]
    adaptive_threshold_m: float
    revisions: tuple[RouteRevision, ...]
    retained_identity_fraction: float


def normalized_route_shape(trip: RobustTrip, count: int = 40) -> tuple[tuple[float, float], ...]:
    """Resample by physical distance in a shared observation coordinate frame.

    ``represent_trip`` is intentionally local to each recording.  Comparing those
    local shapes made different entrances look like different whole routes.  Add
    the observable origin's projected position back so all trips share a frame;
    endpoint samples themselves remain trimmed by :func:`corridor_distance_m`.
    """

    local = resample_polyline(represent_trip(trip, sample_count=count), count)
    earth_radius_m = 6_371_008.8
    latitude_radians = math.radians(trip.points[0].latitude)
    scale = 1.0 / max(0.2, math.cos(latitude_radians))
    origin_x = math.radians(trip.points[0].longitude) * earth_radius_m
    origin_y = earth_radius_m * math.log(math.tan(math.pi / 4.0 + latitude_radians / 2.0))
    return tuple((point[0] * scale + origin_x, point[1] * scale + origin_y) for point in local)


def corridor_distance_m(
    first: tuple[tuple[float, float], ...], second: tuple[tuple[float, float], ...]
) -> float:
    """Robust aligned corridor distance, insensitive to one noisy sample."""

    if len(first) != len(second):
        first, second = resample_polyline(first, 40), resample_polyline(second, 40)
    # Ignore the least informative 5% at either endpoint.  The 88th percentile
    # still preserves sustained alternative streets but rejects isolated spikes.
    distances = [euclidean(a, b) for a, b in zip(first, second)][2:-2]
    p88 = quantile(distances, 0.88)
    rms = math.sqrt(sum(min(value, p88 * 1.35) ** 2 for value in distances) / len(distances))
    return 0.62 * p88 + 0.38 * rms


def _adaptive_threshold(trips: tuple[RobustTrip, ...], shapes: dict[str, tuple[tuple[float, float], ...]]) -> float:
    if len(trips) < 3:
        return 95.0
    ids = list(shapes)
    nearest = []
    for left in ids:
        candidates = [corridor_distance_m(shapes[left], shapes[right]) for right in ids if right != left]
        nearest.append(min(candidates))
    lower = [value for value in nearest if value <= quantile(nearest, 0.65)]
    center = statistics.median(lower)
    mad = statistics.median(abs(value - center) for value in lower)
    observation_floor = statistics.median(trip.median_accuracy_m for trip in trips) * 2.2
    # The cap is a generic corridor-separation safeguard, not tuned to hidden
    # labels; the learned local scale normally controls the result.
    return max(70.0, min(205.0, center + 4.5 * max(4.0, mad) + observation_floor))


def _prototype(member_ids: list[str], shapes: dict[str, tuple[tuple[float, float], ...]]) -> tuple[tuple[float, float], ...]:
    return tuple(
        (
            statistics.median(shapes[trip_id][index][0] for trip_id in member_ids),
            statistics.median(shapes[trip_id][index][1] for trip_id in member_ids),
        )
        for index in range(len(next(iter(shapes.values()))))
    )


def _cluster(trips: tuple[RobustTrip, ...]) -> tuple[list[tuple[list[str], tuple[tuple[float, float], ...]]], float]:
    quality = statistics.median(trip.observation_quality for trip in trips)
    threshold = min(285.0, 56.0 + 156.0 * quality + max(0.0, 0.5 - quality) * 500.0)
    hierarchy = cluster_route_hierarchy(trips, variant_threshold_override_m=threshold)
    return [([*variant.trip_ids], variant.prototype_xy) for variant in hierarchy.variants], hierarchy.variant_threshold_m

class StableRouteRegistry:
    """Matches each recomputed snapshot to durable IDs and records churn."""

    def __init__(self) -> None:
        self.routes: tuple[StableRoute, ...] = ()
        self.next_id = 1

    def update(self, trips: tuple[RobustTrip, ...]) -> StableRouteSnapshot:
        clusters, threshold = _cluster(trips)
        trip_by_id = {trip.trip_id: trip for trip in trips}
        old = {route.route_id: route for route in self.routes}
        candidates = sorted(
            (
                trajectory_distance_m(route.prototype_xy, prototype),
                route.route_id,
                index,
            )
            for route in self.routes
            for index, (_, prototype) in enumerate(clusters)
        )
        matched_old: set[str] = set()
        matched_new: dict[int, str] = {}
        match_limit = max(95.0, threshold * 1.35)
        for distance, old_id, index in candidates:
            if distance <= match_limit and old_id not in matched_old and index not in matched_new:
                matched_old.add(old_id)
                matched_new[index] = old_id
        routes, revisions = [], []
        for index, (members, prototype) in enumerate(clusters):
            nearby_old = tuple(
                route.route_id for route in self.routes if trajectory_distance_m(route.prototype_xy, prototype) <= match_limit
            )
            if index in matched_new:
                route_id = matched_new[index]
                prior = old[route_id]
                movement = trajectory_distance_m(prior.prototype_xy, prototype)
                revision = prior.revision + (movement > 18.0 or set(prior.trip_ids) != set(members))
                event = "merge" if len(nearby_old) > 1 else "update"
                if event == "merge" or movement > 18.0:
                    revisions.append(RouteRevision(len(trips), event, route_id, nearby_old, movement, "stable ID retained while prototype/evidence changed"))
                parents = nearby_old
            else:
                route_id = f"SR{self.next_id:03d}"
                self.next_id += 1
                revision = 1
                parents = nearby_old
                event = "split" if nearby_old else "create"
                revisions.append(RouteRevision(len(trips), event, route_id, nearby_old, 0.0, "new corridor linked to nearby prior identity" if nearby_old else "new observed corridor"))
            quality = statistics.mean(trip_by_id[trip_id].observation_quality for trip_id in members)
            routes.append(StableRoute(route_id, revision, tuple(sorted(members)), prototype, quality, parents))
        for old_id in old.keys() - matched_old:
            if not any(old_id in route.parent_ids for route in routes):
                revisions.append(RouteRevision(len(trips), "retire", old_id, (), 0.0, "prior corridor no longer has current support"))
        retained = len(matched_old) / max(1, len(self.routes)) if self.routes else 1.0
        self.routes = tuple(sorted(routes, key=lambda item: item.route_id))
        assignments = {trip_id: route.route_id for route in self.routes for trip_id in route.trip_ids}
        return StableRouteSnapshot(len(trips), self.routes, assignments, threshold, tuple(revisions), retained)


def assign_stable_routes(trips: tuple[RobustTrip, ...], routes: tuple[StableRoute, ...]) -> dict[str, str]:
    result = {}
    for trip in trips:
        shape = represent_trip(trip)
        result[trip.trip_id] = min(routes, key=lambda route: trajectory_distance_m(shape, route.prototype_xy)).route_id
    return result
