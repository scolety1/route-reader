"""Hierarchical trajectory clustering: OD pair -> family -> route variant."""

from __future__ import annotations

import statistics
from dataclasses import dataclass

from .robust_ingestion import RobustTrip
from .trajectories import represent_trip, trajectory_distance_m


@dataclass(frozen=True)
class RobustRouteVariant:
    variant_id: str
    trip_ids: tuple[str, ...]
    prototype_xy: tuple[tuple[float, float], ...]
    mean_observation_quality: float


@dataclass(frozen=True)
class RobustRouteFamily:
    family_id: str
    variant_ids: tuple[str, ...]


@dataclass(frozen=True)
class RouteHierarchy:
    variants: tuple[RobustRouteVariant, ...]
    families: tuple[RobustRouteFamily, ...]
    variant_to_family: dict[str, str]
    variant_threshold_m: float
    family_threshold_m: float


def _pair_key(left: str, right: str) -> tuple[str, str]:
    return tuple(sorted((left, right)))


def _complete_link_clusters(items: list[str], distances: dict[tuple[str, str], float], threshold: float) -> list[list[str]]:
    clusters = [[item] for item in items]

    def distance(left: list[str], right: list[str]) -> float:
        return max(distances[_pair_key(a, b)] for a in left for b in right)

    while True:
        candidates = [
            (distance(clusters[left], clusters[right]), left, right)
            for left in range(len(clusters))
            for right in range(left + 1, len(clusters))
        ]
        if not candidates:
            break
        best_distance, left, right = min(candidates)
        if best_distance > threshold:
            break
        clusters[left] = clusters[left] + clusters[right]
        del clusters[right]
    return clusters


def _consolidate_prototypes(
    prototypes: list[tuple[list[str], tuple[tuple[float, float], ...], float]],
    threshold: float,
) -> list[tuple[list[str], tuple[tuple[float, float], ...], float]]:
    """Merge complete-link fragments whose robust prototypes still agree.

    Complete linkage prevents corridor chaining but can split one route after a
    partial trace. This second, prototype-level pass recovers that stability while
    retaining a conservative geometric boundary between neighboring streets.
    """

    parents = list(range(len(prototypes)))

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    for left in range(len(prototypes)):
        for right in range(left + 1, len(prototypes)):
            if trajectory_distance_m(prototypes[left][1], prototypes[right][1]) <= threshold:
                a, b = find(left), find(right)
                if a != b:
                    parents[b] = a
    components: dict[int, list[int]] = {}
    for index in range(len(prototypes)):
        components.setdefault(find(index), []).append(index)
    consolidated = []
    for indexes in components.values():
        members = [trip_id for index in indexes for trip_id in prototypes[index][0]]
        total = sum(len(prototypes[index][0]) for index in indexes)
        prototype = tuple(
            (
                sum(prototypes[index][1][point][0] * len(prototypes[index][0]) for index in indexes) / total,
                sum(prototypes[index][1][point][1] * len(prototypes[index][0]) for index in indexes) / total,
            )
            for point in range(len(prototypes[indexes[0]][1]))
        )
        quality = sum(prototypes[index][2] * len(prototypes[index][0]) for index in indexes) / total
        consolidated.append((members, prototype, quality))
    return consolidated


def cluster_route_hierarchy(
    trips: tuple[RobustTrip, ...], variant_threshold_override_m: float | None = None
) -> RouteHierarchy:
    if not trips:
        return RouteHierarchy((), (), {}, 0.0, 0.0)
    representations = {trip.trip_id: represent_trip(trip) for trip in trips}
    quality = statistics.median(trip.observation_quality for trip in trips)
    variant_threshold = (
        variant_threshold_override_m
        if variant_threshold_override_m is not None
        else max(178.0, 135.0 + (1.0 - quality) * 175.0)
    )
    family_threshold = 390.0 + (1.0 - quality) * 230.0
    ids = list(representations)
    pair_distances: dict[tuple[str, str], float] = {}
    for left in range(len(ids)):
        for right in range(left + 1, len(ids)):
            pair_distances[_pair_key(ids[left], ids[right])] = trajectory_distance_m(
                representations[ids[left]], representations[ids[right]]
            )
    groups = _complete_link_clusters(ids, pair_distances, variant_threshold)
    prototypes = []
    trip_by_id = {trip.trip_id: trip for trip in trips}
    for members in groups:
        prototype = tuple(
            (
                sum(representations[trip_id][index][0] for trip_id in members) / len(members),
                sum(representations[trip_id][index][1] for trip_id in members) / len(members),
            )
            for index in range(len(next(iter(representations.values()))))
        )
        prototypes.append((members, prototype, statistics.mean(trip_by_id[item].observation_quality for item in members)))
    prototypes = _consolidate_prototypes(prototypes, variant_threshold)
    prototypes.sort(key=lambda item: (-len(item[0]), -sum(point[1] for point in item[1]) / len(item[1])))
    variants = tuple(
        RobustRouteVariant(f"V{index + 1}", tuple(sorted(members)), prototype, mean_quality)
        for index, (members, prototype, mean_quality) in enumerate(prototypes)
    )
    variant_ids = [variant.variant_id for variant in variants]
    variant_distances: dict[tuple[str, str], float] = {}
    by_id = {variant.variant_id: variant for variant in variants}
    for left in range(len(variant_ids)):
        for right in range(left + 1, len(variant_ids)):
            variant_distances[_pair_key(variant_ids[left], variant_ids[right])] = trajectory_distance_m(
                by_id[variant_ids[left]].prototype_xy, by_id[variant_ids[right]].prototype_xy
            )
    family_groups = _complete_link_clusters(variant_ids, variant_distances, family_threshold) if len(variants) > 1 else [variant_ids]
    families = tuple(RobustRouteFamily(f"F{index + 1}", tuple(group)) for index, group in enumerate(family_groups))
    mapping = {variant_id: family.family_id for family in families for variant_id in family.variant_ids}
    return RouteHierarchy(variants, families, mapping, variant_threshold, family_threshold)


def hierarchy_train_assignments(hierarchy: RouteHierarchy) -> dict[str, str]:
    return {trip_id: variant.variant_id for variant in hierarchy.variants for trip_id in variant.trip_ids}


def assign_route_variants(trips: tuple[RobustTrip, ...], hierarchy: RouteHierarchy) -> dict[str, str]:
    assignments = {}
    for trip in trips:
        representation = represent_trip(trip)
        assignments[trip.trip_id] = min(
            hierarchy.variants,
            key=lambda variant: trajectory_distance_m(representation, variant.prototype_xy),
        ).variant_id
    return assignments
