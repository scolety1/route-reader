"""Quality-aware recurring place and journey discovery for adversarial histories."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from .geo import haversine_m
from .robust_ingestion import RobustTrip
from .types import ODPair, Place


@dataclass(frozen=True)
class RobustDiscovery:
    places: tuple[Place, ...]
    od_pairs: tuple[ODPair, ...]
    trip_places: dict[str, tuple[str, str]]
    rejected_low_quality_endpoints: int


def discover_recurring_journeys_robust(
    trips: tuple[RobustTrip, ...], radius_m: float = 235.0, min_visits: int = 4
) -> RobustDiscovery:
    observations: list[tuple[float, float, float]] = []
    rejected = 0
    for trip in trips:
        endpoints = (
            (trip.origin_latitude, trip.origin_longitude, trip.has_origin_dwell),
            (trip.destination_latitude, trip.destination_longitude, trip.has_destination_dwell),
        )
        for latitude, longitude, has_dwell in endpoints:
            if trip.observation_quality < 0.12 and not has_dwell:
                rejected += 1
                continue
            weight = max(0.2, trip.observation_quality) * (1.25 if has_dwell else 0.65)
            observations.append((latitude, longitude, weight))
    parents = list(range(len(observations)))

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parents[right_root] = left_root

    for left in range(len(observations)):
        for right in range(left + 1, len(observations)):
            if haversine_m(*observations[left][:2], *observations[right][:2]) <= radius_m:
                union(left, right)
    components: dict[int, list[int]] = {}
    for index in range(len(observations)):
        components.setdefault(find(index), []).append(index)
    centers = []
    for members in components.values():
        if len(members) < min_visits:
            continue
        total_weight = sum(observations[index][2] for index in members)
        centers.append(
            (
                sum(observations[index][0] * observations[index][2] for index in members) / total_weight,
                sum(observations[index][1] * observations[index][2] for index in members) / total_weight,
                len(members),
            )
        )
    centers.sort(key=lambda item: (item[1], item[0]))
    places = tuple(
        Place(f"RP{index + 1}", latitude, longitude, count)
        for index, (latitude, longitude, count) in enumerate(centers)
    )

    def nearest(latitude: float, longitude: float, quality: float) -> str:
        if not places:
            return "unknown"
        distance, place_id = min(
            (haversine_m(latitude, longitude, place.latitude, place.longitude), place.place_id)
            for place in places
        )
        acceptance = radius_m * (1.15 + 0.45 * (1.0 - quality))
        return place_id if distance <= acceptance else "unknown"

    trip_places = {
        trip.trip_id: (
            nearest(trip.origin_latitude, trip.origin_longitude, trip.observation_quality),
            nearest(trip.destination_latitude, trip.destination_longitude, trip.observation_quality),
        )
        for trip in trips
    }
    counts = Counter(pair for pair in trip_places.values() if "unknown" not in pair and pair[0] != pair[1])
    od_pairs = []
    for pair, count in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
        if count < min_visits:
            continue
        trip_ids = tuple(trip_id for trip_id, assigned in trip_places.items() if assigned == pair)
        od_pairs.append(ODPair(f"ROD{len(od_pairs) + 1}", pair[0], pair[1], trip_ids))
    return RobustDiscovery(places, tuple(od_pairs), trip_places, rejected)
