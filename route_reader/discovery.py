"""Recurring-place and origin/destination discovery."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from .geo import haversine_m
from .types import ODPair, Place, Trip


@dataclass(frozen=True)
class DiscoveredJourneys:
    places: tuple[Place, ...]
    od_pairs: tuple[ODPair, ...]
    trip_places: dict[str, tuple[str, str]]


def _components(
    observations: list[tuple[float, float]], radius_m: float
) -> list[list[int]]:
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
            if haversine_m(*observations[left], *observations[right]) <= radius_m:
                union(left, right)
    grouped: dict[int, list[int]] = {}
    for index in range(len(observations)):
        grouped.setdefault(find(index), []).append(index)
    return list(grouped.values())


def discover_recurring_journeys(
    trips: tuple[Trip, ...], place_radius_m: float = 140.0, min_visits: int = 3
) -> DiscoveredJourneys:
    observations: list[tuple[float, float]] = []
    for trip in trips:
        observations.append((trip.origin_latitude, trip.origin_longitude))
        observations.append((trip.destination_latitude, trip.destination_longitude))
    components = [
        component
        for component in _components(observations, place_radius_m)
        if len(component) >= min_visits
    ]
    centers = []
    for component in components:
        centers.append(
            (
                sum(observations[index][0] for index in component) / len(component),
                sum(observations[index][1] for index in component) / len(component),
                len(component),
            )
        )
    centers.sort(key=lambda item: (item[1], item[0]))
    places = tuple(
        Place(f"P{index + 1}", latitude, longitude, count)
        for index, (latitude, longitude, count) in enumerate(centers)
    )

    def nearest_place(latitude: float, longitude: float) -> str:
        distances = [
            (haversine_m(latitude, longitude, place.latitude, place.longitude), place.place_id)
            for place in places
        ]
        if not distances or min(distances)[0] > place_radius_m * 1.5:
            return "unknown"
        return min(distances)[1]

    trip_places = {
        trip.trip_id: (
            nearest_place(trip.origin_latitude, trip.origin_longitude),
            nearest_place(trip.destination_latitude, trip.destination_longitude),
        )
        for trip in trips
    }
    counts = Counter(pair for pair in trip_places.values() if "unknown" not in pair)
    od_pairs = []
    for index, (pair, count) in enumerate(
        sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    ):
        if count < min_visits:
            continue
        trip_ids = tuple(
            trip_id for trip_id, trip_pair in trip_places.items() if trip_pair == pair
        )
        od_pairs.append(ODPair(f"OD{index + 1}", pair[0], pair[1], trip_ids))
    return DiscoveredJourneys(places, tuple(od_pairs), trip_places)
