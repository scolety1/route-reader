"""Noise-tolerant trajectory representation, comparison, and clustering."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Sequence

from .geo import clean_trip_xy, euclidean, resample_polyline
from .types import RouteCluster, Trip


def represent_trip(trip: Trip, sample_count: int = 32) -> tuple[tuple[float, float], ...]:
    cleaned = clean_trip_xy(trip.points)
    return resample_polyline(cleaned, count=sample_count)


def discrete_frechet_m(
    first: Sequence[tuple[float, float]], second: Sequence[tuple[float, float]]
) -> float:
    if not first or not second:
        raise ValueError("trajectories must not be empty")
    cache = [[-1.0 for _ in second] for _ in first]
    for left in range(len(first)):
        for right in range(len(second)):
            distance = euclidean(first[left], second[right])
            if left == 0 and right == 0:
                cache[left][right] = distance
            elif left == 0:
                cache[left][right] = max(cache[left][right - 1], distance)
            elif right == 0:
                cache[left][right] = max(cache[left - 1][right], distance)
            else:
                cache[left][right] = max(
                    min(
                        cache[left - 1][right],
                        cache[left - 1][right - 1],
                        cache[left][right - 1],
                    ),
                    distance,
                )
    return cache[-1][-1]


def trajectory_distance_m(
    first: Sequence[tuple[float, float]], second: Sequence[tuple[float, float]]
) -> float:
    if len(first) != len(second):
        first = resample_polyline(first, 32)
        second = resample_polyline(second, 32)
    frechet = discrete_frechet_m(first, second)
    rms = math.sqrt(
        sum(euclidean(a, b) ** 2 for a, b in zip(first, second)) / len(first)
    )
    return 0.58 * frechet + 0.42 * rms


def cluster_routes(
    trips: tuple[Trip, ...], threshold_m: float = 285.0
) -> tuple[RouteCluster, ...]:
    representations = {trip.trip_id: represent_trip(trip) for trip in trips}
    trip_ids = list(representations)
    parents = list(range(len(trip_ids)))

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parents[right_root] = left_root

    for left in range(len(trip_ids)):
        for right in range(left + 1, len(trip_ids)):
            if trajectory_distance_m(
                representations[trip_ids[left]], representations[trip_ids[right]]
            ) <= threshold_m:
                union(left, right)

    raw_groups: dict[int, list[str]] = {}
    for index, trip_id in enumerate(trip_ids):
        raw_groups.setdefault(find(index), []).append(trip_id)

    prototypes: list[tuple[list[str], tuple[tuple[float, float], ...]]] = []
    for member_ids in raw_groups.values():
        prototype = tuple(
            (
                sum(representations[trip_id][point_index][0] for trip_id in member_ids)
                / len(member_ids),
                sum(representations[trip_id][point_index][1] for trip_id in member_ids)
                / len(member_ids),
            )
            for point_index in range(len(next(iter(representations.values()))))
        )
        prototypes.append((sorted(member_ids), prototype))
    prototypes.sort(key=lambda item: -sum(point[1] for point in item[1]) / len(item[1]))
    return tuple(
        RouteCluster(f"R{index + 1}", tuple(member_ids), prototype)
        for index, (member_ids, prototype) in enumerate(prototypes)
    )


def assign_to_routes(
    trips: tuple[Trip, ...], clusters: tuple[RouteCluster, ...]
) -> dict[str, str]:
    assignments: dict[str, str] = {}
    for trip in trips:
        representation = represent_trip(trip)
        assignments[trip.trip_id] = min(
            clusters,
            key=lambda cluster: trajectory_distance_m(representation, cluster.prototype_xy),
        ).cluster_id
    return assignments


def adjusted_rand_index(labels_true: Sequence[str], labels_pred: Sequence[str]) -> float:
    if len(labels_true) != len(labels_pred):
        raise ValueError("label sequences must be the same length")
    if len(labels_true) < 2:
        return 1.0
    contingency = Counter(zip(labels_true, labels_pred))
    true_counts = Counter(labels_true)
    pred_counts = Counter(labels_pred)

    def choose_two(value: int) -> float:
        return value * (value - 1) / 2.0

    pairs = choose_two(len(labels_true))
    sum_cells = sum(choose_two(value) for value in contingency.values())
    sum_true = sum(choose_two(value) for value in true_counts.values())
    sum_pred = sum(choose_two(value) for value in pred_counts.values())
    expected = sum_true * sum_pred / pairs
    maximum = 0.5 * (sum_true + sum_pred)
    if maximum == expected:
        return 1.0
    return (sum_cells - expected) / (maximum - expected)


def majority_truth_mapping(
    predicted: Sequence[str], truth: Sequence[str]
) -> dict[str, str]:
    """Evaluation-only mapping; never used to create clusters or predictions."""

    votes: dict[str, Counter[str]] = {}
    for cluster_id, truth_id in zip(predicted, truth):
        votes.setdefault(cluster_id, Counter())[truth_id] += 1
    return {cluster_id: counts.most_common(1)[0][0] for cluster_id, counts in votes.items()}
