"""Stateful, truth-free online corridor memory.

The learner never reclusters history.  Each completed trip contributes a compact
sequence of direction-aware spatial corridor samples.  Established variants are
updated conservatively; novel geometry must recur across days before becoming a
durable route.  Simulator identities are intentionally absent from this module.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field
from datetime import datetime

from .geo import EARTH_RADIUS_M, euclidean, resample_polyline
from .robust_ingestion import RobustTrip


@dataclass(frozen=True)
class CorridorSample:
    x_m: float
    y_m: float
    heading_bin: int
    progress: float


@dataclass
class CorridorNode:
    node_id: str
    x_m: float
    y_m: float
    heading_bin: int
    support_trips: set[str] = field(default_factory=set)
    variant_ids: set[str] = field(default_factory=set)


@dataclass
class OnlineRouteVariant:
    variant_id: str
    status: str
    trip_ids: list[str]
    support_days: set[str]
    prototype: list[CorridorSample]
    created_at: datetime
    updated_at: datetime
    parent_id: str | None = None
    novelty_reason: str = "unexplained geometry"
    consecutive_misses: int = 0


@dataclass(frozen=True)
class CorridorMatch:
    variant_id: str | None
    state: str
    score: float
    observed_fraction: float
    shared_fraction: float
    decisive_fraction_observed: float
    identity_confidence: float
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class MemoryEvent:
    trip_id: str
    event: str
    variant_id: str | None
    cause: str
    score: float
    durable_variants: int
    candidate_variants: int


@dataclass(frozen=True)
class RouteMemorySnapshot:
    trips_seen: int
    established_variants: int
    repeated_variants: int
    candidate_variants: int
    unresolved_trips: int
    corridor_nodes: int
    shared_corridor_nodes: int
    identity_continuity: float
    identity_churn_per_100_trips: float
    evidence_fragmentation: float


def _global_xy(latitude: float, longitude: float) -> tuple[float, float]:
    """A fixed Web-Mercator-like frame; only local distances are consumed."""

    latitude_radians = math.radians(max(-85.0, min(85.0, latitude)))
    return (
        EARTH_RADIUS_M * math.radians(longitude),
        EARTH_RADIUS_M * math.log(math.tan(math.pi / 4.0 + latitude_radians / 2.0)),
    )


def extract_corridor_samples(trip: RobustTrip, spacing_m: float = 85.0) -> tuple[CorridorSample, ...]:
    """Extract inspectable direction-aware corridor samples at physical spacing."""

    raw = [_global_xy(point.latitude, point.longitude) for point in trip.points]
    if len(raw) < 2:
        return ()
    length = sum(euclidean(a, b) for a, b in zip(raw, raw[1:]))
    count = max(6, min(80, round(length / spacing_m) + 1))
    points = resample_polyline(raw, count)
    samples = []
    for index, point in enumerate(points):
        left = points[max(0, index - 1)]
        right = points[min(len(points) - 1, index + 1)]
        angle = math.atan2(right[1] - left[1], right[0] - left[0])
        heading = int(round((angle % (2 * math.pi)) / (math.pi / 4))) % 8
        samples.append(CorridorSample(point[0], point[1], heading, index / max(1, len(points) - 1)))
    return tuple(samples)


def _heading_distance(left: int, right: int) -> int:
    difference = abs(left - right)
    return min(difference, 8 - difference)


def _local_match(sample: CorridorSample, prototype: list[CorridorSample], tolerance_m: float) -> tuple[float, int]:
    candidates = [
        (euclidean((sample.x_m, sample.y_m), (other.x_m, other.y_m)), index)
        for index, other in enumerate(prototype)
        if _heading_distance(sample.heading_bin, other.heading_bin) <= 1
    ]
    if not candidates:
        return 0.0, -1
    distance, index = min(candidates)
    return max(0.0, 1.0 - distance / tolerance_m), index


def corridor_similarity(
    samples: tuple[CorridorSample, ...], prototype: list[CorridorSample], tolerance_m: float
) -> tuple[float, float, float]:
    """Return robust coverage, ordered coverage, and decisive-middle coverage."""

    if not samples or not prototype:
        return 0.0, 0.0, 0.0
    matches = [_local_match(sample, prototype, tolerance_m) for sample in samples]
    # Endpoint entrances carry little route identity. Weight the middle 80% more.
    weights = [0.35 if sample.progress < 0.10 or sample.progress > 0.90 else 1.0 for sample in samples]
    coverage = sum(weight * score for weight, (score, _) in zip(weights, matches)) / sum(weights)
    indexes = [index for score, index in matches if score > 0.25 and index >= 0]
    ordered = 0.0
    if indexes:
        monotonic = 1 + sum(right >= left - 1 for left, right in zip(indexes, indexes[1:]))
        ordered = monotonic / len(matches)
    middle = [(score, sample) for sample, (score, _) in zip(samples, matches) if 0.25 <= sample.progress <= 0.80]
    decisive = statistics.mean(score for score, _ in middle) if middle else coverage
    return coverage, min(1.0, ordered), decisive


def _blend_prototype(current: list[CorridorSample], incoming: tuple[CorridorSample, ...], support: int) -> list[CorridorSample]:
    count = max(len(current), len(incoming), 8)
    old_xy = resample_polyline([(item.x_m, item.y_m) for item in current], count)
    new_xy = resample_polyline([(item.x_m, item.y_m) for item in incoming], count)
    alpha = min(0.22, 1.0 / max(2, support))
    result = []
    for index, (old, new) in enumerate(zip(old_xy, new_xy)):
        dx, dy = new[0] - old[0], new[1] - old[1]
        # Cap movement per observation. A temporary detour cannot drag the durable
        # centerline wholesale; recurring geometry will move it gradually.
        distance = math.hypot(dx, dy)
        scale = min(1.0, 45.0 / max(1.0, distance))
        x = old[0] + dx * alpha * scale
        y = old[1] + dy * alpha * scale
        prior_heading = current[min(len(current) - 1, round(index * (len(current) - 1) / (count - 1)))].heading_bin
        result.append(CorridorSample(x, y, prior_heading, index / (count - 1)))
    return result


class OnlineCorridorMemory:
    """Append-only OD route memory with novelty lifecycle and shared corridors."""

    def __init__(self, match_tolerance_m: float = 155.0) -> None:
        self.match_tolerance_m = match_tolerance_m
        self.variants: dict[str, OnlineRouteVariant] = {}
        self.nodes: dict[tuple[int, int, int], CorridorNode] = {}
        self.assignments: dict[str, str | None] = {}
        self.matches: dict[str, CorridorMatch] = {}
        self.events: list[MemoryEvent] = []
        self.trips: dict[str, RobustTrip] = {}
        self.next_variant = 1
        self.next_node = 1

    def _scores(self, samples: tuple[CorridorSample, ...]) -> list[tuple[float, float, float, OnlineRouteVariant]]:
        rows = []
        for variant in self.variants.values():
            if variant.status == "expired":
                continue
            coverage, ordered, decisive = corridor_similarity(samples, variant.prototype, self.match_tolerance_m)
            score = 0.52 * coverage + 0.28 * ordered + 0.20 * decisive
            rows.append((score, coverage, decisive, variant))
        return sorted(rows, key=lambda row: (-row[0], -len(row[3].trip_ids), row[3].variant_id))

    def _cause(self, trip: RobustTrip, samples: tuple[CorridorSample, ...], best: tuple[float, float, float, OnlineRouteVariant] | None) -> str:
        if not trip.has_origin_dwell or not trip.has_destination_dwell:
            return "partial_or_truncated_trace"
        if best is None:
            return "first_observed_corridor"
        score, coverage, decisive, variant = best
        start_gap = euclidean((samples[0].x_m, samples[0].y_m), (variant.prototype[0].x_m, variant.prototype[0].y_m))
        end_gap = euclidean((samples[-1].x_m, samples[-1].y_m), (variant.prototype[-1].x_m, variant.prototype[-1].y_m))
        if max(start_gap, end_gap) > 210 and decisive >= 0.70:
            return "endpoint_variation"
        if trip.observation_quality < 0.30:
            return "degraded_sampling_or_gps"
        if coverage >= 0.62 and decisive < 0.45:
            return "early_or_late_divergence"
        if score >= 0.48:
            return "lateral_movement_or_route_evolution"
        return "novel_corridor_or_detour"

    def _new_variant(self, trip: RobustTrip, samples: tuple[CorridorSample, ...], cause: str, parent: str | None) -> OnlineRouteVariant:
        identifier = f"OR{self.next_variant:03d}"
        self.next_variant += 1
        variant = OnlineRouteVariant(identifier, "candidate", [trip.trip_id], {trip.departure.date().isoformat()}, list(samples), trip.departure, trip.departure, parent, cause)
        self.variants[identifier] = variant
        return variant

    def _update_nodes(self, trip_id: str, variant_id: str, samples: tuple[CorridorSample, ...]) -> None:
        for sample in samples[1:-1]:
            key = (round(sample.x_m / 120.0), round(sample.y_m / 120.0), sample.heading_bin)
            node = self.nodes.get(key)
            if node is None:
                node = CorridorNode(f"OC{self.next_node:04d}", sample.x_m, sample.y_m, sample.heading_bin)
                self.next_node += 1
                self.nodes[key] = node
            node.support_trips.add(trip_id)
            node.variant_ids.add(variant_id)

    def observe(self, trip: RobustTrip) -> CorridorMatch:
        """Update memory with exactly one completed trip."""

        if self.trips and trip.departure <= max(item.departure for item in self.trips.values()):
            raise ValueError("online corridor memory requires strictly chronological completed trips")
        samples = extract_corridor_samples(trip)
        self.trips[trip.trip_id] = trip
        if len(samples) < 5:
            match = CorridorMatch(None, "unresolved", 0.0, 0.0, 0.0, 0.0, 0.0, ("too few usable corridor samples",))
            self.assignments[trip.trip_id] = None
            self.matches[trip.trip_id] = match
            self.events.append(MemoryEvent(trip.trip_id, "unresolved", None, "degraded_sampling_or_gps", 0.0, len(self.established_variants), len(self.candidate_variants)))
            return match
        scores = self._scores(samples)
        best = scores[0] if scores else None
        cause = self._cause(trip, samples, best)
        partial = not trip.has_origin_dwell or not trip.has_destination_dwell
        variant = None
        event = "unresolved"
        if best and (best[0] >= 0.70 or (best[3].status == "established" and best[0] >= 0.64 and best[2] >= 0.45)):
            score, coverage, decisive, candidate = best
            margin = score - scores[1][0] if len(scores) > 1 else score
            # If only the shared portion was observed and alternatives are close,
            # retain OD/corridor evidence without inventing a variant assignment.
            if partial and margin < 0.10 and coverage >= 0.55:
                match = CorridorMatch(None, "shared_corridor_only", score, coverage, coverage, decisive, min(0.55, score * 0.65), ("observed portion matches multiple variants", "decisive branch unobserved"))
                self.assignments[trip.trip_id] = None
                self.matches[trip.trip_id] = match
                self.events.append(MemoryEvent(trip.trip_id, "partial_match", None, cause, score, len(self.established_variants), len(self.candidate_variants)))
                return match
            variant = candidate
            event = "reuse"
        elif best and best[0] >= 0.66 and best[3].status in {"candidate", "repeated"}:
            variant = best[3]
            event = "candidate_support"
        else:
            parent = best[3].variant_id if best and best[0] >= 0.45 else None
            variant = self._new_variant(trip, samples, cause, parent)
            event = "candidate_create"
        if trip.trip_id not in variant.trip_ids:
            variant.trip_ids.append(trip.trip_id)
            variant.support_days.add(trip.departure.date().isoformat())
            variant.updated_at = trip.departure
            variant.prototype = _blend_prototype(variant.prototype, samples, len(variant.trip_ids))
        support = len(variant.trip_ids)
        days = len(variant.support_days)
        previous_status = variant.status
        if support >= 5 and days >= 4:
            variant.status = "established"
        elif support >= 3 and days >= 2:
            variant.status = "repeated"
        if variant.status != previous_status:
            event = f"graduate_{variant.status}"
        self._update_nodes(trip.trip_id, variant.variant_id, samples)
        score = best[0] if best and best[3].variant_id == variant.variant_id else (1.0 if support == 1 else 0.55)
        coverage = best[1] if best and best[3].variant_id == variant.variant_id else 0.0
        decisive = best[2] if best and best[3].variant_id == variant.variant_id else 0.0
        confidence = min(1.0, score * min(1.0, support / 5.0) * max(0.25, trip.observation_quality))
        match = CorridorMatch(variant.variant_id, variant.status, score, 1.0, coverage, decisive, confidence, (cause, f"{support} supporting trips across {days} days"))
        self.assignments[trip.trip_id] = variant.variant_id
        self.matches[trip.trip_id] = match
        self.events.append(MemoryEvent(trip.trip_id, event, variant.variant_id, cause, score, len(self.established_variants), len(self.candidate_variants)))
        self._expire_candidates(trip.departure)
        return match

    def _expire_candidates(self, now: datetime) -> None:
        for variant in self.variants.values():
            if variant.status == "candidate" and (now - variant.updated_at).days > 28:
                variant.status = "expired"

    @property
    def established_variants(self) -> tuple[OnlineRouteVariant, ...]:
        return tuple(item for item in self.variants.values() if item.status == "established")

    @property
    def candidate_variants(self) -> tuple[OnlineRouteVariant, ...]:
        return tuple(item for item in self.variants.values() if item.status in {"candidate", "repeated"})

    def prediction_history(self, variant_id: str) -> tuple[RobustTrip, ...]:
        """Direct variant evidence plus high-overlap lineage evidence.

        Candidate splits inherit only observations from a parent whose prototype
        overlaps strongly.  Durations from unrelated corridors are never pooled.
        """

        variant = self.variants[variant_id]
        ids = set(variant.trip_ids)
        if variant.parent_id and variant.parent_id in self.variants:
            parent = self.variants[variant.parent_id]
            coverage, ordered, decisive = corridor_similarity(tuple(variant.prototype), parent.prototype, self.match_tolerance_m)
            if coverage >= 0.82 and ordered >= 0.72 and decisive >= 0.70:
                ids.update(parent.trip_ids)
        return tuple(sorted((self.trips[trip_id] for trip_id in ids), key=lambda item: item.departure))

    def snapshot(self) -> RouteMemorySnapshot:
        resolved = [value for value in self.assignments.values() if value is not None]
        reused = sum(event.event in {"reuse", "candidate_support"} or event.event.startswith("graduate") for event in self.events)
        structural = sum(event.event in {"candidate_create"} for event in self.events)
        durable = self.established_variants
        evidence = [len(self.prediction_history(item.variant_id)) for item in durable]
        fragmentation = 0.0 if not resolved else 1.0 - max((resolved.count(item.variant_id) for item in durable), default=0) / len(resolved)
        return RouteMemorySnapshot(
            len(self.trips),
            len(durable),
            sum(item.status == "repeated" for item in self.variants.values()),
            sum(item.status == "candidate" for item in self.variants.values()),
            sum(value is None for value in self.assignments.values()),
            len(self.nodes),
            sum(len(node.variant_ids) > 1 for node in self.nodes.values()),
            reused / max(1, len(self.events) - 1),
            100.0 * structural / max(1, len(self.events)),
            fragmentation if evidence else 1.0,
        )
