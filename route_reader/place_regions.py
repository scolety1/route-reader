"""Evidence-lifecycle recurring place regions.

Reliable dwell endpoints seed regions.  Truncated endpoints may attach supporting
evidence, but cannot independently become durable personal knowledge.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass

from .geo import haversine_m, latlon_to_xy
from .robust_ingestion import RobustTrip


@dataclass(frozen=True)
class PlaceRegion:
    region_id: str
    latitude: float
    longitude: float
    radius_m: float
    status: str
    observations: int
    reliable_dwell_observations: int
    support_days: int
    arrivals: int
    departures: int
    truncated_support: int
    confidence: float
    evidence_reasons: tuple[str, ...]


@dataclass(frozen=True)
class PlaceRegionDiscovery:
    regions: tuple[PlaceRegion, ...]
    endpoint_assignments: dict[tuple[str, str], str]
    suppressed_truncated_clusters: int


@dataclass(frozen=True)
class _Endpoint:
    trip_id: str
    kind: str
    latitude: float
    longitude: float
    dwell: bool
    quality: float
    day: str
    signature: tuple[int, int]


def _endpoint_rows(trips: tuple[RobustTrip, ...]) -> list[_Endpoint]:
    rows = []
    if not trips:
        return rows
    ref_lat, ref_lon = trips[0].origin_latitude, trips[0].origin_longitude
    for trip in trips:
        endpoints = (
            ("origin", trip.origin_latitude, trip.origin_longitude, trip.has_origin_dwell, trip.destination_latitude, trip.destination_longitude),
            ("destination", trip.destination_latitude, trip.destination_longitude, trip.has_destination_dwell, trip.origin_latitude, trip.origin_longitude),
        )
        for kind, lat, lon, dwell, other_lat, other_lon in endpoints:
            other_x, other_y = latlon_to_xy(other_lat, other_lon, ref_lat, ref_lon)
            rows.append(_Endpoint(trip.trip_id, kind, lat, lon, dwell, trip.observation_quality, trip.departure.date().isoformat(), (round(other_x / 650), round(other_y / 650))))
    return rows


def _center(members: list[_Endpoint]) -> tuple[float, float]:
    reliable = [row for row in members if row.dwell]
    values = reliable or members
    weights = [max(0.12, row.quality) * (1.5 if row.dwell else 0.35) for row in values]
    total = sum(weights)
    return (
        sum(row.latitude * weight for row, weight in zip(values, weights)) / total,
        sum(row.longitude * weight for row, weight in zip(values, weights)) / total,
    )


def _signature(members: list[_Endpoint]) -> set[tuple[int, int]]:
    return {row.signature for row in members if row.dwell}


def discover_place_regions(trips: tuple[RobustTrip, ...]) -> PlaceRegionDiscovery:
    rows = _endpoint_rows(trips)
    seed_rows = [row for row in rows if row.dwell and row.quality >= 0.12]
    groups: list[list[_Endpoint]] = []
    # Compact entrance modes first. This prevents nearby real places from being
    # joined merely because a large campus footprint bridges them spatially.
    for row in sorted(seed_rows, key=lambda item: (-item.quality, item.trip_id, item.kind)):
        candidates = [(haversine_m(row.latitude, row.longitude, *_center(group)), index) for index, group in enumerate(groups)]
        if candidates and min(candidates)[0] <= max(105.0, 2.0 * statistics.median([row.quality * 20.0 + 12.0])):
            groups[min(candidates)[1]].append(row)
        else:
            groups.append([row])
    # Merge entrances into one variable-footprint region only when their repeated
    # journeys point to the same surrounding place(s). Spatial chaining alone is
    # insufficient, which keeps a nearby cafe distinct from a campus.
    changed = True
    while changed:
        changed = False
        for left in range(len(groups)):
            for right in range(left + 1, len(groups)):
                distance = haversine_m(*_center(groups[left]), *_center(groups[right]))
                a, b = _signature(groups[left]), _signature(groups[right])
                overlap = len(a & b) / max(1, min(len(a), len(b)))
                if distance <= 390.0 and overlap >= 0.50 and len(a & b) >= 1:
                    groups[left].extend(groups[right])
                    del groups[right]
                    changed = True
                    break
            if changed:
                break
    # Attach non-dwell endpoints as secondary evidence only. Unattached repeated
    # truncation is counted and exposed, never promoted to a recurring region.
    assignments: dict[tuple[str, str], str] = {}
    unattached: list[_Endpoint] = []
    for row in rows:
        distances = [(haversine_m(row.latitude, row.longitude, *_center(group)), index) for index, group in enumerate(groups)]
        limit = 210.0 + (1.0 - row.quality) * 105.0
        if distances and min(distances)[0] <= limit:
            index = min(distances)[1]
            if row not in groups[index]:
                groups[index].append(row)
        elif not row.dwell:
            unattached.append(row)
    regions = []
    for index, members in enumerate(groups, 1):
        latitude, longitude = _center(members)
        dwell = [row for row in members if row.dwell]
        days = len({row.day for row in dwell})
        arrivals = sum(row.kind == "destination" for row in dwell)
        departures = sum(row.kind == "origin" for row in dwell)
        truncated = len(members) - len(dwell)
        radial = [haversine_m(row.latitude, row.longitude, latitude, longitude) for row in dwell]
        radius = max(45.0, min(420.0, (sorted(radial)[max(0, math.ceil(len(radial) * 0.90) - 1)] if radial else 45.0) + 25.0))
        if len(dwell) >= 7 and days >= 5 and (arrivals >= 2 or departures >= 2):
            status = "strong_recurring"
        elif len(dwell) >= 3 and days >= 3:
            status = "recurring"
        else:
            status = "candidate"
        confidence = min(1.0, 0.08 * len(dwell) + 0.07 * days + 0.12 * min(arrivals, departures) + 0.10 * statistics.mean(row.quality for row in dwell or members))
        reasons = (
            f"{len(dwell)} reliable dwell endpoints across {days} days",
            f"{arrivals} arrivals and {departures} departures",
            f"{truncated} truncated endpoints contributed support but not promotion",
            f"spatial footprint represented as a {radius:.0f}m region",
        )
        region_id = f"PR{index:03d}"
        regions.append(PlaceRegion(region_id, latitude, longitude, radius, status, len(members), len(dwell), days, arrivals, departures, truncated, confidence, reasons))
        for row in members:
            assignments[(row.trip_id, row.kind)] = region_id
    # Count spatial groups of unattached partial endpoints for diagnostics.
    suppressed = 0
    remaining = set(range(len(unattached)))
    while remaining:
        seed = remaining.pop()
        component = {seed}
        frontier = [seed]
        while frontier:
            current = frontier.pop()
            neighbors = {other for other in remaining if haversine_m(unattached[current].latitude, unattached[current].longitude, unattached[other].latitude, unattached[other].longitude) <= 260.0}
            remaining -= neighbors
            component |= neighbors
            frontier.extend(neighbors)
        if len(component) >= 3:
            suppressed += 1
    return PlaceRegionDiscovery(tuple(regions), assignments, suppressed)
