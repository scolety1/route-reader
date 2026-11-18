"""Small, dependency-free geospatial and trajectory helpers."""

from __future__ import annotations

import math
import statistics
from collections.abc import Sequence

from .types import ObservablePoint

EARTH_RADIUS_M = 6_371_008.8


def haversine_m(
    lat1: float, lon1: float, lat2: float, lon2: float
) -> float:
    """Great-circle distance in metres."""

    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = phi2 - phi1
    dlambda = math.radians(lon2 - lon1)
    a = (
        math.sin(dphi / 2.0) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2
    )
    return 2.0 * EARTH_RADIUS_M * math.asin(min(1.0, math.sqrt(a)))


def xy_to_latlon(
    x_m: float, y_m: float, origin_lat: float, origin_lon: float
) -> tuple[float, float]:
    lat = origin_lat + math.degrees(y_m / EARTH_RADIUS_M)
    lon = origin_lon + math.degrees(
        x_m / (EARTH_RADIUS_M * math.cos(math.radians(origin_lat)))
    )
    return lat, lon


def latlon_to_xy(
    lat: float, lon: float, origin_lat: float, origin_lon: float
) -> tuple[float, float]:
    y_m = math.radians(lat - origin_lat) * EARTH_RADIUS_M
    x_m = (
        math.radians(lon - origin_lon)
        * EARTH_RADIUS_M
        * math.cos(math.radians(origin_lat))
    )
    return x_m, y_m


def euclidean(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def polyline_length(points: Sequence[tuple[float, float]]) -> float:
    return sum(euclidean(a, b) for a, b in zip(points, points[1:]))


def interpolate_polyline(
    points: Sequence[tuple[float, float]], distance_m: float
) -> tuple[float, float]:
    if not points:
        raise ValueError("cannot interpolate an empty polyline")
    if distance_m <= 0:
        return points[0]
    remaining = distance_m
    for start, end in zip(points, points[1:]):
        length = euclidean(start, end)
        if remaining <= length:
            ratio = 0.0 if length == 0 else remaining / length
            return (
                start[0] + ratio * (end[0] - start[0]),
                start[1] + ratio * (end[1] - start[1]),
            )
        remaining -= length
    return points[-1]


def resample_polyline(
    points: Sequence[tuple[float, float]], count: int = 32
) -> tuple[tuple[float, float], ...]:
    if count < 2:
        raise ValueError("count must be at least two")
    if not points:
        raise ValueError("cannot resample an empty polyline")
    total = polyline_length(points)
    if total == 0:
        return tuple(points[0] for _ in range(count))
    return tuple(
        interpolate_polyline(points, total * i / (count - 1))
        for i in range(count)
    )


def median_smooth_xy(
    points: Sequence[ObservablePoint], window: int = 3
) -> tuple[tuple[float, float], ...]:
    """Convert to local XY and apply a short median filter."""

    if not points:
        return ()
    ref_lat, ref_lon = points[0].latitude, points[0].longitude
    raw = [latlon_to_xy(p.latitude, p.longitude, ref_lat, ref_lon) for p in points]
    radius = max(0, window // 2)
    smoothed: list[tuple[float, float]] = []
    for index in range(len(raw)):
        values = raw[max(0, index - radius) : min(len(raw), index + radius + 1)]
        smoothed.append(
            (
                statistics.median(v[0] for v in values),
                statistics.median(v[1] for v in values),
            )
        )
    return tuple(smoothed)


def clean_trip_xy(
    points: Sequence[ObservablePoint], max_plausible_speed_mps: float = 42.0
) -> tuple[tuple[float, float], ...]:
    """Remove isolated GPS jumps, then smooth and resample-ready the trace."""

    if len(points) < 3:
        return median_smooth_xy(points)
    ref_lat, ref_lon = points[0].latitude, points[0].longitude
    raw = [latlon_to_xy(p.latitude, p.longitude, ref_lat, ref_lon) for p in points]
    keep = [True] * len(points)
    for index in range(1, len(points) - 1):
        dt_before = (points[index].timestamp - points[index - 1].timestamp).total_seconds()
        dt_after = (points[index + 1].timestamp - points[index].timestamp).total_seconds()
        if dt_before <= 0 or dt_after <= 0:
            keep[index] = False
            continue
        speed_before = euclidean(raw[index - 1], raw[index]) / dt_before
        speed_after = euclidean(raw[index], raw[index + 1]) / dt_after
        direct = euclidean(raw[index - 1], raw[index + 1])
        detour = (
            euclidean(raw[index - 1], raw[index])
            + euclidean(raw[index], raw[index + 1])
            - direct
        )
        if (
            speed_before > max_plausible_speed_mps
            and speed_after > max_plausible_speed_mps
            and detour > 60.0
        ):
            keep[index] = False
    filtered = [point for point, accepted in zip(points, keep) if accepted]
    return median_smooth_xy(filtered, window=3)


def quantile(values: Sequence[float], probability: float) -> float:
    if not values:
        raise ValueError("quantile requires values")
    ordered = sorted(values)
    position = (len(ordered) - 1) * min(1.0, max(0.0, probability))
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction

