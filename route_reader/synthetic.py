"""Deterministic synthetic personal-history fixture.

Truth objects are deliberately separate from observable points.  Production-facing
pipeline code receives only the noisy timestamp/latitude/longitude stream.
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path

from .geo import interpolate_polyline, polyline_length, xy_to_latlon
from .types import ObservablePoint

ORIGIN_LAT = 39.7392
ORIGIN_LON = -104.9903


@dataclass(frozen=True)
class RouteSpec:
    route_id: str
    points_m: tuple[tuple[float, float], ...]
    base_speed_mps: float
    delay_fractions: tuple[float, ...]

    @property
    def length_m(self) -> float:
        return polyline_length(self.points_m)


@dataclass(frozen=True)
class RouteProfile:
    route_id: str
    moving_speed_mps: float
    delays: tuple[tuple[float, float], ...]
    total_seconds: float


@dataclass(frozen=True)
class TripTruth:
    trip_index: int
    departure: datetime
    route_id: str
    actual_duration_seconds: float
    counterfactual_duration_seconds: dict[str, float]


@dataclass(frozen=True)
class SyntheticDataset:
    raw_points: tuple[ObservablePoint, ...]
    truths: tuple[TripTruth, ...]
    true_places_xy: dict[str, tuple[float, float]]
    route_specs: dict[str, RouteSpec]
    seed: int


def canonical_routes() -> dict[str, RouteSpec]:
    return {
        "A": RouteSpec(
            "A",
            ((0, 0), (1200, 80), (2900, 150), (4500, 30), (6000, 0)),
            13.55,
            (0.51,),
        ),
        "B": RouteSpec(
            "B",
            ((0, 0), (1100, 650), (2850, 1020), (4700, 720), (6000, 0)),
            12.65,
            (0.48,),
        ),
        "C": RouteSpec(
            "C",
            ((0, 0), (1150, -620), (2700, -860), (4300, -410), (6000, 0)),
            12.25,
            (0.31, 0.72),
        ),
        "D": RouteSpec(
            "D",
            ((0, 0), (900, 350), (2050, -220), (3350, 540), (4700, 260), (6000, 0)),
            12.45,
            (0.25, 0.70),
        ),
    }


def _gaussian(value: float, center: float, sigma: float) -> float:
    return math.exp(-0.5 * ((value - center) / sigma) ** 2)


def _profile_for(
    spec: RouteSpec, departure: datetime, day_index: int, seed: int
) -> RouteProfile:
    minute = departure.hour * 60 + departure.minute + departure.second / 60.0
    route_number = ord(spec.route_id) - ord("A") + 1
    rng = random.Random(seed * 1_000_003 + day_index * 997 + route_number * 37)
    common_speed_factor = 1.0 + 0.018 * math.sin(day_index * 1.71) + rng.gauss(0, 0.012)
    speed = spec.base_speed_mps * common_speed_factor

    if spec.route_id == "A":
        delays = [max(3.0, 7.0 + 172.0 * _gaussian(minute, 15 * 60 + 41, 10.5) + rng.gauss(0, 4.0))]
    elif spec.route_id == "B":
        friday_penalty = 13.0 if departure.weekday() == 4 else 0.0
        delays = [max(5.0, 16.0 + friday_penalty + 12.0 * _gaussian(minute, 15 * 60 + 55, 13.0) + rng.gauss(0, 4.0))]
    elif spec.route_id == "C":
        # This route is occasionally excellent, but its unscheduled stops make it risky.
        first = max(0.0, rng.gauss(15.0, 22.0))
        second = max(0.0, rng.gauss(13.0, 18.0))
        delays = [first, second]
    else:
        delays = [
            max(5.0, 25.0 + 62.0 * _gaussian(minute, 15 * 60 + 20, 9.0) + rng.gauss(0, 5.0)),
            max(5.0, 14.0 + 58.0 * _gaussian(minute, 16 * 60, 10.0) + rng.gauss(0, 5.0)),
        ]

    positioned = tuple(
        (spec.length_m * fraction, delay)
        for fraction, delay in zip(spec.delay_fractions, delays)
    )
    total = spec.length_m / speed + sum(delays)
    return RouteProfile(spec.route_id, speed, positioned, total)


def _position_at_elapsed(
    spec: RouteSpec, profile: RouteProfile, elapsed_seconds: float
) -> tuple[float, float]:
    elapsed = max(0.0, elapsed_seconds)
    prior_distance = 0.0
    prior_time = 0.0
    for delay_distance, delay_seconds in profile.delays:
        travel_seconds = (delay_distance - prior_distance) / profile.moving_speed_mps
        if elapsed <= prior_time + travel_seconds:
            distance = prior_distance + (elapsed - prior_time) * profile.moving_speed_mps
            return interpolate_polyline(spec.points_m, distance)
        prior_time += travel_seconds
        if elapsed <= prior_time + delay_seconds:
            return interpolate_polyline(spec.points_m, delay_distance)
        prior_time += delay_seconds
        prior_distance = delay_distance
    distance = prior_distance + (elapsed - prior_time) * profile.moving_speed_mps
    return interpolate_polyline(spec.points_m, min(spec.length_m, distance))


def _noisy_point(
    timestamp: datetime,
    xy: tuple[float, float],
    rng: random.Random,
    outlier_probability: float,
) -> ObservablePoint:
    accuracy = rng.uniform(4.0, 10.0)
    x = xy[0] + rng.gauss(0, accuracy * 0.55)
    y = xy[1] + rng.gauss(0, accuracy * 0.55)
    if rng.random() < outlier_probability:
        angle = rng.uniform(0, math.tau)
        distance = rng.uniform(75.0, 155.0)
        x += math.cos(angle) * distance
        y += math.sin(angle) * distance
        accuracy = rng.uniform(35.0, 80.0)
    latitude, longitude = xy_to_latlon(x, y, ORIGIN_LAT, ORIGIN_LON)
    return ObservablePoint(timestamp, latitude, longitude, accuracy)


def _sample_session(
    spec: RouteSpec,
    profile: RouteProfile,
    departure: datetime,
    seed: int,
    day_index: int,
) -> list[ObservablePoint]:
    rng = random.Random(seed * 10_007 + day_index * 313)
    points: list[ObservablePoint] = []

    stationary_t = -240.0
    while stationary_t < 0:
        points.append(
            _noisy_point(
                departure + timedelta(seconds=stationary_t),
                spec.points_m[0],
                rng,
                outlier_probability=0.012,
            )
        )
        stationary_t += rng.uniform(20.0, 34.0)
    points.append(_noisy_point(departure, spec.points_m[0], rng, 0.005))

    elapsed = rng.uniform(7.0, 12.0)
    while elapsed < profile.total_seconds:
        if rng.random() > 0.09:  # occasional dropped sample
            points.append(
                _noisy_point(
                    departure + timedelta(seconds=elapsed),
                    _position_at_elapsed(spec, profile, elapsed),
                    rng,
                    outlier_probability=0.014,
                )
            )
        elapsed += rng.uniform(7.0, 14.0)
    arrival = departure + timedelta(seconds=profile.total_seconds)
    points.append(_noisy_point(arrival, spec.points_m[-1], rng, 0.005))

    stationary_t = rng.uniform(22.0, 34.0)
    while stationary_t <= 240.0:
        points.append(
            _noisy_point(
                arrival + timedelta(seconds=stationary_t),
                spec.points_m[-1],
                rng,
                outlier_probability=0.012,
            )
        )
        stationary_t += rng.uniform(20.0, 34.0)
    return points


def generate_canonical_dataset(seed: int = 20260811, weeks: int = 10) -> SyntheticDataset:
    if not 6 <= weeks <= 10:
        raise ValueError("canonical fixture expects 6-10 weeks")
    routes = canonical_routes()
    fixture_rng = random.Random(seed)
    start_date = datetime(2026, 1, 5)
    base_minutes = [15 * 60 + 12, 15 * 60 + 27, 15 * 60 + 40, 15 * 60 + 52, 16 * 60 + 6]
    all_points: list[ObservablePoint] = []
    truths: list[TripTruth] = []

    for week in range(weeks):
        route_assignment = ["A", "B", "B", "C", "D"]
        fixture_rng.shuffle(route_assignment)
        for weekday in range(5):
            day_index = week * 5 + weekday
            day = start_date + timedelta(days=week * 7 + weekday)
            minute = base_minutes[weekday] + fixture_rng.uniform(-3.2, 3.2)
            departure = day.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(minutes=minute)
            profiles = {
                route_id: _profile_for(spec, departure, day_index, seed)
                for route_id, spec in routes.items()
            }
            selected = route_assignment[weekday]
            profile = profiles[selected]
            all_points.extend(
                _sample_session(routes[selected], profile, departure, seed, day_index)
            )
            truths.append(
                TripTruth(
                    trip_index=day_index,
                    departure=departure,
                    route_id=selected,
                    actual_duration_seconds=profile.total_seconds,
                    counterfactual_duration_seconds={
                        route_id: candidate.total_seconds
                        for route_id, candidate in profiles.items()
                    },
                )
            )

    return SyntheticDataset(
        raw_points=tuple(sorted(all_points, key=lambda point: point.timestamp)),
        truths=tuple(truths),
        true_places_xy={"School": (0.0, 0.0), "Work": (6000.0, 0.0)},
        route_specs=routes,
        seed=seed,
    )


def write_raw_jsonl(dataset: SyntheticDataset, path: Path) -> None:
    """Write only observable fields; hidden truth is never mixed into raw input."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for point in dataset.raw_points:
            payload = asdict(point)
            payload["timestamp"] = point.timestamp.isoformat()
            handle.write(json.dumps(payload, separators=(",", ":")) + "\n")


def read_raw_jsonl(path: Path) -> tuple[ObservablePoint, ...]:
    points: list[ObservablePoint] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            item = json.loads(line)
            points.append(
                ObservablePoint(
                    timestamp=datetime.fromisoformat(item["timestamp"]),
                    latitude=float(item["latitude"]),
                    longitude=float(item["longitude"]),
                    accuracy_m=float(item["accuracy_m"]),
                )
            )
    return tuple(points)
