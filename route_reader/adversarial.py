"""Deterministic multi-place, multi-month adversarial mobility simulator.

The underlying journeys are generated once and may be replayed through different
observation regimes. Hidden identity, delay category, and counterfactual fields are
evaluation-only; runtime modules consume only ``ObservablePoint`` tuples.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from datetime import datetime, timedelta

from .geo import euclidean, interpolate_polyline, polyline_length, xy_to_latlon
from .synthetic import ORIGIN_LAT, ORIGIN_LON
from .types import ObservablePoint


@dataclass(frozen=True)
class ObservationRegime:
    name: str
    interval_range_seconds: tuple[float, float]
    noise_m: float
    drop_probability: float
    outlier_probability: float
    drift_per_sample_m: float
    long_gap_probability: float
    partial_probability: float
    missed_session_probability: float


OBSERVATION_REGIMES = {
    "good": ObservationRegime("good", (4.0, 7.0), 3.0, 0.02, 0.002, 0.15, 0.01, 0.02, 0.01),
    "normal": ObservationRegime("normal", (8.0, 18.0), 7.0, 0.12, 0.012, 0.45, 0.06, 0.10, 0.04),
    "sparse": ObservationRegime("sparse", (22.0, 48.0), 12.0, 0.34, 0.025, 0.90, 0.22, 0.25, 0.10),
    "hostile": ObservationRegime("hostile", (18.0, 72.0), 24.0, 0.46, 0.075, 2.30, 0.38, 0.42, 0.18),
}


@dataclass(frozen=True)
class LifeConfig:
    weeks: int = 20
    route_separability: str = "hard"
    behavior_drift: bool = True
    temporal_signal: str = "strong"
    seed: int = 20260812


@dataclass(frozen=True)
class TrueDelay:
    delay_id: str
    category: str
    route_id: str
    location_xy: tuple[float, float]
    seconds: float
    persistent_target: bool


@dataclass(frozen=True)
class TrueJourney:
    journey_id: str
    origin_id: str
    destination_id: str
    od_id: str
    route_id: str
    route_family_id: str
    departure: datetime
    arrival: datetime
    path_xy: tuple[tuple[float, float], ...]
    moving_speed_mps: float
    delays: tuple[TrueDelay, ...]
    counterfactual_duration_seconds: dict[str, float]
    complication: str

    @property
    def duration_seconds(self) -> float:
        return (self.arrival - self.departure).total_seconds()


@dataclass(frozen=True)
class LifeHistory:
    journeys: tuple[TrueJourney, ...]
    places_xy: dict[str, tuple[float, float]]
    config: LifeConfig


@dataclass(frozen=True)
class ObservedLifeHistory:
    raw_points: tuple[ObservablePoint, ...]
    recorded_journey_ids: frozenset[str]
    regime: ObservationRegime


PLACES_XY = {
    "Home": (0.0, 0.0),
    "School": (6000.0, 1000.0),
    "Work": (10500.0, 1500.0),
    "Grocery": (1750.0, -1250.0),
    "Gym": (-2200.0, 1750.0),
    "Friend": (7600.0, -1950.0),
    "Coffee": (6280.0, 1190.0),
}


def _place_visit(place_id: str, visit_index: int) -> tuple[float, float]:
    spreads = {
        "Home": ((0, 0), (18, -12), (-15, 20)),
        "School": ((-145, -80), (140, 65), (55, -140), (-85, 145)),
        "Work": ((-105, 35), (115, -45), (20, 90)),
        "Grocery": ((-90, -15), (105, 35)),
        "Gym": ((-45, 30), (60, -20)),
        "Friend": ((0, 0), (35, 25)),
        "Coffee": ((-35, 15), (45, -20)),
    }
    center = PLACES_XY[place_id]
    offset = spreads[place_id][visit_index % len(spreads[place_id])]
    return center[0] + offset[0], center[1] + offset[1]


def _generic_path(origin: str, destination: str, bend: float = 0.0) -> tuple[tuple[float, float], ...]:
    start, end = PLACES_XY[origin], PLACES_XY[destination]
    dx, dy = end[0] - start[0], end[1] - start[1]
    length = max(1.0, math.hypot(dx, dy))
    nx, ny = -dy / length, dx / length
    return (
        start,
        (start[0] + dx * 0.33 + nx * bend, start[1] + dy * 0.33 + ny * bend),
        (start[0] + dx * 0.68 + nx * bend * 0.7, start[1] + dy * 0.68 + ny * bend * 0.7),
        end,
    )


def _school_work_routes(separability: str) -> dict[str, tuple[tuple[float, float], ...]]:
    scale = 2.4 if separability == "easy" else 1.0
    return {
        "A": ((6000, 1000), (6800, 1050), (7600, 1150), (8500, 1300), (9400, 1400), (10500, 1500)),
        "B": ((6000, 1000), (6800, 1050), (7600, 1150), (8300, 1150 - 180 * scale), (9100, 1250 - 180 * scale), (9400, 1400), (10500, 1500)),
        "C": ((6000, 1000), (6800, 1050), (7600, 1150), (8500, 1300), (9400, 1400), (9800, 1400 + 420 * scale), (10300, 1400 + 420 * scale), (10500, 1500)),
        "N": ((6000, 1000), (6800, 1050), (7600, 1150), (8500, 1300), (9000, 1300 + 330 * scale), (9800, 1300 + 330 * scale), (10500, 1500)),
        "A_detour": ((6000, 1000), (6800, 1050), (7600, 1150), (8000, 1650), (8750, 1660), (9400, 1400), (10500, 1500)),
    }


ROUTE_FAMILY = {
    "A": "central",
    "A_detour": "central",
    "B": "parallel",
    "C": "late-loop",
    "N": "north-new",
}


def _gaussian(value: float, center: float, sigma: float) -> float:
    return math.exp(-0.5 * ((value - center) / sigma) ** 2)


def _target_profile(
    route_id: str,
    path: tuple[tuple[float, float], ...],
    departure: datetime,
    week: int,
    config: LifeConfig,
    seed_offset: int,
) -> tuple[float, tuple[TrueDelay, ...]]:
    rng = random.Random(config.seed * 1009 + week * 101 + seed_offset * 17 + ord(route_id[0]))
    minute = departure.hour * 60 + departure.minute + departure.second / 60.0
    speed = {"A": 13.4, "A_detour": 13.0, "B": 12.7, "C": 12.8, "N": 13.3}[route_id]
    speed *= 1.0 + rng.gauss(0, 0.018)
    if config.behavior_drift and week >= config.weeks // 2 and route_id == "A":
        speed *= 0.86  # persistent route-wide degradation, not a localized stop
    signal_scale = {"none": 0.0, "weak": 0.45, "strong": 1.0}[config.temporal_signal]
    delays: list[TrueDelay] = []
    if route_id in {"A", "A_detour"}:
        signal = 7.0 + signal_scale * (
            112.0 * _gaussian(minute, 15 * 60 + 30, 11.0)
            + 82.0 * _gaussian(minute, 16 * 60 + 5, 10.0)
        ) + rng.gauss(0, 4.0)
        location = interpolate_polyline(path, polyline_length(path) * 0.58)
        delays.append(TrueDelay("recurring-central", "recurring", route_id, location, max(2.0, signal), True))
    if route_id == "B" and rng.random() < 0.23:
        location = interpolate_polyline(path, polyline_length(path) * 0.66)
        delays.append(TrueDelay("random-parallel", "random", route_id, location, rng.uniform(20, 105), False))
    if route_id == "A_detour":
        location = interpolate_polyline(path, polyline_length(path) * 0.63)
        delays.append(TrueDelay("temporary-construction", "temporary", route_id, location, rng.uniform(55, 105), False))
    if route_id == "C" and rng.random() < 0.58:
        location = interpolate_polyline(path, polyline_length(path) * 0.74)
        delays.append(TrueDelay("behavioral-stop", "behavioral", route_id, location, rng.uniform(35, 90), False))
    return speed, tuple(delays)


def _duration(path: tuple[tuple[float, float], ...], speed: float, delays: tuple[TrueDelay, ...]) -> float:
    return polyline_length(path) / speed + sum(delay.seconds for delay in delays)


def _adapt_endpoints(
    path: tuple[tuple[float, float], ...], start: tuple[float, float], end: tuple[float, float]
) -> tuple[tuple[float, float], ...]:
    points = list(path)
    points[0], points[-1] = start, end
    return tuple(points)


def generate_life_history(config: LifeConfig = LifeConfig()) -> LifeHistory:
    if not 16 <= config.weeks <= 24:
        raise ValueError("adversarial history must cover roughly four to six months")
    target_paths = _school_work_routes(config.route_separability)
    start_date = datetime(2026, 1, 5)
    journeys: list[TrueJourney] = []
    visit_counter = 0

    def add_journey(
        origin: str,
        destination: str,
        departure: datetime,
        route_id: str,
        path: tuple[tuple[float, float], ...],
        family: str,
        week: int,
        complication: str = "none",
        target: bool = False,
    ) -> None:
        nonlocal visit_counter
        index = len(journeys)
        actual_start = _place_visit(origin, visit_counter)
        actual_end = _place_visit(destination, visit_counter + 1)
        visit_counter += 2
        adapted = _adapt_endpoints(path, actual_start, actual_end)
        if complication == "return-toward-origin":
            quarter = interpolate_polyline(adapted, polyline_length(adapted) * 0.22)
            adapted = (adapted[0], quarter, (quarter[0] - 420, quarter[1] - 90), quarter) + adapted[2:]
        if target:
            speed, delays = _target_profile(route_id, adapted, departure, week, config, index)
            counterfactual = {}
            for candidate_id in ("A", "B", "C", "N"):
                if candidate_id == "N" and (not config.behavior_drift or week < config.weeks // 2):
                    continue
                candidate_path = _adapt_endpoints(target_paths[candidate_id], actual_start, actual_end)
                candidate_speed, candidate_delays = _target_profile(candidate_id, candidate_path, departure, week, config, index)
                counterfactual[candidate_id] = _duration(candidate_path, candidate_speed, candidate_delays)
        else:
            rng = random.Random(config.seed + index * 41)
            speed = 12.2 * (1 + rng.gauss(0, 0.025))
            delays_list: list[TrueDelay] = []
            if complication in {"gas-stop", "food-stop", "stationary-intermediate"}:
                location = interpolate_polyline(adapted, polyline_length(adapted) * 0.55)
                delays_list.append(TrueDelay(f"behavior-{index}", "behavioral", route_id, location, rng.uniform(230, 390), False))
            delays = tuple(delays_list)
            counterfactual = {route_id: _duration(adapted, speed, delays)}
        total = _duration(adapted, speed, delays)
        journeys.append(
            TrueJourney(
                journey_id=f"J{index + 1:04d}",
                origin_id=origin,
                destination_id=destination,
                od_id=f"{origin}->{destination}",
                route_id=route_id,
                route_family_id=family,
                departure=departure,
                arrival=departure + timedelta(seconds=total),
                path_xy=adapted,
                moving_speed_mps=speed,
                delays=delays,
                counterfactual_duration_seconds=counterfactual,
                complication=complication,
            )
        )

    complication_cycle = (
        "none", "campus-walk", "parking-wait", "none", "gas-stop", "none",
        "traffic-like-stop", "partial-start", "none", "destination-walk",
        "return-toward-origin", "partial-end", "stationary-intermediate",
    )
    for week in range(config.weeks):
        for weekday in range(5):
            date = start_date + timedelta(days=week * 7 + weekday)
            complication = complication_cycle[(week * 5 + weekday) % len(complication_cycle)]
            add_journey("Home", "School", date.replace(hour=7, minute=32 + (week + weekday) % 12), f"HS{(week + weekday) % 2 + 1}", _generic_path("Home", "School", 260 if (week + weekday) % 2 else -220), f"home-school-{(week + weekday) % 2 + 1}", week, complication)
            if weekday in {0, 1, 3}:
                shifted = config.behavior_drift and week >= config.weeks // 2
                base_minute = {
                    0: 15 * 60 + 5,
                    1: 15 * 60 + 30,
                    3: 15 * 60 + 55,
                }[weekday]
                minute = base_minute + (35 if shifted else 0) + ((week * 3 + weekday) % 9 - 4)
                evolving = config.behavior_drift and week >= config.weeks // 2
                route_cycle = ("N", "B", "N", "A", "C") if evolving else ("A", "B", "C", "A", "B")
                route_id = route_cycle[(week + weekday) % len(route_cycle)]
                if 6 <= week <= 9 and route_id == "A":
                    route_id = "A_detour"
                departure = date.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(minutes=minute)
                add_journey("School", "Work", departure, route_id, target_paths[route_id], ROUTE_FAMILY[route_id], week, complication, target=True)
                add_journey("Work", "Home", date.replace(hour=18, minute=8 + (week + weekday) % 16), f"WH{(week + weekday) % 2 + 1}", _generic_path("Work", "Home", 300 if (week + weekday) % 2 else -280), f"work-home-{(week + weekday) % 2 + 1}", week)
            elif weekday == 2:
                add_journey("School", "Home", date.replace(hour=15, minute=36), "SH1", _generic_path("School", "Home", -150), "school-home", week, complication)
            else:
                add_journey("School", "Coffee", date.replace(hour=15, minute=12), "SC1", _generic_path("School", "Coffee", 35), "school-coffee", week)
                add_journey("Coffee", "Home", date.replace(hour=16, minute=2), "CH1", _generic_path("Coffee", "Home", -180), "coffee-home", week)
            if weekday in {1, 3}:
                add_journey("Home", "Gym", date.replace(hour=20, minute=4), "HG1", _generic_path("Home", "Gym", 120), "home-gym", week)
                add_journey("Gym", "Home", date.replace(hour=21, minute=12), "GH1", _generic_path("Gym", "Home", -120), "gym-home", week)

        saturday = start_date + timedelta(days=week * 7 + 5)
        add_journey("Home", "Grocery", saturday.replace(hour=10, minute=3), "HGrocery", _generic_path("Home", "Grocery", -90), "home-grocery", week)
        if week % 4 == 0:
            # A genuinely separate trip begins only six minutes after arrival.
            prior_arrival = journeys[-1].arrival
            add_journey("Grocery", "Gym", prior_arrival + timedelta(minutes=6), "GGym", _generic_path("Grocery", "Gym", 160), "grocery-gym", week)
            add_journey("Gym", "Home", journeys[-1].arrival + timedelta(minutes=8), "GH2", _generic_path("Gym", "Home", -90), "gym-home", week)
        else:
            add_journey("Grocery", "Home", saturday.replace(hour=10, minute=48), "GHome", _generic_path("Grocery", "Home", 70), "grocery-home", week)
        if week % 5 == 2:
            sunday = start_date + timedelta(days=week * 7 + 6)
            add_journey("Home", "Friend", sunday.replace(hour=17, minute=10), "HF1", _generic_path("Home", "Friend", -420), "home-friend", week, "food-stop")
            add_journey("Friend", "Home", sunday.replace(hour=21, minute=5), "FH1", _generic_path("Friend", "Home", 380), "friend-home", week)

    return LifeHistory(tuple(sorted(journeys, key=lambda item: item.departure)), dict(PLACES_XY), config)


def _journey_position(journey: TrueJourney, elapsed: float) -> tuple[float, float]:
    prior_distance = 0.0
    prior_time = 0.0
    route_length = polyline_length(journey.path_xy)
    delay_positions = sorted(
        ((min(route_length, _nearest_distance(journey.path_xy, delay.location_xy)), delay.seconds) for delay in journey.delays),
        key=lambda item: item[0],
    )
    for distance, delay_seconds in delay_positions:
        travel = (distance - prior_distance) / journey.moving_speed_mps
        if elapsed <= prior_time + travel:
            return interpolate_polyline(journey.path_xy, prior_distance + (elapsed - prior_time) * journey.moving_speed_mps)
        prior_time += travel
        if elapsed <= prior_time + delay_seconds:
            return interpolate_polyline(journey.path_xy, distance)
        prior_time += delay_seconds
        prior_distance = distance
    return interpolate_polyline(journey.path_xy, min(route_length, prior_distance + (elapsed - prior_time) * journey.moving_speed_mps))


def _nearest_distance(path: tuple[tuple[float, float], ...], target: tuple[float, float]) -> float:
    best_distance, best_along = float("inf"), 0.0
    along = 0.0
    for start, end in zip(path, path[1:]):
        dx, dy = end[0] - start[0], end[1] - start[1]
        length_sq = dx * dx + dy * dy
        ratio = 0.0 if length_sq == 0 else max(0.0, min(1.0, ((target[0] - start[0]) * dx + (target[1] - start[1]) * dy) / length_sq))
        projection = (start[0] + ratio * dx, start[1] + ratio * dy)
        distance = euclidean(projection, target)
        if distance < best_distance:
            best_distance = distance
            best_along = along + math.sqrt(length_sq) * ratio
        along += math.sqrt(length_sq)
    return best_along


def observe_life_history(
    history: LifeHistory,
    regime: ObservationRegime | str = "normal",
    seed: int | None = None,
) -> ObservedLifeHistory:
    if isinstance(regime, str):
        regime = OBSERVATION_REGIMES[regime]
    seed = history.config.seed + 71 if seed is None else seed
    all_points: list[ObservablePoint] = []
    recorded: set[str] = set()
    for journey_index, journey in enumerate(history.journeys):
        rng = random.Random(seed * 10007 + journey_index * 313)
        if rng.random() < regime.missed_session_probability:
            continue
        recorded.add(journey.journey_id)
        forced_partial_start = journey.complication == "partial-start"
        forced_partial_end = journey.complication == "partial-end"
        partial = rng.random() < regime.partial_probability
        trim_start = rng.uniform(45, min(150, journey.duration_seconds * 0.28)) if (partial or forced_partial_start) else 0.0
        trim_end = rng.uniform(35, min(130, journey.duration_seconds * 0.24)) if (partial or forced_partial_end) else 0.0
        gap_start, gap_end = -1.0, -1.0
        if rng.random() < regime.long_gap_probability:
            gap_start = rng.uniform(journey.duration_seconds * 0.25, journey.duration_seconds * 0.55)
            gap_end = gap_start + rng.uniform(45, min(180, journey.duration_seconds * 0.25))
        drift_x = drift_y = 0.0

        def add_point(timestamp: datetime, xy: tuple[float, float], outlier_scale: float = 1.0) -> None:
            nonlocal drift_x, drift_y
            if rng.random() < regime.drop_probability:
                return
            drift_x += rng.gauss(0, regime.drift_per_sample_m)
            drift_y += rng.gauss(0, regime.drift_per_sample_m)
            x = xy[0] + drift_x + rng.gauss(0, regime.noise_m)
            y = xy[1] + drift_y + rng.gauss(0, regime.noise_m)
            accuracy = max(2.0, regime.noise_m * rng.uniform(0.7, 1.5))
            if rng.random() < regime.outlier_probability * outlier_scale:
                angle, distance = rng.uniform(0, math.tau), rng.uniform(90, 340)
                x += math.cos(angle) * distance
                y += math.sin(angle) * distance
                accuracy *= 3.0
            lat, lon = xy_to_latlon(x, y, ORIGIN_LAT, ORIGIN_LON)
            all_points.append(ObservablePoint(timestamp, lat, lon, accuracy))

        if trim_start == 0:
            elapsed = -240.0
            while elapsed < 0:
                if journey.complication == "campus-walk" and elapsed < -70:
                    angle = elapsed / 28.0
                    xy = (journey.path_xy[0][0] + math.cos(angle) * 75, journey.path_xy[0][1] + math.sin(angle) * 55)
                else:
                    xy = journey.path_xy[0]
                add_point(journey.departure + timedelta(seconds=elapsed), xy, 0.6)
                elapsed += rng.uniform(*regime.interval_range_seconds)

        elapsed = trim_start
        end_elapsed = max(trim_start + 10, journey.duration_seconds - trim_end)
        while elapsed <= end_elapsed:
            if not (gap_start <= elapsed <= gap_end):
                add_point(journey.departure + timedelta(seconds=elapsed), _journey_position(journey, elapsed))
            elapsed += rng.uniform(*regime.interval_range_seconds)

        if trim_end == 0:
            elapsed = rng.uniform(*regime.interval_range_seconds)
            while elapsed <= 240.0:
                if journey.complication == "destination-walk" and elapsed <= 180:
                    angle = elapsed / 24.0
                    xy = (journey.path_xy[-1][0] + math.cos(angle) * 65, journey.path_xy[-1][1] + math.sin(angle) * 45)
                else:
                    xy = journey.path_xy[-1]
                add_point(journey.arrival + timedelta(seconds=elapsed), xy, 0.6)
                elapsed += rng.uniform(*regime.interval_range_seconds)
    deduplicated: dict[datetime, ObservablePoint] = {}
    for point in sorted(all_points, key=lambda item: item.timestamp):
        deduplicated.setdefault(point.timestamp, point)
    return ObservedLifeHistory(tuple(deduplicated.values()), frozenset(recorded), regime)
