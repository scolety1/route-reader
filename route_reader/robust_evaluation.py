"""Adversarial metrics. This is the only Mission 2 layer allowed to see truth."""

from __future__ import annotations

import math
import statistics
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass

from .adversarial import LifeHistory, ObservedLifeHistory, TrueJourney
from .geo import euclidean, haversine_m, xy_to_latlon
from .robust_ingestion import RobustTrip
from .robust_model import RobustTemporalModel
from .robust_pipeline import RobustLearnedLab, RobustODLearning
from .route_families import assign_route_variants, cluster_route_hierarchy, hierarchy_train_assignments
from .synthetic import ORIGIN_LAT, ORIGIN_LON
from .trajectories import adjusted_rand_index, majority_truth_mapping


@dataclass(frozen=True)
class ColdStartResult:
    history_trips: int
    learned_variants: int
    stage: str
    evaluated_predictions: int
    duration_mae_seconds: float | None
    mean_interval_width_seconds: float | None
    route_ranking_accuracy: float | None


@dataclass(frozen=True)
class RobustMetrics:
    regime: str
    underlying_journeys: int
    recorded_journeys: int
    inferred_trips: int
    segmentation_precision: float
    segmentation_recall: float
    segmentation_f1: float
    place_precision: float
    place_recall: float
    od_precision: float
    od_recall: float
    target_train_trips: int
    discovered_variants: int
    discovered_families: int
    route_variant_ari: float
    route_family_ari: float
    held_out_route_assignment_accuracy: float
    duration_mae_seconds: float
    route_ranking_accuracy: float
    recommendation_regret_seconds: float
    common_route_regret_seconds: float
    overall_median_regret_seconds: float
    time_bucket_regret_seconds: float
    uniform_history_regret_seconds: float
    confidence_coverage: float
    mean_interval_width_seconds: float
    strong_confidence_mae_seconds: float | None
    weak_confidence_mae_seconds: float | None
    delay_precision: float
    delay_recall: float
    delay_regret_improvement_seconds: float
    active_delay_zones: int
    exploration_candidates: int
    exploration_separation_violations: int

    def to_dict(self) -> dict[str, str | int | float | None]:
        return asdict(self)


@dataclass(frozen=True)
class RobustEvaluation:
    metrics: RobustMetrics
    cold_start: tuple[ColdStartResult, ...]
    place_truth_mapping: dict[str, str]
    target_od_id: str | None
    variant_truth_mapping: dict[str, str]
    explicit_failures: tuple[str, ...]


def align_trips_to_truth(
    trips: tuple[RobustTrip, ...], history: LifeHistory, recorded_ids: frozenset[str]
) -> dict[str, TrueJourney]:
    truths = [journey for journey in history.journeys if journey.journey_id in recorded_ids]
    candidates = []
    for trip in trips:
        for truth in truths:
            overlap = max(0.0, (min(trip.arrival, truth.arrival) - max(trip.departure, truth.departure)).total_seconds())
            denominator = max(30.0, min(trip.duration_seconds, truth.duration_seconds))
            score = overlap / denominator
            start_distance = abs((trip.departure - truth.departure).total_seconds())
            if score >= 0.28 or start_distance <= 150:
                candidates.append((score + max(0.0, 1.0 - start_distance / 300.0) * 0.15, trip.trip_id, truth.journey_id))
    matched_trips, matched_truths, result = set(), set(), {}
    truth_by_id = {truth.journey_id: truth for truth in truths}
    for _, trip_id, truth_id in sorted(candidates, reverse=True):
        if trip_id in matched_trips or truth_id in matched_truths:
            continue
        matched_trips.add(trip_id)
        matched_truths.add(truth_id)
        result[trip_id] = truth_by_id[truth_id]
    return result


def _place_mapping(learned: RobustLearnedLab, history: LifeHistory) -> dict[str, str]:
    candidates = []
    for place in learned.discovery.places:
        for truth_name, xy in history.places_xy.items():
            latlon = xy_to_latlon(*xy, ORIGIN_LAT, ORIGIN_LON)
            candidates.append((haversine_m(place.latitude, place.longitude, *latlon), place.place_id, truth_name))
    mapping, used_places, used_truth = {}, set(), set()
    for distance, place_id, truth_name in sorted(candidates):
        if distance > 360 or place_id in used_places or truth_name in used_truth:
            continue
        mapping[place_id] = truth_name
        used_places.add(place_id)
        used_truth.add(truth_name)
    return mapping


def _safe_mean(values: list[float]) -> float:
    return statistics.mean(values) if values else float("nan")


def _evaluate_model_regret(
    od: RobustODLearning,
    aligned_test: list[tuple[RobustTrip, TrueJourney]],
    variant_truth: dict[str, str],
    model: RobustTemporalModel,
    use_zones: bool,
) -> tuple[list[float], list[bool], list[float]]:
    regrets, correct, predicted_actual_errors = [], [], []
    for trip, truth in aligned_test:
        recommendation = model.predict_all(truth.departure, use_delay_zones=use_zones)
        chosen_truth = variant_truth.get(recommendation.fastest_expected.variant_id)
        available = {
            route_id: seconds
            for route_id, seconds in truth.counterfactual_duration_seconds.items()
            if route_id in set(variant_truth.values())
        }
        if chosen_truth not in available or not available:
            continue
        fastest = min(available, key=available.get)
        regrets.append(available[chosen_truth] - available[fastest])
        correct.append(chosen_truth == fastest)
        assigned_variant = od.test_assignments.get(trip.trip_id)
        prediction = next((item for item in recommendation.predictions if item.variant_id == assigned_variant), None)
        if prediction is not None:
            predicted_actual_errors.append(abs(prediction.predicted_seconds - truth.duration_seconds))
    return regrets, correct, predicted_actual_errors


def _cold_start(
    target_train: list[tuple[RobustTrip, TrueJourney]],
    target_test: list[tuple[RobustTrip, TrueJourney]],
) -> tuple[ColdStartResult, ...]:
    results = []
    for requested in (1, 3, 5, 10, 20, 40):
        count = min(requested, len(target_train))
        subset = tuple(item[0] for item in target_train[:count])
        if not subset:
            continue
        hierarchy = cluster_route_hierarchy(subset)
        assignments = hierarchy_train_assignments(hierarchy)
        model = RobustTemporalModel.fit(subset, assignments, ())
        mapping = majority_truth_mapping(
            [assignments[trip.trip_id] for trip, _ in target_train[:count]],
            [truth.route_id for _, truth in target_train[:count]],
        )
        future_trips = tuple(item[0] for item in target_test[:8])
        future_assignments = assign_route_variants(future_trips, hierarchy) if future_trips else {}
        errors, widths, ranking = [], [], []
        for trip, truth in target_test[:8]:
            recommendation = model.predict_all(truth.departure)
            assigned = future_assignments.get(trip.trip_id)
            prediction = next((item for item in recommendation.predictions if item.variant_id == assigned), None)
            if prediction is not None:
                errors.append(abs(prediction.predicted_seconds - truth.duration_seconds))
                widths.append(prediction.interval_high_seconds - prediction.interval_low_seconds)
            chosen = mapping.get(recommendation.fastest_expected.variant_id)
            available = {key: value for key, value in truth.counterfactual_duration_seconds.items() if key in set(mapping.values())}
            if chosen in available and available:
                ranking.append(chosen == min(available, key=available.get))
        mean_width = _safe_mean(widths) if widths else None
        if count < 3:
            stage = "insufficient history"
        elif count < 5:
            stage = "tentative route hypothesis"
        elif count >= 20 and len(hierarchy.variants) <= 5 and mean_width is not None and mean_width <= 220:
            stage = "strong recurring pattern"
        else:
            stage = "useful route estimate"
        results.append(ColdStartResult(count, len(hierarchy.variants), stage, len(errors), _safe_mean(errors) if errors else None, mean_width, _safe_mean(ranking) if ranking else None))
    return tuple(results)


def evaluate_robust_lab(
    learned: RobustLearnedLab, history: LifeHistory, observed: ObservedLifeHistory
) -> RobustEvaluation:
    aligned = align_trips_to_truth(learned.all_trips, history, observed.recorded_journey_ids)
    matched = len(aligned)
    precision = matched / max(1, len(learned.all_trips))
    recall = matched / max(1, len(observed.recorded_journey_ids))
    segmentation_f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    place_mapping = _place_mapping(learned, history)
    place_precision = len(place_mapping) / max(1, len(learned.discovery.places))
    place_recall = len(set(place_mapping.values())) / len(history.places_xy)
    true_train = [aligned[trip.trip_id] for trip in learned.train_trips if trip.trip_id in aligned]
    true_od_counts = Counter(truth.od_id for truth in true_train)
    recurring_true_ods = {pair for pair, count in true_od_counts.items() if count >= 5}
    learned_pairs = []
    target_od_id = None
    for od in learned.discovery.od_pairs:
        pair = (place_mapping.get(od.origin_place_id), place_mapping.get(od.destination_place_id))
        learned_pairs.append(pair)
        if pair == ("School", "Work"):
            target_od_id = od.od_id
    learned_valid = {f"{a}->{b}" for a, b in learned_pairs if a and b}
    od_precision = len(learned_valid & recurring_true_ods) / max(1, len(learned.discovery.od_pairs))
    od_recall = len(learned_valid & recurring_true_ods) / max(1, len(recurring_true_ods))
    failures = []
    if place_precision < 0.8:
        failures.append(f"place discovery over-created {len(learned.discovery.places) - len(place_mapping)} unmatched recurring places")
    if target_od_id is None or target_od_id not in learned.od_learning:
        metrics = RobustMetrics(observed.regime.name, len(history.journeys), len(observed.recorded_journey_ids), len(learned.all_trips), precision, recall, segmentation_f1, place_precision, place_recall, od_precision, od_recall, 0, 0, 0, 0.0, 0.0, 0.0, float("nan"), 0.0, float("nan"), float("nan"), float("nan"), float("nan"), float("nan"), 0.0, float("nan"), None, None, 0.0, 0.0, 0.0, 0, 0, 0)
        failures.append("School->Work recurring journey was not learnable")
        return RobustEvaluation(metrics, (), place_mapping, target_od_id, {}, tuple(failures))

    od = learned.od_learning[target_od_id]
    train_target = [(trip, aligned[trip.trip_id]) for trip in learned.train_trips if trip.trip_id in aligned and aligned[trip.trip_id].od_id == "School->Work" and trip.trip_id in od.train_assignments]
    test_target = [(trip, aligned[trip.trip_id]) for trip in learned.test_trips if trip.trip_id in aligned and aligned[trip.trip_id].od_id == "School->Work" and trip.trip_id in od.test_assignments]
    train_pred = [od.train_assignments[trip.trip_id] for trip, _ in train_target]
    train_truth = [truth.route_id for _, truth in train_target]
    variant_truth = majority_truth_mapping(train_pred, train_truth) if train_pred else {}
    variant_ari = adjusted_rand_index(train_truth, train_pred) if train_pred else 0.0
    family_pred = [od.hierarchy.variant_to_family[item] for item in train_pred]
    family_truth = [truth.route_family_id for _, truth in train_target]
    family_ari = adjusted_rand_index(family_truth, family_pred) if family_pred else 0.0
    assignment_correct = [variant_truth.get(od.test_assignments[trip.trip_id]) == truth.route_id for trip, truth in test_target]

    regrets, ranking, errors = _evaluate_model_regret(od, test_target, variant_truth, od.model, True)
    no_zone_regrets, _, _ = _evaluate_model_regret(od, test_target, variant_truth, od.model, False)
    uniform_regrets, _, _ = _evaluate_model_regret(od, test_target, variant_truth, od.uniform_history_model, True)
    route_counts = Counter(od.model_assignments.values())
    common_variant = route_counts.most_common(1)[0][0]
    train_by_variant: dict[str, list[RobustTrip]] = defaultdict(list)
    for trip, _ in train_target:
        variant = od.model_assignments.get(trip.trip_id)
        if variant:
            train_by_variant[variant].append(trip)
    median_variant = min(train_by_variant, key=lambda item: statistics.median(trip.duration_seconds for trip in train_by_variant[item]))
    common_regrets, median_regrets, bucket_regrets = [], [], []
    coverage, widths, strong_errors, weak_errors = [], [], [], []
    exploration_candidates = exploration_violations = 0
    for trip, truth in test_target:
        available_routes = set(variant_truth.values()) & set(truth.counterfactual_duration_seconds)
        if not available_routes:
            continue
        fastest = min(available_routes, key=lambda item: truth.counterfactual_duration_seconds[item])
        fastest_seconds = truth.counterfactual_duration_seconds[fastest]
        for variant, destination in ((common_variant, common_regrets), (median_variant, median_regrets)):
            route = variant_truth.get(variant)
            if route in available_routes:
                destination.append(truth.counterfactual_duration_seconds[route] - fastest_seconds)
        query_minute = truth.departure.hour * 60 + truth.departure.minute
        bucket_candidates = {}
        for variant, variant_trips in train_by_variant.items():
            nearby = [item.duration_seconds for item in variant_trips if abs(item.departure_minute - query_minute) <= 30]
            if nearby:
                bucket_candidates[variant] = statistics.median(nearby)
        if bucket_candidates:
            route = variant_truth.get(min(bucket_candidates, key=bucket_candidates.get))
            if route in available_routes:
                bucket_regrets.append(truth.counterfactual_duration_seconds[route] - fastest_seconds)
        recommendation = od.model.predict_all(truth.departure)
        assigned = od.test_assignments[trip.trip_id]
        prediction = next((item for item in recommendation.predictions if item.variant_id == assigned), None)
        if prediction:
            actual = truth.duration_seconds
            coverage.append(prediction.interval_low_seconds <= actual <= prediction.interval_high_seconds)
            widths.append(prediction.interval_high_seconds - prediction.interval_low_seconds)
            target = strong_errors if prediction.confidence == "strong" else weak_errors
            target.append(abs(prediction.predicted_seconds - actual))
        if recommendation.exploration_candidate:
            exploration_candidates += 1
            if recommendation.exploration_candidate.variant_id == recommendation.fastest_expected.variant_id or recommendation.exploration_candidate.expected_slowdown_seconds < 0:
                exploration_violations += 1

    # Delay truth: only persistent recurring zones active in the held-out period count positive.
    positives = []
    for _, truth in test_target:
        for delay in truth.delays:
            if delay.persistent_target:
                local = (delay.location_xy[0] - truth.path_xy[0][0], delay.location_xy[1] - truth.path_xy[0][1])
                positives.append((truth.route_id, local))
    unique_positives = []
    for item in positives:
        if not any(item[0] == existing[0] and euclidean(item[1], existing[1]) < 180 for existing in unique_positives):
            unique_positives.append(item)
    matched_positive = set()
    true_positive_zones = 0
    for zone in od.delay_zones:
        route = variant_truth.get(zone.variant_id)
        match = next((index for index, positive in enumerate(unique_positives) if positive[0] == route and euclidean((zone.center_x_m, zone.center_y_m), positive[1]) <= 280), None)
        if match is not None:
            true_positive_zones += 1
            matched_positive.add(match)
    delay_precision = true_positive_zones / max(1, len(od.delay_zones))
    delay_recall = len(matched_positive) / max(1, len(unique_positives))
    if delay_precision < 0.7 and od.delay_zones:
        failures.append("localized-delay inference still confuses some behavioral/random dwell with route delay")
    if _safe_mean(regrets) > _safe_mean(bucket_regrets):
        failures.append("richer temporal model did not beat the broad time-bucket baseline in this scenario")
    if recall < 0.75:
        failures.append("segmentation recall fell below 75%")

    metrics = RobustMetrics(
        observed.regime.name, len(history.journeys), len(observed.recorded_journey_ids), len(learned.all_trips),
        precision, recall, segmentation_f1, place_precision, place_recall, od_precision, od_recall,
        len(train_target), len(od.hierarchy.variants), len(od.hierarchy.families), variant_ari, family_ari,
        _safe_mean(assignment_correct), _safe_mean(errors), _safe_mean(ranking), _safe_mean(regrets),
        _safe_mean(common_regrets), _safe_mean(median_regrets), _safe_mean(bucket_regrets), _safe_mean(uniform_regrets),
        _safe_mean(coverage), _safe_mean(widths), _safe_mean(strong_errors) if strong_errors else None,
        _safe_mean(weak_errors) if weak_errors else None, delay_precision, delay_recall,
        _safe_mean(no_zone_regrets) - _safe_mean(regrets), len(od.delay_zones), exploration_candidates, exploration_violations,
    )
    return RobustEvaluation(metrics, _cold_start(train_target, test_target), place_mapping, target_od_id, variant_truth, tuple(failures))
