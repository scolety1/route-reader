"""Chronological held-out evaluation against hidden synthetic truth."""

from __future__ import annotations

import statistics
from dataclasses import asdict, dataclass

from .geo import haversine_m, xy_to_latlon
from .pipeline import LearnedRouteReader
from .synthetic import ORIGIN_LAT, ORIGIN_LON, SyntheticDataset, TripTruth
from .trajectories import adjusted_rand_index, majority_truth_mapping
from .types import Trip


@dataclass(frozen=True)
class EvaluationMetrics:
    segmented_trips: int
    segmentation_recall: float
    recurring_places_found: int
    recurring_place_recall: float
    recurring_od_train_coverage: float
    discovered_route_variants: int
    train_route_ari: float
    held_out_route_assignment_accuracy: float
    travel_time_mae_seconds: float
    travel_time_mae_without_delay_zones_seconds: float
    delay_feature_mae_improvement_seconds: float
    route_ranking_accuracy: float
    mean_recommendation_regret_seconds: float
    common_route_baseline_accuracy: float
    common_route_baseline_regret_seconds: float
    overall_median_baseline_accuracy: float
    overall_median_baseline_regret_seconds: float
    confidence_interval_coverage: float

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


@dataclass(frozen=True)
class HeldOutDecision:
    departure_iso: str
    actual_route: str
    actual_fastest_route: str
    recommended_route: str
    predicted_seconds: float
    actual_seconds: float
    regret_seconds: float
    confidence: str


@dataclass(frozen=True)
class EvaluationResult:
    metrics: EvaluationMetrics
    cluster_truth_mapping: dict[str, str]
    decisions: tuple[HeldOutDecision, ...]


def _align_truth(trips: tuple[Trip, ...], truths: tuple[TripTruth, ...]) -> dict[str, TripTruth]:
    alignment = {}
    for trip in trips:
        closest = min(
            truths,
            key=lambda truth: abs((truth.departure - trip.departure).total_seconds()),
        )
        distance = abs((closest.departure - trip.departure).total_seconds())
        if distance > 8 * 60:
            raise RuntimeError(f"could not align {trip.trip_id} to fixture truth")
        alignment[trip.trip_id] = closest
    return alignment


def evaluate(
    learned: LearnedRouteReader, dataset: SyntheticDataset
) -> EvaluationResult:
    aligned = _align_truth(learned.all_trips, dataset.truths)
    train_predicted = [
        learned.train_route_assignments[trip.trip_id] for trip in learned.train_trips
    ]
    train_truth = [aligned[trip.trip_id].route_id for trip in learned.train_trips]
    cluster_truth = majority_truth_mapping(train_predicted, train_truth)
    train_ari = adjusted_rand_index(train_truth, train_predicted)

    test_correct = [
        cluster_truth[learned.test_route_assignments[trip.trip_id]]
        == aligned[trip.trip_id].route_id
        for trip in learned.test_trips
    ]
    reverse_mapping = {truth_id: cluster_id for cluster_id, truth_id in cluster_truth.items()}

    route_counts = learned.model.route_trip_counts()
    common_cluster = max(sorted(route_counts), key=lambda key: route_counts[key])
    common_truth = cluster_truth[common_cluster]
    medians = learned.model.historical_medians()
    median_cluster = min(sorted(medians), key=lambda key: medians[key])
    median_truth = cluster_truth[median_cluster]

    absolute_errors = []
    no_zone_errors = []
    coverage = []
    ranking_correct = []
    regrets = []
    common_correct = []
    common_regrets = []
    median_correct = []
    median_regrets = []
    decisions = []

    for trip in learned.test_trips:
        truth = aligned[trip.trip_id]
        recommendation = learned.model.predict_all(truth.departure, use_delay_zones=True)
        no_zone = learned.model.predict_all(truth.departure, use_delay_zones=False)
        predictions_by_truth = {
            cluster_truth[prediction.route_cluster_id]: prediction
            for prediction in recommendation.predictions
        }
        no_zone_by_truth = {
            cluster_truth[prediction.route_cluster_id]: prediction
            for prediction in no_zone.predictions
        }
        actual_route_prediction = predictions_by_truth[truth.route_id]
        actual = truth.counterfactual_duration_seconds[truth.route_id]
        absolute_errors.append(abs(actual_route_prediction.predicted_seconds - actual))
        no_zone_errors.append(
            abs(no_zone_by_truth[truth.route_id].predicted_seconds - actual)
        )
        coverage.append(
            actual_route_prediction.interval_low_seconds
            <= actual
            <= actual_route_prediction.interval_high_seconds
        )

        recommended_truth = cluster_truth[recommendation.best.route_cluster_id]
        fastest_truth = min(
            truth.counterfactual_duration_seconds,
            key=truth.counterfactual_duration_seconds.get,
        )
        fastest_seconds = truth.counterfactual_duration_seconds[fastest_truth]
        regret = truth.counterfactual_duration_seconds[recommended_truth] - fastest_seconds
        ranking_correct.append(recommended_truth == fastest_truth)
        regrets.append(regret)

        common_correct.append(common_truth == fastest_truth)
        common_regrets.append(
            truth.counterfactual_duration_seconds[common_truth] - fastest_seconds
        )
        median_correct.append(median_truth == fastest_truth)
        median_regrets.append(
            truth.counterfactual_duration_seconds[median_truth] - fastest_seconds
        )
        decisions.append(
            HeldOutDecision(
                departure_iso=truth.departure.isoformat(),
                actual_route=truth.route_id,
                actual_fastest_route=fastest_truth,
                recommended_route=recommended_truth,
                predicted_seconds=actual_route_prediction.predicted_seconds,
                actual_seconds=actual,
                regret_seconds=regret,
                confidence=actual_route_prediction.confidence,
            )
        )

    discovered_place_matches = 0
    for true_xy in dataset.true_places_xy.values():
        true_latlon = xy_to_latlon(*true_xy, ORIGIN_LAT, ORIGIN_LON)
        if any(
            haversine_m(*true_latlon, place.latitude, place.longitude) <= 160.0
            for place in learned.discovery.places
        ):
            discovered_place_matches += 1
    primary_od = max(learned.discovery.od_pairs, key=lambda pair: len(pair.trip_ids))
    with_zones_mae = statistics.mean(absolute_errors)
    without_zones_mae = statistics.mean(no_zone_errors)
    metrics = EvaluationMetrics(
        segmented_trips=len(learned.all_trips),
        segmentation_recall=min(1.0, len(learned.all_trips) / len(dataset.truths)),
        recurring_places_found=len(learned.discovery.places),
        recurring_place_recall=discovered_place_matches / len(dataset.true_places_xy),
        recurring_od_train_coverage=len(primary_od.trip_ids) / len(learned.train_trips),
        discovered_route_variants=len(learned.route_clusters),
        train_route_ari=train_ari,
        held_out_route_assignment_accuracy=statistics.mean(test_correct),
        travel_time_mae_seconds=with_zones_mae,
        travel_time_mae_without_delay_zones_seconds=without_zones_mae,
        delay_feature_mae_improvement_seconds=without_zones_mae - with_zones_mae,
        route_ranking_accuracy=statistics.mean(ranking_correct),
        mean_recommendation_regret_seconds=statistics.mean(regrets),
        common_route_baseline_accuracy=statistics.mean(common_correct),
        common_route_baseline_regret_seconds=statistics.mean(common_regrets),
        overall_median_baseline_accuracy=statistics.mean(median_correct),
        overall_median_baseline_regret_seconds=statistics.mean(median_regrets),
        confidence_interval_coverage=statistics.mean(coverage),
    )
    # Reverse mapping is intentionally computed only here, after all learning is done.
    assert set(reverse_mapping) == set(dataset.route_specs)
    return EvaluationResult(metrics, cluster_truth, tuple(decisions))
