"""Run the complete Route Reader proof from raw trace generation to evaluation."""

from __future__ import annotations

import argparse
import json
from datetime import timedelta
from pathlib import Path

from .evaluation import evaluate
from .pipeline import learn_from_raw_points
from .report import build_html_report, format_clock, format_duration
from .synthetic import generate_canonical_dataset, read_raw_jsonl, write_raw_jsonl


def _verdict(metrics: object, preferences: list[str]) -> str:
    checks = [
        metrics.segmentation_recall >= 0.95,
        metrics.recurring_place_recall >= 1.0,
        metrics.discovered_route_variants == 4,
        metrics.train_route_ari >= 0.90,
        metrics.held_out_route_assignment_accuracy >= 0.90,
        metrics.travel_time_mae_seconds <= 30.0,
        metrics.route_ranking_accuracy >= 0.80,
        metrics.mean_recommendation_regret_seconds < metrics.common_route_baseline_regret_seconds,
        metrics.mean_recommendation_regret_seconds < metrics.overall_median_baseline_regret_seconds,
        metrics.delay_feature_mae_improvement_seconds > 0.0,
        len(set(preferences)) >= 2,
    ]
    return "GREEN" if all(checks) else "YELLOW"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=20260811)
    parser.add_argument("--weeks", type=int, default=10)
    parser.add_argument("--output", type=Path, default=Path("artifacts"))
    arguments = parser.parse_args(argv)

    dataset = generate_canonical_dataset(seed=arguments.seed, weeks=arguments.weeks)
    raw_path = arguments.output / "canonical_raw_traces.jsonl"
    write_raw_jsonl(dataset, raw_path)
    # Replay through the serialized raw boundary, proving truth is not passed in memory.
    observable_points = read_raw_jsonl(raw_path)
    learned = learn_from_raw_points(observable_points)
    result = evaluate(learned, dataset)
    metrics = result.metrics
    primary_od = max(learned.discovery.od_pairs, key=lambda pair: len(pair.trip_ids))

    print("ROUTE READER - CANONICAL PERSONAL-HISTORY PROOF")
    print("=" * 52)
    print(f"Raw observations: {len(observable_points):,} across {arguments.weeks} weeks")
    print(f"Trips segmented: {len(learned.all_trips)} (train {len(learned.train_trips)} / held-out {len(learned.test_trips)})")
    print(f"Recurring places: {', '.join(f'{p.place_id} ({p.observations} observations)' for p in learned.discovery.places)}")
    print(f"Recurring journey: {primary_od.origin_place_id} -> {primary_od.destination_place_id} ({len(primary_od.trip_ids)} training trips)")
    print("\nLEARNED ROUTE VARIANTS (hidden truth shown only after evaluation)")
    counts = learned.model.route_trip_counts()
    for cluster in learned.route_clusters:
        print(f"  {cluster.cluster_id}: {counts[cluster.cluster_id]:2d} trips - evaluation match {result.cluster_truth_mapping[cluster.cluster_id]}")

    print("\nINFERRED RECURRING DELAY ZONES")
    for zone in learned.delay_zones:
        print(
            f"  {zone.zone_id} on {zone.route_cluster_id}: {zone.affected_trips} affected trips, "
            f"median {format_duration(zone.median_delay_seconds)}, peak {format_clock(zone.peak_start_minute)}-{format_clock(zone.peak_end_minute)} "
            f"({format_duration(zone.peak_median_delay_seconds)})"
        )

    print("\nTIME-SPECIFIC RECOMMENDATIONS")
    reference = learned.test_trips[0].departure + timedelta(days=14)
    preferences = []
    for minute in (15 * 60 + 12, 15 * 60 + 29, 15 * 60 + 41, 15 * 60 + 52, 16 * 60 + 6):
        departure = reference.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(minutes=minute)
        recommendation = learned.model.predict_all(departure)
        preferences.append(recommendation.best.route_cluster_id)
        print(f"  {format_clock(minute)} -> {recommendation.best.route_cluster_id}")
        for prediction in recommendation.predictions:
            print(
                f"    {prediction.route_cluster_id}: {format_duration(prediction.predicted_seconds)} "
                f"[{format_duration(prediction.interval_low_seconds)}-{format_duration(prediction.interval_high_seconds)}], "
                f"confidence {prediction.confidence}, {prediction.relevant_trips} relevant trips, "
                f"effective n={prediction.effective_sample_size:.1f}"
            )
        if recommendation.exploration_note:
            print(f"    Mix-it-up option: {recommendation.exploration_note}")

    print("\nCHRONOLOGICAL HELD-OUT EVALUATION")
    print(f"  Route clustering ARI:              {metrics.train_route_ari:.3f}")
    print(f"  Held-out route assignment:         {metrics.held_out_route_assignment_accuracy:.0%}")
    print(f"  Travel-time MAE:                   {metrics.travel_time_mae_seconds:.2f}s")
    print(f"  MAE without localized delays:      {metrics.travel_time_mae_without_delay_zones_seconds:.2f}s")
    print(f"  Local-delay feature improvement:   {metrics.delay_feature_mae_improvement_seconds:.2f}s")
    print(f"  Fastest-route ranking accuracy:    {metrics.route_ranking_accuracy:.0%}")
    print(f"  Route Reader mean regret:          {metrics.mean_recommendation_regret_seconds:.2f}s")
    print(f"  Most-common-route baseline:        {metrics.common_route_baseline_regret_seconds:.2f}s ({metrics.common_route_baseline_accuracy:.0%} accurate)")
    print(f"  Lowest-overall-median baseline:    {metrics.overall_median_baseline_regret_seconds:.2f}s ({metrics.overall_median_baseline_accuracy:.0%} accurate)")
    print(f"  Confidence interval coverage:      {metrics.confidence_interval_coverage:.0%}")

    arguments.output.mkdir(parents=True, exist_ok=True)
    metrics_path = arguments.output / "canonical_metrics.json"
    metrics_path.write_text(json.dumps(metrics.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report_path = arguments.output / "route_reader_report.html"
    build_html_report(learned, result, report_path)
    verdict = _verdict(metrics, preferences)
    print(f"\nVERDICT: {verdict}")
    print(f"Raw replay fixture: {raw_path.resolve()}")
    print(f"Metrics:            {metrics_path.resolve()}")
    print(f"Visual report:      {report_path.resolve()}")
    return 0 if verdict == "GREEN" else 1


if __name__ == "__main__":
    raise SystemExit(main())
