"""Run the complete Mission 2 adversarial benchmark and report."""

from __future__ import annotations

import argparse
import math
from datetime import timedelta
from pathlib import Path

from .adversarial import LifeConfig, generate_life_history, observe_life_history
from .benchmark import representative_matrix, run_benchmark, write_benchmark_json
from .benchmark_report import write_benchmark_report
from .robust_evaluation import evaluate_robust_lab
from .robust_pipeline import learn_robust_lab


def _fmt(value: object, digits: int = 2) -> str:
    if value is None:
        return "n/a"
    numeric = float(value)
    return "abstained" if math.isnan(numeric) else f"{numeric:.{digits}f}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("artifacts") / "mission2")
    args = parser.parse_args(argv)

    print("ROUTE READER - MISSION 2 ADVERSARIAL ROBUSTNESS LAB")
    print("=" * 58)
    result = run_benchmark()
    for scenario in result.scenarios:
        m = scenario.metrics
        print(
            f"{scenario.name:25s} segF1={_fmt(m['segmentation_f1'])} "
            f"placeP/R={_fmt(m['place_precision'])}/{_fmt(m['place_recall'])} "
            f"variantARI={_fmt(m['route_variant_ari'])} rank={_fmt(m['route_ranking_accuracy'])} "
            f"regret={_fmt(m['recommendation_regret_seconds'],1)}s interval={_fmt(m['mean_interval_width_seconds'],0)}s"
        )
        for failure in scenario.failures:
            print(f"  exposed: {failure}")

    print("\nCOLD START - NORMAL EVOLVING HISTORY")
    normal = next(item for item in result.scenarios if item.name == "quality-normal")
    for item in normal.cold_start:
        print(
            f"  {item['history_trips']:2d} trips: {item['stage']:27s} "
            f"variants={item['learned_variants']:2d} MAE={_fmt(item['duration_mae_seconds'],1)}s "
            f"interval={_fmt(item['mean_interval_width_seconds'],0)}s rank={_fmt(item['route_ranking_accuracy'])}"
        )

    print("\nINTERPRETABLE DECISION-TIME QUERIES")
    history = generate_life_history(LifeConfig())
    observed = observe_life_history(history, "normal")
    learned = learn_robust_lab(observed.raw_points)
    evaluation = evaluate_robust_lab(learned, history, observed)
    if evaluation.target_od_id and evaluation.target_od_id in learned.od_learning:
        od = learned.od_learning[evaluation.target_od_id]
        reference = max(trip.departure for trip in learned.train_trips) + timedelta(days=3)
        for minute in (15 * 60 + 5, 15 * 60 + 30, 15 * 60 + 55, 16 * 60 + 5, 16 * 60 + 30):
            departure = reference.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(minutes=minute)
            recommendation = od.model.predict_all(departure)
            print(f"  {departure:%H:%M} fastest expected: {recommendation.fastest_expected.variant_id}")
            for prediction in recommendation.predictions:
                print(
                    f"    {prediction.variant_id}: {prediction.predicted_seconds/60:.1f}m "
                    f"[{prediction.interval_low_seconds/60:.1f}-{prediction.interval_high_seconds/60:.1f}], "
                    f"{prediction.confidence}, relevant={prediction.relevant_trips}, recent={prediction.recent_trips}, "
                    f"quality={prediction.mean_observation_quality:.2f}, recent-vs-history={prediction.recent_vs_historical_seconds:+.0f}s"
                )
            if recommendation.exploration_candidate:
                candidate = recommendation.exploration_candidate
                print(
                    f"    exploration only: {candidate.variant_id}, expected {candidate.expected_slowdown_seconds:.0f}s slower, "
                    f"value={candidate.exploration_score:.0f}/100 - {candidate.reason}"
                )

    json_path = args.output / "benchmark.json"
    html_path = args.output / "benchmark_report.html"
    write_benchmark_json(result, json_path)
    write_benchmark_report(result, html_path)
    print(f"\nVERDICT: {result.verdict}")
    print(f"Machine results: {json_path.resolve()}")
    print(f"Human report:    {html_path.resolve()}")
    return 0 if result.verdict == "GREEN" else 1


if __name__ == "__main__":
    raise SystemExit(main())
