"""Mission 3 canonical, multi-seed, sensitivity, and replay benchmark."""

from __future__ import annotations

import argparse
import json
import statistics
from dataclasses import asdict
from pathlib import Path

from .adaptive_model import AdaptiveJourneyModel, AdaptiveRouteModel
from .adversarial import LifeConfig, generate_life_history, observe_life_history
from .incremental_replay import milestone_steps, replay_route_history
from .place_regions import discover_place_regions
from .robust_evaluation import align_trips_to_truth, evaluate_robust_lab
from .robust_ingestion import segment_trips_robust
from .robust_pipeline import learn_robust_lab
from .route_families import cluster_route_hierarchy
from .stable_routes import StableRouteRegistry
from .trajectories import adjusted_rand_index


SEEDS = (20260812, 20260829, 20260907, 20261003, 20261111)


def _safe_mean(values: list[float]) -> float | None:
    return statistics.mean(values) if values else None


def _percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    values = sorted(values)
    return values[round((len(values) - 1) * p)]


def _scenario(seed: int, signal: str, regime: str = "normal") -> dict[str, object]:
    history = generate_life_history(LifeConfig(seed=seed, behavior_drift=False, temporal_signal=signal))
    observed = observe_life_history(history, regime)
    learned = learn_robust_lab(observed.raw_points)
    evaluation = evaluate_robust_lab(learned, history, observed)
    aligned = align_trips_to_truth(learned.all_trips, history, observed.recorded_journey_ids)
    adaptive_regrets: list[float] = []
    heldout_errors: list[float] = []
    abstentions = 0
    selected: dict[str, str] = {}
    delay_enabled = 0
    interval_widths: list[float] = []
    coverage: list[float] = []
    od = learned.od_learning.get(evaluation.target_od_id or "")
    if od:
        models = []
        for variant_id in sorted(set(od.model_assignments.values())):
            trips = tuple(trip for trip in learned.train_trips if od.model_assignments.get(trip.trip_id) == variant_id)
            identity = min(1.0, len(trips) / 8.0) * statistics.mean(trip.observation_quality for trip in trips)
            model = AdaptiveRouteModel(variant_id, trips, od.delay_zones, identity)
            models.append(model)
            selected[variant_id] = model.selected_model
            delay_enabled += model.selected_model == "temporal_delay"
        journey_model = AdaptiveJourneyModel(tuple(models))
        mapping = evaluation.variant_truth_mapping
        for trip in learned.test_trips:
            truth = aligned.get(trip.trip_id)
            if not truth or truth.od_id != "School->Work" or trip.trip_id not in od.test_assignments:
                continue
            recommendation = journey_model.predict_all(truth.departure)
            if recommendation.fastest_expected is None:
                abstentions += 1
            # Latent policy quality is measured even when the user-facing object
            # honestly abstains; this separates decision quality from trust gating.
            if recommendation.predictions:
                chosen = mapping.get(recommendation.predictions[0].variant_id)
                available = {route: value for route, value in truth.counterfactual_duration_seconds.items() if route in set(mapping.values())}
                if chosen in available and available:
                    adaptive_regrets.append(available[chosen] - min(available.values()))
            assigned = od.test_assignments.get(trip.trip_id)
            prediction = next((item for item in recommendation.predictions if item.variant_id == assigned), None)
            if prediction:
                heldout_errors.append(abs(prediction.expected_seconds - truth.duration_seconds))
                interval_widths.append(prediction.interval_high_seconds - prediction.interval_low_seconds)
                coverage.append(float(prediction.interval_low_seconds <= truth.duration_seconds <= prediction.interval_high_seconds))
    place_regions = discover_place_regions(learned.train_trips)
    confirmed = [region for region in place_regions.regions if region.status != "candidate"]
    return {
        "seed": seed,
        "signal": signal,
        "regime": regime,
        "selected_models": selected,
        "adaptive_regret_seconds": _safe_mean(adaptive_regrets),
        "adaptive_heldout_mae_seconds": _safe_mean(heldout_errors),
        "overall_median_regret_seconds": evaluation.metrics.overall_median_regret_seconds,
        "time_bucket_regret_seconds": evaluation.metrics.time_bucket_regret_seconds,
        "richer_mission2_regret_seconds": evaluation.metrics.recommendation_regret_seconds,
        "model_selection_regret_seconds": None if not adaptive_regrets else _safe_mean(adaptive_regrets) - min(evaluation.metrics.overall_median_regret_seconds, evaluation.metrics.time_bucket_regret_seconds, evaluation.metrics.recommendation_regret_seconds),
        "abstentions": abstentions,
        "delay_enabled_routes": delay_enabled,
        "interval_coverage": _safe_mean(coverage),
        "mean_interval_width_seconds": _safe_mean(interval_widths),
        "confirmed_places": len(confirmed),
        "candidate_places": sum(region.status == "candidate" for region in place_regions.regions),
        "suppressed_truncated_clusters": place_regions.suppressed_truncated_clusters,
    }


def _route_quality(regime: str) -> dict[str, object]:
    history = generate_life_history()
    observed = observe_life_history(history, regime)
    trips = segment_trips_robust(observed.raw_points)
    aligned = align_trips_to_truth(trips, history, observed.recorded_journey_ids)
    train = tuple(sorted(trips, key=lambda item: item.departure)[: int(len(trips) * 0.8)])
    target = tuple(trip for trip in train if trip.trip_id in aligned and aligned[trip.trip_id].od_id == "School->Work")
    registry = StableRouteRegistry()
    snapshots = [registry.update(target[: min(count, len(target))]) for count in (5, 10, 20, 40, len(target))]
    final = snapshots[-1]
    learned = learn_robust_lab(observed.raw_points)
    evaluation = evaluate_robust_lab(learned, history, observed)
    od = learned.od_learning.get(evaluation.target_od_id or '')
    runtime_variants = None
    if od:
        by_id = {trip.trip_id: trip for trip in learned.train_trips}
        od_trips = tuple(by_id[trip_id] for trip_id in od.trip_ids if trip_id in by_id)
        runtime_variants = len(StableRouteRegistry().update(od_trips).routes)
    ari = adjusted_rand_index([aligned[trip.trip_id].route_id for trip in target], [final.assignments[trip.trip_id] for trip in target])
    return {
        "regime": regime,
        "trips": len(target),
        "variants": len(final.routes),
        "runtime_discovered_od_variants": runtime_variants,
        "ari": ari,
        "threshold_m": final.adaptive_threshold_m,
        "snapshots": [{"trips": item.history_trips, "variants": len(item.routes), "retained_identity_fraction": item.retained_identity_fraction, "revision_events": len(item.revisions)} for item in snapshots],
    }


def _sensitivity() -> list[dict[str, object]]:
    history = generate_life_history()
    observed = observe_life_history(history, "normal")
    trips = segment_trips_robust(observed.raw_points)
    aligned = align_trips_to_truth(trips, history, observed.recorded_journey_ids)
    target = tuple(trip for trip in trips[: int(len(trips) * 0.8)] if trip.trip_id in aligned and aligned[trip.trip_id].od_id == "School->Work")
    rows = []
    for threshold in (160.0, 180.0, 200.0, 220.0, 250.0):
        hierarchy = cluster_route_hierarchy(target, variant_threshold_override_m=threshold)
        assignments = {trip_id: variant.variant_id for variant in hierarchy.variants for trip_id in variant.trip_ids}
        rows.append({"route_threshold_m": threshold, "variants": len(hierarchy.variants), "ari": adjusted_rand_index([aligned[trip.trip_id].route_id for trip in target], [assignments[trip.trip_id] for trip in target])})
    return rows


def run_benchmark() -> dict[str, object]:
    scenarios = [_scenario(seed, signal) for seed in SEEDS for signal in ("strong", "weak", "none")]
    distributions = {}
    for signal in ("strong", "weak", "none"):
        rows = [row for row in scenarios if row["signal"] == signal]
        regrets = [float(row["adaptive_regret_seconds"]) for row in rows if row["adaptive_regret_seconds"] is not None]
        selection_regrets = [float(row["model_selection_regret_seconds"]) for row in rows if row["model_selection_regret_seconds"] is not None]
        distributions[signal] = {
            "adaptive_regret_median": statistics.median(regrets) if regrets else None,
            "adaptive_regret_iqr": [_percentile(regrets, 0.25), _percentile(regrets, 0.75)],
            "adaptive_regret_worst": max(regrets) if regrets else None,
            "material_simple_baseline_failures": sum(float(row["adaptive_regret_seconds"]) > min(float(row["overall_median_regret_seconds"]), float(row["time_bucket_regret_seconds"])) + 5.0 for row in rows if row["adaptive_regret_seconds"] is not None),
            "model_selection_regret_median": statistics.median(selection_regrets) if selection_regrets else None,
        }
    canonical_history = generate_life_history()
    canonical_observed = observe_life_history(canonical_history, "normal")
    canonical_trips = segment_trips_robust(canonical_observed.raw_points)
    aligned = align_trips_to_truth(canonical_trips, canonical_history, canonical_observed.recorded_journey_ids)
    target = tuple(trip for trip in canonical_trips if trip.trip_id in aligned and aligned[trip.trip_id].od_id == "School->Work")
    replay = replay_route_history(target)
    route_quality = [_route_quality(regime) for regime in ("good", "normal", "sparse", "hostile")]
    weak_failures = distributions["weak"]["material_simple_baseline_failures"]
    strong_failures = distributions["strong"]["material_simple_baseline_failures"]
    good = next(item for item in route_quality if item["regime"] == "good")
    verdict = "GREEN" if weak_failures == 0 and strong_failures == 0 and good["runtime_discovered_od_variants"] <= 10 and good["variants"] <= 10 else "YELLOW"
    return {
        "verdict": verdict,
        "seeds": list(SEEDS),
        "scenarios": scenarios,
        "distributions": distributions,
        "route_quality": route_quality,
        "threshold_sensitivity": _sensitivity(),
        "incremental_milestones": [asdict(step) for step in milestone_steps(replay)],
        "known_limitations": [
            "strong-signal per-route graduation does not always recover the joint richer ranker's full held-out advantage",
            "sparse/hostile route identities remain unsuitable for confident recommendation and are gated by abstention",
            "place regions use dwell evidence inferred from recording context, not ground-truth dwell duration",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("artifacts/mission3-results.json"))
    args = parser.parse_args()
    result = run_benchmark()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"verdict": result["verdict"], "output": str(args.output), "distributions": result["distributions"]}, indent=2))


if __name__ == "__main__":
    main()
