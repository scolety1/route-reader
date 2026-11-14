"""Mission 4 online-memory benchmark, ablation, and machine-readable audit."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path

from .adversarial import LifeConfig, generate_life_history, observe_life_history
from .churn_instrumentation import instrument_batch_churn
from .mission4_evaluation import strict_walk_forward, walk_forward_dict
from .online_corridors import OnlineCorridorMemory
from .robust_evaluation import align_trips_to_truth
from .robust_ingestion import segment_trips_robust
from .trajectories import adjusted_rand_index


SEEDS = (20260812, 20260829, 20260907, 20261003, 20261111)


def _quantile(values: list[float], probability: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * probability)]


def _target(config: LifeConfig, regime: str):
    history = generate_life_history(config)
    observed = observe_life_history(history, regime)
    trips = segment_trips_robust(observed.raw_points)
    aligned = align_trips_to_truth(trips, history, observed.recorded_journey_ids)
    target = tuple(trip for trip in trips if trip.trip_id in aligned and aligned[trip.trip_id].od_id == "School->Work")
    return history, observed, target, {trip.trip_id: aligned[trip.trip_id] for trip in target}


def _memory_metrics(target, truth_by_trip) -> tuple[dict[str, object], OnlineCorridorMemory]:
    memory = OnlineCorridorMemory()
    snapshots = []
    for trip in target:
        memory.observe(trip)
        if len(memory.trips) in {5, 10, 20, 40, len(target)}:
            snapshots.append(asdict(memory.snapshot()))
    durable = memory.established_variants
    assignments = [(truth_by_trip[trip_id].route_id, variant_id) for trip_id, variant_id in memory.assignments.items() if variant_id and memory.variants[variant_id].status == "established"]
    purity_numerator = 0
    truth_fragments: dict[str, Counter[str]] = defaultdict(Counter)
    majority_routes = []
    for variant in durable:
        votes = Counter(truth_by_trip[item].route_id for item in variant.trip_ids if item in truth_by_trip)
        if votes:
            purity_numerator += votes.most_common(1)[0][1]
            majority_routes.append(votes.most_common(1)[0])
        for trip_id in variant.trip_ids:
            if trip_id in truth_by_trip:
                truth_fragments[truth_by_trip[trip_id].route_id][variant.variant_id] += 1
    purity = purity_numerator / max(1, sum(len(variant.trip_ids) for variant in durable))
    fragmentation_numerator = sum(sum(values.values()) - max(values.values()) for values in truth_fragments.values() if values)
    fragmentation_denominator = sum(sum(values.values()) for values in truth_fragments.values())
    duplicate_durable = sum(max(0, count - 1) for count in Counter(route for route, _ in majority_routes).values())
    impure_durable = sum(count / max(1, len(variant.trip_ids)) < 0.75 for variant, (_, count) in zip(durable, majority_routes)) if majority_routes else 0
    labels_true = [left for left, _ in assignments]
    labels_pred = [right for _, right in assignments]
    ari = adjusted_rand_index(labels_true, labels_pred) if len(assignments) >= 2 else None
    emerging_ids = [trip.trip_id for trip in target if truth_by_trip[trip.trip_id].route_id == "N"]
    discovery_delay = None
    if emerging_ids:
        first_index = next(index for index, trip in enumerate(target, 1) if trip.trip_id == emerging_ids[0])
        established_n = set()
        for variant in durable:
            votes = Counter(truth_by_trip[item].route_id for item in variant.trip_ids if item in truth_by_trip)
            if votes and votes.most_common(1)[0][0] == "N":
                established_n.add(variant.variant_id)
        event = next((item for item in memory.events if item.event == "graduate_established" and item.variant_id in established_n), None)
        if event:
            graduation_index = next(index for index, trip in enumerate(target, 1) if trip.trip_id == event.trip_id)
            discovery_delay = max(1, graduation_index - first_index + 1)
    detour_statuses = [memory.variants[variant_id].status for trip_id, variant_id in memory.assignments.items() if variant_id and truth_by_trip[trip_id].route_id == "A_detour"]
    return {
        "trips": len(target),
        "established_variants": len(durable),
        "repeated_variants": sum(item.status == "repeated" for item in memory.variants.values()),
        "candidate_variants": sum(item.status == "candidate" for item in memory.variants.values()),
        "expired_candidates": sum(item.status == "expired" for item in memory.variants.values()),
        "durable_route_purity": purity,
        "durable_route_ari": ari,
        "duplicate_durable_variants": duplicate_durable,
        "impure_durable_variants": impure_durable,
        "evidence_fragmentation": fragmentation_numerator / max(1, fragmentation_denominator),
        "identity_churn_per_100_trips": memory.snapshot().identity_churn_per_100_trips,
        "identity_continuity": memory.snapshot().identity_continuity,
        "unresolved_trips": memory.snapshot().unresolved_trips,
        "shared_corridor_nodes": memory.snapshot().shared_corridor_nodes,
        "corridor_nodes": memory.snapshot().corridor_nodes,
        "emerging_route_discovery_delay_trips": discovery_delay,
        "temporary_detour_became_established": "established" in detour_statuses,
        "event_causes": dict(Counter(event.cause for event in memory.events)),
        "event_types": dict(Counter(event.event for event in memory.events)),
        "snapshots": snapshots,
    }, memory


def _policy(result, name: str) -> dict[str, object]:
    return asdict(next(item for item in result.policies if item.policy == name))


def run_benchmark() -> dict[str, object]:
    # Same underlying history across observation qualities.
    route_regimes = []
    regime_cache = {}
    for regime in ("good", "normal", "sparse", "hostile"):
        _, _, target, truth = _target(LifeConfig(), regime)
        metrics, memory = _memory_metrics(target, truth)
        route_regimes.append({"regime": regime, **metrics})
        regime_cache[regime] = (target, truth, memory)
    # Instrument the architecture being replaced on canonical normal history.
    normal_target, normal_truth, _ = regime_cache["normal"]
    mission3_churn = instrument_batch_churn(normal_target)
    mission4_normal = next(item for item in route_regimes if item["regime"] == "normal")
    ablation = {
        "mission3_batch": mission3_churn,
        "mission4_online": {
            "final_durable_variants": mission4_normal["established_variants"],
            "candidate_variants": mission4_normal["candidate_variants"] + mission4_normal["repeated_variants"],
            "identity_churn_per_100_trips": mission4_normal["identity_churn_per_100_trips"],
            "historical_identity_reassignments": 0,
            "evidence_fragmentation": mission4_normal["evidence_fragmentation"],
        },
    }
    scenarios = []
    canonical_walk_forward = {}
    for seed in SEEDS:
        for signal in ("strong", "weak", "none"):
            _, _, target, truth = _target(LifeConfig(seed=seed, behavior_drift=False, temporal_signal=signal), "normal")
            result = strict_walk_forward(target, truth)
            adaptive = _policy(result, "adaptive")
            overall = _policy(result, "overall_median")
            bucket = _policy(result, "time_bucket")
            kernel = _policy(result, "temporal_kernel")
            scenarios.append({
                "seed": seed,
                "signal": signal,
                "adaptive": adaptive,
                "overall_median": overall,
                "time_bucket": bucket,
                "temporal_kernel": kernel,
                "oracle": _policy(result, "oracle"),
                "interval_coverage": result.interval_coverage,
                "mean_interval_width_seconds": result.mean_interval_width_seconds,
                "leakage_violations": result.leakage_violations,
                "model_selection_errors": result.model_selection_errors,
            })
            if seed == SEEDS[0]:
                canonical_walk_forward[signal] = walk_forward_dict(result)
    distributions = {}
    for signal in ("strong", "weak", "none"):
        rows = [item for item in scenarios if item["signal"] == signal]
        regrets = [item["adaptive"]["mean_regret_seconds"] for item in rows if item["adaptive"]["mean_regret_seconds"] is not None]
        widths = [item["mean_interval_width_seconds"] for item in rows if item["mean_interval_width_seconds"] is not None]
        simple_failures = sum(item["adaptive"]["mean_regret_seconds"] > min(item["overall_median"]["mean_regret_seconds"], item["time_bucket"]["mean_regret_seconds"]) + 5.0 for item in rows if item["adaptive"]["mean_regret_seconds"] is not None and item["overall_median"]["mean_regret_seconds"] is not None and item["time_bucket"]["mean_regret_seconds"] is not None)
        distributions[signal] = {
            "adaptive_regret_median": statistics.median(regrets),
            "adaptive_regret_iqr": [_quantile(regrets, 0.25), _quantile(regrets, 0.75)],
            "adaptive_regret_worst": max(regrets),
            "material_simple_baseline_failures": simple_failures,
            "abstention_rate_median": statistics.median(item["adaptive"]["abstention_rate"] for item in rows),
            "interval_width_median_seconds": statistics.median(widths),
            "interval_coverage_median": statistics.median(item["interval_coverage"] for item in rows if item["interval_coverage"] is not None),
        }
    good = next(item for item in route_regimes if item["regime"] == "good")
    normal = next(item for item in route_regimes if item["regime"] == "normal")
    gates = {
        "route_churn_reduced": mission4_normal["established_variants"] <= max(6, mission3_churn["final_structural_variants"] * 0.60),
        "good_normal_paradox_removed": abs(good["established_variants"] - normal["established_variants"]) <= 1,
        "alternatives_preserved": min(good["established_variants"], normal["established_variants"]) >= 3,
        "new_route_learnable": normal["emerging_route_discovery_delay_trips"] is not None and normal["emerging_route_discovery_delay_trips"] <= 16,
        "temporary_detour_contained": not normal["temporary_detour_became_established"],
        "weak_signal_safe": distributions["weak"]["material_simple_baseline_failures"] == 0,
        "strict_walk_forward": sum(item["leakage_violations"] for item in scenarios) == 0,
        "confidence_useful": distributions["strong"]["interval_width_median_seconds"] < 218.0,
        "strong_signal_improved": canonical_walk_forward["strong"]["policies"][6]["mean_regret_seconds"] < 18.0,
    }
    verdict = "GREEN" if all(gates.values()) else "YELLOW"
    return {
        "verdict": verdict,
        "gates": gates,
        "mission3_churn_instrumentation": mission3_churn,
        "route_regimes": route_regimes,
        "ablation": ablation,
        "scenarios": scenarios,
        "distributions": distributions,
        "canonical_walk_forward": canonical_walk_forward,
        "explicit_failures": [name for name, passed in gates.items() if not passed],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("artifacts/mission4-results.json"))
    args = parser.parse_args()
    result = run_benchmark()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False, default=str) + "\n", encoding="utf-8")
    print(json.dumps({"verdict": result["verdict"], "gates": result["gates"], "distributions": result["distributions"]}, indent=2))


if __name__ == "__main__":
    main()
