"""Mission 5 multi-seed sequential-decision benchmark and audit artifact."""

from __future__ import annotations

import argparse
import inspect
import json
import statistics
from dataclasses import asdict, replace
from pathlib import Path

from .mission5_evaluation import (
    SequentialReplayResult,
    build_partial_feedback_fixture,
    replay_dict,
    strict_partial_feedback_replay,
)
from .off_policy import evaluate_logged_policy
from .online_corridors import OnlineCorridorMemory
from .ranking_policy import ExplorationConfig
from . import off_policy, partial_feedback, ranking_policy, user_behavior


SEEDS = (
    20260812,
    20260829,
    20260907,
    20261003,
    20261111,
    20261201,
    20261219,
    20270107,
    20270123,
    20270211,
)


def _quantile(values: list[float], probability: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * probability)]


def _policy_records(result: SequentialReplayResult):
    return tuple(item for item in result.records if item.phase != "cold_start")


def _mean(records, field: str) -> float | None:
    values = [getattr(item, field) for item in records if getattr(item, field) is not None]
    return statistics.mean(values) if values else None


def _scenario_summary(result: SequentialReplayResult) -> dict[str, object]:
    records = _policy_records(result)
    probabilities = [
        log.decision.exploration.probability if log.decision.exploration else 0.0
        for log in result.logs
    ]
    third = max(1, len(probabilities) // 3)
    selected_expected_cost = sum(
        log.decision.exploration.expected_cost_seconds
        for log in result.logs
        if log.decision.exploration is not None
        and log.decision.policy_selected_action == log.decision.exploration.action_id
    )
    noncompliance = sum(
        record.followed_selected_action is False
        for record in records
    )
    return {
        "config": result.config,
        "total_trips": len(result.records),
        "policy_decisions": len(records),
        "mission5_mean_recommended_regret_seconds": _mean(records, "recommended_pseudo_regret_seconds"),
        "mission5_mean_selected_regret_seconds": _mean(records, "selected_pseudo_regret_seconds"),
        "mission4_mean_regret_seconds": _mean(records, "mission4_regret_seconds"),
        "overall_median_mean_regret_seconds": _mean(records, "overall_median_regret_seconds"),
        "realized_user_mean_regret_seconds": _mean(records, "realized_user_regret_seconds"),
        "full_cumulative_selected_regret_seconds": result.checkpoints[-1].cumulative_selected_pseudo_regret_seconds if result.checkpoints else None,
        "full_cumulative_recommended_regret_seconds": result.checkpoints[-1].cumulative_recommended_pseudo_regret_seconds if result.checkpoints else None,
        "full_cumulative_realized_user_regret_seconds": result.checkpoints[-1].cumulative_realized_user_regret_seconds if result.checkpoints else None,
        "exploration_opportunities": result.exploration_opportunities,
        "exploration_choices": result.exploration_choices,
        "exploration_realized_regret_seconds": result.checkpoints[-1].cumulative_exploration_regret_seconds if result.checkpoints else None,
        "exploration_expected_cost_seconds": selected_expected_cost,
        "early_mean_exploration_probability": statistics.mean(probabilities[:third]) if probabilities else 0.0,
        "late_mean_exploration_probability": statistics.mean(probabilities[-third:]) if probabilities else 0.0,
        "temporal_graduation_decision": result.temporal_graduation_decision,
        "post_graduation_mean_regret_seconds": result.post_graduation_mean_regret_seconds,
        "policy_convergence_last_ten": result.policy_convergence_last_ten,
        "detected_drift_decisions": sum(log.decision.detected_drift for log in result.logs),
        "noncompliant_decisions": noncompliance,
        "action_observation_counts": result.action_observation_counts,
        "unresolved_variant_outcomes": result.unresolved_variant_outcomes,
        "propensity_frozen_before_outcome": result.propensity_frozen_before_outcome,
        "leakage_violations": result.leakage_violations,
        "checkpoints": [asdict(item) for item in result.checkpoints],
    }


def _distribution(rows: list[dict[str, object]]) -> dict[str, object]:
    m5 = [float(item["mission5_mean_recommended_regret_seconds"]) for item in rows]
    m4 = [float(item["mission4_mean_regret_seconds"]) for item in rows]
    overall = [float(item["overall_median_mean_regret_seconds"]) for item in rows]
    exploration = [float(item["exploration_realized_regret_seconds"]) for item in rows]
    convergence = [int(item["temporal_graduation_decision"]) for item in rows if item["temporal_graduation_decision"] is not None]
    return {
        "mission5_regret_median_seconds": statistics.median(m5),
        "mission5_regret_iqr_seconds": [_quantile(m5, 0.25), _quantile(m5, 0.75)],
        "mission5_regret_worst_seconds": max(m5),
        "mission4_regret_median_seconds": statistics.median(m4),
        "median_improvement_over_mission4_seconds": statistics.median(left - right for left, right in zip(m4, m5)),
        "material_simple_baseline_failures": sum(value > min(base, prior) + 5.0 for value, base, prior in zip(m5, overall, m4)),
        "exploration_regret_median_seconds": statistics.median(exploration),
        "exploration_regret_iqr_seconds": [_quantile(exploration, 0.25), _quantile(exploration, 0.75)],
        "graduated_seeds": len(convergence),
        "graduation_decision_median": statistics.median(convergence) if convergence else None,
    }


def _emerging_summary(result: SequentialReplayResult, truth_by_trip) -> dict[str, object]:
    truth_by_decision = {item.decision_id: item for item in result.evaluation_truth}
    emerging_actions = {
        truth_by_decision[trip_id].planned_action
        for trip_id, truth in truth_by_trip.items()
        if truth.route_id == "N" and trip_id in truth_by_decision
    }
    action = next(iter(emerging_actions), None)
    opportunities = choices = first_opportunity = None
    if action:
        opportunity_indices = [
            index
            for index, log in enumerate(result.logs, 1)
            if log.decision.exploration is not None and log.decision.exploration.action_id == action
        ]
        choice_indices = [
            index
            for index, log in enumerate(result.logs, 1)
            if log.decision.exploration is not None
            and log.decision.exploration.action_id == action
            and log.decision.policy_selected_action == action
        ]
        opportunities = len(opportunity_indices)
        choices = len(choice_indices)
        first_opportunity = opportunity_indices[0] if opportunity_indices else None
    count = result.action_observation_counts.get(action, 0) if action else 0
    return {
        "opaque_action": action,
        "resolved_observations": count,
        "reached_repeated": count >= 3,
        "reached_established": count >= 5,
        "exploration_opportunities": opportunities,
        "exploration_choices": choices,
        "first_exploration_opportunity": first_opportunity,
        "became_ranking_contender": bool(action and any(log.decision.policy_actions["mission5_ranking"] == action for log in result.logs)),
    }


def _ope_summary(result: SequentialReplayResult) -> dict[str, object]:
    validations = []
    for validation in result.ope:
        validations.append({
            "target_policy": validation.target_policy,
            "simulator_true_mean_seconds": validation.simulator_true_mean_seconds,
            "truth_supported_decisions": validation.truth_supported_decisions,
            "estimates": [
                {
                    **asdict(estimate),
                    "absolute_error_seconds": (
                        abs(estimate.estimated_mean_seconds - validation.simulator_true_mean_seconds)
                        if estimate.estimated_mean_seconds is not None and validation.simulator_true_mean_seconds is not None
                        else None
                    ),
                }
                for estimate in validation.estimates
            ],
        })
    # Deliberately remove target-policy support to verify the guardrail itself.
    poor_logs = []
    for log in result.logs:
        target = log.decision.policy_actions["mission5_ranking"]
        probabilities = {action: 0.0 for action in log.behavior_propensities}
        probabilities[target] = 0.001
        recipient = log.actual_action if log.actual_action != target else next((action for action in probabilities if action != target), target)
        probabilities[recipient] += 0.999
        poor_logs.append(
            replace(
                log,
                behavior_propensities=probabilities,
                actual_action_propensity=probabilities[log.actual_action],
            )
        )
    poor = evaluate_logged_policy(tuple(poor_logs), "mission5_ranking")
    return {
        "adequate_support": validations,
        "poor_support_guardrail": [asdict(item) for item in poor],
    }


def _spatial_regression(trips) -> dict[str, object]:
    memory = OnlineCorridorMemory()
    historical_reassignments = 0
    for trip in trips:
        before = dict(memory.assignments)
        memory.observe(trip)
        historical_reassignments += sum(memory.assignments.get(key) != value for key, value in before.items())
    snapshot = memory.snapshot()
    return {
        "established_variants": len(memory.established_variants),
        "historical_identity_reassignments": historical_reassignments,
        "identity_churn_per_100_trips": snapshot.identity_churn_per_100_trips,
        "identity_continuity": snapshot.identity_continuity,
        "shared_corridor_nodes": snapshot.shared_corridor_nodes,
    }


def run_benchmark() -> tuple[dict[str, object], SequentialReplayResult]:
    scenarios: dict[str, list[dict[str, object]]] = {signal: [] for signal in ("strong", "weak", "none")}
    canonical: dict[str, SequentialReplayResult] = {}
    canonical_fixtures = {}
    for signal in ("strong", "weak", "none"):
        for seed in SEEDS:
            trips, truth = build_partial_feedback_fixture(
                seed=seed,
                temporal_signal=signal,
                behavior_drift=False,
            )
            result = strict_partial_feedback_replay(
                trips,
                truth,
                seed=seed,
                temporal_signal=signal,
                behavior_drift=False,
                user_behavior="mixed",
            )
            scenarios[signal].append(_scenario_summary(result))
            if seed == SEEDS[0]:
                canonical[signal] = result
                canonical_fixtures[signal] = (trips, truth)

    drift_rows = []
    emerging_rows = []
    spatial_trips = None
    for seed in SEEDS[:5]:
        trips, truth = build_partial_feedback_fixture(
            seed=seed,
            temporal_signal="strong",
            behavior_drift=True,
        )
        result = strict_partial_feedback_replay(
            trips,
            truth,
            seed=seed,
            temporal_signal="strong",
            behavior_drift=True,
            user_behavior="mixed",
        )
        drift_rows.append(_scenario_summary(result))
        emerging_rows.append(_emerging_summary(result, truth))
        if spatial_trips is None:
            spatial_trips = trips

    strong_trips, strong_truth = canonical_fixtures["strong"]
    user_rows = {}
    for behavior in ("compliant", "mixed", "habitual"):
        user_rows[behavior] = _scenario_summary(
            strict_partial_feedback_replay(
                strong_trips,
                strong_truth,
                seed=SEEDS[0],
                temporal_signal="strong",
                behavior_drift=False,
                user_behavior=behavior,
            )
        )
    sensitivities = {
        "mostly_exploitative": ExplorationConfig(maximum_expected_cost_seconds=25.0, maximum_probability=0.03),
        "conservative_epsilon": ExplorationConfig(maximum_expected_cost_seconds=35.0, maximum_probability=0.08),
        "uncertainty_driven": ExplorationConfig(),
    }
    sensitivity_rows = {}
    for name, config in sensitivities.items():
        sensitivity_rows[name] = _scenario_summary(
            strict_partial_feedback_replay(
                strong_trips,
                strong_truth,
                seed=SEEDS[0],
                temporal_signal="strong",
                behavior_drift=False,
                user_behavior="mixed",
                exploration_config=config,
            )
        )

    distributions = {signal: _distribution(rows) for signal, rows in scenarios.items()}
    source = "\n".join(inspect.getsource(module) for module in (partial_feedback, ranking_policy, user_behavior, off_policy))
    canonical_ope = _ope_summary(canonical["strong"])
    ranking_validation = next(
        item for item in canonical_ope["adequate_support"] if item["target_policy"] == "mission5_ranking"
    )
    dr_validation = next(item for item in ranking_validation["estimates"] if item["estimator"] == "doubly_robust_clipped")
    poor_support = canonical_ope["poor_support_guardrail"]
    no_signal = scenarios["none"]
    gates = {
        "true_partial_feedback": all(item["leakage_violations"] == 0 for rows in scenarios.values() for item in rows),
        "structural_truth_boundary": all(token not in source for token in ("from .adversarial", "TrueJourney", "counterfactual_duration_seconds")),
        "decision_aligned_selector": "pairwise_probability" in inspect.getsource(ranking_policy.RegretRankingPolicy),
        "strong_signal_improvement": distributions["strong"]["median_improvement_over_mission4_seconds"] >= 2.0,
        "weak_signal_safe": distributions["weak"]["material_simple_baseline_failures"] == 0,
        "no_signal_exploration_decays": statistics.median(item["late_mean_exploration_probability"] for item in no_signal) < statistics.median(item["early_mean_exploration_probability"] for item in no_signal),
        "drift_detected": sum(item["detected_drift_decisions"] > 0 for item in drift_rows) >= 3,
        "emerging_route_established": sum(item["reached_established"] for item in emerging_rows) >= 3,
        "propensity_correctness": all(item["propensity_frozen_before_outcome"] for rows in scenarios.values() for item in rows),
        "ope_adequate_support": dr_validation["reliable"] and dr_validation["absolute_error_seconds"] <= 12.0,
        "ope_poor_support_guarded": all(not item["reliable"] for item in poor_support),
        "recommendation_actual_separate": user_rows["habitual"]["noncompliant_decisions"] > 0,
    }
    spatial = _spatial_regression(spatial_trips or strong_trips)
    gates["mission4_spatial_stable"] = spatial["established_variants"] == 4 and spatial["historical_identity_reassignments"] == 0
    verdict = "GREEN" if all(gates.values()) else "YELLOW"
    result = {
        "verdict": verdict,
        "gates": gates,
        "explicit_failures": [name for name, passed in gates.items() if not passed],
        "exploration_defaults": asdict(ExplorationConfig()),
        "definitions": {
            "selected_pseudo_regret": "simulator counterfactual duration of the policy-selected action minus the same-decision fastest feasible route; evaluation-only",
            "recommended_pseudo_regret": "the same quantity for the non-exploratory recommendation",
            "realized_user_regret": "actual user action duration minus the same-decision fastest feasible route",
            "exploration_regret": "selected pseudo-regret only on decisions where the stochastic exploration candidate was selected",
        },
        "distributions": distributions,
        "scenarios": scenarios,
        "canonical": {signal: replay_dict(value) for signal, value in canonical.items()},
        "drift": drift_rows,
        "emerging_route": emerging_rows,
        "user_behavior": user_rows,
        "logging_policy_sensitivity": sensitivity_rows,
        "off_policy_evaluation": canonical_ope,
        "mission4_selector_ablation": {
            signal: {
                "mission4_median_regret_seconds": distributions[signal]["mission4_regret_median_seconds"],
                "mission5_median_regret_seconds": distributions[signal]["mission5_regret_median_seconds"],
                "mission5_improvement_seconds": distributions[signal]["median_improvement_over_mission4_seconds"],
            }
            for signal in ("strong", "weak", "none")
        },
        "mission4_spatial_regression": spatial,
        "localized_delay_gate": "closed: ObservedTripOutcome has no localized-delay evidence, so temporal_delay inherits temporal_kernel rather than using simulator hindsight",
        "temporary_effect": {
            "canonical_ephemeral_action_counts": canonical["strong"].action_observation_counts,
            "maximum_established_candidates": sum(count >= 5 for count in canonical["strong"].action_observation_counts.values()),
            "interpretation": "the temporary detour did not receive enough clean repeated outcomes to become an established ranking action",
        },
    }
    return result, canonical["strong"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("artifacts/mission5-results.json"))
    parser.add_argument("--log-output", type=Path, default=Path("artifacts/mission5-policy-log.jsonl"))
    args = parser.parse_args()
    result, canonical = run_benchmark()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False, default=str) + "\n", encoding="utf-8")
    args.log_output.parent.mkdir(parents=True, exist_ok=True)
    args.log_output.write_text(
        "".join(json.dumps(asdict(log), sort_keys=True, default=str) + "\n" for log in canonical.logs),
        encoding="utf-8",
    )
    print(json.dumps({"verdict": result["verdict"], "gates": result["gates"], "distributions": result["distributions"]}, indent=2))


if __name__ == "__main__":
    main()
