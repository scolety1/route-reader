"""Small, guarded off-policy estimator set for frozen partial-feedback logs."""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass

from .partial_feedback import FrozenDecisionLog


@dataclass(frozen=True)
class OPEEstimate:
    estimator: str
    target_policy: str
    estimated_mean_seconds: float | None
    standard_error_seconds: float | None
    support_coverage: float
    effective_sample_size: float
    maximum_importance_weight: float
    unsupported_decisions: int
    reliable: bool
    reason: str


def _standard_error(values: list[float]) -> float | None:
    if len(values) < 2:
        return None
    return statistics.stdev(values) / math.sqrt(len(values))


def _estimate_map(log: FrozenDecisionLog) -> dict[str, float]:
    return {item.action_id: item.expected_seconds for item in log.decision.estimates}


def evaluate_logged_policy(
    logs: tuple[FrozenDecisionLog, ...],
    target_policy: str,
    *,
    clip_weight: float = 10.0,
    minimum_supported_propensity: float = 0.02,
) -> tuple[OPEEstimate, ...]:
    """Estimate a deterministic target policy's mean duration from chosen actions."""

    if clip_weight <= 0:
        raise ValueError("importance-weight clip must be positive")
    complete = tuple(log for log in logs if log.observed_outcome is not None)
    if not complete:
        raise ValueError("off-policy evaluation requires completed logs")
    for log in complete:
        if target_policy not in log.decision.policy_actions:
            raise ValueError(f"target policy {target_policy} was not frozen in every decision")

    weights: list[float] = []
    matched_outcomes: list[float] = []
    ips_terms: list[float] = []
    dr_terms: list[float] = []
    supported = 0
    unsupported = 0
    maximum_raw_weight = 0.0
    for log in complete:
        target = log.decision.policy_actions[target_policy]
        propensity = log.behavior_propensities.get(target, 0.0)
        q = _estimate_map(log)
        if propensity >= minimum_supported_propensity and target in q:
            supported += 1
        else:
            unsupported += 1
        actual = log.actual_action
        outcome = log.observed_outcome.duration_seconds
        matched = actual == target and propensity > 0
        raw_weight = 1.0 / propensity if matched else 0.0
        maximum_raw_weight = max(maximum_raw_weight, raw_weight)
        weight = min(clip_weight, raw_weight)
        weights.append(weight)
        if matched:
            matched_outcomes.append(outcome)
        ips_terms.append(weight * outcome)
        q_target = q.get(target)
        q_actual = q.get(actual)
        if q_target is not None:
            correction = weight * (outcome - q_actual) if matched and q_actual is not None else 0.0
            dr_terms.append(q_target + correction)

    n = len(complete)
    support_coverage = supported / n
    weight_sum = sum(weights)
    ess = weight_sum * weight_sum / sum(weight * weight for weight in weights) if any(weights) else 0.0
    reliability = support_coverage >= 0.70 and ess >= 6.0 and unsupported <= n * 0.30
    reason = (
        "adequate logged-action support"
        if reliability
        else "historical logging support is inadequate; do not treat the value estimate as precise"
    )

    ips_value = statistics.mean(ips_terms)
    snips_value = sum(weight * outcome for weight, outcome in zip((w for w in weights if w > 0), matched_outcomes)) / weight_sum if weight_sum else None
    # Keep aligned contributions for uncertainty rather than pretending the
    # matched sample itself is independent of the importance weighting.
    snips_terms = [weight * (log.observed_outcome.duration_seconds if log.actual_action == log.decision.policy_actions[target_policy] else 0.0) for weight, log in zip(weights, complete)]
    if snips_value is not None and weight_sum:
        snips_terms = [n * value / weight_sum for value in snips_terms]
    return (
        OPEEstimate("ips_clipped", target_policy, ips_value, _standard_error(ips_terms), support_coverage, ess, min(clip_weight, maximum_raw_weight), unsupported, reliability, reason),
        OPEEstimate("snips_clipped", target_policy, snips_value, _standard_error(snips_terms) if snips_value is not None else None, support_coverage, ess, min(clip_weight, maximum_raw_weight), unsupported, reliability, reason),
        OPEEstimate("doubly_robust_clipped", target_policy, statistics.mean(dr_terms) if dr_terms else None, _standard_error(dr_terms), support_coverage, ess, min(clip_weight, maximum_raw_weight), unsupported, reliability and len(dr_terms) == n, reason),
    )
