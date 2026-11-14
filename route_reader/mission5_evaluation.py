"""Evaluation-only sequential replay with simulator truth behind a hard boundary.

This is the only Mission 5 module that imports ``adversarial``. Runtime policy
objects receive ``DecisionContext`` and one ``ObservedTripOutcome`` after the
actual action and all propensities have been frozen.
"""

from __future__ import annotations

import random
import statistics
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import timedelta

from .adversarial import (
    LifeConfig,
    TrueJourney,
    _adapt_endpoints,
    _duration,
    _school_work_routes,
    _target_profile,
    generate_life_history,
    observe_life_history,
)
from .off_policy import OPEEstimate, evaluate_logged_policy
from .partial_feedback import DecisionContext, FrozenDecisionLog, ObservedTripOutcome, PartialFeedbackState
from .ranking_policy import ExplorationConfig, RegretRankingPolicy
from .robust_evaluation import align_trips_to_truth
from .robust_ingestion import RobustTrip, segment_trips_robust
from .user_behavior import choose_user_action


@dataclass(frozen=True)
class SimulationEvaluationTruth:
    """Never accepted by a runtime learner, ranker, or exploration interface."""

    decision_id: str
    available_duration_seconds: dict[str, float]
    fastest_action: str
    planned_action: str


@dataclass(frozen=True)
class SequentialDecisionRecord:
    sequence: int
    decision_id: str
    phase: str
    history_outcomes_before: int
    recommended_action: str | None
    policy_selected_action: str | None
    actual_action: str
    fastest_action: str
    selected_was_exploration: bool
    followed_selected_action: bool | None
    selected_pseudo_regret_seconds: float
    recommended_pseudo_regret_seconds: float
    realized_user_regret_seconds: float
    mission4_regret_seconds: float | None
    overall_median_regret_seconds: float | None
    outcome_resolved_to_variant: bool


@dataclass(frozen=True)
class RegretCheckpoint:
    decisions: int
    cumulative_selected_pseudo_regret_seconds: float
    cumulative_recommended_pseudo_regret_seconds: float
    cumulative_realized_user_regret_seconds: float
    cumulative_exploitation_regret_seconds: float
    cumulative_exploration_regret_seconds: float
    exploration_choices: int


@dataclass(frozen=True)
class PolicyValueValidation:
    target_policy: str
    simulator_true_mean_seconds: float | None
    truth_supported_decisions: int
    estimates: tuple[OPEEstimate, ...]


@dataclass(frozen=True)
class SequentialReplayResult:
    config: dict[str, object]
    records: tuple[SequentialDecisionRecord, ...]
    logs: tuple[FrozenDecisionLog, ...]
    evaluation_truth: tuple[SimulationEvaluationTruth, ...]
    checkpoints: tuple[RegretCheckpoint, ...]
    ope: tuple[PolicyValueValidation, ...]
    action_observation_counts: dict[str, int]
    unresolved_variant_outcomes: int
    exploration_opportunities: int
    exploration_choices: int
    temporal_graduation_decision: int | None
    post_graduation_mean_regret_seconds: float | None
    policy_convergence_last_ten: float | None
    propensity_frozen_before_outcome: bool
    leakage_violations: int


def build_partial_feedback_fixture(
    *,
    seed: int,
    temporal_signal: str,
    behavior_drift: bool,
    regime: str = "normal",
    weeks: int = 20,
) -> tuple[tuple[RobustTrip, ...], dict[str, TrueJourney]]:
    history = generate_life_history(
        LifeConfig(weeks=weeks, seed=seed, temporal_signal=temporal_signal, behavior_drift=behavior_drift)
    )
    observed = observe_life_history(history, regime)
    trips = segment_trips_robust(observed.raw_points)
    aligned = align_trips_to_truth(trips, history, observed.recorded_journey_ids)
    target = tuple(
        trip
        for trip in trips
        if trip.trip_id in aligned and aligned[trip.trip_id].od_id == "School->Work"
    )
    return target, {trip.trip_id: aligned[trip.trip_id] for trip in target}


def _past_quality(state: PartialFeedbackState) -> float:
    qualities = [outcome.observation_quality for _, outcome in state.outcomes[-6:]]
    return statistics.median(qualities) if qualities else 0.65


def _duration_for_action(truth: TrueJourney, hidden_action: str) -> float:
    if hidden_action == truth.route_id:
        return truth.duration_seconds
    if hidden_action not in truth.counterfactual_duration_seconds:
        raise ValueError(f"simulator has no feasible outcome for action {hidden_action}")
    return truth.counterfactual_duration_seconds[hidden_action]


def _evaluation_duration_for_known_detour(
    truth: TrueJourney,
    *,
    seed: int,
    temporal_signal: str,
    behavior_drift: bool,
) -> float:
    """Complete an evaluation fixture for a still-known temporary detour.

    The base simulator omits A_detour from ordinary counterfactual dictionaries.
    If partial-feedback history established it, evaluation must still score it
    without using that knowledge to remove the runtime action. Construction is
    active only in its original week 6-9 window; afterward the temporary delay is
    removed while the physical path remains traversable.
    """

    start_date = truth.departure.replace(month=1, day=5, hour=0, minute=0, second=0, microsecond=0)
    week = max(0, (truth.departure.date() - start_date.date()).days // 7)
    config = LifeConfig(seed=seed, temporal_signal=temporal_signal, behavior_drift=behavior_drift)
    path = _adapt_endpoints(_school_work_routes("hard")["A_detour"], truth.path_xy[0], truth.path_xy[-1])
    seed_offset = max(0, int(truth.journey_id[1:]) - 1)
    speed, delays = _target_profile("A_detour", path, truth.departure, week, config, seed_offset)
    if not 6 <= week <= 9:
        delays = tuple(delay for delay in delays if delay.category != "temporary")
    return _duration(path, speed, delays)


def _checkpoints(records: tuple[SequentialDecisionRecord, ...]) -> tuple[RegretCheckpoint, ...]:
    policy_records = tuple(item for item in records if item.phase != "cold_start")
    if not policy_records:
        return ()
    points = sorted(set(min(len(policy_records), point) for point in (10, 20, 40, len(policy_records))))
    result = []
    for point in points:
        prefix = policy_records[:point]
        exploration = [item for item in prefix if item.selected_was_exploration]
        exploitation = [item for item in prefix if not item.selected_was_exploration]
        result.append(
            RegretCheckpoint(
                point,
                sum(item.selected_pseudo_regret_seconds for item in prefix),
                sum(item.recommended_pseudo_regret_seconds for item in prefix),
                sum(item.realized_user_regret_seconds for item in prefix),
                sum(item.selected_pseudo_regret_seconds for item in exploitation),
                sum(item.selected_pseudo_regret_seconds for item in exploration),
                len(exploration),
            )
        )
    return tuple(result)


def _validate_ope(
    logs: tuple[FrozenDecisionLog, ...],
    truths: tuple[SimulationEvaluationTruth, ...],
) -> tuple[PolicyValueValidation, ...]:
    truth_by_id = {item.decision_id: item for item in truths}
    result = []
    for policy in (
        "most_common",
        "overall_median",
        "time_bucket",
        "recency",
        "temporal_kernel",
        "temporal_delay",
        "mission4_adaptive",
        "mission5_ranking",
    ):
        values = []
        for log in logs:
            action = log.decision.policy_actions[policy]
            truth = truth_by_id[log.decision.context.decision_id]
            if action in truth.available_duration_seconds:
                values.append(truth.available_duration_seconds[action])
        estimates = evaluate_logged_policy(logs, policy)
        result.append(
            PolicyValueValidation(
                policy,
                statistics.mean(values) if values else None,
                len(values),
                estimates,
            )
        )
    return tuple(result)


def strict_partial_feedback_replay(
    trips: tuple[RobustTrip, ...],
    truth_by_trip: dict[str, TrueJourney],
    *,
    seed: int,
    temporal_signal: str,
    behavior_drift: bool,
    user_behavior: str = "mixed",
    exploration_config: ExplorationConfig = ExplorationConfig(),
    warmup_decisions: int = 10,
    minimum_established_evidence: int = 5,
) -> SequentialReplayResult:
    """Run one chronology; reveal exactly one actual-action outcome per decision."""

    ordered = tuple(sorted((trip for trip in trips if trip.trip_id in truth_by_trip), key=lambda item: item.departure))
    state = PartialFeedbackState()
    policy = RegretRankingPolicy(state, exploration_config)
    rng = random.Random(seed * 7919 + 503)
    hidden_to_opaque: dict[str, str] = {}
    opaque_to_hidden: dict[str, str] = {}

    def opaque(hidden: str) -> str:
        if hidden not in hidden_to_opaque:
            identifier = f"route-{len(hidden_to_opaque) + 1:03d}"
            hidden_to_opaque[hidden] = identifier
            opaque_to_hidden[identifier] = hidden
        return hidden_to_opaque[hidden]

    records: list[SequentialDecisionRecord] = []
    logs: list[FrozenDecisionLog] = []
    evaluation_truths: list[SimulationEvaluationTruth] = []
    leakage_violations = 0
    exploration_opportunities = exploration_choices = 0
    for sequence, trip in enumerate(ordered, 1):
        truth = truth_by_trip[trip.trip_id]
        planned_action = opaque(truth.route_id)
        available = {
            opaque(route_id): duration
            for route_id, duration in truth.counterfactual_duration_seconds.items()
        }
        available[planned_action] = truth.duration_seconds
        if "A_detour" in hidden_to_opaque and hidden_to_opaque["A_detour"] not in available:
            available[hidden_to_opaque["A_detour"]] = _evaluation_duration_for_known_detour(
                truth,
                seed=seed,
                temporal_signal=temporal_signal,
                behavior_drift=behavior_drift,
            )
        fastest = min(available, key=available.get)
        history_before = len(state.outcomes)
        established = state.known_actions(minimum_established_evidence)
        # Repeated (3+) routes may gather bounded exploration evidence but only
        # established (5+) routes can become ordinary recommendations.
        eligible = state.known_actions(3) if established else ()
        # Feasibility comes only from previously observed route memory. A known
        # action unexpectedly absent from simulator truth is an evaluation-fixture
        # defect, not a reason for hidden truth to edit the runtime candidate set.
        missing_truth = set(eligible) - set(available)
        if missing_truth:
            raise ValueError(f"evaluation fixture lacks known feasible actions: {sorted(missing_truth)}")
        decision = None
        choice = None
        if sequence > warmup_decisions and eligible:
            minimum_count = min(state.action_counts()[action] for action in eligible)
            context = DecisionContext(
                trip.trip_id,
                truth.departure,
                truth.departure.weekday(),
                truth.departure.hour * 60 + truth.departure.minute + truth.departure.second / 60.0,
                _past_quality(state),
                min(0.98, 0.45 + 0.07 * minimum_count),
            )
            decision = policy.decide(context, eligible, random_draw=rng.random())
            habitual = max(eligible, key=lambda action: (state.action_counts()[action], action))
            choice = choose_user_action(
                decision,
                user_behavior,
                random_draw=rng.random(),
                habitual_action=habitual,
                planned_action=planned_action,
            )
            actual_action = choice.actual_action
            frozen = FrozenDecisionLog(
                decision,
                user_behavior,
                choice.logging_propensities,
                actual_action,
                choice.actual_action_propensity,
                truth.departure,
            )
            if decision.exploration is not None:
                exploration_opportunities += 1
            selected_was_exploration = bool(
                decision.exploration is not None
                and decision.policy_selected_action == decision.exploration.action_id
            )
            exploration_choices += int(selected_was_exploration)
        else:
            actual_action = planned_action
            frozen = None
            selected_was_exploration = False

        hidden_actual = opaque_to_hidden[actual_action]
        actual_true_duration = available[actual_action]
        measurement_offset = trip.duration_seconds - truth.duration_seconds
        observed_duration = max(1.0, actual_true_duration + measurement_offset)
        resolved = (
            trip.has_origin_dwell
            and trip.has_destination_dwell
            and trip.observation_quality >= 0.18
        )
        outcome = ObservedTripOutcome(
            trip.trip_id,
            actual_action,
            truth.departure + timedelta(seconds=observed_duration),
            observed_duration,
            trip.observation_quality,
            resolved,
        )
        evaluation_truth = SimulationEvaluationTruth(
            trip.trip_id,
            available,
            fastest,
            planned_action,
        )
        evaluation_truths.append(evaluation_truth)
        if frozen is not None:
            completed = frozen.complete(outcome)
            logs.append(completed)
            leakage_violations += int(history_before != len(state.outcomes))
        # The single runtime update happens only after the decision and logging
        # probabilities are immutable. Evaluation truth is never passed here.
        state.observe(
            DecisionContext(
                trip.trip_id,
                truth.departure,
                truth.departure.weekday(),
                truth.departure.hour * 60 + truth.departure.minute + truth.departure.second / 60.0,
                trip.observation_quality,
                0.90 if resolved else 0.30,
            ),
            outcome,
        )

        selected = decision.policy_selected_action if decision else None
        recommended = decision.recommended_action if decision else None
        selected_duration = available[selected] if selected else available[actual_action]
        recommended_duration = available[recommended] if recommended else available[actual_action]
        actual_duration = available[actual_action]
        mission4_regret = None
        overall_regret = None
        if decision:
            mission4_action = decision.policy_actions["mission4_adaptive"]
            overall_action = decision.policy_actions["overall_median"]
            mission4_regret = available[mission4_action] - available[fastest]
            overall_regret = available[overall_action] - available[fastest]
        records.append(
            SequentialDecisionRecord(
                sequence,
                trip.trip_id,
                "cold_start" if decision is None else "exploration" if selected_was_exploration else "exploitation",
                history_before,
                recommended,
                selected,
                actual_action,
                fastest,
                selected_was_exploration,
                choice.followed_selected_action if choice else None,
                selected_duration - available[fastest],
                recommended_duration - available[fastest],
                actual_duration - available[fastest],
                mission4_regret,
                overall_regret,
                resolved,
            )
        )

    records_tuple = tuple(records)
    logs_tuple = tuple(logs)
    truth_tuple = tuple(item for item in evaluation_truths if item.decision_id in {log.decision.context.decision_id for log in logs})
    policy_records = tuple(item for item in records_tuple if item.phase != "cold_start")
    graduation = next(
        (
            index
            for index, (record, log) in enumerate(zip(policy_records, logs_tuple), 1)
            if log.decision.recommended_action != log.decision.fallback_action
            and record.recommended_pseudo_regret_seconds + 1.0 < (record.overall_median_regret_seconds or 0.0)
        ),
        None,
    )
    post = policy_records[graduation:] if graduation is not None else ()
    last_ten = policy_records[-10:]
    propensity_order_ok = all(
        log.observed_outcome is not None and log.frozen_at < log.observed_outcome.completed_at
        for log in logs_tuple
    )
    return SequentialReplayResult(
        {
            "seed": seed,
            "temporal_signal": temporal_signal,
            "behavior_drift": behavior_drift,
            "user_behavior": user_behavior,
            "warmup_decisions": warmup_decisions,
            "minimum_established_evidence": minimum_established_evidence,
            "exploration": asdict(exploration_config),
        },
        records_tuple,
        logs_tuple,
        truth_tuple,
        _checkpoints(records_tuple),
        _validate_ope(logs_tuple, truth_tuple) if logs_tuple else (),
        dict(state.action_counts()),
        sum(not record.outcome_resolved_to_variant for record in records_tuple),
        exploration_opportunities,
        exploration_choices,
        graduation,
        statistics.mean(item.recommended_pseudo_regret_seconds for item in post) if post else None,
        statistics.mean(item.recommended_pseudo_regret_seconds <= 1.0 for item in last_ten) if last_ten else None,
        propensity_order_ok,
        leakage_violations,
    )


def replay_dict(result: SequentialReplayResult, *, include_logs: bool = False) -> dict[str, object]:
    payload = asdict(result)
    if not include_logs:
        payload.pop("logs", None)
        payload.pop("evaluation_truth", None)
        payload.pop("records", None)
    return payload
