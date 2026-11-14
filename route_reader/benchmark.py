"""Representative adversarial benchmark matrix and machine-readable results."""

from __future__ import annotations

import json
import math
import statistics
import time
from dataclasses import asdict, dataclass
from datetime import timedelta
from pathlib import Path

from .adversarial import LifeConfig, LifeHistory, generate_life_history, observe_life_history
from .robust_evaluation import RobustEvaluation, align_trips_to_truth, evaluate_robust_lab
from .robust_pipeline import learn_robust_lab


@dataclass(frozen=True)
class ScenarioSpec:
    name: str
    observation_regime: str
    route_separability: str = "hard"
    behavior_drift: bool = True
    temporal_signal: str = "strong"
    history_weeks: int | None = None


@dataclass(frozen=True)
class ScenarioResult:
    name: str
    dimensions: dict[str, str | int | bool]
    metrics: dict[str, str | int | float | None]
    cold_start: list[dict[str, str | int | float | None]]
    failures: list[str]
    drift_recency_mae_seconds: float | None
    drift_uniform_mae_seconds: float | None
    runtime_seconds: float


@dataclass(frozen=True)
class BenchmarkResult:
    verdict: str
    scenarios: tuple[ScenarioResult, ...]
    mission1_regression_command: str

    def to_dict(self) -> dict[str, object]:
        return {
            "verdict": self.verdict,
            "scenarios": [asdict(scenario) for scenario in self.scenarios],
            "mission1_regression_command": self.mission1_regression_command,
        }


def representative_matrix() -> tuple[ScenarioSpec, ...]:
    return (
        ScenarioSpec("quality-good", "good"),
        ScenarioSpec("quality-normal", "normal"),
        ScenarioSpec("quality-sparse", "sparse"),
        ScenarioSpec("quality-hostile", "hostile"),
        ScenarioSpec("easy-overlap-normal", "normal", route_separability="easy"),
        ScenarioSpec("static-strong-signal", "normal", behavior_drift=False, temporal_signal="strong"),
        ScenarioSpec("static-weak-signal", "normal", behavior_drift=False, temporal_signal="weak"),
        ScenarioSpec("static-no-signal", "normal", behavior_drift=False, temporal_signal="none"),
        ScenarioSpec("cold-history-6-weeks", "normal", history_weeks=6),
        ScenarioSpec("medium-history-12-weeks", "normal", history_weeks=12),
    )


def _truncate(history: LifeHistory, weeks: int | None) -> LifeHistory:
    if weeks is None:
        return history
    cutoff = min(journey.departure for journey in history.journeys) + timedelta(days=weeks * 7)
    return LifeHistory(
        tuple(journey for journey in history.journeys if journey.departure < cutoff),
        history.places_xy,
        history.config,
    )


def _drift_probe(evaluation: RobustEvaluation, learned: object, history: LifeHistory, observed: object) -> tuple[float | None, float | None]:
    if not history.config.behavior_drift or not evaluation.target_od_id:
        return None, None
    od = learned.od_learning.get(evaluation.target_od_id)
    if od is None:
        return None, None
    aligned = align_trips_to_truth(learned.all_trips, history, observed.recorded_journey_ids)
    recency_errors, uniform_errors = [], []
    for trip in learned.test_trips:
        truth = aligned.get(trip.trip_id)
        if truth is None or truth.od_id != "School->Work" or truth.route_id != "A":
            continue
        variant = od.test_assignments.get(trip.trip_id)
        if variant is None:
            continue
        for model, destination in ((od.model, recency_errors), (od.uniform_history_model, uniform_errors)):
            prediction = next(
                (item for item in model.predict_all(truth.departure).predictions if item.variant_id == variant),
                None,
            )
            if prediction:
                destination.append(abs(prediction.predicted_seconds - truth.duration_seconds))
    return (
        statistics.mean(recency_errors) if recency_errors else None,
        statistics.mean(uniform_errors) if uniform_errors else None,
    )


def run_scenario(spec: ScenarioSpec) -> tuple[ScenarioResult, RobustEvaluation]:
    started = time.perf_counter()
    config = LifeConfig(
        route_separability=spec.route_separability,
        behavior_drift=spec.behavior_drift,
        temporal_signal=spec.temporal_signal,
    )
    history = _truncate(generate_life_history(config), spec.history_weeks)
    observed = observe_life_history(history, spec.observation_regime)
    learned = learn_robust_lab(observed.raw_points)
    evaluation = evaluate_robust_lab(learned, history, observed)
    recency_mae, uniform_mae = _drift_probe(evaluation, learned, history, observed)
    dimensions = {
        "observation_quality": spec.observation_regime,
        "route_separability": spec.route_separability,
        "behavior_drift": spec.behavior_drift,
        "temporal_signal": spec.temporal_signal,
        "history_weeks": spec.history_weeks or config.weeks,
    }
    result = ScenarioResult(
        spec.name,
        dimensions,
        evaluation.metrics.to_dict(),
        [asdict(item) for item in evaluation.cold_start],
        list(evaluation.explicit_failures),
        recency_mae,
        uniform_mae,
        time.perf_counter() - started,
    )
    return result, evaluation


def _metric(scenarios: dict[str, ScenarioResult], scenario: str, metric: str) -> float:
    value = scenarios[scenario].metrics[metric]
    return float(value) if value is not None else float("nan")


def benchmark_verdict(results: tuple[ScenarioResult, ...]) -> str:
    by_name = {result.name: result for result in results}
    normal = by_name["quality-normal"]
    sparse = by_name["quality-sparse"]
    hostile = by_name["quality-hostile"]
    strong = by_name["static-strong-signal"]
    no_signal = by_name["static-no-signal"]
    checks = (
        _metric(by_name, "quality-normal", "segmentation_f1") >= 0.88,
        _metric(by_name, "quality-normal", "route_variant_ari") >= 0.70,
        _metric(by_name, "quality-normal", "route_ranking_accuracy") >= 0.70,
        _metric(by_name, "quality-normal", "recommendation_regret_seconds") < _metric(by_name, "quality-normal", "common_route_regret_seconds"),
        _metric(by_name, "static-strong-signal", "recommendation_regret_seconds") < _metric(by_name, "static-strong-signal", "time_bucket_regret_seconds"),
        _metric(by_name, "static-strong-signal", "delay_regret_improvement_seconds") > 0,
        _metric(by_name, "static-no-signal", "active_delay_zones") == 0,
        _metric(by_name, "quality-sparse", "mean_interval_width_seconds") > _metric(by_name, "quality-good", "mean_interval_width_seconds"),
        _metric(by_name, "quality-hostile", "segmentation_f1") < _metric(by_name, "quality-normal", "segmentation_f1"),
        _metric(by_name, "quality-normal", "exploration_separation_violations") == 0,
    )
    return "GREEN" if all(checks) else "YELLOW"


def run_benchmark(specs: tuple[ScenarioSpec, ...] | None = None) -> BenchmarkResult:
    specs = representative_matrix() if specs is None else specs
    results = []
    for spec in specs:
        result, _ = run_scenario(spec)
        results.append(result)
    result_tuple = tuple(results)
    return BenchmarkResult(
        benchmark_verdict(result_tuple) if specs == representative_matrix() else "UNSCORED",
        result_tuple,
        "python -m route_reader.demo && python -m unittest discover -s tests -v",
    )


def write_benchmark_json(result: BenchmarkResult, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result.to_dict(), indent=2, sort_keys=True, allow_nan=True) + "\n", encoding="utf-8")
