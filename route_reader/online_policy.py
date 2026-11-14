"""Journey-level safety gate for online corridor prediction histories."""

from __future__ import annotations

from dataclasses import dataclass

from .adaptive_model import AdaptiveJourneyModel, AdaptiveRecommendation, AdaptiveRouteModel
from .online_corridors import OnlineCorridorMemory


@dataclass(frozen=True)
class JourneyGraduationEvidence:
    overall_walk_forward_mae_seconds: float | None
    time_bucket_walk_forward_mae_seconds: float | None
    temporal_lift_seconds: float | None
    richer_models_permitted: bool
    reason: str


def _aggregate(models: tuple[AdaptiveRouteModel, ...], model_name: str) -> tuple[float | None, int]:
    weighted = total = 0.0
    for model in models:
        evidence = next(item for item in model.evidence if item.model == model_name)
        if evidence.walk_forward_mae_seconds is not None and evidence.validation_predictions >= 4:
            weighted += evidence.walk_forward_mae_seconds * evidence.validation_predictions
            total += evidence.validation_predictions
    return (weighted / total, int(total)) if total else (None, 0)


class OnlineAdaptiveJourneyModel:
    """Apply OD-level evidence safety before route-level adaptive prediction.

    Mission 3 selected complexity independently per fragmented route.  With online
    memory we can also ask whether broad time structure improves chronological MAE
    across the recurring journey.  If it has not, all route models back off to the
    overall median.  This gate uses observations only, never counterfactual regret.
    """

    def __init__(self, route_models: tuple[AdaptiveRouteModel, ...]):
        self.route_models = route_models
        overall, overall_n = _aggregate(route_models, "overall_median")
        bucket, bucket_n = _aggregate(route_models, "time_bucket")
        lift = None if overall is None or bucket is None else overall - bucket
        permitted = lift is not None and min(overall_n, bucket_n) >= 12 and lift >= 2.0
        if not permitted:
            for model in route_models:
                if model.selected_model in {"time_bucket", "recency", "temporal_kernel", "temporal_delay"}:
                    model.selected_model = "overall_median"
        reason = (
            f"broad time buckets improved {min(overall_n, bucket_n)} chronological predictions by {lift:.1f}s"
            if permitted and lift is not None
            else "journey-level temporal structure has not earned deployment; overall medians retained"
        )
        self.graduation = JourneyGraduationEvidence(overall, bucket, lift, permitted, reason)
        self._delegate = AdaptiveJourneyModel(route_models)

    def predict_all(self, departure) -> AdaptiveRecommendation:
        return self._delegate.predict_all(departure)


def build_online_route_models(memory: OnlineCorridorMemory) -> tuple[AdaptiveRouteModel, ...]:
    models = []
    for variant in memory.established_variants:
        history = memory.prediction_history(variant.variant_id)
        identity = min(1.0, len(variant.trip_ids) / 8.0)
        models.append(AdaptiveRouteModel(variant.variant_id, history, route_identity_confidence=identity))
    return tuple(models)
