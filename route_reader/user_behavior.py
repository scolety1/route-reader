"""Recommendation-following models with exact combined logging propensities."""

from __future__ import annotations

import math
from dataclasses import dataclass

from .partial_feedback import RankingDecision


@dataclass(frozen=True)
class UserBehaviorConfig:
    name: str
    recommendation_follow_probability: float
    familiarity_bias: float
    planned_route_bias: float


USER_BEHAVIORS = {
    "compliant": UserBehaviorConfig("compliant", 0.86, 3.0, 6.0),
    "mixed": UserBehaviorConfig("mixed", 0.42, 3.0, 30.0),
    "habitual": UserBehaviorConfig("habitual", 0.26, 9.0, 12.0),
}


@dataclass(frozen=True)
class UserChoice:
    actual_action: str
    actual_action_propensity: float
    logging_propensities: dict[str, float]
    conditional_propensities: dict[str, float]
    followed_selected_action: bool


def _conditional_distribution(
    selected_action: str,
    actions: tuple[str, ...],
    config: UserBehaviorConfig,
    habitual_action: str | None,
    planned_action: str | None,
) -> dict[str, float]:
    follow = config.recommendation_follow_probability
    weights = {action: 1.0 for action in actions}
    if habitual_action in weights:
        weights[habitual_action] *= config.familiarity_bias
    if planned_action in weights:
        weights[planned_action] *= config.planned_route_bias
    total = sum(weights.values())
    distribution = {action: (1.0 - follow) * weight / total for action, weight in weights.items()}
    distribution[selected_action] += follow
    return distribution


def choose_user_action(
    decision: RankingDecision,
    behavior: str,
    *,
    random_draw: float,
    habitual_action: str | None = None,
    planned_action: str | None = None,
) -> UserChoice:
    """Sample the actual route and retain its marginal logging probability.

    The marginal integrates over the Route Reader policy's already-frozen
    randomized choice. This is the propensity required to evaluate actual route
    outcomes, rather than a probability reconstructed after completion.
    """

    if behavior not in USER_BEHAVIORS:
        raise ValueError(f"unknown user behavior: {behavior}")
    config = USER_BEHAVIORS[behavior]
    actions = tuple(sorted(set(decision.eligible_actions) | ({planned_action} if planned_action else set())))
    conditional_by_selection = {
        selected: _conditional_distribution(selected, actions, config, habitual_action, planned_action)
        for selected, probability in decision.policy_propensities.items()
        if probability > 0
    }
    marginal = {action: 0.0 for action in actions}
    for selected, selection_probability in decision.policy_propensities.items():
        if selection_probability <= 0:
            continue
        conditional = conditional_by_selection[selected]
        for action, probability in conditional.items():
            marginal[action] += selection_probability * probability
    conditional = conditional_by_selection[decision.policy_selected_action]
    threshold = min(max(random_draw, 0.0), math.nextafter(1.0, 0.0))
    running = 0.0
    actual = actions[-1]
    for action in actions:
        running += conditional[action]
        if threshold < running:
            actual = action
            break
    if not math.isclose(sum(marginal.values()), 1.0, abs_tol=1e-9):
        raise AssertionError("logging propensities must sum to one")
    return UserChoice(
        actual,
        marginal[actual],
        marginal,
        conditional,
        actual == decision.policy_selected_action,
    )
