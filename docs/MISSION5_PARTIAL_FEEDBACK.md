# Mission 5: partial-feedback route ranking

Verdict: **YELLOW**. Mission 5 establishes the correct sequential-learning and logging architecture, preserves Mission 4 spatial stability, behaves safely with weak/no temporal signal, and validates useful SNIPS/doubly-robust estimates against simulator truth. It does not yet produce dependable strong-signal lift or establish emerging routes quickly enough under imperfect user compliance.

## Executable information boundary

Runtime code is split from evaluation truth:

- `partial_feedback.py` defines opaque `DecisionContext`, `ObservedTripOutcome`, action estimates, decisions, and frozen logs. An outcome contains one action and one duration; no field can contain outcomes for unchosen actions.
- `ranking_policy.py`, `user_behavior.py`, and `off_policy.py` do not import the simulator, `TrueJourney`, or `SimulationEvaluationTruth`.
- `mission5_evaluation.py` is explicitly evaluation-only. It owns `SimulationEvaluationTruth`, freezes the policy and combined user/logging propensity, completes the selected action, and only then uses counterfactual truth for experimental regret.
- `PartialFeedbackState.observe` applies one chronological update. Unresolved `shared_corridor_only` outcomes remain OD observations but are excluded from variant counts, predictions, calibration, ranking, confidence, and exploration.

The sequential order is context → known/repeated route set → duration estimates → ranking → bounded exploration draw → user action draw → frozen propensity log → one observed outcome → runtime update → evaluation-only regret. Tests inspect both dataclass fields and runtime imports so leakage is an architectural failure, not a reporting convention.

## Decision and safety policy

The trusted fallback is the established route with the lowest overall observed median. Mission 5 directly estimates context-matched pairwise win probability by weekday/departure window with Beta(2,2) shrinkage. It may depart from the fallback only after at least four matched comparisons, a win probability of at least 0.56, and no more than 20 seconds of disagreement with the contextual duration estimate. This is a route-order objective, not an MAE selector.

Outcome prediction still supplies expected duration and uncertainty. The research candidates retained in every decision are most-common, overall median, departure bucket, recency, temporal kernel, gated temporal-plus-delay, the partial-log Mission 4 MAE selector, and the Mission 5 pairwise selector.

The localized-delay gate remains closed in Mission 5: `ObservedTripOutcome` carries no supported localized event feature. `temporal_delay` therefore inherits `temporal_kernel`; simulator hindsight never reactivates the richer feature.

## Exploration and route lifecycle

Default limits are inspectable:

- maximum predicted penalty: 45 seconds;
- maximum probability: 0.15;
- minimum material uncertainty: 38 seconds;
- four-decision cooldown after selected exploration;
- route-identity confidence at least 0.55;
- recent observation quality at least 0.30;
- no exploration of a materially worse route.

Five clean observations are still required for an ordinary recommendation, matching Mission 4 establishment. A repeated route with three or four observations may only be a bounded exploration candidate. The evidence object keeps `fastest_expected_action`, recommendation, exploration candidate, expected exploration cost, maturity, uncertainty, advantage, confidence, drift, and abstention separate.

## Propensities and user choice

`RankingDecision.policy_propensities` is created with the decision and validated to sum to one. `choose_user_action` then models compliant, mixed, or habitual behavior. It analytically marginalizes over Route Reader's randomized action to produce the combined actual-route logging distribution. `FrozenDecisionLog` verifies that the actual-action propensity exactly matches this distribution before accepting a later outcome.

Recommendation, policy-selected exploration, and actual action are distinct. On canonical strong history, the mixed user ignored the selected action 14/46 times and the habitual user 15/46 times. Algorithm recommendation regret is therefore reported separately from actual-user realized regret.

## Sequential results

Ten seeds per stationary signal, 46 policy decisions after ten cold-start trips:

| Signal | Mission 5 median regret | IQR | Worst | Mission 4 median | Paired median lift | Material losses |
|---|---:|---:|---:|---:|---:|---:|
| Strong | 17.7s | 15.7–19.4s | 23.2s | 19.1s | 0.2s | 1/10 |
| Weak | 11.3s | 8.8–11.6s | 15.0s | 12.0s | 1.8s | 0/10 |
| None | 2.1s | 1.7–3.9s | 7.7s | 2.1s | 0.0s | 0/10 |

Six of ten strong seeds make an evidence-backed temporal departure from the fallback, at a median decision 25.5. This is directionally useful but not the material family-wide improvement required for GREEN. Canonical strong cumulative recommendation regret is 260.6/504.7/764.5/894.3 seconds at 10/20/40/46 decisions. Selected-action regret, including exploration, is 183.7/475.9/771.1/900.9 seconds. Three selected explorations incurred 83.5 seconds of counterfactual regret and 49.3 seconds of predicted penalty.

Weak signal has no material simple-baseline failures. No-signal exploration probability declines across the ten-seed family; canonical early-to-late mean probability falls from 0.030 to 0.010. Median cumulative no-signal exploration regret is 64.2 seconds, or about 1.4 seconds per policy decision. That cost is visible and remains a reason not to treat the policy as production-ready.

Drift is flagged in four of five drift seeds, but mean recommendation regret is 26.3 seconds versus 26.1 for the partial-log Mission 4 selector. All five emerging routes reach repeated status; only one reaches five clean observations and becomes a ranking contender. This failure is the clearest evidence that sparse controlled exploration plus non-compliance cannot yet close the lifecycle quickly enough.

Temporary `A_detour` observations remain provisional in the canonical stationary history. If another seed establishes the path while construction is active, the evaluation fixture continues to score the known physical detour after the temporary delay expires rather than using hidden truth to remove it from the runtime candidate set.

## Offline policy evaluation

Mission 5 uses three closely related estimators because each exposes a different failure mode:

- clipped IPS (maximum weight 10) is transparent but can be biased by clipping and finite normalization;
- clipped SNIPS normalizes the observed importance mass;
- clipped doubly robust combines frozen duration estimates with propensity-weighted residual correction.

For the canonical Mission 5 target policy, simulator truth is 392.0 seconds. SNIPS estimates 395.3 seconds (3.3 seconds absolute error); doubly robust estimates 395.0 (3.1 seconds error), with support coverage 100% and effective sample size 25.2. Clipped IPS estimates 356.5 (35.5 seconds error), so it is not the preferred estimator despite adequate nominal support.

The poor-support stress sets target propensity below the 0.02 support floor. Every estimator reports zero support coverage, 46 unsupported decisions, a clipped maximum weight of 10, and `reliable=false`. Values remain in the audit artifact to demonstrate instability, but must not be used as policy claims.

## Spatial and historical status

Mission 4 remains constant: four durable variants, zero historical identity reassignments, 13.3 structural events per 100 trips, and 0.864 identity continuity in the Mission 5 spatial regression. Mission 5 does not change `online_corridors.py` or the historical Missions 1–4 algorithms.

## Remaining failures

1. Strong-signal paired median lift is only 0.2 seconds and one of ten seeds materially loses to a comparable simple selector.
2. Emerging-route establishment succeeds in only one of five drift seeds.
3. Drift is detected, but the resulting policy does not improve average drift regret.
4. Canonical no-signal exploration is small per decision but still costs 31.6 realized counterfactual seconds; the ten-seed median is 64.2 seconds.
5. Prediction/calibration use only action-level duration and observation quality; shared-corridor and localized-delay attribution are preserved as compatibility gates, not fully integrated partial-feedback models.
6. Logged-policy validation is simulator-local. Real personal data will not provide counterfactual truth, making support diagnostics mandatory.

## Reproduction

```powershell
python -m unittest discover -s tests -v
python -m route_reader.demo
python -m route_reader.robust_demo
python -m route_reader.mission3_benchmark --output artifacts\mission3-results.json
python -m route_reader.mission4_benchmark --output artifacts\mission4-results.json
python -m route_reader.mission5_benchmark --output artifacts\mission5-results.json --log-output artifacts\mission5-policy-log.jsonl
python -m route_reader.mission5_report --input artifacts\mission5-results.json --output artifacts\mission5-report.html
```

Machine-readable results are `artifacts/mission5-results.json`; the counterfactual-free canonical policy log is `artifacts/mission5-policy-log.jsonl`; the visual summary is `artifacts/mission5-report.html`.
