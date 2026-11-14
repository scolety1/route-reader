# Mission 4: online corridor memory

Verdict: **YELLOW**. Online route memory solves most structural churn, removes the good-versus-normal fragmentation paradox, preserves real alternatives, contains temporary detours, and passes a strict leakage audit. Stabilized evidence alone does not solve strong-signal policy selection, and one of five strict weak-signal seeds still favors a different simple baseline in hindsight.

## Root-cause instrumentation

Mission 3 replay recomputed and relabeled the entire history after every arrival. On canonical normal history this produced 23 structural-creation events, 17 historical identity reassignments, and 18 final structural variants. Causes were:

- partial or truncated traces: 12/23;
- lateral geometry or route evolution: 4/23;
- prototype drift or threshold reinterpretation: 3/23;
- large divergence or a genuine new route: 3/23;
- initial hypothesis: 1/23.

The central hypothesis was partly correct: batch reinterpretation fragmented route evidence. It was not the sole cause of poor strong-signal ranking.

## Architecture

`online_corridors.py` is the main runtime path. Each completed trip is physically resampled into direction-aware corridor samples in a common metric frame. The learner compares the new sequence to durable prototypes, records reusable 120-metre corridor cells, and updates only the selected identity. Historical assignments never change.

Novelty has four states: unresolved trip, candidate variant, repeated variant, and established variant. Establishment requires at least five trips across four days. Candidates expire after 28 unsupported days. Prototype movement is capped per observation, preventing one detour from dragging an established route centerline.

Partial traces may match shared corridor structure without receiving a complete variant identity. Late-diverging alternatives share corridor cells but maintain separate durable variants. Prediction evidence is inherited from a parent only under high whole-corridor overlap; unrelated durations are not pooled.

`online_policy.py` adds an OD-level chronological graduation gate. Richer route models are permitted only when broad temporal structure has improved prior prediction error across the journey. The gate never reads counterfactual outcomes.

## Route-memory ablation

On canonical normal history:

| Metric | Mission 3 batch replay | Mission 4 online memory |
|---|---:|---:|
| Final structural/durable variants | 18 | 4 |
| Provisional variants | embedded in structure | 2 |
| Historical reassignments | 17 | 0 |
| Evidence fragmentation | high/recomputed | .13 evaluation fragmentation |

Good and normal GPS each learn four durable variants; good purity is 98.2% and normal purity 98.1%. The temporary `A_detour` remains provisional. The emerging `N` route graduates after 14 good-GPS or 16 normal-GPS target observations from its first appearance. Sparse learns three durable coarse identities and hostile only one; unresolved/provisional evidence is retained instead of forcing fine assignment.

## Strict policy evaluation

At every decision, the evaluator freezes structure, candidate routes, model selection, calibration cutoff, recommendation, and abstention before revealing the outcome. Simple policies and adaptive Route Reader use identical online-memory variants and history. Counterfactual truth is evaluation-only. Across all 15 five-seed signal runs, leakage violations are zero.

Five-seed adaptive results:

| Signal | Median regret | IQR | Worst | Material failures vs best comparable simple policy |
|---|---:|---:|---:|---:|
| Strong | 18.7s | 13.7–19.2s | 26.2s | 1/5 |
| Weak | 12.1s | 8.2–12.5s | 16.2s | 1/5 |
| None | 0.0s | 0.0–1.2s | 3.4s | 0/5 |

Canonical strong regret improves modestly from Mission 3's 21.7s to 19.2s, while the same-memory time bucket reaches 15.5s. Canonical weak adaptive and overall-median regret both equal 8.1s; no-signal adaptive and median are both 0.0s. Stabilization provides adequate observations, but prediction-MAE-based graduation still does not reliably choose the route-ranking policy with lowest future regret. Observational personal history does not directly reveal counterfactual route outcomes, which is now the dominant blocker.

Median five-seed interval widths fall from Mission 3's roughly 218–281s to 164–193s, with median coverage around 94%. Route uncertainty, provisional state, and partial matching reduce identity confidence and can trigger abstention. Adaptive abstention remains high (65–84% median), so confidence is more useful but still conservative.

Localized delay zones remain compatible: shared corridor nodes expose a future attribution level, while use of any delay predictor still requires Mission 3's positive chronological contribution gate. Mission 4 does not claim a canonical delay-enabled route.

## Remaining failures

1. Strong-signal policy selection still misses much of the time-bucket advantage.
2. One strict weak-signal seed is 7.1s worse than the hindsight-best comparable time bucket; chronological evidence did not justify that bucket prospectively.
3. Abstention is frequent even with mature normal history.
4. Sparse and hostile data support coarse corridor memory, not reliable fine variants.
5. Emerging route graduation can take 16 target observations because recurrence is deliberately conservative.
6. Corridor cells are geometric observations, not mapped road segments.

## Reproduction

```powershell
python -m unittest discover -s tests -v
python -m route_reader.demo
python -m route_reader.robust_demo
python -m route_reader.mission3_benchmark --output artifacts\mission3-results.json
python -m route_reader.mission4_benchmark --output artifacts\mission4-results.json
python -m route_reader.mission4_report --input artifacts\mission4-results.json --output artifacts\mission4-report.html
```
