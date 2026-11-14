# Mission 3: adaptive personal-route learning

Verdict: **YELLOW**. Mission 3 fixes the unacceptable weak-signal regression and the recurring-place over-creation problem, and introduces chronological selection, empirical calibration, abstention, stable IDs/lineage, and one-trip replay. It does not yet make route identity stable enough for real-device traces.

## Runtime design

- `place_regions.py` seeds place regions only from quality-qualified dwell endpoints. Partial/truncated endpoints may support an existing region but cannot promote themselves. Status, footprint, arrivals, departures, day support, confidence, and reasons are exposed.
- `stable_routes.py` provides durable `SR###` IDs and explicit create/update/merge/split/retire revisions across snapshots. The hierarchy accepts an observable-quality threshold override; simulator labels remain evaluation-only.
- `adaptive_model.py` compares overall median, broad time bucket, recency, temporal kernel, and temporal-plus-delay. Every score is rolling-origin and uses only prior observations. Minimum evidence and a seconds-denominated complexity penalty favor simplicity. Delay features additionally require positive walk-forward contribution.
- Prediction confidence separates empirical interval, support, calibration cutoff, model-selection confidence, route-identity confidence, observation quality, and departure-time familiarity. Unsupported or indistinguishable choices abstain. Exploration remains a separate non-fastest object.
- `incremental_replay.py` updates knowledge one trip at a time; `mission3_benchmark.py` evaluates canonical fixtures, five deterministic seeds, 5/10/20/40/full snapshots, and threshold sensitivity.

## Canonical results

Place regions on normal observations: **1.00 precision / .86 recall**, versus Mission 2 **.55 / .86**. Six repeated truncated clusters were suppressed. The inferred campus footprint is represented as a region; nearby recurring places remain separately seeded by their dwell/transition evidence.

The production Mission 3 route-quality setting reduces canonical good-GPS fragmentation from **14 to 9 variants** while raising ARI from **.696 to .799** and preserving 100% held-out assignment. Normal remains useful at **8 variants, .892 ARI, 80% assignment** in the preserved primary pipeline. Sensitivity is material: on the independently scoped target history, 160/180/200/220/250m produces 22/16/14/13/9 variants and ARI .592/.665/.593/.587/.075. Route thresholds remain a blocker.

Canonical static weak signal selects the overall median and matches its **5.4s regret**, eliminating Mission 2's **20.9s** regression. Canonical no-signal remains **0.0s**. In canonical strong signal, one route graduates to the temporal kernel, but adaptive joint decision regret remains **21.7s**, versus Mission 2 rich ranker's **12.3s**. This failure prevents GREEN.

Empirical intervals are calibrated only from residuals whose timestamps precede deployment. The five-seed scenario artifact records held-out coverage and widths. User-facing decisions commonly abstain because route intervals overlap; this is honest but currently conservative.

## Multi-seed results

Five seeds per temporal setting:

| Signal | Adaptive regret median | IQR | Worst | Material failures vs best simple |
|---|---:|---:|---:|---:|
| Strong | 6.4s | 4.1–21.7s | 42.8s | 1/5 |
| Weak | 8.3s | 5.4–8.8s | 18.1s | 0/5 |
| None | 0.0s | 0.0–0.0s | 33.5s | 0/5 |

The no-signal worst case is not a selector regression: the comparable simple baselines fail on the same seed. Still, it demonstrates fixture-dependent route-ranking risk.

## Incremental behavior and remaining failures

At 5 trips the replay has three route hypotheses and abstains on route identity. At 10 and 20 trips it retains IDs but abstains because alternatives overlap. Models are initially overall medians; recency graduates only after more evidence. Merge/split lineage is explicit. However, the target replay still reaches 18 structural variants by the full 60 trips. Stable names make churn interpretable; they do not make the underlying clustering sufficiently stable.

Explicit remaining failures:

1. Good-GPS fragmentation is reduced in the primary quality setting but not eliminated in the stricter incremental lineage view.
2. Strong-signal selection graduates locally but does not consistently preserve the richer joint-ranking upside.
3. Sparse and hostile route ARI remain near zero; recommendation must abstain.
4. Calibration is conservative and frequently returns “no reliable advantage.”
5. Place dwell is inferred from recording context; real device dwell semantics remain unvalidated.
6. Delay gating is implemented, but no canonical adaptive route earned `temporal_delay`; the feature remains promising rather than proven.

## Reproduction

```powershell
python -m unittest discover -s tests -v
python -m route_reader.demo
python -m route_reader.robust_demo
python -m route_reader.mission3_benchmark --output artifacts\mission3-results.json
python -m route_reader.mission3_report --input artifacts\mission3-results.json --output artifacts\mission3-report.html
```

Machine output is `artifacts/mission3-results.json`; the visual report is `artifacts/mission3-report.html`.
