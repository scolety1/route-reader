# Mission 2 adversarial robustness laboratory

Mission 2 adds a deterministic five-month life history with seven places, more than ten recurring OD patterns, overlapping route variants, a new route introduced halfway through history, construction mutation, schedule shift, persistent route degradation, and structured/random/temporary/behavioral dwell categories.

The same underlying 413 journeys replay through four independent observation regimes:

- **good** — 4–7 second sampling, low noise and almost no loss;
- **normal** — irregular phone-style sampling, moderate noise and loss;
- **sparse** — 22–48 second intervals, substantial drops and missing spans;
- **hostile** — severe noise, drift, outliers, partial traces, gaps, and missing sessions.

Run the complete ten-scenario matrix:

```powershell
python -m route_reader.robust_demo
```

Outputs are written to `artifacts/mission2/benchmark.json` and `artifacts/mission2/benchmark_report.html`.

## Algorithm changes

- Multi-trip session segmentation joins aligned intermediate stops but splits short back-to-back journeys with a direction change.
- Each inferred trip receives an observation-quality score used by later stages.
- Place discovery rejects the weakest partial endpoints and reports over-creation/merging explicitly.
- Complete-link trajectory clustering prevents corridor chaining; a conservative prototype pass repairs partial-trace fragmentation. Results are represented as OD → route family → variant.
- Two high-quality observations may enter prediction as an explicitly insufficient-confidence emerging route; singletons do not.
- A 28-day recency half-life adapts personal history after drift while a uniform-history ablation measures whether it helped.
- Confidence width responds to effective sample size, observation quality, temporal unfamiliarity, reliability, and recent/history disagreement.
- Delay zones require recurrence or temporal lift, minimum affected trips, and recent support. Expired temporary zones are rejected.
- Fastest expected route and conservative exploration value are separate data structures and outputs.

## Reading the benchmark honestly

The matrix is intentionally not perfect. Normal conditions retain useful route ranking while place discovery still creates false places from partial traces. Sparse GPS damages variant identity and widens intervals. Hostile GPS causes major segmentation/route failure and may leave no calibratable held-out prediction; the report calls this an abstention rather than inventing confidence. Cold-start ranking is non-monotonic as new variants appear, even while uncertainty generally contracts.

Delay-zone labels are evaluation categories only. Runtime output always says recurring localized dwell/delay pattern and never claims to recover a signal program.

Mission 1 remains an independent regression path:

```powershell
python -m route_reader.demo
python -m unittest discover -s tests -v
```
