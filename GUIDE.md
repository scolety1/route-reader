# Route Reader — Mission 1 proof

This technical guide describes the canonical demo and the later research experiments. Generated reports are produced locally; operational handoff documents are excluded from this public snapshot.

Route Reader learns a person's repeated routes from their own noisy timestamped location history. This repository is a simulation-first technical proof, not a generic navigator: it discovers recurring places and journeys, separates route variants, learns time-dependent performance and localized dwell patterns, then ranks known personal routes with uncertainty and evidence.

The canonical fixture contains ten weeks of weekday School → Work trips, four distinct route corridors, noisy and irregular GPS samples, dropped points, outliers, natural speed variation, and repeatable time-dependent delay zones. Hidden route/place/counterfactual truth is isolated in the simulator and is used only by evaluation.

## Run

Python 3.11+; no third-party runtime dependencies.

```powershell
python -m route_reader.demo
python -m unittest discover -s tests -v
```

The demo regenerates and replays `artifacts/canonical_raw_traces.jsonl`, prints evidence-bearing recommendations and held-out metrics, and writes `artifacts/route_reader_report.html` plus `artifacts/canonical_metrics.json`.

## Canonical acceptance thresholds

- 50 trips segmented from the raw stream; chronological 40/10 train/test split.
- two recurring places and the dominant recurring OD pair recovered;
- four route variants, route-clustering ARI ≥ 0.90;
- held-out travel-time MAE ≤ 30 seconds;
- held-out fastest-route accuracy ≥ 0.80;
- mean recommendation regret below both required non-temporal baselines;
- at least one recurring localized delay pattern and positive delay-feature ablation;
- the preferred learned route changes across departure-time windows.

These are fixture regression bounds, not claims about general real-world performance.

## Project map

- `route_reader/synthetic.py` — deterministic world and raw-only JSONL boundary
- `route_reader/ingestion.py` — sessionization and trip segmentation
- `route_reader/discovery.py` — recurring places and OD pairs
- `route_reader/trajectories.py` — resampling, discrete Fréchet distance, clustering
- `route_reader/delays.py` — repeated localized slowdown/dwell discovery
- `route_reader/model.py` — temporal prediction, uncertainty, evidence, exploration hints
- `route_reader/pipeline.py` — truth-free learner orchestration
- `route_reader/evaluation.py` — held-out truth alignment, counterfactual ranking, baselines
- `route_reader/report.py` / `demo.py` — visible proof and artifacts
- `docs/ARCHITECTURE.md` — experiment and privacy boundaries

## Mission 2 robustness laboratory

Mission 1 remains the stable canonical proof. Mission 2 attacks its assumptions with a five-month, seven-place life history; overlapping and evolving routes; difficult segmentation; drift; and good/normal/sparse/hostile GPS replays.

```powershell
python -m route_reader.robust_demo
```

The command runs a representative ten-scenario benchmark and creates:

- `artifacts/mission2/benchmark.json`
- `artifacts/mission2/benchmark_report.html`

See `docs/MISSION2.md` for algorithms, experiment boundaries, and interpretation guardrails.

## Mission 5 partial-feedback research

Mission 5 adds strict chosen-action-only replay, decision-aligned pairwise ranking, controlled exploration, propensity logging, simulated user non-compliance, and guarded off-policy evaluation. It is intentionally **YELLOW**: the architecture is sound, but strong-signal lift and emerging-route convergence are not yet dependable.

```powershell
python -m route_reader.mission5_benchmark
python -m route_reader.mission5_report
```

See `docs/MISSION5_PARTIAL_FEEDBACK.md`; the report command generates `artifacts/mission5-report.html` locally.
