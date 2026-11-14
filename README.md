# Route Reader

A Python research prototype that learns recurring route choices from noisy timestamped location streams. It separates simulator truth from the learner and evaluates predictions on chronological holdouts.

## What works today

- Sessionization, recurring-place discovery, trajectory resampling and discrete Fréchet-based clustering.
- Time-dependent travel estimates, localized delay patterns, uncertainty and evidence-bearing rankings.
- Robustness, adaptive corridor-memory and partial-feedback experiments.
- A deterministic synthetic demo that generates a standalone HTML report.

## Contribution and attribution

A project by [Spencer Colety](https://github.com/scolety1), developed with AI assistance. The local source history records `scolety1` as the project author. This is presented as AI-assisted project work, not a claim that every line was independently hand-written. [Provenance](PROVENANCE.md) identifies the source baseline and publication changes.

The engineering decisions to examine are the explicit data/evidence boundaries, deterministic domain logic, and testable treatment of uncertainty described below.

## Run it

Python 3.11+; no third-party runtime dependencies (verified with Python 3.14.6). Run from the repository root:

```sh
python -m unittest discover -s tests -v
python -m route_reader.demo
```

Open `artifacts/route_reader_report.html`. All generated locations and trips are synthetic. The demo creates its own input traces; no personal GPS files are needed. [Additional experiments](GUIDE.md) are separate from the canonical demo.

## Start reading here

1. [Pipeline](route_reader/pipeline.py): the raw-input/learner boundary.
2. [Trajectories](route_reader/trajectories.py): resampling and route distance.
3. [Temporal model](route_reader/model.py) and [evaluation](route_reader/evaluation.py).
4. [Partial-feedback policy](route_reader/online_policy.py) and [architecture](docs/ARCHITECTURE.md).

## Verification and limits

Verified November 14, 2026: 70 tests passed. The canonical synthetic demo passed its acceptance gates: clustering ARI 1.0, held-out travel-time MAE 12.17 seconds, fastest-route ranking accuracy 90%. These are fixed-fixture regression results, not real-world accuracy.

Simulation-first, not navigation software. Real GPS, device behavior and route generalization are unverified. Mission 5 partial-feedback work remains exploratory: dependable strong-signal lift and emerging-route convergence are not established.
