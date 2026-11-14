# Architecture and experiment notes

## Boundary and flow

The learning entry point accepts only timestamp, latitude, longitude, and reported accuracy. Simulator truth is defined separately and is imported only by evaluation. The demo writes observable JSONL and reads it back before learning, making the boundary executable rather than documentary.

1. Split daily raw sessions on long gaps. Reject short apparent-motion bursts caused by GPS outliers; join sustained movement across intersection dwell.
2. Cluster trip endpoints within 140 m to discover recurring places and count directed place pairs to discover recurring journeys.
3. Remove isolated jumps, median-smooth, and arc-length-resample each journey. Cluster a graph of trajectories whose blended discrete-Fréchet/RMS distance is within 285 m.
4. Detect sustained sub-3 m/s interior segments. Spatially cluster repeated events by learned route to produce recurring delay zones and time-window summaries.
5. For each route, predict stable movement time with a broad Gaussian time-of-day kernel and localized dwell with a narrower kernel when a zone shows a sharp temporal peak. Recency and weekday similarity affect weights. Sparse routes fall back to their historical median with low confidence.
6. Rank expected duration. Evidence includes intervals, effective sample size, nearby trip count, departure-time distance, recency, reliability, and expected localized dwell. Exploration is only surfaced when a less-tested route is close; it never replaces the default recommendation.

## Evaluation discipline

The final two chronological weeks are held out. Place discovery, route prototypes, delay zones, and model parameters use only the first eight weeks. Held-out paths are assigned to fixed training prototypes. Simulator labels are mapped to learned clusters only after fitting, inside evaluation.

Metrics cover segmentation and recurring-place/OD recovery, clustering ARI, held-out route assignment, travel-time MAE, interval coverage, fastest-route accuracy, and counterfactual regret. Ablations compare predictions with and without localized dwell features. Baselines always select the most commonly observed route or the route with the lowest all-time median.

## Mission 5 sequential-decision boundary

The partial-feedback runtime accepts opaque action IDs, departure context, and exactly one `ObservedTripOutcome` after a decision. It has no simulator imports or counterfactual container. Duration estimation, pairwise route ranking, exploration, user action, and evaluation are separate layers. The ranking policy has an established overall-median fallback; repeated but unestablished routes can only receive budgeted exploration.

Every policy probability and combined user/logging probability is validated and frozen before completion. The evaluation-only replay owns `SimulationEvaluationTruth` and may calculate counterfactual regret only after the actual route is immutable. Partial shared-corridor outcomes update chronological OD history but not variant evidence. See `MISSION5_PARTIAL_FEEDBACK.md` for thresholds, estimator assumptions, and measured limitations.

## Privacy architecture

- Raw location history and all learned artifacts stay local; the proof has no network dependency.
- No telemetry, analytics, advertising identifier, account, credential, or cloud store exists.
- Observable traces contain no simulator labels. Future ingestion can replace the JSONL reader without changing the learner boundary.
- A production version should encrypt the local store, minimize raw retention after feature extraction, support deletion/export, keep derived place/route identifiers pseudonymous, and make any off-device processing explicit opt-in. Those controls are not simulated here.

## Important assumptions

The V0 expects one dominant trip per raw session, endpoints with several minutes of stationary evidence, a stable coordinate system, and repeated exposure to each route/time window. The synthetic street corridors are more separable than many real urban alternatives. GPS multipath, tunnels, mode changes, shared route trunks, seasonality, closures, and concept drift remain unproven. Delay zones are inferred repeated dwell patterns, never claims about actual signal programs.
