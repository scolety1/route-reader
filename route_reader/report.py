"""Compact dependency-free HTML report for the canonical experiment."""

from __future__ import annotations

import html
import statistics
from datetime import datetime, timedelta
from pathlib import Path

from .evaluation import EvaluationResult
from .pipeline import LearnedRouteReader


def format_duration(seconds: float) -> str:
    minutes, remainder = divmod(max(0, round(seconds)), 60)
    return f"{minutes}m {remainder:02d}s"


def format_clock(minute: float) -> str:
    hour, minute_value = divmod(int(round(minute)), 60)
    suffix = "AM" if hour < 12 else "PM"
    display_hour = hour % 12 or 12
    return f"{display_hour}:{minute_value:02d} {suffix}"


def _prediction_chart(learned: LearnedRouteReader, reference: datetime) -> str:
    minutes = list(range(15 * 60 + 5, 16 * 60 + 11, 5))
    series: dict[str, list[float]] = {}
    for minute in minutes:
        departure = reference.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(minutes=minute)
        for prediction in learned.model.predict_all(departure).predictions:
            series.setdefault(prediction.route_cluster_id, []).append(prediction.predicted_seconds / 60.0)
    all_values = [value for values in series.values() for value in values]
    low, high = min(all_values) - 0.25, max(all_values) + 0.25
    width, height = 760, 300
    left, top, plot_width, plot_height = 55, 24, 670, 225

    def x_at(index: int) -> float:
        return left + index * plot_width / (len(minutes) - 1)

    def y_at(value: float) -> float:
        return top + (high - value) * plot_height / (high - low)

    colors = ["#4fd1c5", "#f6ad55", "#90cdf4", "#f687b3", "#b794f4"]
    lines = []
    for grid in range(5):
        value = low + grid * (high - low) / 4
        y = y_at(value)
        lines.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_width}" y2="{y:.1f}" class="grid"/>')
        lines.append(f'<text x="{left - 8}" y="{y + 4:.1f}" text-anchor="end">{value:.1f}m</text>')
    for index, minute in enumerate(minutes):
        if index % 3 == 0 or index == len(minutes) - 1:
            x = x_at(index)
            lines.append(f'<text x="{x:.1f}" y="{top + plot_height + 25}" text-anchor="middle">{format_clock(minute).replace(" ", "&nbsp;")}</text>')
    legend = []
    for index, (cluster_id, values) in enumerate(sorted(series.items())):
        color = colors[index % len(colors)]
        points = " ".join(f"{x_at(i):.1f},{y_at(value):.1f}" for i, value in enumerate(values))
        lines.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="3"/>')
        legend.append(f'<span><i style="background:{color}"></i>{cluster_id}</span>')
    return (
        f'<div class="legend">{"".join(legend)}</div>'
        f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="Predicted travel time by departure time">'
        + "".join(lines)
        + "</svg>"
    )


def build_html_report(
    learned: LearnedRouteReader,
    evaluation: EvaluationResult,
    output_path: Path,
) -> None:
    metrics = evaluation.metrics
    primary_od = max(learned.discovery.od_pairs, key=lambda pair: len(pair.trip_ids))
    route_counts = learned.model.route_trip_counts()
    window_specs = [
        (15 * 60 + 5, 15 * 60 + 24),
        (15 * 60 + 25, 15 * 60 + 39),
        (15 * 60 + 40, 15 * 60 + 54),
        (15 * 60 + 55, 16 * 60 + 10),
    ]
    performance_rows = []
    for cluster_id in sorted(route_counts):
        route_trips = [
            trip
            for trip in learned.train_trips
            if learned.train_route_assignments[trip.trip_id] == cluster_id
        ]
        cells = []
        for start, end in window_specs:
            durations = [
                trip.duration_seconds
                for trip in route_trips
                if start <= trip.departure_minute <= end
            ]
            cells.append(format_duration(statistics.median(durations)) if durations else "—")
        performance_rows.append(
            f"<tr><th>{cluster_id}</th><td>{len(route_trips)}</td>"
            + "".join(f"<td>{cell}</td>" for cell in cells)
            + "</tr>"
        )

    delay_rows = "".join(
        "<tr>"
        f"<th>{zone.zone_id}</th><td>{zone.route_cluster_id}</td>"
        f"<td>({zone.center_x_m:.0f}m, {zone.center_y_m:.0f}m)</td>"
        f"<td>{zone.affected_trips}</td><td>{format_duration(zone.median_delay_seconds)}</td>"
        f"<td>{format_clock(zone.peak_start_minute)}–{format_clock(zone.peak_end_minute)}</td>"
        f"<td>{format_duration(zone.peak_median_delay_seconds)}</td></tr>"
        for zone in learned.delay_zones
    )
    decision_rows = "".join(
        "<tr>"
        f"<td>{html.escape(item.departure_iso[0:16].replace('T', ' '))}</td>"
        f"<td>{item.actual_fastest_route}</td><td>{item.recommended_route}</td>"
        f"<td>{format_duration(item.regret_seconds)}</td><td>{item.confidence}</td></tr>"
        for item in evaluation.decisions
    )
    reference = learned.test_trips[0].departure
    css = """
    :root{color-scheme:dark;--bg:#0b1220;--panel:#111c30;--text:#eef6ff;--muted:#9eb0c9;--accent:#4fd1c5;--line:#26364f}
    *{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:15px/1.5 system-ui,sans-serif}
    main{max-width:1120px;margin:auto;padding:36px 24px 64px}h1{font-size:40px;margin:0 0 4px}h2{margin-top:34px;font-size:22px}
    .sub{color:var(--muted);margin:0 0 26px}.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:12px}
    .card,.panel{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:18px}.value{font-size:28px;font-weight:750;color:var(--accent)}
    .label,.note{color:var(--muted)}table{width:100%;border-collapse:collapse;background:var(--panel);border-radius:12px;overflow:hidden}
    th,td{padding:10px 12px;border-bottom:1px solid var(--line);text-align:left}thead th{color:var(--muted);font-size:12px;text-transform:uppercase}
    svg{width:100%;background:var(--panel);border-radius:12px}svg text{fill:var(--muted);font-size:11px}.grid{stroke:var(--line);stroke-width:1}
    .legend{display:flex;gap:16px;margin:8px 0}.legend span{color:var(--muted)}.legend i{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:6px}
    code{color:#b9f5ef}footer{margin-top:34px;color:var(--muted)}
    """
    document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Route Reader — canonical proof</title><style>{css}</style></head><body><main>
<h1>Route Reader</h1><p class="sub">Canonical simulation-first proof · seed 20260811 · chronological 40/10 split</p>
<section class="cards">
<div class="card"><div class="value">{metrics.discovered_route_variants}</div><div class="label">route variants independently found</div></div>
<div class="card"><div class="value">{metrics.route_ranking_accuracy:.0%}</div><div class="label">held-out fastest-route accuracy</div></div>
<div class="card"><div class="value">{metrics.travel_time_mae_seconds:.1f}s</div><div class="label">held-out travel-time MAE</div></div>
<div class="card"><div class="value">{metrics.mean_recommendation_regret_seconds:.1f}s</div><div class="label">mean recommendation regret</div></div>
</section>
<h2>Discovery</h2><div class="panel"><b>{primary_od.origin_place_id} → {primary_od.destination_place_id}</b> was the recurring journey, covering {len(primary_od.trip_ids)}/{len(learned.train_trips)} training trips. Two recurring places were found. Route clustering ARI against truth (evaluation only): <b>{metrics.train_route_ari:.2f}</b>.</div>
<h2>Predicted route performance changes with departure time</h2>{_prediction_chart(learned, reference)}
<h2>Observed performance by time window</h2><table><thead><tr><th>Learned route</th><th>Trips</th>{''.join(f'<th>{format_clock(a)}–{format_clock(b)}</th>' for a,b in window_specs)}</tr></thead><tbody>{''.join(performance_rows)}</tbody></table>
<h2>Inferred recurring delay zones</h2><p class="note">These are repeated localized dwell patterns inferred from traces—not recovered traffic-signal schedules.</p>
<table><thead><tr><th>Zone</th><th>Route</th><th>Local position</th><th>Affected trips</th><th>Median dwell</th><th>Peak window</th><th>Peak dwell</th></tr></thead><tbody>{delay_rows}</tbody></table>
<h2>Held-out decisions</h2><table><thead><tr><th>Departure</th><th>True fastest</th><th>Recommended</th><th>Regret</th><th>Confidence</th></tr></thead><tbody>{decision_rows}</tbody></table>
<h2>Baseline comparison</h2><section class="cards">
<div class="card"><div class="value">{metrics.mean_recommendation_regret_seconds:.1f}s</div><div class="label">Route Reader regret · {metrics.route_ranking_accuracy:.0%} accuracy</div></div>
<div class="card"><div class="value">{metrics.common_route_baseline_regret_seconds:.1f}s</div><div class="label">most-common-route regret · {metrics.common_route_baseline_accuracy:.0%} accuracy</div></div>
<div class="card"><div class="value">{metrics.overall_median_baseline_regret_seconds:.1f}s</div><div class="label">lowest-overall-median regret · {metrics.overall_median_baseline_accuracy:.0%} accuracy</div></div>
</section>
<h2>Localized-delay ablation</h2><div class="panel">MAE with inferred delay zones: <b>{metrics.travel_time_mae_seconds:.1f}s</b>. Without localized delay features: <b>{metrics.travel_time_mae_without_delay_zones_seconds:.1f}s</b>. Improvement: <b>{metrics.delay_feature_mae_improvement_seconds:.1f}s</b>.</div>
<footer>All location processing is local. The fixture contains no network calls, telemetry, credentials, routing APIs, or hidden labels in learner input.</footer>
</main></body></html>"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(document, encoding="utf-8")
