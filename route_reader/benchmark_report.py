"""Compact HTML rendering of adversarial benchmark cliffs and failures."""

from __future__ import annotations

import html
import math
from pathlib import Path

from .benchmark import BenchmarkResult


def _number(value: object, digits: int = 2, suffix: str = "") -> str:
    if value is None:
        return "—"
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return html.escape(str(value))
    if math.isnan(numeric):
        return "abstained"
    return f"{numeric:.{digits}f}{suffix}"


def write_benchmark_report(result: BenchmarkResult, path: Path) -> None:
    rows = []
    for scenario in result.scenarios:
        metrics = scenario.metrics
        rows.append(
            "<tr>"
            f"<th>{html.escape(scenario.name)}</th>"
            f"<td>{_number(metrics['segmentation_f1'])}</td>"
            f"<td>{_number(metrics['place_precision'])}/{_number(metrics['place_recall'])}</td>"
            f"<td>{_number(metrics['route_variant_ari'])}</td>"
            f"<td>{_number(metrics['route_ranking_accuracy'], 0, '%') if isinstance(metrics['route_ranking_accuracy'], str) else _number(float(metrics['route_ranking_accuracy']) * 100 if metrics['route_ranking_accuracy'] is not None else None, 0, '%')}</td>"
            f"<td>{_number(metrics['recommendation_regret_seconds'], 1, 's')}</td>"
            f"<td>{_number(metrics['confidence_coverage'] * 100 if metrics['confidence_coverage'] is not None else None, 0, '%')}</td>"
            f"<td>{_number(metrics['mean_interval_width_seconds'], 0, 's')}</td>"
            f"<td>{_number(metrics['delay_precision'])}/{_number(metrics['delay_recall'])}</td>"
            "</tr>"
        )
    normal = next(item for item in result.scenarios if item.name == "quality-normal")
    cold_rows = "".join(
        "<tr>"
        f"<th>{item['history_trips']}</th><td>{html.escape(str(item['stage']))}</td>"
        f"<td>{item['learned_variants']}</td><td>{item['evaluated_predictions']}</td>"
        f"<td>{_number(item['duration_mae_seconds'], 1, 's')}</td>"
        f"<td>{_number(item['mean_interval_width_seconds'], 0, 's')}</td>"
        f"<td>{_number(item['route_ranking_accuracy'] * 100 if item['route_ranking_accuracy'] is not None else None, 0, '%')}</td></tr>"
        for item in normal.cold_start
    )
    failure_sections = "".join(
        f"<article><h3>{html.escape(scenario.name)}</h3><ul>"
        + ("".join(f"<li>{html.escape(failure)}</li>" for failure in scenario.failures) or "<li>No evaluator-level failure flag.</li>")
        + "</ul></article>"
        for scenario in result.scenarios
        if scenario.failures or scenario.name in {"quality-sparse", "quality-hostile"}
    )
    css = """
    :root{color-scheme:dark;--bg:#08111d;--panel:#111e2f;--line:#263b55;--text:#edf6ff;--muted:#9ab0c8;--good:#5ee1b4;--warn:#ffca6b}
    *{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:14px/1.5 system-ui,sans-serif}main{max-width:1240px;margin:auto;padding:34px 22px 70px}
    h1{font-size:38px;margin:0}.sub{color:var(--muted)}.verdict{display:inline-block;padding:7px 12px;border:1px solid var(--good);color:var(--good);border-radius:999px;font-weight:700}
    h2{margin-top:34px}table{width:100%;border-collapse:collapse;background:var(--panel);border:1px solid var(--line)}th,td{padding:9px 10px;border-bottom:1px solid var(--line);text-align:left}thead th{color:var(--muted);font-size:11px;text-transform:uppercase}
    .failures{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:12px}.failures article{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:14px}.failures h3{margin:0}.note{color:var(--muted)}code{color:#a9efe0}
    """
    document = f"""<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Route Reader adversarial robustness lab</title><style>{css}</style></head><body><main>
    <h1>Route Reader robustness lab</h1><p class="sub">Five-month synthetic life history · multiple places/OD pairs · route overlap/evolution · four GPS regimes</p><span class="verdict">{result.verdict}</span>
    <h2>Benchmark matrix</h2><p class="note">“Abstained” means no held-out prediction had enough stable route identity for calibration; it is preferable to fabricated confidence.</p>
    <table><thead><tr><th>Scenario</th><th>Seg F1</th><th>Place P/R</th><th>Variant ARI</th><th>Rank acc.</th><th>Regret</th><th>Interval coverage</th><th>Mean width</th><th>Delay P/R</th></tr></thead><tbody>{''.join(rows)}</tbody></table>
    <h2>Cold-start progression · normal evolving scenario</h2><table><thead><tr><th>Trips</th><th>Responsible stage</th><th>Variants</th><th>Predictions</th><th>Duration MAE</th><th>Mean width</th><th>Rank acc.</th></tr></thead><tbody>{cold_rows}</tbody></table>
    <h2>Exposed failures</h2><div class="failures">{failure_sections}</div>
    <h2>Interpretation guardrails</h2><p class="note">Delay zones are inferred localized dwell patterns, never traffic-signal schedules. Counterfactual route times and synthetic identities exist only inside evaluation. Runtime recommendation modules receive timestamped coordinates and learned history.</p>
    </main></body></html>"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(document, encoding="utf-8")
