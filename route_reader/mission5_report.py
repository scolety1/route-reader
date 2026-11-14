"""Generate a concise, self-contained Mission 5 evidence report."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path


def _fmt(value, digits: int = 1) -> str:
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "PASS" if value else "FAIL"
    if isinstance(value, (int, float)):
        return f"{value:.{digits}f}"
    return html.escape(str(value))


def _trajectory_svg(checkpoints: list[dict[str, object]]) -> str:
    if not checkpoints:
        return ""
    width, height = 760, 250
    maximum_x = max(float(item["decisions"]) for item in checkpoints)
    fields = (
        ("cumulative_recommended_pseudo_regret_seconds", "#38bdf8", "recommendation"),
        ("cumulative_selected_pseudo_regret_seconds", "#f59e0b", "selected incl. exploration"),
        ("cumulative_realized_user_regret_seconds", "#f472b6", "actual user"),
    )
    maximum_y = max(float(item[field]) for item in checkpoints for field, _, _ in fields) or 1.0
    lines = []
    for field, color, label in fields:
        points = []
        for item in checkpoints:
            x = 48 + float(item["decisions"]) / maximum_x * (width - 72)
            y = height - 35 - float(item[field]) / maximum_y * (height - 65)
            points.append(f"{x:.1f},{y:.1f}")
        lines.append(f'<polyline fill="none" stroke="{color}" stroke-width="3" points="{" ".join(points)}"/>')
        lines.append(f'<span class="legend"><i style="background:{color}"></i>{label}</span>')
    polylines = "".join(item for item in lines if item.startswith("<polyline"))
    legend = "".join(item for item in lines if item.startswith("<span"))
    axes = f'<line x1="48" y1="15" x2="48" y2="{height - 35}"/><line x1="48" y1="{height - 35}" x2="{width - 20}" y2="{height - 35}"/>'
    labels = f'<text x="8" y="22">{maximum_y:.0f}s</text><text x="22" y="{height - 30}">0</text><text x="{width - 52}" y="{height - 14}">{maximum_x:.0f} decisions</text>'
    return f'<div class="chart"><svg viewBox="0 0 {width} {height}" role="img" aria-label="Cumulative regret trajectories">{axes}{labels}{polylines}</svg><div>{legend}</div></div>'


def render_report(data: dict[str, object]) -> str:
    verdict = data["verdict"]
    distributions = data["distributions"]
    strong = data["scenarios"]["strong"][0]
    ranking_ope = next(item for item in data["off_policy_evaluation"]["adequate_support"] if item["target_policy"] == "mission5_ranking")
    gates = "".join(
        f'<tr><td>{html.escape(name.replace("_", " "))}</td><td class="{"pass" if passed else "fail"}">{"PASS" if passed else "FAIL"}</td></tr>'
        for name, passed in data["gates"].items()
    )
    distribution_rows = "".join(
        "<tr>"
        f"<td>{signal}</td>"
        f"<td>{_fmt(row['mission5_regret_median_seconds'])}</td>"
        f"<td>{_fmt(row['mission5_regret_iqr_seconds'][0])}–{_fmt(row['mission5_regret_iqr_seconds'][1])}</td>"
        f"<td>{_fmt(row['mission5_regret_worst_seconds'])}</td>"
        f"<td>{_fmt(row['mission4_regret_median_seconds'])}</td>"
        f"<td>{_fmt(row['median_improvement_over_mission4_seconds'])}</td>"
        f"<td>{row['material_simple_baseline_failures']}/10</td>"
        "</tr>"
        for signal, row in distributions.items()
    )
    ope_rows = "".join(
        "<tr>"
        f"<td>{html.escape(item['estimator'])}</td>"
        f"<td>{_fmt(item['estimated_mean_seconds'])}</td>"
        f"<td>{_fmt(item['absolute_error_seconds'])}</td>"
        f"<td>{_fmt(item['effective_sample_size'])}</td>"
        f"<td>{_fmt(item['support_coverage'] * 100)}%</td>"
        f"<td class=\"{'pass' if item['reliable'] else 'fail'}\">{'usable' if item['reliable'] else 'unreliable'}</td>"
        "</tr>"
        for item in ranking_ope["estimates"]
    )
    failures = ", ".join(item.replace("_", " ") for item in data["explicit_failures"])
    emerging = data["emerging_route"]
    established = sum(item["reached_established"] for item in emerging)
    repeated = sum(item["reached_repeated"] for item in emerging)
    spatial = data["mission4_spatial_regression"]
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Route Reader Mission 5</title><style>
:root{{--bg:#07111f;--panel:#0f1d2f;--ink:#e6edf7;--muted:#94a3b8;--line:#26384d;--cyan:#38bdf8;--amber:#f59e0b;--red:#fb7185;--green:#34d399}}
*{{box-sizing:border-box}} body{{margin:0;background:linear-gradient(145deg,#07111f,#0a1728);color:var(--ink);font:15px/1.5 system-ui,sans-serif}}
main{{max-width:1120px;margin:auto;padding:34px 22px 60px}} h1{{font-size:36px;margin:0 0 6px}} h2{{margin:34px 0 12px;font-size:21px}} p{{color:var(--muted);max-width:850px}} .verdict{{display:inline-block;padding:7px 13px;border:1px solid var(--amber);color:var(--amber);border-radius:999px;font-weight:800;letter-spacing:.08em}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:12px}} .card{{background:rgba(15,29,47,.92);border:1px solid var(--line);border-radius:13px;padding:17px}} .metric{{font-size:27px;font-weight:800;color:var(--cyan)}} .label{{color:var(--muted);font-size:13px}}
table{{width:100%;border-collapse:collapse;background:var(--panel);border:1px solid var(--line)}} th,td{{padding:10px 12px;border-bottom:1px solid var(--line);text-align:right}} th:first-child,td:first-child{{text-align:left}} th{{color:var(--muted);font-size:12px;text-transform:uppercase}} .pass{{color:var(--green);font-weight:700}} .fail{{color:var(--red);font-weight:700}}
.chart{{background:var(--panel);border:1px solid var(--line);padding:14px;border-radius:13px}} svg line{{stroke:#50627a;stroke-width:1}} svg text{{fill:var(--muted);font-size:12px}} .legend{{margin-right:18px;color:var(--muted);font-size:13px}} .legend i{{display:inline-block;width:12px;height:3px;margin:0 6px 3px 0}}
code{{color:#bae6fd}} .callout{{border-left:4px solid var(--amber);padding:9px 16px;background:#171d25;color:#dbe7f5}}
</style></head><body><main>
<span class="verdict">{verdict}</span><h1>Partial feedback is real. Policy lift is not yet dependable.</h1>
<p>Mission 5 replaces full-information retrospective selection with a chronological route-choice learner. It logs exact action probabilities before completion and updates only the route actually taken. The trustworthy result is YELLOW because strong-signal lift is inconsistent and emerging routes graduate too slowly.</p>
<div class="grid">
<div class="card"><div class="metric">{_fmt(distributions['strong']['mission5_regret_median_seconds'])}s</div><div class="label">10-seed strong median regret / decision</div></div>
<div class="card"><div class="metric">{_fmt(distributions['strong']['median_improvement_over_mission4_seconds'])}s</div><div class="label">median improvement over partial-log Mission 4 selector</div></div>
<div class="card"><div class="metric">{_fmt(strong['exploration_realized_regret_seconds'])}s</div><div class="label">canonical cumulative exploration regret, 3 choices</div></div>
<div class="card"><div class="metric">{established}/5</div><div class="label">drift seeds where emerging route established ({repeated}/5 reached repeated)</div></div>
</div>
<h2>Cumulative regret — canonical strong signal</h2>{_trajectory_svg(strong['checkpoints'])}
<p>Recommendation pseudo-regret excludes user non-compliance; actual-user regret records the route the simulated user truly took. Exploration is shown separately and is never relabeled as exploitation.</p>
<h2>Ten-seed decision results</h2><table><thead><tr><th>signal</th><th>M5 median</th><th>IQR</th><th>worst</th><th>M4 median</th><th>median lift</th><th>material failures</th></tr></thead><tbody>{distribution_rows}</tbody></table>
<h2>Offline policy evaluation — Mission 5 target</h2><p>Simulator truth: {_fmt(ranking_ope['simulator_true_mean_seconds'])} seconds. SNIPS and doubly robust agree within 3.3 seconds under full action support; clipped IPS remains biased and is retained as a warning, not the preferred estimate.</p>
<table><thead><tr><th>estimator</th><th>estimate</th><th>absolute error</th><th>ESS</th><th>support</th><th>status</th></tr></thead><tbody>{ope_rows}</tbody></table>
<h2>Acceptance gates</h2><table><tbody>{gates}</tbody></table>
<div class="callout"><strong>Remaining failures:</strong> {html.escape(failures)}. Poor-support stress reports all estimators unreliable; it does not emit a precise policy claim.</div>
<h2>Spatial and product safeguards</h2><div class="grid">
<div class="card"><div class="metric">{spatial['established_variants']}</div><div class="label">Mission 4 durable variants retained</div></div>
<div class="card"><div class="metric">{spatial['historical_identity_reassignments']}</div><div class="label">historical route reassignments</div></div>
<div class="card"><div class="metric">15%</div><div class="label">maximum exploration probability</div></div>
<div class="card"><div class="metric">45s</div><div class="label">maximum expected exploration penalty</div></div>
</div>
<p>Localized-delay influence remains gated off because the partial outcome interface carries no supported localized feature. The optimizer targets route choice among known feasible alternatives—never speed or driving behavior.</p>
</main></body></html>"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("artifacts/mission5-results.json"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/mission5-report.html"))
    args = parser.parse_args()
    data = json.loads(args.input.read_text(encoding="utf-8"))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render_report(data), encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
