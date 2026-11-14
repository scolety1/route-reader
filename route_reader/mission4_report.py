"""Render the Mission 4 machine audit as a compact self-contained HTML report."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path


def _seconds(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.1f}s"


def render(data: dict[str, object]) -> str:
    churn = data["mission3_churn_instrumentation"]
    online = data["ablation"]["mission4_online"]
    routes = "".join(
        f"<tr><th>{row['regime']}</th><td>{row['established_variants']}</td><td>{row['candidate_variants'] + row['repeated_variants']}</td>"
        f"<td>{row['durable_route_purity']:.1%}</td><td>{row['identity_churn_per_100_trips']:.1f}</td>"
        f"<td>{row['unresolved_trips']}</td></tr>" for row in data["route_regimes"]
    )
    signals = "".join(
        f"<tr><th>{name}</th><td>{_seconds(row['adaptive_regret_median'])}</td>"
        f"<td>{_seconds(row['adaptive_regret_iqr'][0])}–{_seconds(row['adaptive_regret_iqr'][1])}</td>"
        f"<td>{_seconds(row['adaptive_regret_worst'])}</td><td>{row['material_simple_baseline_failures']}/5</td>"
        f"<td>{row['abstention_rate_median']:.0%}</td><td>{_seconds(row['interval_width_median_seconds'])}</td></tr>"
        for name, row in data["distributions"].items()
    )
    causes = "".join(f"<li>{html.escape(name.replace('_', ' '))}: <strong>{count}</strong></li>" for name, count in churn["causes"].items())
    gates = "".join(f"<li class='{'pass' if passed else 'fail'}'>{'✓' if passed else '✗'} {html.escape(name.replace('_', ' '))}</li>" for name, passed in data["gates"].items())
    return f"""<!doctype html><html><head><meta charset='utf-8'><title>Route Reader Mission 4</title>
<style>body{{margin:0;background:#0c1219;color:#eaf1f8;font:15px system-ui}}main{{max-width:1100px;margin:auto;padding:40px}}h1{{font-size:36px;margin-bottom:3px}}
.verdict{{font-weight:800;color:#f5c768}}section{{background:#16212c;border:1px solid #26384a;border-radius:12px;padding:20px;margin:18px 0}}
table{{border-collapse:collapse;width:100%}}th,td{{text-align:left;border-bottom:1px solid #314457;padding:9px}}.pass{{color:#93dfb2}}.fail{{color:#ff9f9f}}code{{color:#9fd5ff}}</style></head>
<body><main><h1>Route Reader — Mission 4</h1><p class='verdict'>Verdict: {data['verdict']}</p>
<p>Append-only corridor memory substantially stabilizes personal route identity. Strong-signal exploitation and one strict weak-seed safety failure remain blockers.</p>
<section><h2>Mission 3 churn instrumentation</h2><p>{churn['structural_creation_events']} structural creations and {churn['historical_identity_reassignments']} historical reassignments over {churn['trips']} trips.</p><ul>{causes}</ul></section>
<section><h2>Representation ablation</h2><p>Batch replay: <strong>{churn['final_structural_variants']}</strong> structural variants. Online memory: <strong>{online['final_durable_variants']}</strong> durable variants, {online['candidate_variants']} provisional, and <strong>zero</strong> historical reassignments.</p></section>
<section><h2>Route memory by observation regime</h2><table><tr><th>Regime</th><th>Durable</th><th>Provisional</th><th>Purity</th><th>Churn/100</th><th>Unresolved</th></tr>{routes}</table></section>
<section><h2>Five-seed strict walk-forward policy</h2><table><tr><th>Signal</th><th>Median regret</th><th>IQR</th><th>Worst</th><th>Simple failures</th><th>Abstention</th><th>Interval</th></tr>{signals}</table></section>
<section><h2>Acceptance gates</h2><ul>{gates}</ul></section>
<p>Reproduce with <code>python -m route_reader.mission4_benchmark --output artifacts/mission4-results.json</code>.</p></main></body></html>"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("artifacts/mission4-results.json"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/mission4-report.html"))
    args = parser.parse_args()
    data = json.loads(args.input.read_text(encoding="utf-8"))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render(data), encoding="utf-8")
    print(args.output.resolve())


if __name__ == "__main__":
    main()
