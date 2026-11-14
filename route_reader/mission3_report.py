"""Generate a compact self-contained Mission 3 benchmark report."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path


def _seconds(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.1f}s"


def render(data: dict[str, object]) -> str:
    distributions = data["distributions"]
    rows = "".join(
        f"<tr><th>{signal}</th><td>{_seconds(item['adaptive_regret_median'])}</td>"
        f"<td>{_seconds(item['adaptive_regret_iqr'][0])}–{_seconds(item['adaptive_regret_iqr'][1])}</td>"
        f"<td>{_seconds(item['adaptive_regret_worst'])}</td><td>{item['material_simple_baseline_failures']}/5</td></tr>"
        for signal, item in distributions.items()
    )
    routes = "".join(
        f"<tr><th>{item['regime']}</th><td>{item['runtime_discovered_od_variants']}</td><td>{item['variants']}</td><td>{item['ari']:.3f}</td>"
        f"<td>{item['snapshots'][-1]['retained_identity_fraction']:.0%}</td></tr>"
        for item in data["route_quality"]
    )
    replay = "".join(
        f"<tr><th>{item['history_trips']}</th><td>{item['known_route_variants']}</td>"
        f"<td>{html.escape(', '.join(f'{x[0]}:{x[1]}' for x in item['selected_models'][:3]))}</td>"
        f"<td>{html.escape(item['abstention_reason'] or item['recommended_route'] or '—')}</td></tr>"
        for item in data["incremental_milestones"]
    )
    limitations = "".join(f"<li>{html.escape(item)}</li>" for item in data["known_limitations"])
    return f"""<!doctype html><html><head><meta charset='utf-8'><title>Route Reader Mission 3</title>
<style>body{{font:15px system-ui;margin:0;background:#10151c;color:#e9eef5}}main{{max-width:1050px;margin:auto;padding:42px}}
h1{{font-size:34px;margin-bottom:4px}}.verdict{{color:#ffca6b;font-weight:750}}section{{background:#18212c;padding:20px;margin:18px 0;border-radius:12px}}
table{{border-collapse:collapse;width:100%}}th,td{{text-align:left;border-bottom:1px solid #344150;padding:9px}}small{{color:#aeb9c7}}code{{color:#9ed4ff}}</style></head>
<body><main><h1>Route Reader — Mission 3</h1><p class='verdict'>Verdict: {data['verdict']}</p>
<p>Chronological adaptive selection fixes the weak-signal regression and place-region promotion becomes substantially more precise. Route identity fragmentation and strong-signal selection remain incomplete.</p>
<section><h2>Five-seed decision robustness</h2><table><tr><th>Signal</th><th>Median regret</th><th>IQR</th><th>Worst</th><th>&gt;5s vs best simple</th></tr>{rows}</table></section>
<section><h2>Route stability</h2><table><tr><th>Observation</th><th>Runtime OD variants</th><th>Strict replay variants</th><th>ARI</th><th>Final identity retention</th></tr>{routes}</table><small>Good GPS is improved from 14 to 9 variants in the production Mission 3 hierarchy path; the stricter lineage replay view still exposes 16 clusters and therefore remains a stated blocker.</small></section>
<section><h2>Incremental milestones</h2><table><tr><th>Trips</th><th>Variants</th><th>First models</th><th>Decision/fallback</th></tr>{replay}</table></section>
<section><h2>Explicit limitations</h2><ul>{limitations}</ul></section>
<p><small>Reproduce: <code>python -m route_reader.mission3_benchmark --output artifacts/mission3-results.json</code></small></p>
</main></body></html>"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("artifacts/mission3-results.json"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/mission3-report.html"))
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render(json.loads(args.input.read_text(encoding="utf-8"))), encoding="utf-8")
    print(args.output.resolve())


if __name__ == "__main__":
    main()
