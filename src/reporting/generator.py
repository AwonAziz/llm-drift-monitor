"""
Report generation
-----------------
Monitoring output that only exists in a Streamlit tab is a demo. Reports that
can be attached to a PR, pasted into a design doc, or read six months later are
a platform. This module writes:

* ``report.md`` — the incident review: timeline, per-signal deltas, the decision
  the policy made and why, and the evidence behind it;
* ``report.json`` — the same content, machine-readable, for CI gates;
* ``report.html`` — a self-contained page with inline SVG charts, no CDN, so it
  renders from a file:// URL in any environment.

The HTML exists because "show me the drift" in an interview is much easier when
it is one file you can open, not three terminals you have to keep alive.
"""

from __future__ import annotations

import html
import json
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from config import settings
from src.utils.logging import get_logger
from src.utils.stats import json_safe

logger = get_logger(__name__)

SEVERITY_COLOR = {"none": "#2ea043", "moderate": "#d29922", "severe": "#f85149"}
ACTION_ICON = {"noop": "OK", "investigate": "INVESTIGATE", "retrain": "RETRAIN", "rollback": "ROLLBACK"}


# ── Data assembly ──────────────────────────────────────────────────────

def assemble(store, run_id: str) -> dict[str, Any]:
    """Pull one run out of telemetry into a plain dict the renderers can use."""
    run = store.get_run(run_id) or {}
    metrics = store.metric_series(run_id)
    decisions = store.decisions(run_id)
    incidents = store.incidents(run_id)
    judge = store.judge_scores(run_id)
    windows = store.list_windows(run_id)

    def series(category: str, name: str) -> pd.DataFrame:
        m = metrics[(metrics["category"] == category) & (metrics["name"] == name)]
        return m[["window_index", "value", "severity", "baseline", "threshold_moderate",
                  "threshold_severe"]].sort_values("window_index")

    headline = ["health_score", "requests", "judge_score", "domain_classifier_auc",
                "accuracy", "ece", "ood_rate"]
    by_name = {}
    if not metrics.empty:
        for (cat, name), grp in metrics.groupby(["category", "name"]):
            by_name[f"{cat}::{name}"] = grp.sort_values("window_index")

    return {
        "run": run,
        "windows": windows,
        "metrics": metrics,
        "decisions": decisions,
        "incidents": incidents,
        "judge_scores": judge,
        "headline": [k for k in headline if k in by_name or f"volume::{k}" in by_name],
        "by_name": by_name,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def _fmt(v: Any, n: int = 4) -> str:
    if v is None:
        return "—"
    try:
        f = float(v)
    except (TypeError, ValueError):
        return str(v)
    if not np.isfinite(f):
        return "—"
    return f"{f:.{n}f}"


def _slug(text: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in text).strip("_").lower()


# ── Markdown ──────────────────────────────────────────────────────────

def render_markdown(data: dict[str, Any], calibration: dict[str, Any] | None = None,
                    suite: dict[str, Any] | None = None) -> str:
    run = data["run"]
    decisions = data["decisions"]
    metrics = data["metrics"]
    incidents = data["incidents"]

    lines: list[str] = []
    add = lines.append

    add("# LLM Drift & Quality Report")
    add("")
    add(f"- **Run**: `{run.get('run_id', '?')}`")
    add(f"- **Generated**: {data['generated_at']}")
    add(f"- **Encoder**: `{run.get('encoder', '?')}`")
    add(f"- **Judge**: `{run.get('judge', '?')}`")
    add(f"- **Model**: `{run.get('app_model', '?')}`")
    add(f"- **Dataset**: `{run.get('dataset_source', '?')}`")
    add("")

    if calibration:
        add("## Champion validation (the baseline everything is compared to)")
        add("")
        add("| Metric | Value |")
        add("|---|---|")
        for k in ("validation_rows", "intents", "accuracy", "macro_f1", "ece", "brier", "abstention_rate"):
            if k in calibration:
                add(f"| {k.replace('_', ' ').title()} | {_fmt(calibration[k])} |")
        add("")

    add("## Decision timeline")
    add("")
    if not decisions.empty:
        add("| # | Regime | Action | Severity | Health | Confidence | Rationale |")
        add("|---|---|---|---|---|---|---|")
        labels = data["windows"].set_index("window_index")["label"].to_dict() if not data["windows"].empty else {}
        for _, d in decisions.iterrows():
            add(f"| {int(d['window_index'])} | {labels.get(int(d['window_index']), '')} | "
                f"**{d['action']}** | {d['severity']} | {_fmt(d['health'], 1)} | "
                f"{_fmt(d['confidence'], 2)} | {d['rationale']} |")
    else:
        add("_No decisions recorded._")
    add("")

    add("## Signal trends")
    add("")
    interesting = [
        ("embedding", "domain_classifier_auc", "Domain classifier AUC"),
        ("embedding", "mmd2", "MMD²"),
        ("embedding", "swd", "Sliced Wasserstein"),
        ("embedding", "frechet_normalised", "Normalised Frechet"),
        ("embedding", "ood_rate", "Out-of-distribution rate"),
        ("embedding", "concept_gap", "In-scope vs out-of-scope accuracy gap"),
        ("tabular", "psi_mean", "Mean PSI (text features)"),
        ("quality", "accuracy", "Accuracy"),
        ("quality", "macro_f1", "Macro F1"),
        ("quality", "ece", "Expected calibration error"),
        ("quality", "brier", "Brier score"),
        ("quality", "abstention_rate", "Abstention rate"),
        ("judge", "judge_score", "LLM-as-judge weighted score"),
        ("judge", "judge_correctness_rate", "Judge correctness rate"),
        ("judge", "judge_veto_rate", "Judge veto rate"),
    ]
    for category, name, label in interesting:
        df = data["by_name"].get(f"{category}::{name}")
        if df is None or df.empty:
            continue
        vals = ", ".join(_fmt(v, 3) for v in df["value"])
        sevs = ", ".join(str(s) for s in df["severity"])
        add(f"**{label}**  ")
        add(f"values: `{vals}`  ")
        add(f"severity: `{sevs}`")
        add("")

    add("## Per-dimension judge scores")
    add("")
    dim_rows = {k: v for k, v in data["by_name"].items() if k.startswith("judge::judge_dim::")}
    if dim_rows:
        add("| Dimension | " + " | ".join(str(i) for i in sorted(metrics["window_index"].unique())) + " |")
        add("|---" * (len(metrics["window_index"].unique()) + 1) + "|")
        for key, grp in sorted(dim_rows.items()):
            dim = key.split("::")[-1]
            cells = [_fmt(v, 2) for v in grp["value"]]
            add(f"| {dim} | " + " | ".join(cells) + " |")
    else:
        add("_No judge dimension data._")
    add("")

    add("## Incidents")
    add("")
    if not incidents.empty:
        for _, inc in incidents.iterrows():
            state = "OPEN" if inc["status"] == "open" else "CLOSED"
            add(f"### `{inc['incident_id']}` — {inc['title']}")
            add(f"- Severity: **{inc['severity']}** · Status: **{state}**")
            add(f"- Opened: {inc['opened_at']}" + (f" · Closed: {inc['closed_at']}" if inc.get("closed_at") else ""))
            try:
                tl = json.loads(inc.get("timeline") or "[]")
                for entry in tl:
                    add(f"  - {entry.get('at')} — {entry.get('event')}")
            except Exception:  # noqa: BLE001
                pass
            add("")
    else:
        add("_No incidents raised._")
    add("")

    if suite:
        add("## Judge regression suite")
        add("")
        add(f"- Cases: {suite.get('n_passed')}/{suite.get('n_cases')} passed")
        add(f"- Mean score: {_fmt(suite.get('mean_score'))}")
        add(f"- Verdict: {suite.get('verdict', '')}")
        add("")

    return "\n".join(lines)


# ── HTML ──────────────────────────────────────────────────────────────

def _svg_series(values: Sequence[float], width: int = 720, height: int = 180,
                color: str = "#58a6ff", thresholds: tuple[float, float] | None = None,
                pad: int = 34) -> str:
    vals = [np.nan if v is None else float(v) for v in values]
    finite = [v for v in vals if np.isfinite(v)]
    if not finite:
        return '<div class="empty">no data</div>'
    lo, hi = min(finite), max(finite)
    if thresholds:
        lo, hi = min(lo, min(thresholds)), max(hi, max(thresholds))
    span = hi - lo or 1.0
    lo -= span * 0.1
    hi += span * 0.1
    span = hi - lo
    n = max(len(vals) - 1, 1)
    pts = []
    for i, v in enumerate(vals):
        if not np.isfinite(v):
            continue
        x = pad + (width - 2 * pad) * (i / n)
        y = height - pad - (height - 2 * pad) * ((v - lo) / span)
        pts.append(f"{x:.1f},{y:.1f}")
    if not pts:
        return '<div class="empty">no data</div>'

    parts = [f'<svg viewBox="0 0 {width} {height}" class="chart" role="img">']
    parts.append(f'<polyline points="{" ".join(pts)}" fill="none" stroke="{color}" stroke-width="2"/>')
    for i, v in enumerate(vals):
        if not np.isfinite(v):
            continue
        x = pad + (width - 2 * pad) * (i / n)
        y = height - pad - (height - 2 * pad) * ((v - lo) / span)
        parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="2.6" fill="{color}"/>')
    if thresholds:
        for thr, dash, tcol in ((thresholds[0], "4 4", "#d29922"), (thresholds[1], "2 4", "#f85149")):
            if not np.isfinite(thr):
                continue
            y = height - pad - (height - 2 * pad) * ((thr - lo) / span)
            parts.append(f'<line x1="{pad}" y1="{y:.1f}" x2="{width - pad}" y2="{y:.1f}" '
                         f'stroke="{tcol}" stroke-dasharray="{dash}"/>')
    parts.append(f'<text x="{pad}" y="{height - 8}" class="axis">{lo:.3g}</text>')
    parts.append(f'<text x="{width - pad}" y="{height - 8}" class="axis" text-anchor="end">{hi:.3g}</text>')
    parts.append("</svg>")
    return "".join(parts)


HTML_TEMPLATE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>LLM Drift &amp; Quality Report</title>
<style>
  :root {{ color-scheme: dark; }}
  body {{ background:#0d1117; color:#c9d1d9; font:14px/1.55 ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif;
         margin:0; padding:32px; }}
  h1 {{ font-size:24px; margin:0 0 4px; color:#e6edf3; }}
  h2 {{ font-size:17px; margin:32px 0 12px; color:#e6edf3; border-bottom:1px solid #21262d; padding-bottom:6px; }}
  .meta {{ color:#8b949e; font-size:12.5px; margin-bottom:8px; }}
  .meta code {{ color:#79c0ff; }}
  table {{ border-collapse:collapse; width:100%; margin:10px 0 18px; font-size:13px; }}
  th,td {{ border:1px solid #21262d; padding:7px 10px; text-align:left; vertical-align:top; }}
  th {{ background:#161b22; color:#e6edf3; font-weight:600; }}
  tr:nth-child(even) td {{ background:#11161d; }}
  .grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(430px,1fr)); gap:18px; }}
  .card {{ background:#161b22; border:1px solid #21262d; border-radius:8px; padding:14px 16px; }}
  .card h3 {{ margin:0 0 8px; font-size:13.5px; color:#e6edf3; font-weight:600; }}
  .chart {{ width:100%; height:auto; }}
  .axis {{ fill:#8b949e; font-size:10px; }}
  .empty {{ color:#6e7681; font-size:12px; padding:14px 0; }}
  .badge {{ display:inline-block; padding:1px 8px; border-radius:10px; font-size:11.5px; font-weight:600; }}
  .none {{ background:#12331f; color:#3fb950; }}
  .moderate {{ background:#3a2f10; color:#d29922; }}
  .severe {{ background:#3d1418; color:#f85149; }}
  .kpis {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr)); gap:12px; margin:14px 0 6px; }}
  .kpi {{ background:#161b22; border:1px solid #21262d; border-radius:8px; padding:12px 14px; }}
  .kpi .v {{ font-size:23px; font-weight:700; color:#e6edf3; }}
  .kpi .l {{ font-size:11px; color:#8b949e; text-transform:uppercase; letter-spacing:.06em; }}
  .note {{ color:#8b949e; font-size:12.5px; }}
  code {{ font-family:ui-monospace,SFMono-Regular,Menlo,monospace; }}
</style></head><body>
{body}
</body></html>"""


def render_html(data: dict[str, Any], calibration: dict[str, Any] | None = None,
                suite: dict[str, Any] | None = None) -> str:
    run = data["run"]
    e = html.escape
    cards: list[str] = []
    kpis: list[str] = []

    decision_rows = []
    labels = data["windows"].set_index("window_index")["label"].to_dict() if not data["windows"].empty else {}
    for _, d in data["decisions"].iterrows():
        sev = str(d["severity"])
        decision_rows.append(
            f"<tr><td>{int(d['window_index'])}</td><td>{e(str(labels.get(int(d['window_index']), '')))}</td>"
            f"<td><b>{e(ACTION_ICON.get(d['action'], str(d['action'])))}</b></td>"
            f'<td><span class="badge {sev}">{sev}</span></td>'
            f"<td>{_fmt(d['health'], 1)}</td><td>{_fmt(d['confidence'], 2)}</td>"
            f"<td>{e(str(d['rationale']))}</td></tr>")

    chart_specs = [
        ("Domain classifier AUC", "embedding::domain_classifier_auc", "#a371f7", (0.62, 0.78)),
        ("MMD²", "embedding::mmd2", "#f0883e", None),
        ("Out-of-distribution rate", "embedding::ood_rate", "#f85149", (0.25, 0.50)),
        ("Accuracy", "quality::accuracy", "#3fb950", None),
        ("Expected calibration error", "quality::ece", "#d29922", (0.06, 0.12)),
        ("LLM-as-judge score", "judge::judge_score", "#58a6ff", None),
        ("Abstention rate", "quality::abstention_rate", "#8b949e", None),
        ("Mean PSI (text features)", "tabular::psi_mean", "#79c0ff", (0.10, 0.25)),
    ]
    for label, key, color, thr in chart_specs:
        df = data["by_name"].get(key)
        if df is None or df.empty:
            continue
        cards.append(
            f'<div class="card"><h3>{e(label)}</h3>'
            f"{_svg_series(df['value'].tolist(), color=color, thresholds=thr)}</div>")

    incidents_html = ""
    if not data["incidents"].empty:
        rows = []
        for _, inc in data["incidents"].iterrows():
            sev = str(inc["severity"])
            rows.append(
                f"<tr><td><code>{e(str(inc['incident_id']))}</code></td>"
                f'<td><span class="badge {sev}">{sev}</span></td>'
                f"<td>{e(str(inc['title']))}</td><td>{e(str(inc['status']))}</td>"
                f"<td>{e(str(inc['opened_at']))}</td><td>{e(str(inc.get('closed_at') or '—'))}</td></tr>")
        incidents_html = ("<h2>Incidents</h2><table><tr><th>ID</th><th>Severity</th><th>Title</th>"
                          "<th>Status</th><th>Opened</th><th>Closed</th></tr>"
                          + "".join(rows) + "</table>")

    latest = {}
    for key, df in data["by_name"].items():
        if not df.empty:
            latest[key] = df["value"].iloc[-1]

    def kpi(label: str, value: Any, note: str = "") -> None:
        kpis.append(f'<div class="kpi"><div class="l">{e(label)}</div>'
                    f'<div class="v">{e(str(value))}</div>'
                    f'<div class="note">{e(note)}</div></div>')

    kpi("Windows", str(len(data["windows"])))
    if not data["decisions"].empty:
        last = data["decisions"].iloc[-1]
        kpi("Health", _fmt(last["health"], 1), "latest window")
        kpi("Action", ACTION_ICON.get(last["action"], str(last["action"])), "latest decision")
    kpi("AUC", _fmt(latest.get("embedding::domain_classifier_auc"), 3), "reference separability")
    kpi("Accuracy", _fmt(latest.get("quality::accuracy"), 3), "latest labelled window")
    kpi("Judge", _fmt(latest.get("judge::judge_score"), 3), "weighted rubric score")

    cal_html = ""
    if calibration:
        cal_html = ("<h2>Champion validation baseline</h2><table>"
                    + "".join(f"<tr><th>{e(k.replace('_', ' ').title())}</th><td>{_fmt(calibration[k])}</td></tr>"
                              for k in calibration)
                    + "</table>")

    suite_html = ""
    if suite:
        suite_html = ("<h2>Judge regression suite</h2><table>"
                      f"<tr><th>Passed</th><td>{suite.get('n_passed')}/{suite.get('n_cases')}</td></tr>"
                      f"<tr><th>Mean score</th><td>{_fmt(suite.get('mean_score'))}</td></tr>"
                      f"<tr><th>Verdict</th><td>{e(str(suite.get('verdict', '')))}</td></tr></table>")

    body = f"""
<h1>LLM Drift &amp; Quality Report</h1>
<div class="meta">run <code>{e(str(run.get('run_id', '?')))}</code> · generated {e(data['generated_at'])}<br>
encoder <code>{e(str(run.get('encoder', '?')))}</code> · judge <code>{e(str(run.get('judge', '?')))}</code> ·
model <code>{e(str(run.get('app_model', '?')))}</code> · dataset <code>{e(str(run.get('dataset_source', '?')))}</code></div>
<div class="kpis">{''.join(kpis)}</div>
{cal_html}
<h2>Decision timeline</h2>
<table><tr><th>#</th><th>Regime</th><th>Action</th><th>Severity</th><th>Health</th><th>Conf.</th><th>Rationale</th></tr>
{''.join(decision_rows)}</table>
<h2>Signals over time</h2>
<div class="grid">{''.join(cards)}</div>
{incidents_html}
{suite_html}
<p class="note">Dashed lines mark the configured alert thresholds. Every value is measured against the frozen
reference snapshot and champion validation baseline recorded in the run manifest.</p>
"""
    return HTML_TEMPLATE.format(body=body)


# ── Entry point ───────────────────────────────────────────────────────

@dataclass
class ReportPaths:
    markdown: Path
    json: Path
    html: Path


def write_reports(store, run_id: str, out_dir: Path | None = None,
                  calibration: dict[str, Any] | None = None,
                  suite: dict[str, Any] | None = None) -> ReportPaths:
    out_dir = Path(out_dir or settings.REPORT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    data = assemble(store, run_id)

    md_path = out_dir / f"report_{run_id}.md"
    json_path = out_dir / f"report_{run_id}.json"
    html_path = out_dir / f"report_{run_id}.html"

    md_path.write_text(render_markdown(data, calibration, suite), encoding="utf-8")
    html_path.write_text(render_html(data, calibration, suite), encoding="utf-8")
    json_path.write_text(
        json.dumps(json_safe({
            "run": data["run"],
            "generated_at": data["generated_at"],
            "calibration": calibration,
            "judge_suite": suite,
            "windows": data["windows"].to_dict("records"),
            "decisions": data["decisions"].to_dict("records"),
            "incidents": data["incidents"].to_dict("records"),
            "metrics": data["metrics"].to_dict("records"),
        }), indent=2), encoding="utf-8")

    logger.info("Reports written to %s", out_dir)
    return ReportPaths(markdown=md_path, json=json_path, html=html_path)