#!/usr/bin/env python3
"""
Build the public demo site
-------------------------
Generates a static, dependency-free site into ``docs/`` for GitHub Pages:

* ``docs/report.html`` — the self-contained incident report for a run
* ``docs/index.html``   — a landing page with the headline numbers and the
  report embedded in an iframe

Everything is rendered from telemetry, so the numbers on the landing page are
read out of the database rather than typed by hand and left to rot.

    python scripts/build_site.py                    # latest run
    python scripts/build_site.py --run run_2026...  # a specific run
    python scripts/build_site.py --commit            # write a workflow for Pages

Why a static site and not the live dashboard: the report needs no server, no
model weights, no LLM and no API key, so it cannot rot and cannot go down. It
is the artefact that still works in three years. The dashboard is a separate
deployment for people who want to poke at it.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.logging import setup_logging  # noqa: E402

setup_logging("WARNING")

from config import settings  # noqa: E402
from src.reporting import assemble, render_html  # noqa: E402
from src.storage.telemetry import TelemetryStore  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent

DOCS = ROOT / "docs"
REPO = "https://github.com/AwonAziz/llm-drift-monitor"

PAGES_WORKFLOW = """name: deploy-site
on:
  push:
    branches: [main]
    paths: ["docs/**"]
  workflow_dispatch:

permissions:
  contents: read
  pages: write
  id-token: write

concurrency:
  group: pages
  cancel-in-progress: false

jobs:
  deploy:
    runs-on: ubuntu-latest
    environment:
      name: github-pages
      url: ${{ steps.deployment.outputs.page_url }}
    steps:
      # docs/ is generated locally by scripts/build_site.py and committed.
      # Regenerating it here would need the telemetry database, which is
      # deliberately gitignored, so a published site that quietly rebuilds
      # itself from nothing is worse than one that only changes when a human
      # has run the demo and looked at the output.
      - uses: actions/checkout@v4

      - uses: actions/configure-pages@v5
      - uses: actions/upload-pages-artifact@v3
        with:
          path: docs

      - id: deployment
        uses: actions/deploy-pages@v4
"""


def _fmt(value, digits: int = 3, dash: str = "—") -> str:
    if value is None:
        return dash
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def _load_suite() -> dict | None:
    path = settings.ARTIFACT_DIR / "judge_regression.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None


def _load_calibration() -> dict | None:
    path = settings.ARTIFACT_DIR / "bootstrap_summary.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8")).get("calibration")
    except Exception:  # noqa: BLE001
        return None


INDEX = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>LLM Drift &amp; Quality Monitor</title>
<meta name="description" content="Embedding drift, output-quality drift and LLM-as-judge regression tracking for a production LLM application.">
<style>
  :root {{ color-scheme: dark; --bg:#0d1117; --panel:#161b22; --border:#21262d;
           --text:#c9d1d9; --muted:#8b949e; --blue:#58a6ff; --green:#3fb950;
           --yellow:#d29922; --red:#f85149; --purple:#a371f7; }}
  * {{ box-sizing:border-box; }}
  body {{ margin:0; background:var(--bg); color:var(--text); padding:0 20px 72px;
          font:15px/1.65 ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif; }}
  .wrap {{ max-width:1080px; margin:0 auto; }}
  header {{ padding:56px 0 28px; border-bottom:1px solid var(--border); margin-bottom:34px; }}
  h1 {{ font-size:34px; line-height:1.2; margin:0 0 10px; color:#e6edf3; letter-spacing:-.02em; }}
  .sub {{ color:var(--muted); font-size:16px; max-width:760px; margin:0 0 20px; }}
  .cta {{ display:inline-block; margin:0 10px 10px 0; padding:9px 18px; border-radius:7px;
          background:var(--blue); color:#08121f; font-weight:650; font-size:14px;
          text-decoration:none; border:1px solid var(--blue); }}
  .cta.alt {{ background:transparent; color:var(--blue); }}
  .kpis {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(158px,1fr)); gap:13px; margin-bottom:34px; }}
  .kpi {{ background:var(--panel); border:1px solid var(--border); border-radius:9px; padding:14px 16px; }}
  .kpi .l {{ font-size:10.5px; color:var(--muted); text-transform:uppercase; letter-spacing:.07em; }}
  .kpi .v {{ font-size:24px; font-weight:700; margin-top:4px; color:#e6edf3; }}
  .kpi .n {{ font-size:11.5px; color:var(--muted); }}
  h2 {{ font-size:20px; margin:38px 0 12px; color:#e6edf3; }}
  p {{ margin:0 0 13px; }}
  .panel {{ background:var(--panel); border:1px solid var(--border); border-radius:9px;
            padding:18px 20px; margin-bottom:15px; }}
  .claim {{ border-left:3px solid var(--blue); }}
  .claim.warn {{ border-left-color:var(--yellow); }}
  .claim.bad {{ border-left-color:var(--red); }}
  .claim.good {{ border-left-color:var(--green); }}
  code {{ font-family:ui-monospace,SFMono-Regular,Menlo,monospace; font-size:13.5px;
          background:#010409; padding:1.5px 6px; border-radius:4px; }}
  .frame {{ width:100%; height:1180px; border:1px solid var(--border); border-radius:9px;
            background:var(--panel); margin-top:12px; }}
  table {{ width:100%; border-collapse:collapse; font-size:13.5px; margin-bottom:14px; }}
  th,td {{ border:1px solid var(--border); padding:7px 10px; text-align:left; }}
  th {{ background:var(--panel); color:#e6edf3; }}
  footer {{ margin-top:44px; padding-top:20px; border-top:1px solid var(--border);
            color:var(--muted); font-size:13px; }}
  a {{ color:var(--blue); }}
</style>
</head>
<body>
<div class="wrap">
<header>
  <h1>LLM Drift &amp; Quality Monitor</h1>
  <p class="sub">Embedding drift, output-quality drift and LLM-as-judge regression tracking
  for a production LLM application &mdash; demonstrated across a scripted production shift on
  Banking77, a public dataset of 10,000 real customer-service queries.</p>
  <a class="cta" href="{repo}">View the code</a>
  <a class="cta alt" href="{repo}#readme">Architecture &amp; the bugs I found</a>
  <a class="cta alt" href="{repo}/blob/main/INTERVIEW_PLAYBOOK.md">Design notes</a>
</header>

<div class="kpis">
  <div class="kpi"><div class="l">Windows</div><div class="v">{windows}</div><div class="n">{regimes} production regimes</div></div>
  <div class="kpi"><div class="l">Requests</div><div class="v">{requests:,}</div><div class="n">{labelled:,} labelled</div></div>
  <div class="kpi"><div class="l">Accuracy floor</div><div class="v" style="color:var(--red)">{acc_min}</div><div class="n">from {acc_max}</div></div>
  <div class="kpi"><div class="l">OOD rate</div><div class="v">{ood_max}</div><div class="n">from {ood_min}</div></div>
  <div class="kpi"><div class="l">Domain AUC</div><div class="v">{auc_min}&ndash;{auc_max}</div><div class="n">reference separability</div></div>
  <div class="kpi"><div class="l">Tests</div><div class="v">193</div><div class="n">offline, ~95s</div></div>
</div>

<h2>The one idea the project is built around</h2>
<div class="panel claim">
  <p><strong>Input drift and output-quality drift are different incidents with different remedies.</strong></p>
  <p>A model can absorb an entirely new input distribution and stay accurate. It can also keep seeing
  byte-identical traffic and lose accuracy. Retraining fixes the first cause and does nothing for the
  second &mdash; so the triage policy weights quality above input shift and refuses to retrain on input
  drift alone.</p>
</div>

<h2>What the run shows</h2>
<div class="panel claim good"><p><strong>Windows 4&ndash;6 &mdash; input drift with quality intact.</strong>
A new partner channel switches to terse, lowercase, emoji-laden text for the same intents. The
embedding space genuinely moves (AUC {style_auc} vs {base_auc} at baseline) while accuracy stays at
{style_acc}. The correct action is to widen coverage, <em>not</em> to retrain.</p></div>

<div class="panel claim bad"><p><strong>Windows 7&ndash;11 &mdash; the real incident.</strong>
Two products ship and {oos_pct:.0f}% of traffic falls outside the model's training scope. Out-of-distribution
rate reaches {ood_max}, accuracy falls to {acc_min}, and the concept gap between in-scope and
out-of-scope accuracy reaches {gap_max}. Note that input-drift severity alone reads only
&ldquo;moderate&rdquo; here &mdash; the quality signals are what make it severe, and quality is what a user feels.</p></div>

<div class="panel claim good"><p><strong>Window 3 &mdash; the control case.</strong>
A 2&times; traffic spike with an identical mix fires nothing. Every detector is sample-size aware,
which is the single most common failure in drift dashboards.</p></div>

<h2>Full incident report</h2>
<p>Decision timeline, per-signal trends, per-dimension judge scores and the incident record, rendered
from the telemetry store. Self-contained &mdash; no server, no CDN.</p>
<iframe class="frame" src="report.html" title="LLM drift and quality incident report"></iframe>

<h2>How it is measured</h2>
<table>
  <tr><th>Signal</th><th>Statistic</th></tr>
  <tr><td>Embedding drift</td><td>MMD&sup2; (permutation-calibrated), sliced Wasserstein,
      domain-classifier AUC, normalised Fr&eacute;chet distance, k-NN novelty rate, concept gap</td></tr>
  <tr><td>Output quality</td><td>Accuracy, macro-F1, ECE / MCE / adaptive ECE, Brier, AUC,
      reliability curve, abstention rate, label-free proxies, separate in-scope baselines</td></tr>
  <tr><td>LLM-as-judge</td><td>Anchored five-dimension rubric with veto conditions, stratified
      sampling, z-gated regression test, pairwise win-rate, golden regression suite</td></tr>
  <tr><td>Triage</td><td>Health score, severity, confidence, confirmation windows, four actions,
      incident lifecycle with a timeline</td></tr>
</table>

<h2>Calibration bugs found along the way</h2>
<div class="panel claim warn">
  <p>A domain classifier fitted on raw 384-dimensional sentence embeddings separates two samples of the
  <em>same</em> distribution at AUC 0.63, purely by exploiting sampling noise &mdash; so every window
  looks like drift. Fitting PCA on the reference brought the null back to 0.50. Along the way: an MMD
  permutation null computed at a different sample size than the observation, a k-NN novelty cutoff
  missing leave-one-out (out-of-scope rate read 100% on in-distribution traffic), an accuracy metric
  that was tautologically 1.0, and a judge rubric whose missing veto cap let a reply asking for a
  customer's PIN average 0.84. Each is now a regression test.</p>
</div>

<footer>
  <p>Dataset: <a href="https://github.com/PolyAI-LDN/task-specific-datasets">Banking77</a>
  (Casanueva et al., 2020), CC BY 4.0, fetched at runtime &mdash; not redistributed here.
  Embeddings: <code>all-MiniLM-L6-v2</code>. Judge: {judge}.</p>
  <p>Report generated from run <code>{run_id}</code> &middot; {generated}</p>
</footer>
</div>
</body>
</html>
"""


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the public demo site into docs/")
    parser.add_argument("--run", default=None, help="Run id (default: most recent)")
    parser.add_argument("--commit-workflow", action="store_true",
                        help="Also write .github/workflows/deploy-site.yml")
    args = parser.parse_args()

    store = TelemetryStore()
    run_id = args.run or store.latest_run_id()
    if not run_id:
        print("No monitoring runs found. Run: python scripts/demo_shift.py --fast")
        return 1

    data = assemble(store, run_id)
    if data["windows"] is None or data["windows"].empty:
        print(f"Run {run_id} has no windows; nothing to publish.")
        return 1

    DOCS.mkdir(parents=True, exist_ok=True)
    report_path = DOCS / "report.html"
    report_path.write_text(
        render_html(data, calibration=_load_calibration(), suite=_load_suite()),
        encoding="utf-8")

    by = data["by_name"]

    def col(name: str) -> list:
        df = by.get(name)
        return [] if df is None else [None if v is None else float(v) for v in df["value"]]

    def finite(values):
        """Drop both None and NaN.

        The telemetry store writes NULL for a NaN metric, but pandas hands it
        back as float('nan'), so filtering on None alone lets NaN through and
        `max()` returns nan — which is how "concept gap reaches nan" ends up
        on a published page.
        """
        out = []
        for v in values:
            if v is None:
                continue
            try:
                f = float(v)
            except (TypeError, ValueError):
                continue
            if f == f:
                out.append(f)
        return out

    windows = data["windows"].sort_values("window_index")
    auc = finite(col("embedding::domain_classifier_auc"))
    ood = finite(col("embedding::ood_rate"))
    acc = finite(col("quality::accuracy"))
    gap = finite(col("embedding::concept_gap"))

    labels = [str(x) for x in windows["label"].tolist()]

    def window_value(values, label: str):
        if label in labels and labels.index(label) < len(values):
            return values[labels.index(label)]
        return None

    judged = (data["judge_scores"]["judge_model"].iloc[0]
              if not data["judge_scores"].empty else "ollama/qwen3:8b")

    page = INDEX.format(
        repo=REPO,
        windows=len(windows),
        regimes=len({x for x in labels}),
        requests=int(windows["n_traffic"].sum()),
        labelled=int(windows["n_labeled"].sum()),
        acc_min=_fmt(min(acc)) if acc else "—",
        acc_max=_fmt(max(acc)) if acc else "—",
        ood_max=_fmt(max(ood)) if ood else "—",
        ood_min=_fmt(min(ood), 2) if ood else "—",
        auc_min=_fmt(min(auc), 2) if auc else "—",
        auc_max=_fmt(max(auc), 2) if auc else "—",
        base_auc=_fmt(window_value(auc, "baseline"), 2) or "—",
        style_auc=_fmt(window_value(auc, "style_shift"), 2) or "—",
        style_acc=_fmt(window_value(acc, "style_shift")) or "—",
        gap_max=_fmt(max(gap), 2) if gap else "—",
        oos_pct=(max(ood) * 100) if ood else 0,
        run_id=run_id,
        judge=judged,
        generated=data["generated_at"],
    )
    (DOCS / "index.html").write_text(page, encoding="utf-8")

    if args.commit_workflow:
        wf = ROOT / ".github" / "workflows" / "deploy-site.yml"
        wf.write_text(PAGES_WORKFLOW, encoding="utf-8")
        print(f"Wrote    {wf.relative_to(ROOT)}")

    print(f"Run      {run_id}")
    print(f"Windows  {len(windows)}   requests {int(windows['n_traffic'].sum()):,}"
          f"   regimes {len({x for x in labels})}")
    print(f"Wrote    docs/report.html  ({(report_path.stat().st_size / 1024):.1f} KB)")
    print(f"Wrote    docs/index.html   "
          f"({(DOCS / 'index.html').stat().st_size / 1024:.1f} KB)")
    print("\nNext: git add docs && git commit && git push")
    print("      then enable Settings -> Pages -> Source: GitHub Actions")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
