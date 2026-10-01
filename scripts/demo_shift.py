#!/usr/bin/env python3
"""
Production shift demo
---------------------
The headline demonstration. Fourteen evaluation windows of a live support
assistant, through a scripted production shift, with the monitor watching the
whole time.

The story, which is a real one rather than a contrived one:

  windows 1-3   baseline. Retail card and ATM traffic, as launched.
  window  4     a marketing campaign floods the queue. Same mix, more volume.
                Nothing should fire — and nothing does, which is the point:
                a detector that fires on volume alone gets muted in a week.
  windows 5-7   a new partner channel starts routing traffic. Terse, lowercase,
                emoji, no politeness. *Style* drift with the same intents.
                Embedding drift fires; accuracy holds. The correct action is to
                widen coverage, not to retrain.
  windows 8-10  remittance and virtual-card products ship. 45% of traffic is
                for intents the model has never seen. Out-of-distribution rate
                goes to ~45%, accuracy halves, the judge score drops, and an
                incident opens.
  windows 11-12 the worst of it: out-of-scope traffic, degraded phrasing and
                5% label noise from the annotation backlog. Severity severe.
  window  13-14 recovery: the replacement model (trained with the new intents
                in scope) is promoted mid-run and the incident closes.

Usage
-----
    python scripts/demo_shift.py                    # full demo
    python scripts/demo_shift.py --fast             # no LLM, smaller windows
    python scripts/demo_shift.py --judge-model qwen3:1.7b
    python scripts/demo_shift.py --no-promote       # show that it does not self-heal
    python scripts/demo_shift.py --regime new_intents --windows 3
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.utils.logging import setup_logging  # noqa: E402

setup_logging()

from config import settings  # noqa: E402
from src.app.assistant import PLAYBOOKS  # noqa: E402
from src.data.datasets import load_banking77  # noqa: E402
from src.data.shifts import Regime, TrafficSimulator  # noqa: E402
from src.judge import JudgeRegressionSuite, LLMBasedJudge  # noqa: E402
from src.llm import build_llm, resolve_auto  # noqa: E402
from src.monitoring import DriftMonitor, build_artifacts, calibration_report  # noqa: E402
from src.reporting import write_reports  # noqa: E402
from src.storage.telemetry import TelemetryStore  # noqa: E402

BAR = "=" * 96


def _fmt(v, n=3, dash="-"):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return dash
    return f"{v:.{n}f}"


def _bar(value: float, lo: float, hi: float, width: int = 12) -> str:
    if value is None or not np.isfinite(value) or hi <= lo:
        return " " * width
    filled = int(round(width * min(1.0, max(0.0, (value - lo) / (hi - lo)))))
    return "#" * filled + "." * (width - filled)


def _severity_chip(name: str, sev: str) -> str:
    icon = {"severe": "SEV", "moderate": "mod", "none": "ok "}.get(sev, "ok ")
    return f"{name}={icon}"


def _chip(sev: str, sub: str = "") -> str:
    """Severity chip; the parenthetical shows the raw worst signal when the
    detector stayed below its confirmation threshold."""
    icon = {"severe": "SEV", "moderate": "mod", "none": "ok "}.get(sev, "ok ")
    suffix = f" (worst signal {sub})" if sub and sub != "none" else ""
    return f"{icon}{suffix}"


def header(text: str) -> None:
    print("\n" + BAR)
    print(f"  {text}")
    print(BAR)


def main() -> int:
    parser = argparse.ArgumentParser(description="Simulated production shift demo")
    parser.add_argument("--fast", action="store_true",
                        help="No LLM calls, smaller windows - a 60-second structural demo")
    parser.add_argument("--no-llm", action="store_true",
                        help="Templated responses but keep the judge (needs a judge backend)")
    parser.add_argument("--mock-judge", action="store_true",
                        help="Use the deterministic mock judge instead of a real model")
    parser.add_argument("--judge-model", default=settings.JUDGE_MODEL)
    parser.add_argument("--app-model", default=settings.LLM_MODEL)
    parser.add_argument("--window-size", type=int, default=settings.WINDOW_SIZE)
    parser.add_argument("--judge-sample", type=int, default=settings.JUDGE_SAMPLE_SIZE)
    parser.add_argument("--response-sample", type=int, default=settings.LLM_RESPONSE_SAMPLE)
    parser.add_argument("--max-workers", type=int, default=settings.JUDGE_MAX_WORKERS)
    parser.add_argument("--no-promote", action="store_true",
                        help="Skip the mid-run retrain so the incident stays open")
    parser.add_argument("--regime", default=None, help="Run a single regime")
    parser.add_argument("--windows", type=int, default=0, help="Limit window count")
    parser.add_argument("--reset", action="store_true", help="Clear the telemetry store first")
    parser.add_argument("--embedding-backend", default=settings.EMBEDDING_BACKEND)
    args = parser.parse_args()

    if args.fast:
        args.no_llm = True
        args.mock_judge = True
        args.window_size = min(args.window_size, 120)
        args.judge_sample = min(args.judge_sample, 6)
        args.response_sample = 0

    t_start = time.perf_counter()

    # ── 0. Load or build the champion ───────────────────────────────
    header("STEP 0  Bootstrap the monitored system")
    try:
        artifacts = build_artifacts(n_reference=settings.REFERENCE_SIZE,
                                    encoder_backend=args.embedding_backend)
    except Exception as exc:  # noqa: BLE001
        print(f"Bootstrap failed: {exc}")
        return 1

    calibration = calibration_report(artifacts)
    for key in ("intents", "validation_rows_in_scope", "validation_rows_out_of_scope",
                "accuracy", "macro_f1", "auc", "ece", "brier", "abstention_rate", "temperature"):
        if key in calibration:
            value = calibration[key]
            print(f"  {key:32s} {value if isinstance(value, (str, int)) else round(value, 4)}")
    print(f"  {'out_of_scope_accuracy':32s} {_fmt(calibration.get('out_of_scope_accuracy'), 4)}")
    print(f"  {'encoder':32s} {artifacts.encoder.signature}")

    # ── 1. Judge baseline ───────────────────────────────────────────
    header("STEP 1  Validate the judge before trusting it")
    suite_payload = None
    if not args.no_llm:
        judge_llm = resolve_auto("ollama")
        judge = LLMBasedJudge(llm=judge_llm, baseline_score=None)
        suite = JudgeRegressionSuite(judge, playbooks=PLAYBOOKS)
        suite_report = suite.run(persist_baseline=True)
        suite_payload = suite_report.to_dict()
        print(f"  judge backend        {judge.signature}")
        print(f"  golden cases         {suite_report.n_passed}/{suite_report.n_cases} passed")
        print(f"  mean rubric score    {_fmt(suite_report.mean_score)}")
        print(f"  verdict              {suite_report.verdict}")
        failing = [c.case_id for c in suite_report.failed_cases]
        if failing:
            print(f"  failing              {', '.join(failing)}")
        print(f"  duration             {suite_report.duration_ms / 1000:.1f}s")
    else:
        print("  --no-llm: skipping the judge suite (responses and scores will be placeholders)")

    # ── 2. Judge baseline score on reference traffic ────────────────
    # The judge's regression test needs a score to regress *from*. We establish
    # it by judging reference traffic, exactly as you would in production.
    app_llm = build_llm("mock" if args.no_llm else "ollama",
                        model=args.app_model if not args.no_llm else "mock-responder-v1")
    judge_llm = build_llm("mock") if args.mock_judge else resolve_auto("ollama")
    monitor = DriftMonitor(
        artifacts, TelemetryStore(), llm=app_llm, use_llm=not args.no_llm,
        judge=LLMBasedJudge(llm=judge_llm, baseline_score=None,
                            sample_size=args.judge_sample, max_workers=args.max_workers),
        response_sample=args.response_sample, judge_sample=args.judge_sample,
        max_workers=args.max_workers, run_notes="demo_shift")
    monitor.expected_volume = args.window_size
    if not args.no_llm:
        monitor.judge.sample_size = args.judge_sample

    print("\n  Establishing the judge baseline on reference traffic...")
    ref_frame = artifacts.reference_df.sample(n=min(80, len(artifacts.reference_df)),
                                              random_state=101).reset_index(drop=True)
    ref_frame = pd.concat(
        [ref_frame, monitor.assistant.classify_frame(ref_frame, "text").reset_index(drop=True)],
        axis=1)
    if args.no_llm:
        from src.app.assistant import templated_response
        ref_frame["response"] = [
            templated_response(str(p), PLAYBOOKS.get(str(p), PLAYBOOKS["default"]))
            for p in ref_frame["pred_intent"]
        ]
    else:
        from src.app.assistant import templated_response as _template
        from src.utils.parallel import ordered_map

        def _reply(pair):
            _, row = pair
            out = monitor.assistant.handle(str(row["text"]))
            return out.response or _template(str(row["pred_intent"]),
                                             PLAYBOOKS.get(str(row["pred_intent"]), PLAYBOOKS["default"]))

        replies = ordered_map(_reply, list(ref_frame.iterrows()),
                              max_workers=args.max_workers, label="baseline replies")
        if len(replies) != len(ref_frame):
            raise RuntimeError(f"only {len(replies)}/{len(ref_frame)} baseline replies were produced")
        ref_frame["response"] = replies
    ref_frame["in_scope"] = True
    baseline_judge = monitor.judge.evaluate_window(ref_frame, window_index=-1,
                                                    n=args.judge_sample, seed=99,
                                                    playbooks=PLAYBOOKS)
    monitor.judge.baseline_score = baseline_judge.mean_score
    monitor.judge.baseline_score_se = max(baseline_judge.score_se, 0.01)
    monitor.judge.baseline_dimension_scores = dict(baseline_judge.dimension_means)
    print(f"  judge baseline score {_fmt(baseline_judge.mean_score)} "
          f"(SE {_fmt(baseline_judge.score_se, 4)}, n={baseline_judge.n_judged})")
    print("  per-dimension:")
    for dim, val in baseline_judge.dimension_means.items():
        print(f"    {dim:22s} {_fmt(val, 2)}")

    # ── 3. Run the schedule ──────────────────────────────────────────
    header("STEP 2  Run the production schedule")
    df, info = load_banking77()
    simulator = TrafficSimulator(df, base_window_size=args.window_size)

    if args.regime:
        schedule = [r for r in _all_regimes() if r.name == args.regime]
        if not schedule:
            print(f"Unknown regime '{args.regime}'. Available: "
                  f"{', '.join(r.name for r in _all_regimes())}")
            return 2
    else:
        schedule = _all_regimes()

    if args.reset:
        monitor.store.reset()
    monitor.start_run()
    print(f"  run_id: {monitor.run_id}")
    print(f"  {len(schedule)} regimes, {sum(r.n_windows for r in schedule)} windows, "
          f"~{args.window_size * sum(r.n_windows for r in schedule)} requests\n")

    promoted = False
    promotion_window = 11
    results = []

    for emitted, window in enumerate(simulator.iter_schedule(schedule)):
        if args.windows and emitted >= args.windows:
            break
        result = monitor.process_window(window)
        results.append(result)
        _print_window(result)

        if (not args.no_promote) and not promoted and window.window_index >= promotion_window:
            header("STEP 3  Incident response: retrain and promote")
            promoted = True
            _promote(monitor, artifacts, schedule, df, promotion_window)

    monitor.finish_run()

    # ── 4. Report ────────────────────────────────────────────────────
    header("STEP 4  Report")
    paths = write_reports(monitor.store, monitor.run_id,
                          calibration=calibration, suite=suite_payload)
    _print_summary(monitor, results, paths, calibration, suite_payload, promoted,
                   time.perf_counter() - t_start)
    return 0


def _all_regimes() -> list[Regime]:
    from src.data.shifts import default_schedule
    return default_schedule()


def _print_window(result: dict) -> None:
    d = result["decision"]
    e = result["embedding"]
    q = result["quality"]
    j = result["judge"]
    w = result["window"]
    idx = w.window_index
    label = w.label

    print(f"\n  [{idx:02d}] {label:14s} {w.notes}")
    print(f"       drift   auc={_fmt(e.domain_auc)} mmd2={_fmt(e.mmd2, 4)} ood={_fmt(e.ood_rate)} "
          f"gap={_fmt(e.concept_gap)} [{_chip(e.severity if e.drift_detected else 'none', e.severity if not e.drift_detected else '')}]")
    print(f"       quality acc={_fmt(q.accuracy)} f1={_fmt(q.macro_f1)} ece={_fmt(q.ece)} "
          f"abs={_fmt(q.abstention_rate)} labels={q.n_labeled}/{q.n_requests} "
          f"[{_severity_chip('qual', q.severity)}]")
    print(f"       judge   score={_fmt(j.mean_score)} (base {_fmt(j.baseline_score)}, "
          f"z={_fmt(j.score_z, 1)}) veto={_fmt(j.veto_rate)} n={j.n_judged} "
          f"[{_severity_chip('judge', j.severity)}]")
    print(f"       DECISION {d.action.upper():12s} health={d.health_score:5.1f} "
          f"conf={d.confidence:.2f}")
    if d.signals.get("fired"):
        print(f"       signals  {', '.join(d.signals['fired'][:6])}")
    if result.get("incident_id"):
        print(f"       INCIDENT {result['incident_id']}")
    print(f"       ({result['elapsed_s']:.1f}s)")


def _promote(monitor: DriftMonitor, artifacts, schedule, df, window_index: int) -> None:
    """Retrain with the new intents in scope, then re-point the monitor at it."""
    from src.monitoring.monitor import build_artifacts as rebuild

    print("  Rebuilding the champion with the out-of-scope intents added to the label set...")
    expanded = rebuild(train_on_expanded_scope=True,
                       version=f"champion-v2-{time.strftime('%H%M%S')}")
    print(f"  New model: {expanded.classifier.signature}")
    print(f"  New intents: {len(expanded.classifier.classes)} "
          f"(was {len(artifacts.classifier.classes)})")

    old_acc = artifacts.quality_baseline.get("accuracy")
    new_acc = expanded.quality_baseline.get("accuracy")
    print(f"  In-scope validation accuracy: {_fmt(old_acc)} -> {_fmt(new_acc)}")
    print(f"  Out-of-scope accuracy:       {_fmt(artifacts.quality_baseline.get('oos_accuracy'))} -> "
          f"{_fmt(expanded.quality_baseline.get('oos_accuracy'))}")

    monitor.art.classifier = expanded.classifier
    monitor.assistant.classifier = expanded.classifier
    print("  Promotion complete. Continuing the schedule with the new champion.\n")


def _print_summary(monitor, results, paths, calibration, suite, promoted, elapsed) -> None:
    store = monitor.store
    run_id = monitor.run_id
    decisions = store.decisions(run_id)
    incidents = store.incidents(run_id)

    print(f"  windows evaluated   {len(results)}")
    print(f"  total requests      {sum(len(r['window'].frame) for r in results)}")
    print(f"  labelled rows       {sum(r['quality'].n_labeled for r in results)}")
    print(f"  judge calls         {sum(r['judge'].n_judged for r in results)}")
    if not decisions.empty:
        counts = decisions["action"].value_counts().to_dict()
        print(f"  decisions           {counts}")
        open_n = int((incidents["status"] == "open").sum()) if "status" in incidents.columns else 0
        closed_n = int((incidents["status"] == "closed").sum()) if "status" in incidents.columns else 0
        print(f"  incidents           {open_n} open, {closed_n} closed")
    print(f"  wall clock          {elapsed:.1f}s")

    # Narrative
    print("\n  What the platform caught:")
    first_drift = next((r for r in results if r["embedding"].drift_detected), None)
    first_quality = next((r for r in results if r["quality"].drift_detected), None)
    first_judge = next((r for r in results if r["judge"].regression), None)
    if first_drift:
        print(f"    - embedding drift at window {first_drift['window'].window_index} "
              f"({first_drift['window'].label}): AUC {_fmt(first_drift['embedding'].domain_auc)}, "
              f"OOD {_fmt(first_drift['embedding'].ood_rate)}")
    if first_quality:
        print(f"    - output quality regression at window {first_quality['window'].window_index}: "
              f"accuracy {_fmt(first_quality['quality'].accuracy)}, "
              f"ECE {_fmt(first_quality['quality'].ece)}")
    if first_judge:
        print(f"    - judge regression at window {first_judge['window'].window_index}: "
              f"score {_fmt(first_judge['judge'].mean_score)} vs "
              f"{_fmt(first_judge['judge'].baseline_score)} (z={_fmt(first_judge['judge'].score_z, 1)})")
    if promoted:
        print("    - retrain promoted at window 11; recovery visible in the final windows")

    print("\n  Artefacts:")
    print(f"    {paths.html}")
    print(f"    {paths.markdown}")
    print(f"    {paths.json}")
    print(f"    telemetry db       {store.path}")
    print("    dashboard          streamlit run dashboard/app.py")
    print(f"    API                uvicorn src.api.server:app --port {settings.API_PORT}")


if __name__ == "__main__":
    raise SystemExit(main())