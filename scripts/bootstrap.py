#!/usr/bin/env python3
"""
Bootstrap
---------
One-time setup: fetch the dataset, train the champion, freeze the reference
snapshot and quality baseline, and validate the judge against its golden suite.

    python scripts/bootstrap.py                  # champion-v1, in-scope intents only
    python scripts/bootstrap.py --expanded       # champion-v2, new intents in scope
    python scripts/bootstrap.py --no-llm         # skip the judge suite

``--expanded`` builds the *replacement* model used by the demo's recovery
stage: trained on the reference plus the intents that only appear in production
after the shift. Keeping the two champions comparable is what lets the demo
show that retraining fixed the incident rather than that the traffic went away.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.logging import setup_logging  # noqa: E402

setup_logging()

from config import settings  # noqa: E402
from src.judge import JudgeRegressionSuite, LLMBasedJudge  # noqa: E402
from src.llm import resolve_auto  # noqa: E402
from src.monitoring import calibration_report  # noqa: E402
from src.monitoring.monitor import build_artifacts  # noqa: E402
from src.utils.stats import json_safe  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Bootstrap the drift monitor")
    parser.add_argument("--expanded", action="store_true",
                        help="Train the replacement model covering the out-of-scope intents too")
    parser.add_argument("--reference-size", type=int, default=settings.REFERENCE_SIZE)
    parser.add_argument("--validation-size", type=int, default=400)
    parser.add_argument("--embedding-backend", default=settings.EMBEDDING_BACKEND)
    parser.add_argument("--version", default=None)
    parser.add_argument("--no-llm", action="store_true", help="Skip the LLM judge regression suite")
    parser.add_argument("--suite-cases", type=int, default=0,
                        help="Run only the first N golden cases (0 = all)")
    parser.add_argument("--force-suite-baseline", action="store_true",
                        help="Overwrite the stored judge baseline even if the suite regressed")
    args = parser.parse_args()

    t0 = time.perf_counter()
    print("\n" + "=" * 72)
    print("LLM DRIFT MONITOR - BOOTSTRAP")
    print("=" * 72)

    artifacts = build_artifacts(
        n_reference=args.reference_size,
        n_validation=args.validation_size,
        encoder_backend=args.embedding_backend,
        train_on_expanded_scope=args.expanded,
        version=args.version,
    )

    calibration = calibration_report(artifacts)
    print("\nChampion validation")
    for k, v in calibration.items():
        print(f"  {k:20s} {v if isinstance(v, str) else (f'{v:.4f}' if isinstance(v, float) else v)}")

    suite_payload = None
    if not args.no_llm:
        from src.app.assistant import PLAYBOOKS
        from src.judge.agreement import DEFAULT_SUITE

        llm = resolve_auto(settings.JUDGE_PROVIDER)
        judge = LLMBasedJudge(llm=llm, baseline_score=None)
        cases = DEFAULT_SUITE[: args.suite_cases] if args.suite_cases else DEFAULT_SUITE
        suite = JudgeRegressionSuite(judge, cases=cases, playbooks=PLAYBOOKS)
        report = suite.run(persist_baseline=args.force_suite_baseline)
        suite_payload = report.to_dict()

        print("\nJudge regression suite")
        print(f"  backend          {judge.signature}")
        print(f"  passed           {report.n_passed}/{report.n_cases}")
        print(f"  mean score       {report.mean_score:.3f}")
        print(f"  verdict          {report.verdict}")
        for case in report.failed_cases:
            print(f"    FAIL {case.case_id:24s} score={_fmt(case.score)} min={case.expected_min} veto={case.vetoed}")
        if suite_payload and report.flipped_cases:
            print("  flipped since baseline:")
            for f in report.flipped_cases:
                print(f"    {f['case_id']}: {f['was_passed']} -> {f['now_passed']}")

    summary = {
        "calibration": calibration,
        "judge_suite": suite_payload,
        "artifacts": artifacts.summary(),
        "bootstrap_seconds": round(time.perf_counter() - t0, 1),
    }
    (settings.ARTIFACT_DIR / "bootstrap_summary.json").write_text(
        json.dumps(json_safe(summary), indent=2), encoding="utf-8")

    print(f"\nArtifacts written to {settings.ARTIFACT_DIR}")
    print(f"  reference snapshot : {settings.REFERENCE_SNAPSHOT.name}")
    print(f"  classifier         : {artifacts.classifier.version} "
          f"({len(artifacts.classifier.classes)} intents)")
    print("  run manifest       : run_manifest.json")
    print(f"\nBootstrap complete in {summary['bootstrap_seconds']}s.")
    print("\nNext:")
    print("  python scripts/demo_shift.py          # the full simulated production shift")
    print("  uvicorn src.api.server:app --port 8000 # serving + monitoring API")
    print("  streamlit run dashboard/app.py        # dashboard")
    return 0


def _fmt(v):
    return "n/a" if v is None or v != v else f"{v:.3f}"


if __name__ == "__main__":
    raise SystemExit(main())