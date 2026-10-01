#!/usr/bin/env python3
"""
Judge regression suite as a standalone command.

    python scripts/eval_judge.py                          # current configured judge
    python scripts/eval_judge.py --model qwen3:14b        # evaluate a candidate
    python scripts/eval_judge.py --show-baseline          # print the stored baseline
    python scripts/eval_judge.py --rebaseline             # accept the current results

Run this in CI. It is the guard that stops a judge upgrade from silently
changing what your dashboard means.

Exit codes: 0 pass, 1 regression, 2 not comparable (signature changed),
3 the judge backend was unavailable.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import settings  # noqa: E402
from src.app.assistant import PLAYBOOKS  # noqa: E402
from src.judge import DEFAULT_SUITE, JudgeRegressionSuite, LLMBasedJudge  # noqa: E402
from src.judge.agreement import SUITE_PATH  # noqa: E402
from src.llm import resolve_auto  # noqa: E402
from src.utils.logging import setup_logging  # noqa: E402

setup_logging()

BAR = "-" * 84


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate the LLM judge against the golden suite")
    parser.add_argument("--model", default=settings.JUDGE_MODEL)
    parser.add_argument("--provider", default=settings.JUDGE_PROVIDER)
    parser.add_argument("--n", type=int, default=0, help="Run only the first N cases")
    parser.add_argument("--rebaseline", action="store_true",
                        help="Overwrite the stored baseline with the current results")
    parser.add_argument("--show-baseline", action="store_true",
                        help="Print the stored baseline and exit")
    parser.add_argument("--workers", type=int, default=settings.JUDGE_MAX_WORKERS)
    args = parser.parse_args()

    if args.show_baseline:
        if not SUITE_PATH.exists():
            print("No stored baseline. Run without --show-baseline first.")
            return 3
        print(SUITE_PATH.read_text(encoding="utf-8"))
        return 0

    print(BAR)
    print("LLM JUDGE REGRESSION SUITE")
    print(BAR)

    llm = resolve_auto(args.provider if args.provider != "auto" else None)
    if llm.provider == "mock":
        print("\nNo real judge backend available — using the deterministic mock.")
        print("This validates the pipeline, not the judge. Start Ollama or set a key.\n")
    judge = LLMBasedJudge(llm=llm, max_workers=args.workers)
    print(f"backend   {judge.signature}")
    print(f"rubric    {settings.JUDGE_RUBRIC_VERSION}")
    print(f"cases     {len(DEFAULT_SUITE)}")
    print()

    suite = JudgeRegressionSuite(judge, cases=DEFAULT_SUITE[: args.n] if args.n else DEFAULT_SUITE,
                                 playbooks=PLAYBOOKS)
    report = suite.run(persist_baseline=args.rebaseline)

    print()
    print(f"passed      {report.n_passed}/{report.n_cases}")
    print(f"mean score  {report.mean_score:.4f}")
    print(f"duration    {report.duration_ms / 1000:.1f}s")
    print()
    for dim, mean in sorted(report.dimension_means.items()):
        delta = report.dimension_deltas.get(dim)
        delta_txt = f"  ({delta:+.3f} vs baseline)" if delta is not None else ""
        print(f"  {dim:20s} {mean:.2f}{delta_txt}")

    if report.failed_cases:
        print("\nFAILED CASES")
        for case in report.failed_cases:
            print(f"  {case.case_id:24s} score={case.score:.3f} min={case.expected_min} "
                  f"veto={case.vetoed or 'none'}")
            if case.rationale:
                print(f"    {case.rationale[:150]}")

    if report.flipped_cases:
        print("\nFLIPPED SINCE BASELINE")
        for flip in report.flipped_cases:
            print(f"  {flip['case_id']:24s} {flip['was_passed']} -> {flip['now_passed']}")

    print()
    print(report.verdict)

    if report.regression:
        return 1
    if "not comparable" in report.verdict.lower():
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())