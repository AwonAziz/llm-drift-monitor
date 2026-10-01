#!/usr/bin/env python3
"""
Export a report for an existing run.

The demo writes reports as its last step; this is for the case where you want
one without re-running anything — after a manual retrain, for a teammate, or to
attach evidence to a ticket.

    python scripts/export_report.py                     # latest run
    python scripts/export_report.py --run run_2026...   # a specific run
    python scripts/export_report.py --out ./evidence    # somewhere else
    python scripts/export_report.py --open             # open the HTML afterwards
"""

from __future__ import annotations

import argparse
import json
import sys
import webbrowser
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import settings  # noqa: E402
from src.reporting import write_reports  # noqa: E402
from src.storage.telemetry import TelemetryStore  # noqa: E402
from src.utils.logging import setup_logging  # noqa: E402

setup_logging()


def main() -> int:
    parser = argparse.ArgumentParser(description="Export drift reports for a monitoring run")
    parser.add_argument("--run", default=None, help="Run id (default: the most recent run)")
    parser.add_argument("--out", default=None, help="Output directory (default: data/reports)")
    parser.add_argument("--open", action="store_true", help="Open the HTML report in a browser")
    args = parser.parse_args()

    store = TelemetryStore()
    run_id = args.run or store.latest_run_id()
    if not run_id:
        print("No runs found. Run: python scripts/demo_shift.py --fast")
        return 1

    run = store.get_run(run_id)
    if run is None:
        print(f"No run '{run_id}'. Available:")
        for row in store.list_runs(20).itertuples():
            print(f"  {row.run_id}  {row.status}  {row.started_at}")
        return 2

    calibration = _load_calibration()
    suite = _load_suite()

    paths = write_reports(store, run_id, out_dir=args.out,
                          calibration=calibration, suite=suite)
    print(f"Run      {run_id} ({run.get('status')})")
    print(f"Encoder  {run.get('encoder')}")
    print(f"Judge    {run.get('judge')}")
    print(f"Windows  {len(store.list_windows(run_id))}")
    print()
    for path in (paths.html, paths.markdown, paths.json):
        print(f"  {path}")
    if args.open and paths.html.exists():
        webbrowser.open(paths.html.resolve().as_uri())
    return 0


def _load_calibration() -> dict | None:
    path = settings.ARTIFACT_DIR / "bootstrap_summary.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8")).get("calibration")
    except Exception:  # noqa: BLE001
        return None


def _load_suite() -> dict | None:
    path = settings.ARTIFACT_DIR / "judge_regression.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None


if __name__ == "__main__":
    raise SystemExit(main())