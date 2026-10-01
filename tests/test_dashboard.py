"""
Dashboard smoke tests
---------------------
Streamlit scripts are ordinary Python that runs on page load, which means a
typo in one is invisible to `pytest` and invisible to a health check — the
server reports "ok" while every page shows a traceback. That is exactly how a
one-character bug (`pivot(run_id)` instead of `pivot(store, run_id)`) shipped
and was only found by opening the browser.

`AppTest` runs the real script headlessly against a temporary telemetry store,
so a broken dashboard now fails CI instead of failing silently in a demo.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from config import settings
from src.storage.telemetry import MetricPoint, TelemetryStore
from src.utils.logging import setup_logging

setup_logging("WARNING")

pytest.importorskip("streamlit", reason="dashboard deps not installed")

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture()
def seeded_store(tmp_path, monkeypatch):
    """A small but complete run: windows, metrics, a decision and an incident."""
    db = tmp_path / "telemetry.db"
    store = TelemetryStore(db)

    run = store.start_run(encoder="enc/test", judge="mock/judge",
                          app_model="champion-test", dataset_source="bundled")
    for i, label in enumerate(["baseline", "baseline", "style_shift", "new_intents",
                               "mixed_crisis", "recovery"]):
        handle = store.open_window(run, i, label)
        store.log_metrics(handle, [
            MetricPoint("embedding", "domain_classifier_auc", 0.52 + 0.04 * i,
                        0.62, 0.78, "none" if i < 2 else "moderate", "auc"),
            MetricPoint("embedding", "mmd2", 0.001 + 0.002 * i, 0.003, 0.015, "none", "mmd2"),
            MetricPoint("embedding", "ood_rate", 0.04 + 0.05 * i, 0.25, 0.50, "none", "fraction"),
            MetricPoint("embedding", "concept_gap", 0.0 if i < 3 else 0.5, 0.2, 0.4,
                        "none", "accuracy"),
            MetricPoint("quality", "accuracy", 0.97 - 0.08 * i, None, None, "none", "accuracy"),
            MetricPoint("quality", "macro_f1", 0.95 - 0.07 * i, None, None, "none", "f1"),
            MetricPoint("quality", "ece", 0.04 + 0.05 * i, 0.06, 0.12, "none", "ece"),
            MetricPoint("quality", "brier", 0.06 + 0.02 * i, 0.15, 0.22, "none", "brier"),
            MetricPoint("quality", "abstention_rate", 0.02, 0.15, 0.30, "none", "fraction"),
            MetricPoint("quality", "label_coverage", 0.5, None, None, "none", "fraction"),
            MetricPoint("quality", "confidence_mean", 0.9, None, None, "none", "confidence"),
            MetricPoint("quality", "proxy_accuracy", 0.9, None, None, "none", "accuracy"),
            MetricPoint("quality", "accuracy_in_scope", 0.95, None, None, "none", "accuracy"),
            MetricPoint("quality", "accuracy_out_of_scope", 0.0, None, None, "none", "accuracy"),
            MetricPoint("judge", "judge_score", 0.8 - 0.03 * i, 0.85, None, "none", "weighted_0_1"),
            MetricPoint("judge", "judge_correctness_rate", 0.8, None, None, "none", "fraction"),
            MetricPoint("judge", "judge_veto_rate", 0.1, None, None, "none", "fraction"),
            MetricPoint("judge", "judge_pairwise_win_rate", 0.5, None, None, "none", "fraction"),
            MetricPoint("judge", "judge_dim::task_correctness", 4.5, 4.6, None, "none", "score_1_5"),
            MetricPoint("judge", "judge_dim::groundedness", 4.4, 4.7, None, "none", "score_1_5"),
            MetricPoint("judge", "judge_dim::tone", 4.8, 4.7, None, "none", "score_1_5"),
            MetricPoint("tabular", "psi_mean", 0.05 + 0.02 * i, 0.10, 0.25, "none", "psi"),
            MetricPoint("tabular", "psi::text_length", 0.06, 0.10, 0.25, "none", "psi"),
            MetricPoint("volume", "requests", 100 + 10 * i, None, None, "none", "count"),
        ])
        store.log_judge_scores(handle, [{
            "sample_hash": 1, "sample_snippet": f"card question {i}",
            "dimension": "groundedness", "score": 4.0,
            "rationale": "Playbook-faithful.", "judge_model": "mock", "rubric_version": "v1",
            "in_scope": 1, "verdict": "pass",
        }])
        store.log_traffic(handle, [{
            "request_id": f"r{i}", "text_hash": 1000 + i, "text_snippet": f"card question {i}",
            "in_scope": 1 if i < 3 else 0, "gold_intent": "card_arrival" if i < 3 else None,
            "pred_intent": "card_arrival", "confidence": 0.9,
            "abstained": 0, "response": "Your card arrives in 3-5 working days.",
            "app_latency_ms": 12.0,
        }])
        store.log_decision(handle, "investigate" if i >= 3 else "noop",
                           "severe" if i >= 4 else "none", 40.0 if i >= 4 else 95.0,
                           {"fired": ["quality:accuracy:severe"]} if i >= 3 else {},
                           "quality regression" if i >= 3 else "all clear",
                           confidence=0.8)
        store.close_window(handle, 100, 50)

    incident = store.open_incident(run, "run", "severe", "Quality regression",
                                   {"fired": ["quality:accuracy:severe"]})
    store.append_incident_timeline(incident, "observed", {"window": 3})
    store.finish_run(run)

    monkeypatch.setattr(settings, "TELEMETRY_DB", db)
    monkeypatch.setattr("src.storage.telemetry.settings.TELEMETRY_DB", db)
    return run


def _run_dashboard():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(ROOT / "dashboard" / "app.py"), default_timeout=180)
    at.run()
    return at


class TestDashboardRenders:
    def test_script_executes_without_exception(self, seeded_store):
        """The whole point: a dashboard that 500s on load must fail here."""
        at = _run_dashboard()
        assert not at.exception, [str(e.value) for e in at.exception]

    def test_header_and_kpis_render(self, seeded_store):
        at = _run_dashboard()
        body = " ".join(str(m.value) for m in at.markdown)
        assert "LLM Drift & Quality Monitor" in body
        labels = [str(m.label) for m in at.metric]
        assert {"Health", "Decision", "Domain AUC", "Accuracy", "Judge"} <= set(labels)

    def test_all_five_tabs_present(self, seeded_store):
        at = _run_dashboard()
        assert [t.label for t in at.tabs] == [
            "Overview", "Embedding drift", "Output quality", "LLM-as-judge", "Run detail"]

    def test_run_selector_lists_the_run(self, seeded_store):
        at = _run_dashboard()
        assert at.selectbox
        # The label shows the run id without its "run_" prefix for readability.
        assert any(seeded_store[4:] in str(opt) for opt in at.selectbox[0].options)

    def test_charts_are_built(self, seeded_store):
        at = _run_dashboard()
        assert len(at.get("plotly_chart")) >= 8, "expected charts on the default tab set"

    def test_dataframes_render(self, seeded_store):
        at = _run_dashboard()
        assert len(at.dataframe) >= 1

    def test_incident_is_surfaced(self, seeded_store):
        at = _run_dashboard()
        body = " ".join(str(m.value) for m in at.markdown)
        assert "INCIDENT" in body.upper() or "Quality regression" in body


class TestDashboardResilience:
    def test_empty_store_does_not_crash(self, tmp_path, monkeypatch):
        """A fresh checkout with no telemetry must render, not traceback."""
        db = tmp_path / "empty.db"
        TelemetryStore(db)  # create the schema
        monkeypatch.setattr(settings, "TELEMETRY_DB", db)
        monkeypatch.setattr("src.storage.telemetry.settings.TELEMETRY_DB", db)
        at = _run_dashboard()
        assert not at.exception, [str(e.value) for e in at.exception]
        assert at.error, "the dashboard should say it has no runs, not fail silently"

    def test_single_window_run_renders(self, tmp_path, monkeypatch):
        db = tmp_path / "one.db"
        store = TelemetryStore(db)
        run = store.start_run(encoder="e", judge="j")
        handle = store.open_window(run, 0, "baseline")
        store.log_metrics(handle, [MetricPoint("embedding", "domain_classifier_auc", 0.5)])
        store.log_decision(handle, "noop", "none", 100.0, {}, "fine")
        store.close_window(handle, 10, 0)
        monkeypatch.setattr(settings, "TELEMETRY_DB", db)
        monkeypatch.setattr("src.storage.telemetry.settings.TELEMETRY_DB", db)
        at = _run_dashboard()
        assert not at.exception, [str(e.value) for e in at.exception]
