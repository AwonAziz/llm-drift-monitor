"""End-to-end: a full simulated production shift through the monitor.

This is the test that would catch a wiring regression between the detectors,
the orchestrator and the telemetry store — the kind that unit tests miss and
that only shows up when you run the demo at 6pm on a Friday.

The whole suite runs on the hashing encoder and the mock judge, so it needs no
model download and no LLM. The assertions are about the *shape of the story*:
clean baseline, drift on a changed mix, quality collapse when out-of-scope
traffic arrives, and a decision that never recommends a retrain on input drift
alone.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from config import settings
from src.data.shifts import Regime, TrafficSimulator
from src.monitoring.monitor import DriftMonitor, build_artifacts, text_features
from src.reporting import assemble, render_html, render_markdown, write_reports
from src.storage.telemetry import TelemetryStore

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def artifacts():
    return build_artifacts(n_reference=700, n_validation=300,
                           encoder_backend="hashing", version="it-v1")


@pytest.fixture(scope="module")
def monitor(artifacts, tmp_path_factory):
    from src.judge import LLMBasedJudge
    from src.llm import build_llm

    store = TelemetryStore(tmp_path_factory.mktemp("e2e") / "telemetry.db")
    mon = DriftMonitor(
        artifacts, store, llm=build_llm("mock"), use_llm=True,
        judge=LLMBasedJudge(llm=build_llm("mock"), baseline_score=0.80,
                            baseline_score_se=0.05, sample_size=6, max_workers=1),
        response_sample=6, judge_sample=6, max_workers=1, run_notes="pytest",
    )
    mon.expected_volume = 120
    return mon


@pytest.fixture(scope="module")
def dataset_frame():
    from src.data.datasets import load_banking77

    df, _ = load_banking77()
    return df


SHORT_SCHEDULE = [
    Regime("baseline", 2, 0.0, "polished", description="calm"),
    Regime("new_intents", 2, 0.50, "polished", description="new products ship"),
]


@pytest.fixture(scope="module")
def results(monitor, dataset_frame):
    simulator = TrafficSimulator(dataset_frame, base_window_size=120)
    monitor.start_run()
    out = [monitor.process_window(w) for w in simulator.iter_schedule(SHORT_SCHEDULE)]
    monitor.finish_run()
    return out


class TestTextFeatures:
    def test_shape_and_types(self):
        frame = text_features(["Hello, where is my card?", "cash machine ate it 🙏"])
        assert len(frame) == 2
        assert set(frame.columns) >= {"text_length", "word_count", "unique_word_ratio",
                                      "digit_ratio", "upper_ratio", "emoji_flag",
                                      "is_question", "has_currency"}
        assert frame.select_dtypes("number").shape == frame.shape

    def test_handles_empty_and_unicode(self):
        frame = text_features(["", "🙂"])
        assert np.isfinite(frame.to_numpy()).all()

    def test_emoji_is_detected(self):
        assert bool(text_features(["great 👀"]).iloc[0]["emoji_flag"])
        assert not bool(text_features(["great"]).iloc[0]["emoji_flag"])


class TestRun:
    def test_every_window_produced_a_decision(self, monitor, results):
        assert len(results) == 4
        decisions = monitor.store.decisions(monitor.run_id)
        assert len(decisions) == 4
        assert set(decisions["action"]) <= {"noop", "investigate", "retrain", "rollback"}

    def test_traffic_was_persisted(self, monitor, results):
        total = monitor.store.list_windows(monitor.run_id)["n_traffic"].sum()
        expected = sum(len(r["window"].frame) for r in results)
        assert total == expected

    def test_all_metric_families_are_written(self, monitor):
        metrics = monitor.store.metric_series(monitor.run_id)
        assert set(metrics["category"]) == {"embedding", "tabular", "quality", "judge", "volume"}
        for name in ("domain_classifier_auc", "mmd2", "ood_rate", "accuracy", "ece",
                     "judge_score", "judge_veto_rate", "psi_mean"):
            assert name in set(metrics["name"]), name

    def test_every_metric_is_json_safe(self, monitor):
        for value in monitor.store.metric_series(monitor.run_id).get("extra", pd.Series(dtype=str)):
            if value:
                json.loads(value)

    def test_baseline_windows_are_healthy(self, monitor, results):
        for result in results[:2]:
            assert result["quality"].accuracy >= 0.75
            assert result["quality"].severity == "none"

    def test_out_of_scope_traffic_is_caught(self, results):
        """The point of the whole exercise: a scope change degrades quality."""
        shocked = results[-1]
        assert shocked["quality"].accuracy < results[0]["quality"].accuracy - 0.15
        assert shocked["embedding"].ood_rate > results[0]["embedding"].ood_rate

    def test_quality_gap_between_scopes_is_measured(self, results):
        shocked = results[-1]
        assert shocked["quality"].out_of_scope_accuracy < 0.2
        assert shocked["quality"].in_scope_accuracy > 0.6
        assert shocked["embedding"].concept_gap > 0.3

    def test_policy_never_retrains_on_input_drift_alone(self, results):
        for result in results:
            if result["quality"].drift_detected:
                continue
            assert result["decision"].action != "retrain"

    def test_judge_ran_on_every_window(self, monitor, results):
        for result in results:
            assert result["judge"].n_judged > 0
        scores = monitor.store.judge_scores(monitor.run_id)
        assert not scores.empty
        assert set(scores["dimension"]) == {d.key for d in
                                            __import__("src.judge.rubric", fromlist=["RUBRIC"]).RUBRIC}

    def test_run_manifest_records_provenance(self, monitor):
        run = monitor.store.get_run(monitor.run_id)
        assert run["encoder"] and run["judge"] and run["app_model"]
        snapshot = json.loads(run["config_snapshot"])
        assert "thresholds" in snapshot and "judge_sample" in snapshot


class TestReports:
    def test_markdown_renders(self, monitor):
        data = assemble(monitor.store, monitor.run_id)
        md = render_markdown(data, calibration={"accuracy": 0.9},
                             suite={"n_passed": 8, "n_cases": 8, "verdict": "ok"})
        assert "# LLM Drift & Quality Report" in md
        assert "Decision timeline" in md
        assert "Incidents" in md

    def test_html_is_self_contained(self, monitor):
        data = assemble(monitor.store, monitor.run_id)
        html = render_html(data, calibration={"accuracy": 0.9})
        assert html.startswith("<!doctype html>")
        assert "http://" not in html.replace("http://localhost", "")
        assert "<svg" in html

    def test_writes_all_three_formats(self, monitor, tmp_path):
        paths = write_reports(monitor.store, monitor.run_id, out_dir=tmp_path,
                              calibration={"accuracy": 0.9})
        for path in (paths.markdown, paths.json, paths.html):
            assert path.exists() and path.stat().st_size > 500
        payload = json.loads(paths.json.read_text(encoding="utf-8"))
        assert payload["run"]["run_id"] == monitor.run_id
        assert payload["metrics"]


class TestExpandedChampion:
    def test_widening_scope_fixes_out_of_scope_accuracy(self, artifacts):
        """Retraining with the new intents in scope is the actual remedy.

        Measured directly on held-out text from the intents that were out of
        scope for v1 — the same comparison the demo's recovery stage makes.
        """
        from src.data.datasets import load_banking77

        df, _ = load_banking77()
        probe = df[df["category"].isin(settings.OUT_OF_SCOPE_INTENTS) & ~df["category"].isin(
            [i for i in settings.LAUNCH_INTENTS])]
        if probe.empty:
            pytest.skip("no out-of-scope text available")
        probe = probe.sample(min(300, len(probe)), random_state=3).reset_index(drop=True)

        v1 = artifacts.classifier.predict_frame(probe, "text")
        v1_acc = float((v1["pred_intent"].to_numpy() == probe["category"].to_numpy()).mean())

        expanded = build_artifacts(n_reference=700, n_validation=300,
                                   encoder_backend="hashing",
                                   train_on_expanded_scope=True, version="it-v2")
        v2 = expanded.classifier.predict_frame(probe, "text")
        v2_acc = float((v2["pred_intent"].to_numpy() == probe["category"].to_numpy()).mean())

        assert len(expanded.classifier.classes) > len(artifacts.classifier.classes)
        assert v1_acc < 0.1, f"v1 unexpectedly handles out-of-scope traffic ({v1_acc:.2f})"
        assert v2_acc > 0.6, f"expanded champion still fails on new intents ({v2_acc:.2f})"
