"""Tests for the drift orchestrator policy and the telemetry store.

The orchestrator is where monitoring becomes behaviour, so the tests are about
*decisions*: does it refuse to retrain on input drift alone, does it demand
confirmation, does it keep working without labels.
"""

from __future__ import annotations

import json

import pytest

from src.monitoring.orchestrator import (
    ACTION_INVESTIGATE,
    ACTION_NOOP,
    ACTION_RETRAIN,
    Decision,
    DriftOrchestrator,
    summarise,
    volume_anomaly,
)
from src.storage.telemetry import MetricPoint, TelemetryStore


def _emb(detected=False, severity="none"):
    return {"drift_detected": detected, "severity": severity, "n_labeled": 0,
            "mmd_p_value": 0.01 if detected else 0.4}


def _qual(detected=False, severity="none", signals=None, n_labeled=60):
    return {"drift_detected": detected, "severity": severity,
            "severity_by_signal": signals or {}, "n_labeled": n_labeled,
            "n_requests": n_labeled or 0}


def _judge(regression=False, severity="none", n=20):
    return {"regression": regression, "severity": severity, "n_judged": n,
            "dimension_deltas": {}, "mean_score": 0.8}


class TestPolicy:
    def test_all_clear_is_noop(self):
        o = DriftOrchestrator()
        d = o.decide(embedding=_emb(), quality=_qual(), judge=_judge())
        assert d.action == ACTION_NOOP
        assert d.health_score == pytest.approx(100.0)
        assert d.signals["fired"] == []

    def test_input_drift_alone_does_not_retrain(self):
        """The headline policy: new distribution, same accuracy -> widen scope.

        Retraining cannot help while the new traffic is unlabelled, and a policy
        that fires a rebuild here gets the rebuild turned off.
        """
        o = DriftOrchestrator()
        d = o.decide(embedding=_emb(True, "moderate"), quality=_qual(), judge=_judge())
        assert d.action == ACTION_INVESTIGATE
        assert "not a model problem" in d.rationale or "scope" in d.rationale.lower()

    def test_quality_regression_without_input_drift_blames_the_model(self):
        o = DriftOrchestrator(confirm_windows=1)
        d = o.decide(embedding=_emb(), quality=_qual(True, "moderate", {"accuracy": "moderate"}),
                     judge=_judge())
        assert d.action == ACTION_INVESTIGATE
        assert "dependenc" in d.rationale.lower() or "artifact" in d.rationale.lower()

    def test_sustained_severe_quality_regression_retrains(self):
        o = DriftOrchestrator(confirm_windows=2, auto_retrain=True)
        signals = {"accuracy": "severe", "ece": "severe"}
        first = o.decide(embedding=_emb(True, "severe"),
                         quality=_qual(True, "severe", signals), judge=_judge())
        assert first.action == ACTION_INVESTIGATE, "must not act on a single window"
        second = o.decide(embedding=_emb(True, "severe"),
                          quality=_qual(True, "severe", signals), judge=_judge())
        assert second.action == ACTION_RETRAIN
        assert second.signals["sustained"] is True
        assert second.signals["consecutive"] >= 2

    def test_confirmation_resets_when_signals_clear(self):
        o = DriftOrchestrator(confirm_windows=3)
        o.decide(embedding=_emb(True, "moderate"), quality=_qual(), judge=_judge())
        o.decide(embedding=_emb(), quality=_qual(), judge=_judge())
        d = o.decide(embedding=_emb(True, "moderate"), quality=_qual(), judge=_judge())
        assert d.signals["consecutive"] == 1

    def test_judge_regression_investigates_the_judge_first(self):
        o = DriftOrchestrator(confirm_windows=1)
        d = o.decide(embedding=_emb(), quality=_qual(),
                     judge=_judge(True, "moderate", n=25))
        assert d.action == ACTION_INVESTIGATE
        assert "judge itself" in d.rationale.lower()

    def test_health_decreases_with_severity(self):
        o = DriftOrchestrator()
        healthy = o.decide(embedding=_emb(), quality=_qual(), judge=_judge()).health_score
        bad = o.decide(embedding=_emb(True, "severe"),
                       quality=_qual(True, "severe", {"accuracy": "severe"}),
                       judge=_judge(True, "severe")).health_score
        assert bad < healthy

    def test_quality_outweighs_embedding_in_the_health_score(self):
        """A user getting a wrong answer costs more than a moved input distribution."""
        o = DriftOrchestrator()
        only_input = o.decide(embedding=_emb(True, "severe"), quality=_qual(), judge=_judge())
        only_quality = o.decide(embedding=_emb(), quality=_qual(True, "severe", {"accuracy": "severe"}),
                                judge=_judge())
        assert only_quality.health_score < only_input.health_score

    def test_confidence_reflects_evidence(self):
        o = DriftOrchestrator(confirm_windows=1)
        thin = o.decide(embedding=_emb(), quality=_qual(n_labeled=0), judge=_judge(n=0))
        strong = o.decide(embedding=_emb(True), quality=_qual(n_labeled=80), judge=_judge(n=30))
        assert strong.confidence > thin.confidence

    def test_signal_key_is_none_when_nothing_fires(self):
        o = DriftOrchestrator()
        d = o.decide(embedding=_emb(), quality=_qual(), judge=_judge())
        assert d.signals["signal_key"] is None

    def test_decision_serialises(self):
        o = DriftOrchestrator()
        payload = o.decide(embedding=_emb(), quality=_qual(), judge=_judge()).to_dict()
        json.dumps(payload)
        assert payload["policy_version"] == "policy-v1"

    def test_summary_is_readable(self):
        o = DriftOrchestrator()
        text = summarise(o.decide(embedding=_emb(), quality=_qual(), judge=_judge()))
        assert "noop" in text and "health" in text


class TestIncidentLifecycle:
    def test_escalation_opens_and_recovery_closes(self, store: TelemetryStore):
        o = DriftOrchestrator(confirm_windows=1)
        run = store.start_run()
        h1 = store.open_window(run, 0, "baseline")
        h2 = store.open_window(run, 1, "new_intents")
        o.update_incident(store, run, h1.window_id,
                          Decision("noop", "none", 100.0, 1.0, "fine", {}))
        incident = o.update_incident(store, run, h1.window_id,
                                     Decision("investigate", "severe", 40.0, 0.9, "bad", {}))
        assert incident is not None
        assert len(store.incidents(run, "open")) == 1

        o.update_incident(store, run, h2.window_id,
                          Decision("noop", "none", 100.0, 1.0, "recovered", {}))
        assert len(store.incidents(run, "open")) == 0
        assert len(store.incidents(run, "closed")) == 1

    def test_timeline_records_observations(self, store: TelemetryStore):
        o = DriftOrchestrator()
        run = store.start_run()
        h = store.open_window(run, 0, "a")
        o.update_incident(store, run, h.window_id,
                          Decision("retrain", "severe", 30.0, 1.0, "x", {}))
        for i in range(1, 3):
            hh = store.open_window(run, i, f"w{i}")
            o.update_incident(store, run, hh.window_id,
                              Decision("investigate", "severe", 35.0, 0.9, "x", {}))
        row = store.incidents(run).iloc[0]
        events = [e["event"] for e in json.loads(row["timeline"])]
        assert "opened" in events and "observed" in events


class TestVolumeAnomaly:
    def test_normal_volume_is_quiet(self):
        assert volume_anomaly(250, 250)["anomaly"] is False

    def test_spike_is_detected(self):
        result = volume_anomaly(800, 250)
        assert result["anomaly"] is True
        assert result["direction"] == "spike"

    def test_drop_is_detected(self):
        assert volume_anomaly(40, 250)["direction"] == "drop"

    def test_small_baseline_is_ignored(self):
        assert volume_anomaly(0, 0)["anomaly"] is False


class TestTelemetryStore:
    def test_run_window_metric_round_trip(self, store: TelemetryStore):
        run = store.start_run(encoder="enc", judge="judge")
        h = store.open_window(run, 0, "baseline")
        store.log_metrics(h, [
            MetricPoint(category="embedding", name="mmd2", value=0.001,
                        threshold_moderate=0.003, severity="none", unit="mmd2"),
            MetricPoint(category="quality", name="accuracy", value=0.91, severity="none"),
        ])
        store.close_window(h, 250, 140)
        store.log_traffic(h, [{"text_hash": 1, "pred_intent": "a", "gold_intent": "a",
                               "confidence": 0.9, "in_scope": 1, "abstained": 0,
                               "response": "ok", "text_snippet": "x"}])
        store.log_decision(h, "noop", "none", 100.0, {}, "all good", confidence=0.3)

        assert store.latest_run_id() == run
        assert len(store.metric_series(run)) == 2
        assert len(store.traffic_for_window(h.window_id)) == 1
        # close_window is the authoritative labelled count; n_traffic is
        # recounted from the traffic table so call order cannot corrupt it.
        assert store.get_window(run, 0)["n_labeled"] == 140
        assert store.decisions(run).iloc[0]["confidence"] == pytest.approx(0.3)

    def test_traffic_count_is_order_independent(self, store: TelemetryStore):
        run = store.start_run()
        h = store.open_window(run, 0, "w")
        store.close_window(h, 0, 0)
        store.log_traffic(h, [{"text_hash": 1}, {"text_hash": 2}])
        assert store.get_window(run, 0)["n_traffic"] == 2

    def test_metrics_are_upserted_not_duplicated(self, store: TelemetryStore):
        run = store.start_run()
        h = store.open_window(run, 0, "w")
        store.log_metrics(h, [MetricPoint("embedding", "mmd2", 0.001)])
        store.log_metrics(h, [MetricPoint("embedding", "mmd2", 0.002)])
        assert len(store.metric_series(run)) == 1
        assert store.metric_series(run).iloc[0]["value"] == pytest.approx(0.002)

    def test_judge_scores_round_trip(self, store: TelemetryStore):
        run = store.start_run()
        h = store.open_window(run, 0, "w")
        store.log_judge_scores(h, [{"sample_hash": 1, "dimension": "tone", "score": 3,
                                    "judge_model": "m", "rubric_version": "v1"}])
        assert len(store.judge_scores(run)) == 1

    def test_reset_clears_everything(self, store: TelemetryStore):
        run = store.start_run()
        h = store.open_window(run, 0, "w")
        store.log_metrics(h, [MetricPoint("embedding", "mmd2", 0.1)])
        store.reset()
        assert store.metric_series(run).empty
        assert store.latest_run_id() is None

    def test_metric_pivot_includes_labels(self, store: TelemetryStore):
        run = store.start_run()
        h = store.open_window(run, 0, "style_shift")
        store.log_metrics(h, [MetricPoint("embedding", "mmd2", 0.1)])
        wide = store.metric_pivot(run)
        assert "mmd2" in wide.columns
        assert wide.iloc[0]["label"] == "style_shift"

    def test_nan_values_stored_as_null(self, store: TelemetryStore):
        run = store.start_run()
        h = store.open_window(run, 0, "w")
        store.log_metrics(h, [MetricPoint("quality", "accuracy", float("nan"))])
        assert store.metric_series(run).iloc[0]["value"] is None

    def test_store_survives_reopening_the_file(self, tmp_path):
        path = tmp_path / "t.db"
        first = TelemetryStore(path)
        run = first.start_run()
        h = first.open_window(run, 0, "w")
        first.log_metrics(h, [MetricPoint("embedding", "mmd2", 0.5)])
        first.close()

        second = TelemetryStore(path)
        assert second.metric_series(run).iloc[0]["value"] == pytest.approx(0.5)

    def test_missing_columns_are_migrated(self, store: TelemetryStore):
        """A database from an earlier schema must not need a manual wipe."""
        import sqlite3

        with sqlite3.connect(store.path) as conn:
            cols = {r[1] for r in conn.execute("PRAGMA table_info(decisions)")}
            if "confidence" in cols:
                conn.execute("ALTER TABLE decisions DROP COLUMN confidence")
        reopened = TelemetryStore(store.path)
        run = reopened.start_run()
        h = reopened.open_window(run, 0, "w")
        reopened.log_decision(h, "noop", "none", 100.0, {}, "ok", confidence=0.5)
        assert reopened.decisions(run).iloc[0]["confidence"] == pytest.approx(0.5)