"""
The monitor
-----------
Wires the pieces into one loop: serve → log → detect → decide → persist.

Per evaluation window:

1. **Serve.** Classify every request; generate LLM responses for a sampled
   subset (configurable). LLM cost is the constraint that shapes real eval
   pipelines, so this is sampled by default and the sample size is recorded.
2. **Embed.** Encode the window's text with the same encoder that built the
   reference snapshot, and stamp the encoder signature.
3. **Detect.** Embedding drift (MMD / sliced Wasserstein / domain classifier /
   normalised Frechet / novelty), tabular drift on cheap text-derived features,
   and concept gap between in-scope and out-of-scope accuracy.
4. **Measure quality.** On labelled rows: accuracy, macro-F1, ECE, MCE, Brier,
   reliability curve, abstention, worst intents. On all rows: confidence
   distribution and the label-free proxy.
5. **Judge.** Stratified LLM-as-judge sample against the rubric, with a
   regression test against the frozen judge baseline.
6. **Decide.** The orchestrator combines everything into one action and manages
   the incident lifecycle.
7. **Persist.** Every metric, judgement, decision and incident goes to the
   telemetry store so the dashboard and the report generator have the same data.

The class is deliberately synchronous and side-effect-explicit. In a real
deployment step 1 would be the existing serving path and steps 2-7 would run on
a schedule; here the boundary is explicit so both halves can be reasoned about.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from config import settings
from src.app.assistant import PLAYBOOKS, IntentClassifier, SupportAssistant, templated_response
from src.data.datasets import build_reference, load_banking77, write_dataset
from src.data.shifts import Regime, TrafficSimulator, TrafficWindow
from src.drift import EmbeddingDriftDetector, TabularDriftDetector
from src.embeddings import build_encoder
from src.judge import LLMBasedJudge
from src.monitoring.orchestrator import DriftOrchestrator, summarise, volume_anomaly
from src.quality import OutputQualityMonitor, snapshot_baseline
from src.storage.reference import ReferenceSnapshot
from src.storage.telemetry import MetricPoint, TelemetryStore
from src.utils.logging import get_logger, log_dict, set_log_context
from src.utils.parallel import ordered_map
from src.utils.stats import stable_hash

logger = get_logger(__name__)


@dataclass
class BootstrapArtifacts:
    """Everything the monitor needs at serving time, built once."""

    dataset_info: Any
    reference_df: pd.DataFrame
    validation_df: pd.DataFrame
    classifier: IntentClassifier
    encoder: Any
    snapshot: ReferenceSnapshot
    quality_baseline: dict[str, float]
    tabular_detector: TabularDriftDetector
    judge_baseline_responses: dict[int, str] = field(default_factory=dict)

    def summary(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset_info.to_dict(),
            "reference_rows": len(self.reference_df),
            "validation_rows": len(self.validation_df),
            "intents": len(self.classifier.classes),
            "encoder": self.encoder.signature,
            "quality_baseline": {k: (round(v, 4) if isinstance(v, float) and v == v else v)
                                 for k, v in self.quality_baseline.items()},
        }


def text_features(texts: Sequence[str]) -> pd.DataFrame:
    """
    Cheap tabular features derived from the text.

    Deliberately shallow. Column-level drift on these is the *fast* signal that
    tells an on-call engineer something concrete ("median length halved, 30%
    lowercase"), while the embedding detectors tell them that something changed
    without saying what. Having both is what makes a drift alert actionable.
    """
    s = pd.Series([str(t) for t in texts])
    words = s.str.split().str.len().fillna(0)
    return pd.DataFrame({
        "text_length": s.str.len().astype(float),
        "word_count": words.astype(float),
        "unique_word_ratio": (s.str.split().map(lambda w: len(set(w)) / max(len(w), 1))).astype(float),
        "digit_ratio": (s.str.count(r"\d") / s.str.len().clip(lower=1)).astype(float),
        "upper_ratio": (s.str.count(r"[A-Z]") / s.str.len().clip(lower=1)).astype(float),
        "punct_ratio": (s.str.count(r"[^\w\s]") / s.str.len().clip(lower=1)).astype(float),
        "is_question": s.str.endswith("?").astype(float),
        "emoji_flag": s.str.contains(
            "[\U0001F300-\U0001FAFF☀-➿]", regex=True, na=False).astype(float),
        "has_currency": s.str.contains(r"[£$€]|\bgbp\b|\busd\b|\beur\b", case=False,
                                       regex=True, na=False).astype(float),
    })


class DriftMonitor:
    """Owns the serving path, the detectors and the telemetry for one run."""

    def __init__(self, artifacts: BootstrapArtifacts, store: TelemetryStore,
                 llm=None, use_llm: bool = True, judge: LLMBasedJudge | None = None,
                 orchestrator: DriftOrchestrator | None = None,
                 response_sample: int = settings.LLM_RESPONSE_SAMPLE,
                 judge_sample: int = settings.JUDGE_SAMPLE_SIZE,
                 max_workers: int = settings.JUDGE_MAX_WORKERS,
                 run_notes: str = ""):
        self.art = artifacts
        self.store = store
        self.assistant = SupportAssistant(artifacts.classifier, llm=llm, use_llm=use_llm)
        self.embedding_detector = EmbeddingDriftDetector(artifacts.snapshot)
        self.tabular_detector = artifacts.tabular_detector
        self.quality_monitor = OutputQualityMonitor(artifacts.quality_baseline)
        self.judge = judge or LLMBasedJudge(
            baseline_score=artifacts.quality_baseline.get("judge_score"),
            baseline_score_se=artifacts.quality_baseline.get("judge_score_se", 0.0),
            baseline_scores=artifacts.quality_baseline.get("judge_dimensions", {}),
            sample_size=judge_sample, max_workers=max_workers)
        self.orchestrator = orchestrator or DriftOrchestrator()
        self.response_sample = response_sample
        self.judge_sample = judge_sample
        self.max_workers = max_workers
        self.run_notes = run_notes
        self.expected_volume = settings.WINDOW_SIZE
        self.run_id: str | None = None

    # ── run lifecycle ──────────────────────────────────────────────
    def start_run(self) -> str:
        self.run_id = self.store.start_run(
            encoder=self.art.encoder.signature,
            judge=self.judge.signature,
            app_model=self.art.classifier.signature,
            dataset_source=self.art.dataset_info.source,
            notes=self.run_notes,
            config_snapshot={
                "window_size": settings.WINDOW_SIZE,
                "thresholds": settings.THRESHOLDS.to_dict(),
                "judge_sample": self.judge_sample,
                "response_sample": self.response_sample,
            },
        )
        set_log_context(run_id=self.run_id)
        return self.run_id

    def finish_run(self, status: str = "complete", notes: str = "") -> None:
        if self.run_id:
            self.store.finish_run(self.run_id, status, notes)

    # ── one window ──────────────────────────────────────────────────
    def process_window(self, window: TrafficWindow) -> dict[str, Any]:
        assert self.run_id, "call start_run() first"
        t_start = time.perf_counter()
        set_log_context(window=window.label)
        handle = self.store.open_window(
            self.run_id, window.window_index, label=window.label, notes=window.notes)

        frame = window.frame.copy()
        logger.info("Window %d '%s': %d requests (%.0f%% out of scope, %.0f%% labelled)",
                    window.window_index, window.label, len(frame),
                    100 * (1 - frame["in_scope"].mean()), 100 * frame["gold_intent"].notna().mean())

        # 1. Serve — classify everything, generate responses for a sample.
        preds = self.art.classifier.predict_frame(frame, "text")
        frame = pd.concat([frame.reset_index(drop=True), preds.reset_index(drop=True)], axis=1)
        frame["margin"] = frame.get("margin", pd.Series(0.0, index=frame.index))

        sampled_idx = self._response_sample_index(len(frame))
        responses: dict[int, str] = {}
        if sampled_idx:
            def _respond(i: int) -> tuple[int, str, float]:
                row = frame.iloc[i]
                out = self.assistant.handle(str(row["text"]))
                return i, out.response, out.app_latency_ms

            for i, resp, lat in ordered_map(_respond, sampled_idx,
                                            max_workers=self.max_workers, label="responses"):
                responses[i] = resp
                frame.at[i, "response"] = resp
                frame.at[i, "app_latency_ms"] = lat

        if "response" not in frame.columns:
            frame["response"] = ""
        if "app_latency_ms" not in frame.columns:
            frame["app_latency_ms"] = np.nan
        # Ungenerated rows get the templated reply: the classifier verdict is
        # what the detectors consume, and a fabricated-but-deterministic body is
        # better than an empty string that skews length statistics.
        for i in range(len(frame)):
            if not frame.at[i, "response"]:
                playbook = PLAYBOOKS.get(str(frame.at[i, "pred_intent"]), PLAYBOOKS["default"])
                frame.at[i, "response"] = templated_response(str(frame.at[i, "pred_intent"]), playbook)

        # 2. Embed + 3a. embedding drift
        emb = self.art.encoder.encode(frame["text"].astype(str).tolist())
        is_correct = np.where(frame["gold_intent"].notna(),
                              (frame["pred_intent"].to_numpy() == frame["gold_intent"].fillna("").to_numpy()),
                              False)
        emb_result = self.embedding_detector.detect(
            emb, in_scope=frame["in_scope"].to_numpy(dtype=bool),
            is_correct=is_correct.astype(float))

        # 3b. tabular drift on cheap text features
        feats = text_features(frame["text"].tolist())
        tab_result = self.tabular_detector.detect(feats)

        # 4. output quality
        quality_result = self.quality_monitor.evaluate(
            frame.assign(labeled=frame["gold_intent"].notna()))

        # 5. LLM-as-judge
        judge_result = self.judge.evaluate_window(
            frame, window_index=window.window_index, n=self.judge_sample,
            seed=window.window_index, playbooks=PLAYBOOKS)

        vol = volume_anomaly(len(frame), self.expected_volume)

        # 6. Decide
        decision = self.orchestrator.decide(
            embedding=emb_result.to_dict(), quality=quality_result.to_dict(),
            judge=judge_result.to_dict(), volume=vol, window_index=window.window_index)

        # 7. Persist
        self.store.log_traffic(handle, self._records(frame))
        self.store.log_metrics(handle, self._metric_points(
            window, emb_result, tab_result, quality_result, judge_result, vol))
        if judge_result.samples:
            self.store.log_judge_scores(handle, [r for j in judge_result.samples for r in j.flat()])
        self.store.log_decision(handle, decision.action, decision.severity,
                                decision.health_score, decision.signals, decision.rationale,
                                confidence=decision.confidence)
        self.store.close_window(handle, len(frame), int(frame["gold_intent"].notna().sum()))
        incident_id = self.orchestrator.update_incident(
            self.store, self.run_id, handle.window_id, decision,
            extra={"regime": window.label, "notes": window.notes})

        elapsed = time.perf_counter() - t_start
        log_dict(logger, "window complete", {
            "action": decision.action, "health": round(decision.health_score, 1),
            "emb": emb_result.severity, "qual": quality_result.severity,
            "judge": judge_result.severity, "secs": round(elapsed, 1),
        })
        logger.info("  %s", summarise(decision))
        return {
            "window": window, "handle": handle, "decision": decision,
            "embedding": emb_result, "tabular": tab_result,
            "quality": quality_result, "judge": judge_result,
            "volume": vol, "incident_id": incident_id, "elapsed_s": elapsed,
        }

    def run_schedule(self, schedule: Sequence[Regime] | None = None,
                     simulator: TrafficSimulator | None = None,
                     on_window=None) -> list[dict[str, Any]]:
        if simulator is None:
            df, _ = load_banking77()
            simulator = TrafficSimulator(df, base_window_size=settings.WINDOW_SIZE)
        results = []
        for window in simulator.iter_schedule(schedule):
            res = self.process_window(window)
            results.append(res)
            if on_window:
                on_window(res)
        return results

    # ── helpers ─────────────────────────────────────────────────────
    def _response_sample_index(self, n: int) -> list[int]:
        """Spread the LLM response sample evenly across the window, deterministically.

        Evenly spaced rather than random so a re-run of the same window generates
        the same replies, and so the sample is not accidentally concentrated in
        whichever intent the simulator happened to emit first.
        """
        if self.response_sample <= 0 or n == 0:
            return []
        k = min(self.response_sample, n)
        picks = np.linspace(0, n - 1, num=k).round().astype(int)
        return sorted(pd.unique(picks).tolist())

    @staticmethod
    def _records(frame: pd.DataFrame) -> list[dict[str, Any]]:
        out = []
        for _, row in frame.iterrows():
            gold = row["gold_intent"]
            out.append({
                "request_id": f"req_{stable_hash(str(row['text']) + str(row.name)) % 10**12}",
                "text_hash": stable_hash(str(row["text"])) % (10 ** 12),
                "text_snippet": str(row["text"])[:240],
                "gold_intent": None if pd.isna(gold) else str(gold),
                "in_scope": int(bool(row["in_scope"])),
                "pred_intent": str(row["pred_intent"]),
                "confidence": float(row["confidence"]),
                "abstained": int(bool(row["abstained"])),
                "response": str(row["response"])[:1200],
                "app_latency_ms": float(row["app_latency_ms"]) if pd.notna(row["app_latency_ms"]) else None,
            })
        return out

    @staticmethod
    def _metric_points(window, emb, tab, quality, judge, vol) -> list[MetricPoint]:
        pts: list[MetricPoint] = []

        # Embedding drift. Each tuple is (metric name, live value, moderate
        # threshold, severe threshold, unit, vote key in the detector's signals).
        for name, value, mod, sev, unit, vote in (
            ("mmd2", emb.mmd2, settings.MMD_MODERATE, settings.MMD_SEVERE, "mmd2", "mmd"),
            ("swd", emb.swd, settings.SWD_MODERATE, settings.SWD_SEVERE, "cosine", "sliced_wasserstein"),
            ("centroid_cosine_shift", emb.centroid_cosine_shift,
             settings.CENTROID_COS_MODERATE, settings.CENTROID_COS_SEVERE, "cosine", "centroid_shift"),
            ("frechet_normalised", emb.frechet, settings.FRECHET_MODERATE,
             settings.FRECHET_SEVERE, "ratio", "frechet"),
            ("domain_classifier_auc", emb.domain_auc, settings.DOMAIN_AUC_MODERATE,
             settings.DOMAIN_AUC_SEVERE, "auc", "domain_classifier"),
            ("ood_rate", emb.ood_rate, settings.NOVELTY_MODERATE, settings.NOVELTY_SEVERE,
             "fraction", "novelty"),
            ("concept_gap", emb.concept_gap, 0.20, 0.40, "accuracy", "concept"),
        ):
            pts.append(MetricPoint(
                category="embedding", name=name, value=value,
                threshold_moderate=mod, threshold_severe=sev,
                severity=str(emb.signals.get(vote, "none")), unit=unit,
                extra={"p_value": emb.mmd_p_value if name == "mmd2" else None,
                       "significant": emb.signals.get("significant")}))

        # Tabular drift
        for col in tab.columns:
            pts.append(MetricPoint(
                category="tabular", name=f"psi::{col.column}", value=col.psi,
                threshold_moderate=settings.PSI_MODERATE, threshold_severe=settings.PSI_SEVERE,
                severity=col.severity, unit="psi",
                extra={"ks_p_value": col.ks_p_value, "kind": col.kind}))
        pts.append(MetricPoint(category="tabular", name="psi_mean", value=tab.psi_mean,
                               threshold_moderate=settings.PSI_MODERATE,
                               threshold_severe=settings.PSI_SEVERE, unit="psi"))

        # Output quality. `moderate`/`severe` are the absolute alert thresholds.
        # Accuracy, macro-F1, confidence and coverage are recorded without one:
        # their verdict comes from the delta against the frozen baseline, which
        # travels in the `baseline` column instead.
        for name, value, moderate, severe, unit in [
            ("accuracy", quality.accuracy, None, None, "accuracy"),
            ("macro_f1", quality.macro_f1, None, None, "f1"),
            ("ece", quality.ece, settings.ECE_WARN, settings.ECE_CRITICAL, "ece"),
            ("brier", quality.brier, settings.BRIER_WARN, settings.BRIER_CRITICAL, "brier"),
            ("abstention_rate", quality.abstention_rate, settings.ABSTENTION_JUMP,
             settings.ABSTENTION_JUMP * 2, "fraction"),
            ("confidence_mean", quality.confidence_mean, None, None, "confidence"),
            ("label_coverage", quality.label_coverage, None, None, "fraction"),
            ("proxy_accuracy", quality.proxy_accuracy, None, None, "accuracy"),
            ("accuracy_in_scope", quality.in_scope_accuracy, None, None, "accuracy"),
            ("accuracy_out_of_scope", quality.out_of_scope_accuracy, None, None, "accuracy"),
        ]:
            pts.append(MetricPoint(
                category="quality", name=name, value=value,
                baseline=(quality.accuracy - quality.accuracy_delta
                          if name == "accuracy" and np.isfinite(quality.accuracy_delta) else None),
                threshold_moderate=moderate, threshold_severe=severe,
                severity=quality.severity_by_signal.get(
                    {"accuracy": "accuracy", "ece": "ece", "brier": "brier",
                     "abstention_rate": "abstention"}.get(name, ""), "none"),
                unit=unit))

        # Judge
        pts.append(MetricPoint(category="judge", name="judge_score", value=judge.mean_score,
                               baseline=judge.baseline_score,
                               severity=judge.severity, unit="weighted_0_1",
                               extra={"n_judged": judge.n_judged, "se": judge.score_se,
                                      "z": judge.score_z, "model": judge.judge_model,
                                      "rubric": judge.rubric_version}))
        pts.append(MetricPoint(category="judge", name="judge_correctness_rate",
                               value=judge.correctness_rate, severity=judge.severity,
                               unit="fraction", extra={"n": judge.n_judged}))
        pts.append(MetricPoint(category="judge", name="judge_veto_rate", value=judge.veto_rate,
                               severity=judge.severity, unit="fraction"))
        pts.append(MetricPoint(category="judge", name="judge_pairwise_win_rate",
                               value=judge.pairwise_win_rate, severity=judge.severity, unit="fraction"))
        for dim, mean in judge.dimension_means.items():
            pts.append(MetricPoint(category="judge", name=f"judge_dim::{dim}", value=mean,
                                   baseline=judge.dimension_deltas.get(dim), severity=judge.severity,
                                   unit="score_1_5"))

        # Volume
        pts.append(MetricPoint(category="volume", name="requests", value=vol.get("current_n"),
                               severity="moderate" if vol.get("anomaly") else "none", unit="count",
                               extra={"expected": vol.get("expected_n"), "ratio": vol.get("ratio")}))
        pts.append(MetricPoint(category="volume", name="health_score",
                               value=None, severity="none", unit="score"))
        return pts


# ── Bootstrap ──────────────────────────────────────────────────────────

def build_artifacts(n_reference: int = settings.REFERENCE_SIZE,
                    n_validation: int = 400,
                    encoder_backend: str = settings.EMBEDDING_BACKEND,
                    train_on_expanded_scope: bool = False,
                    version: str | None = None) -> BootstrapArtifacts:
    """
    Train the champion, freeze the reference snapshot and the quality baseline.

    ``train_on_expanded_scope=True`` produces the *replacement* model used by the
    recovery stage of the demo: trained on the reference plus the out-of-scope
    intents, so the new traffic stops being out-of-scope. That contrast — same
    traffic, better model — is what shows the platform can distinguish "the
    world changed" from "we changed".
    """
    df, info = load_banking77()
    write_dataset(df, info)

    reference_df = build_reference(df, n=n_reference)
    train_df = reference_df
    scope_intents = set(settings.LAUNCH_INTENTS)
    if train_on_expanded_scope:
        scope_intents |= set(settings.OUT_OF_SCOPE_INTENTS)
        extra = df[df["category"].isin(settings.OUT_OF_SCOPE_INTENTS)]
        extra = extra.rename(columns={"category": "gold_intent"})
        extra["in_scope"] = False
        train_df = pd.concat([reference_df, extra[["text", "gold_intent", "in_scope"]]],
                             ignore_index=True)
        logger.info("Expanded-scope training set: %d rows, %d intents in scope",
                    len(train_df), len(scope_intents))

    # Hold out real labelled rows for validation, excluding anything used in
    # training. Balanced 50/50 in-scope / out-of-scope *relative to the scope
    # this model was trained on*: the in-scope half defines what "healthy"
    # means, the out-of-scope half makes the scope gap measurable and keeps
    # being meaningful after a promotion widens the scope.
    used = set(train_df["text"].astype(str))
    pool = df[~df["text"].astype(str).isin(used)].copy()
    pool["is_in_scope"] = pool["category"].isin(scope_intents)
    rng = np.random.default_rng(11)
    half = max(1, n_validation // 2)
    picks: list[pd.Series] = []
    for flag in (True, False):
        sub = pool[pool["is_in_scope"] == flag]
        if sub.empty:
            continue
        take = min(half, len(sub))
        idx = rng.choice(len(sub), size=take, replace=False)
        picks.append(sub.iloc[idx])
    validation_df = pd.concat(picks).sample(frac=1.0, random_state=3).reset_index(drop=True)
    validation_df = validation_df.rename(columns={"category": "gold_intent"})
    validation_df["in_scope"] = validation_df["is_in_scope"]
    validation_df = validation_df[["text", "gold_intent", "in_scope"]]
    logger.info("Validation set: %d rows (%.0f%% in scope, %.0f%% out of scope)",
                len(validation_df), 100 * validation_df["in_scope"].mean(),
                100 * (1 - validation_df["in_scope"].mean()))

    version = version or ("champion-v2-expanded" if train_on_expanded_scope else "champion-v1")
    classifier = IntentClassifier.train(train_df, version=version)

    # The model's realised scope is its own class list, not the intent name we
    # asked for. Deriving `in_scope` from the classifier keeps the validation
    # labels honest if training silently dropped a class.
    validation_df["in_scope"] = validation_df["gold_intent"].isin(classifier.classes)

    val_preds = classifier.predict_frame(validation_df, "text")
    validation_df = pd.concat([validation_df.reset_index(drop=True),
                               val_preds.reset_index(drop=True)], axis=1)
    classifier.fit_temperature(validation_df)

    # Recompute predictions under the fitted temperature before baselining.
    val_preds = classifier.predict_frame(validation_df, "text")
    validation_df["pred_intent"] = val_preds["pred_intent"]
    validation_df["confidence"] = val_preds["confidence"]
    validation_df["abstained"] = val_preds["abstained"]

    full_baseline = snapshot_baseline(validation_df)
    oos_accuracy = full_baseline.get("oos_accuracy")
    oos_rows = full_baseline.get("n_out_of_scope", 0)
    quality_baseline = dict(snapshot_baseline(validation_df[validation_df["in_scope"]]))
    quality_baseline["oos_accuracy"] = oos_accuracy
    quality_baseline["n_out_of_scope"] = oos_rows
    quality_baseline["n_validation_total"] = len(validation_df)
    logger.info("Quality baseline (in scope, n=%d): accuracy=%.4f f1=%.4f ece=%.4f brier=%.4f | "
                "out-of-scope accuracy=%.4f (n=%d)",
                quality_baseline.get("n_validation", 0),
                quality_baseline.get("accuracy", float("nan")),
                quality_baseline.get("macro_f1", float("nan")),
                quality_baseline.get("ece", float("nan")),
                quality_baseline.get("brier", float("nan")),
                oos_accuracy if oos_accuracy is not None else float("nan"), oos_rows)

    encoder = build_encoder(backend=encoder_backend, reference_texts=reference_df["text"].tolist())
    ref_emb = encoder.encode(reference_df["text"].tolist())
    snapshot = ReferenceSnapshot.build(
        reference_df, ref_emb, intent_col="gold_intent",
        confidences=validation_df["confidence"].to_numpy(dtype=float),
        correctness=(validation_df["pred_intent"].to_numpy()
                    == validation_df["gold_intent"].to_numpy()).astype(float),
        quality=quality_baseline,
        meta={"encoder": encoder.signature, "classifier": classifier.signature,
              "dataset_source": info.source, "version": version},
    )
    snapshot.save()

    tabular_detector = TabularDriftDetector(text_features(reference_df["text"].tolist()))
    reference_df.to_csv(settings.REFERENCE_DIR / "reference.csv", index=False)
    validation_df.to_csv(settings.REFERENCE_DIR / "validation.csv", index=False)
    classifier.save()

    return BootstrapArtifacts(
        dataset_info=info, reference_df=reference_df, validation_df=validation_df,
        classifier=classifier, encoder=encoder, snapshot=snapshot,
        quality_baseline=quality_baseline, tabular_detector=tabular_detector,
    )


def calibration_report(art: BootstrapArtifacts) -> dict[str, Any]:
    """Champion validation metrics — the numbers the demo opens with."""
    q = art.quality_baseline
    return {
        "validation_rows_in_scope": q.get("n_validation", 0),
        "validation_rows_out_of_scope": q.get("n_out_of_scope", 0),
        "validation_rows_total": q.get("n_validation_total", 0),
        "intents": len(art.classifier.classes),
        "temperature": art.classifier.temperature,
        "accuracy": q.get("accuracy"),
        "accuracy_se": q.get("accuracy_se"),
        "macro_f1": q.get("macro_f1"),
        "auc": q.get("auc"),
        "ece": q.get("ece"),
        "brier": q.get("brier"),
        "abstention_rate": q.get("abstention_rate"),
        "out_of_scope_accuracy": q.get("oos_accuracy"),
        "encoder": art.encoder.signature,
    }