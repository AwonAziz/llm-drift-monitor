"""
FastAPI serving + monitoring API
--------------------------------
Two audiences in one process, because they need the same objects:

* **Inference** — ``/predict``, ``/predict/batch``. Classify a message, draft a
  reply, log the full request/response/latency envelope.
* **Monitoring** — everything under ``/monitoring``. Current signal values,
  metric series, decisions, incidents, and an on-demand judge run.

The monitoring endpoints read the same SQLite store the simulator writes, so the
API reflects the last processed window rather than a parallel computation that
could disagree with it. A monitoring surface that computes its own answer is a
monitoring surface you cannot trust during an incident.
"""

from __future__ import annotations

import json
import time
from contextlib import asynccontextmanager
from typing import Any

import numpy as np
import pandas as pd
from fastapi import BackgroundTasks, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse

from ..app.assistant import PLAYBOOKS, IntentClassifier, SupportAssistant, templated_response
from ..drift import EmbeddingDriftDetector
from ..judge import LLMBasedJudge
from ..judge.rubric import RUBRIC_VERSION
from ..llm import build_llm
from ..quality import OutputQualityMonitor
from ..storage.reference import ReferenceSnapshot
from ..storage.telemetry import TelemetryStore
from ..utils.logging import get_logger
from .schemas import (
    BatchPredictRequest,
    BatchPredictResponse,
    DecisionResponse,
    DriftStatusResponse,
    EvaluateRequest,
    EvaluateResponse,
    HealthResponse,
    IncidentResponse,
    JudgeStatusResponse,
    MetricsResponse,
    ModelInfoResponse,
    PredictRequest,
    PredictResponse,
    QualityStatusResponse,
    RunSummaryResponse,
    SeriesResponse,
)

logger = get_logger(__name__)

_state: dict[str, Any] = {
    "classifier": None,
    "snapshot": None,
    "detector": None,
    "store": None,
    "assistant": None,
    "judge": None,
    "quality_monitor": None,
    "encoder": None,
    "started_at": time.time(),
    "requests": 0,
    "errors": 0,
}


def _load_classifier() -> IntentClassifier | None:
    try:
        clf = IntentClassifier.load()
    except FileNotFoundError:
        logger.warning("No classifier artifact. Run scripts/bootstrap.py.")
        return None
    return clf


def _load_snapshot() -> ReferenceSnapshot | None:
    try:
        return ReferenceSnapshot.load()
    except FileNotFoundError:
        logger.warning("No reference snapshot. Run scripts/bootstrap.py.")
        return None


@asynccontextmanager
async def lifespan(app: FastAPI):
    clf = _load_classifier()
    snap = _load_snapshot()
    _state["classifier"] = clf
    _state["snapshot"] = snap
    _state["store"] = TelemetryStore()
    if clf:
        _state["assistant"] = SupportAssistant(clf, llm=build_llm())
        _state["judge"] = LLMBasedJudge()
        _state["quality_monitor"] = OutputQualityMonitor(
            snap.quality if snap else {})
    if snap:
        _state["detector"] = EmbeddingDriftDetector(snap)
        from ..embeddings import build_encoder
        try:
            _state["encoder"] = build_encoder(reference_texts=snap.meta.get("reference_texts") or None)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Encoder unavailable: %s", exc)
    logger.info("API ready")
    yield
    logger.info("API shutting down")


app = FastAPI(
    title="LLM Drift Monitor — Serving + Monitoring API",
    description=(
        "Inference endpoints for the monitored assistant, and a read model over the "
        "same telemetry the monitor writes."
    ),
    version="1.0.0",
    lifespan=lifespan,
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


# ── helpers ────────────────────────────────────────────────────────────

def _get_store() -> TelemetryStore:
    """
    The store is opened lazily and cached.

    The monitoring endpoints must stay answerable even when startup never ran —
    an ops endpoint that 500s because a file could not be opened is worse than
    one that reports "no data yet".
    """
    if _state["store"] is None:
        try:
            _state["store"] = TelemetryStore()
        except Exception as exc:  # noqa: BLE001
            logger.error("Telemetry store unavailable: %s", exc)
            _state["store"] = _NullStore()
    return _state["store"]


class _NullStore:
    """Stands in for the store when the database cannot be opened."""

    def latest_run_id(self, finished_only: bool = False) -> None:
        return None

    def list_runs(self, limit: int = 20):
        return pd.DataFrame()

    def metric_series(self, run_id: str, category: str | None = None, name: str | None = None):
        return pd.DataFrame()

    def metric_pivot(self, run_id: str):
        return pd.DataFrame()

    def list_windows(self, run_id: str):
        return pd.DataFrame()

    def decisions(self, run_id: str):
        return pd.DataFrame()

    def incidents(self, run_id: str | None = None, status: str | None = None):
        return pd.DataFrame()

    def judge_scores(self, run_id: str):
        return pd.DataFrame()

    def traffic_for_window(self, window_id: str):
        return pd.DataFrame()


def _get_assistant() -> SupportAssistant | None:
    if _state["assistant"] is None and _state["classifier"] is not None:
        _state["assistant"] = SupportAssistant(_state["classifier"], llm=build_llm())
    return _state["assistant"]


def _require_classifier() -> IntentClassifier:
    if _state["classifier"] is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Run scripts/bootstrap.py.")
    return _state["classifier"]


def _resolve_run(run_id: str | None) -> str:
    store = _get_store()
    resolved = run_id or store.latest_run_id()
    if not resolved:
        raise HTTPException(status_code=404, detail="No monitoring run recorded. Run scripts/demo_shift.py.")
    if run_id and store.get_run(resolved) is None:
        # Returning an unknown run silently would answer with empty windows and
        # read like "the system is healthy" — the worst possible failure mode
        # for a monitoring endpoint.
        raise HTTPException(status_code=404, detail=f"No monitoring run '{run_id}'.")
    return resolved


def _num(value: Any) -> float | None:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return None if not np.isfinite(f) else f


def _delta(current: float | None, baseline: float | None) -> float | None:
    if current is None or baseline is None:
        return None
    return round(current - baseline, 4)


def _latest_metric(run_id: str, category: str, name: str) -> dict[str, Any] | None:
    df = _get_store().metric_series(run_id, category, name)
    if df.empty:
        return None
    return df.sort_values("window_index").iloc[-1].to_dict()


# ── inference ──────────────────────────────────────────────────────────

@app.post("/predict", response_model=PredictResponse, tags=["Inference"])
async def predict(request: PredictRequest) -> PredictResponse:
    """Classify a customer message and draft a reply from the intent playbook."""
    clf = _require_classifier()
    t0 = time.perf_counter()
    try:
        if request.intent:
            known = [c for c in clf.classes if c == request.intent]
            if not known:
                raise HTTPException(status_code=422, detail=f"Unknown intent '{request.intent}'")
            pred = clf.predict([request.text])[0]
            pred["intent"] = request.intent
        else:
            pred = clf.predict([request.text])[0]

        playbook = PLAYBOOKS.get(pred["intent"], PLAYBOOKS["default"])
        if pred["abstained"]:
            playbook = PLAYBOOKS["default"]

        if request.generate_response:
            assistant = _get_assistant()
            try:
                prompt = assistant.build_responder_prompt(
                    request.text, pred["intent"], playbook, pred["confidence"])
                reply = assistant.llm.generate(prompt).text
            except Exception as exc:  # noqa: BLE001
                logger.warning("LLM unavailable (%s); templated reply.", exc)
                reply = templated_response(pred["intent"], playbook)
        else:
            reply = templated_response(pred["intent"], playbook)
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        _state["errors"] += 1
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    latency = (time.perf_counter() - t0) * 1000
    _state["requests"] += 1
    in_scope = pred["intent"] in set(clf.classes)
    return PredictResponse(
        request_id=f"req_{int(time.time_ns()) % 10**12}",
        text=request.text,
        pred_intent=pred["intent"],
        intent_confidence=round(pred["confidence"], 4),
        intent_margin=round(pred["margin"], 4),
        abstained=pred["abstained"],
        in_scope_known=in_scope,
        response=reply,
        top3=pred["top3"],
        latency_ms=round(latency, 3),
        model_version=clf.version,
    )


@app.post("/predict/batch", response_model=BatchPredictResponse, tags=["Inference"])
async def predict_batch(request: BatchPredictRequest) -> BatchPredictResponse:
    """Classify up to 200 messages without LLM generation (classification only)."""
    _require_classifier()
    if len(request.instances) > 200:
        raise HTTPException(status_code=400, detail="Max 200 instances per batch")
    t0 = time.perf_counter()
    preds = _state["classifier"].predict([i.text for i in request.instances])
    clf = _state["classifier"]
    out = []
    for req, pred in zip(request.instances, preds):
        playbook = PLAYBOOKS.get(pred["intent"], PLAYBOOKS["default"])
        reply = ("" if not request.generate_response
                 else templated_response(pred["intent"], playbook))
        out.append(PredictResponse(
            request_id=f"req_{int(time.time_ns()) % 10**12}",
            text=req.text, pred_intent=pred["intent"],
            intent_confidence=round(pred["confidence"], 4),
            intent_margin=round(pred["margin"], 4),
            abstained=pred["abstained"],
            in_scope_known=pred["intent"] in set(clf.classes),
            response=reply, top3=pred["top3"], latency_ms=0.0,
            model_version=clf.version))
    _state["requests"] += len(out)
    return BatchPredictResponse(predictions=out, n=len(out),
                                total_latency_ms=round((time.perf_counter() - t0) * 1000, 3))


@app.get("/health", response_model=HealthResponse, tags=["Operations"])
async def health() -> HealthResponse:
    clf = _state["classifier"]
    judge = _state["judge"]
    return HealthResponse(
        status="healthy" if clf is not None else "uninitialised",
        model_loaded=clf is not None,
        reference_loaded=_state["snapshot"] is not None,
        version=clf.version if clf else "none",
        intents=len(clf.classes) if clf else 0,
        encoder=_state["encoder"].signature if _state["encoder"] else "none",
        judge=judge.signature if judge else "none",
        uptime_seconds=round(time.time() - _state["started_at"], 1),
    )


@app.get("/model/info", response_model=ModelInfoResponse, tags=["Operations"])
async def model_info() -> ModelInfoResponse:
    clf = _require_classifier()
    snap = _state["snapshot"]
    q = (snap.quality if snap else {}) or {}
    return ModelInfoResponse(
        version=clf.version, intents=len(clf.classes),
        temperature=clf.temperature, abstain_threshold=clf.abstain_threshold,
        encoder=_state["encoder"].signature if _state["encoder"] else "none",
        in_scope_accuracy=_num(q.get("accuracy")),
        out_of_scope_accuracy=_num(q.get("oos_accuracy")),
        macro_f1=_num(q.get("macro_f1")),
        ece=_num(q.get("ece")),
        brier=_num(q.get("brier")),
    )


# ── monitoring read model ──────────────────────────────────────────────

@app.get("/monitoring/runs", response_model=list[RunSummaryResponse], tags=["Monitoring"])
async def runs(limit: int = Query(10, ge=1, le=100)) -> list[RunSummaryResponse]:
    store = _get_store()
    df = store.list_runs(limit)
    if df.empty:
        return []
    out = []
    for _, row in df.iterrows():
        windows = store.list_windows(str(row["run_id"]))
        out.append(RunSummaryResponse(
            run_id=str(row["run_id"]),
            started_at=str(row.get("started_at") or ""),
            finished_at=None if pd.isna(row.get("finished_at")) else str(row.get("finished_at")),
            status=str(row.get("status") or "unknown"),
            encoder=None if pd.isna(row.get("encoder")) else str(row.get("encoder")),
            judge=None if pd.isna(row.get("judge")) else str(row.get("judge")),
            app_model=None if pd.isna(row.get("app_model")) else str(row.get("app_model")),
            dataset_source=None if pd.isna(row.get("dataset_source")) else str(row.get("dataset_source")),
            windows=0 if windows.empty else int(len(windows)),
        ))
    return out


@app.get("/monitoring/status", response_model=MetricsResponse, tags=["Monitoring"])
async def metrics(run_id: str | None = None) -> MetricsResponse:
    """One-glance current state of the monitored system."""
    try:
        resolved = _resolve_run(run_id)
    except HTTPException:
        return MetricsResponse(run_id=None, windows=0, requests=int(_state["requests"]),
                               health_score=None, action=None, severity=None,
                               domain_auc=None, ood_rate=None, accuracy=None, ece=None,
                               judge_score=None, judge_delta=None, open_incidents=0)
    windows = _get_store().list_windows(resolved)
    decisions = _get_store().decisions(resolved)
    incidents = _get_store().incidents(resolved, status="open")

    def latest(cat, name):
        row = _latest_metric(resolved, cat, name)
        return _num(row["value"]) if row else None

    judge_row = _latest_metric(resolved, "judge", "judge_score")
    last = decisions.iloc[-1].to_dict() if not decisions.empty else {}
    return MetricsResponse(
        run_id=resolved,
        windows=int(len(windows)),
        requests=int(windows["n_traffic"].sum()) if not windows.empty else 0,
        health_score=_num(last.get("health")),
        action=str(last.get("action")) if last else None,
        severity=str(last.get("severity")) if last else None,
        domain_auc=latest("embedding", "domain_classifier_auc"),
        ood_rate=latest("embedding", "ood_rate"),
        accuracy=latest("quality", "accuracy"),
        ece=latest("quality", "ece"),
        judge_score=latest("judge", "judge_score"),
        judge_delta=_num(judge_row.get("baseline")) - _num(judge_row.get("value"))
        if judge_row else None,
        open_incidents=int(len(incidents)),
    )


@app.get("/monitoring/drift", response_model=DriftStatusResponse, tags=["Monitoring"])
async def drift_status(run_id: str | None = None) -> DriftStatusResponse:
    resolved = _resolve_run(run_id)
    windows = _get_store().list_windows(resolved)
    if windows.empty:
        raise HTTPException(status_code=404, detail="No windows in this run.")

    def latest(name):
        return _latest_metric(resolved, "embedding", name)

    auc = latest("domain_classifier_auc")
    mmd = latest("mmd2")
    extra = mmd.get("extra") if mmd else None
    signals: dict[str, Any] = {}
    if isinstance(extra, str):
        signals = json.loads(extra)
    window_row = windows.sort_values("window_index").iloc[-1]
    return DriftStatusResponse(
        run_id=resolved,
        window_index=int(window_row["window_index"]),
        regime=window_row["label"],
        drift_detected=str(auc["severity"]) != "none" if auc else False,
        severity=max([str(r["severity"]) for r in [auc] if r] or ["none"],
                     key=lambda s: {"none": 0, "moderate": 1, "severe": 2}.get(s, 0)),
        domain_auc=_num(auc["value"]) if auc else None,
        mmd2=_num(mmd["value"]) if mmd else None,
        mmd_p_value=signals.get("p_value") if isinstance(signals, dict) else None,
        sliced_wasserstein=_num((latest("swd") or {}).get("value")),
        centroid_cosine_shift=_num((latest("centroid_cosine_shift") or {}).get("value")),
        frechet=_num((latest("frechet_normalised") or {}).get("value")),
        ood_rate=_num((latest("ood_rate") or {}).get("value")),
        concept_gap=_num((latest("concept_gap") or {}).get("value")),
        significance=signals.get("significant") if isinstance(signals, dict) else None,
        signals=signals if isinstance(signals, dict) else {},
    )


@app.get("/monitoring/quality", response_model=QualityStatusResponse, tags=["Monitoring"])
async def quality_status(run_id: str | None = None) -> QualityStatusResponse:
    resolved = _resolve_run(run_id)
    windows = _get_store().list_windows(resolved)
    if windows.empty:
        raise HTTPException(status_code=404, detail="No windows in this run.")
    last = windows.sort_values("window_index").iloc[-1]
    wid = last["window_id"]

    def latest(name):
        return _num((_latest_metric(resolved, "quality", name) or {}).get("value"))

    traffic = _get_store().traffic_for_window(wid)
    worst: list[dict[str, Any]] = []
    labeled = traffic[traffic["gold_intent"].notna()] if not traffic.empty else traffic
    if not labeled.empty:
        per = labeled.assign(ok=labeled["pred_intent"] == labeled["gold_intent"]) \
                      .groupby("gold_intent")["ok"].agg(["mean", "count"]).sort_values("mean")
        worst = [{"intent": str(i), "accuracy": round(float(r["mean"]), 4), "n": int(r["count"])}
                 for i, r in per.head(5).iterrows()]

    sev_row = _latest_metric(resolved, "quality", "accuracy")
    severity = str(sev_row["severity"]) if sev_row else "none"
    q_baseline = (_state["snapshot"].quality if _state["snapshot"] else {}) or {}
    return QualityStatusResponse(
        window_index=int(last["window_index"]),
        n_requests=int(last["n_traffic"] or 0),
        n_labeled=int(last["n_labeled"] or 0),
        label_coverage=latest("label_coverage"),
        accuracy=latest("accuracy"),
        macro_f1=latest("macro_f1"),
        ece=latest("ece"),
        brier=latest("brier"),
        abstention_rate=latest("abstention_rate"),
        accuracy_delta=_delta(latest("accuracy"), _num(q_baseline.get("accuracy"))),
        ece_delta=_delta(latest("ece"), _num(q_baseline.get("ece"))),
        proxy_accuracy=latest("proxy_accuracy"),
        in_scope_accuracy=latest("accuracy_in_scope"),
        out_of_scope_accuracy=latest("accuracy_out_of_scope"),
        worst_intents=worst,
        severity=severity,
        signals={"window": last["label"]},
    )


@app.get("/monitoring/judge", response_model=JudgeStatusResponse, tags=["Monitoring"])
async def judge_status(run_id: str | None = None) -> JudgeStatusResponse:
    resolved = _resolve_run(run_id)
    row = _latest_metric(resolved, "judge", "judge_score")
    if row is None:
        raise HTTPException(status_code=404, detail="No judge results in this run.")
    extra = json.loads(row.get("extra") or "{}") if isinstance(row.get("extra"), str) else {}
    dims = _get_store().metric_series(resolved, "judge")
    dim_scores = {r["name"].split("::")[-1]: _num(r["value"])
                  for _, r in dims[dims["name"].str.startswith("judge_dim::")].iterrows()}
    return JudgeStatusResponse(
        window_index=int(row["window_index"]),
        judge_model=str(extra.get("model", "unknown")),
        rubric_version=str(_latest_metric(resolved, "judge", "judge_score").get("extra") and RUBRIC_VERSION),
        baseline_score=_num(row.get("baseline")),
        score=_num(row.get("value")),
        score_delta=(_num(row.get("value")) - _num(row.get("baseline"))
                     if _num(row.get("baseline")) is not None else None),
        score_z=_num(extra.get("z")),
        dimension_scores={k: v for k, v in dim_scores.items() if v is not None},
        veto_rate=_num((_latest_metric(resolved, "judge", "judge_veto_rate") or {}).get("value")),
        pairwise_win_rate=_num((_latest_metric(resolved, "judge", "judge_pairwise_win_rate") or {}).get("value")),
        n_judged=int(extra.get("n_judged", 0)),
        regression=str(row.get("severity")) != "none",
        severity=str(row.get("severity")),
    )


@app.get("/monitoring/series", response_model=SeriesResponse, tags=["Monitoring"])
async def metric_series(run_id: str | None = None, category: str = "embedding",
                        name: str = "domain_classifier_auc") -> SeriesResponse:
    resolved = _resolve_run(run_id)
    df = _get_store().metric_series(resolved, category, name)
    points = [
        {"window_index": int(r["window_index"]), "value": _num(r["value"]),
         "severity": r["severity"], "baseline": _num(r["baseline"]),
         "threshold_moderate": _num(r["threshold_moderate"]),
         "threshold_severe": _num(r["threshold_severe"]),
         "label": r.get("label")}
        for _, r in df.sort_values("window_index").iterrows()
    ]
    return SeriesResponse(run_id=resolved, category=category, name=name,
                          unit=df["unit"].iloc[0] if not df.empty else None, points=points)


@app.get("/monitoring/decisions", response_model=list[DecisionResponse], tags=["Monitoring"])
async def decisions(run_id: str | None = None) -> list[DecisionResponse]:
    # List endpoints degrade to an empty list when nothing has run yet. A 404
    # here forces the dashboard to special-case "no runs" on every list, and
    # an empty list is a truthful answer: there are no decisions.
    try:
        resolved = _resolve_run(run_id)
    except HTTPException:
        return []
    df = _get_store().decisions(resolved)
    out = []
    for _, d in df.iterrows():
        signals = d.get("signals")
        out.append(DecisionResponse(
            window_id=str(d["window_id"]), window_index=int(d["window_index"]),
            action=str(d["action"]), severity=str(d["severity"]),
            health=_num(d["health"]) or 0.0, confidence=_num(d.get("confidence")),
            rationale=str(d["rationale"] or ""),
            signals=json.loads(signals) if isinstance(signals, str) and signals else {},
        ))
    return out


@app.get("/monitoring/incidents", response_model=list[IncidentResponse], tags=["Monitoring"])
async def incidents(run_id: str | None = None, status: str | None = None) -> list[IncidentResponse]:
    try:
        resolved = _resolve_run(run_id)
    except HTTPException:
        return []
    df = _get_store().incidents(resolved, status)
    out = []
    for _, inc in df.iterrows():
        timeline = inc.get("timeline")
        signals = inc.get("signals")
        # pd.isna on an object column holding both None and strings is unreliable,
        # so closed_at is checked by type as well as by null-ness.
        closed = inc.get("closed_at")
        out.append(IncidentResponse(
            incident_id=str(inc["incident_id"]), severity=str(inc["severity"]),
            title=str(inc["title"]) if isinstance(inc.get("title"), str) else "",
            status=str(inc["status"]),
            opened_at=str(inc["opened_at"]),
            closed_at=str(closed) if isinstance(closed, str) else None,
            timeline=json.loads(timeline) if isinstance(timeline, str) and timeline else [],
            signals=json.loads(signals) if isinstance(signals, str) and signals else {},
        ))
    return out


@app.post("/monitoring/evaluate", response_model=EvaluateResponse, tags=["Monitoring"])
async def run_judge(request: EvaluateRequest, background_tasks: BackgroundTasks) -> EvaluateResponse:
    """Re-judge a sample from the most recent window on demand."""
    resolved = _resolve_run(getattr(request, "run_id", None))
    judge = _state["judge"]
    if judge is None:
        raise HTTPException(status_code=503, detail="Judge backend unavailable")
    windows = _get_store().list_windows(resolved)
    if windows.empty:
        raise HTTPException(status_code=404, detail="No windows in this run.")
    wid = windows.sort_values("window_index").iloc[-1]["window_id"]
    traffic = _get_store().traffic_for_window(wid)
    if traffic.empty:
        raise HTTPException(status_code=404, detail="No traffic in the latest window")
    frame = traffic.rename(columns={"text_snippet": "text"}).copy()
    frame["gold_intent"] = frame["gold_intent"]
    t0 = time.perf_counter()
    result = judge.evaluate_window(frame, window_index=int(windows.sort_values("window_index").iloc[-1]["window_index"]),
                                   n=request.n, seed=request.seed, playbooks=PLAYBOOKS)
    return EvaluateResponse(
        run_id=resolved, window_index=result.window_index, n_judged=result.n_judged,
        score=_num(result.mean_score), baseline_score=_num(result.baseline_score),
        severity=result.severity, regression=result.regression,
        dimension_scores=result.dimension_means,
        unparseable=result.n_unparseable,
        elapsed_ms=round((time.perf_counter() - t0) * 1000, 1),
    )


@app.get("/metrics/prometheus", tags=["Operations"], response_class=PlainTextResponse)
async def prometheus() -> str:
    """Prometheus text exposition for the current signals."""
    lines = [
        f"ldm_http_requests_total {int(_state['requests'])}",
        f"ldm_http_errors_total {int(_state['errors'])}",
        f"ldm_uptime_seconds {round(time.time() - _state['started_at'], 1)}",
    ]
    try:
        resolved = _resolve_run(None)
    except HTTPException:
        return "\n".join(lines)
    for cat, name, labels in [
        ("embedding", "domain_classifier_auc", "signal=\"domain_classifier_auc\""),
        ("embedding", "ood_rate", "signal=\"ood_rate\""),
        ("embedding", "mmd2", "signal=\"mmd2\""),
        ("quality", "accuracy", "signal=\"accuracy\""),
        ("quality", "ece", "signal=\"ece\""),
        ("judge", "judge_score", "signal=\"judge_score\""),
    ]:
        row = _latest_metric(resolved, cat, name)
        if row and _num(row["value"]) is not None:
            lines.append(f'ldm_drift_value{{{labels}}} {_num(row["value"])}')
    windows = _get_store().list_windows(resolved)
    if not windows.empty:
        total = int(windows["n_traffic"].sum())
        lines.append(f"ldm_windows_total {len(windows)}")
        lines.append(f"ldm_requests_total {total}")
    open_inc = _get_store().incidents(resolved, status="open")
    lines.append(f"ldm_open_incidents {len(open_inc)}")
    return "\n".join(lines)
