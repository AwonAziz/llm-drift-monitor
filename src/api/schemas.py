"""Pydantic schemas for the serving and monitoring API."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=4000,
                      examples=["My card has not arrived after two weeks"])
    intent: str | None = Field(None, description="Force an intent instead of classifying")
    generate_response: bool = Field(True, description="Skip the LLM for a fast classification-only call")


class PredictResponse(BaseModel):
    request_id: str
    text: str
    pred_intent: str
    intent_confidence: float
    intent_margin: float
    abstained: bool
    in_scope_known: bool | None = None
    response: str
    top3: list[dict[str, Any]] = Field(default_factory=list)
    latency_ms: float
    model_version: str


class BatchPredictRequest(BaseModel):
    instances: list[PredictRequest] = Field(..., min_length=1, max_length=200)
    generate_response: bool = Field(False)


class BatchPredictResponse(BaseModel):
    predictions: list[PredictResponse]
    n: int
    total_latency_ms: float


class HealthResponse(BaseModel):
    status: Literal["healthy", "degraded", "uninitialised"]
    model_loaded: bool
    reference_loaded: bool
    version: str
    intents: int
    encoder: str
    judge: str
    uptime_seconds: float


class ModelInfoResponse(BaseModel):
    version: str
    intents: int
    temperature: float
    abstain_threshold: float
    encoder: str
    in_scope_accuracy: float | None
    out_of_scope_accuracy: float | None
    macro_f1: float | None
    ece: float | None
    brier: float | None


class MetricsResponse(BaseModel):
    run_id: str | None
    windows: int
    requests: int
    health_score: float | None
    action: str | None
    severity: str | None
    domain_auc: float | None
    ood_rate: float | None
    accuracy: float | None
    ece: float | None
    judge_score: float | None
    judge_delta: float | None
    open_incidents: int


class DriftStatusResponse(BaseModel):
    run_id: str | None
    window_index: int | None
    regime: str | None
    drift_detected: bool
    severity: str
    domain_auc: float | None
    mmd2: float | None
    mmd_p_value: float | None
    sliced_wasserstein: float | None
    centroid_cosine_shift: float | None
    frechet: float | None
    ood_rate: float | None
    concept_gap: float | None
    significance: bool | None
    signals: dict[str, Any] = Field(default_factory=dict)
    recommendation: str = ""


class QualityStatusResponse(BaseModel):
    window_index: int | None
    n_requests: int
    n_labeled: int
    label_coverage: float | None
    accuracy: float | None
    macro_f1: float | None
    ece: float | None
    brier: float | None
    abstention_rate: float | None
    accuracy_delta: float | None
    ece_delta: float | None
    proxy_accuracy: float | None
    in_scope_accuracy: float | None
    out_of_scope_accuracy: float | None
    worst_intents: list[dict[str, Any]] = Field(default_factory=list)
    reliability: list[list[float | None]] = Field(default_factory=list)
    severity: str
    signals: dict[str, Any] = Field(default_factory=dict)
    recommendation: str = ""


class JudgeStatusResponse(BaseModel):
    window_index: int | None
    judge_model: str
    rubric_version: str
    baseline_score: float | None
    score: float | None
    score_delta: float | None
    score_z: float | None
    dimension_scores: dict[str, float] = Field(default_factory=dict)
    dimension_deltas: dict[str, float] = Field(default_factory=dict)
    veto_rate: float | None
    pairwise_win_rate: float | None
    n_judged: int
    regression: bool
    severity: str
    recommendation: str = ""


class DecisionResponse(BaseModel):
    window_id: str
    window_index: int
    action: str
    severity: str
    health: float
    confidence: float | None
    rationale: str
    signals: dict[str, Any] = Field(default_factory=dict)


class RunSummaryResponse(BaseModel):
    run_id: str
    started_at: str
    finished_at: str | None
    status: str
    encoder: str | None
    judge: str | None
    app_model: str | None
    dataset_source: str | None
    windows: int


class SeriesResponse(BaseModel):
    run_id: str
    category: str
    name: str
    unit: str | None = None
    points: list[dict[str, Any]] = Field(default_factory=list)


class IncidentResponse(BaseModel):
    incident_id: str
    severity: str
    title: str
    status: str
    opened_at: str
    closed_at: str | None
    timeline: list[dict[str, Any]] = Field(default_factory=list)
    signals: dict[str, Any] = Field(default_factory=dict)


class EvaluateRequest(BaseModel):
    n: int = Field(24, ge=1, le=200, description="Sample size to judge")
    seed: int = Field(0, ge=0)


class EvaluateResponse(BaseModel):
    run_id: str | None
    window_index: int
    n_judged: int
    score: float | None
    baseline_score: float | None
    severity: str
    regression: bool
    dimension_scores: dict[str, float]
    unparseable: int
    elapsed_ms: float