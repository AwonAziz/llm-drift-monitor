"""
The system under observation
----------------------------
A retail-banking support assistant, built the way a small team would actually
build one:

    customer text ──► intent classifier (TF-IDF + calibrated logistic regression)
                   ──► intent playbook lookup
                   ──► LLM drafts a customer-facing reply from the playbook
                   ──► structured AssistantOutput

Everything the monitoring platform measures flows through this object, which
means the platform is not evaluating a notebook — it is evaluating a serving
path with the same shapes a real system produces (confidence, latency,
abstention, free-text output).

The classifier is *calibrated* on purpose. A raw ``LogisticRegression`` on
high-dimensional sparse features is badly overconfident, and an uncalibrated
score makes every downstream quality metric — ECE, Brier, the abstention
threshold, the judge's confidence weighting — meaningless. Calibrating costs
one extra cross-validation pass and makes the monitoring numbers mean something.
"""

from __future__ import annotations

import json
import pickle
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from config import settings
from src.llm.client import BaseLLM, build_llm
from src.utils.logging import get_logger
from src.utils.stats import stable_hash

logger = get_logger(__name__)

MODEL_PATH = settings.ARTIFACT_DIR / "assistant" / "classifier.joblib"
META_PATH = settings.ARTIFACT_DIR / "assistant" / "meta.json"


def humanise(intent: str) -> str:
    return intent.replace("_", " ").strip().capitalize()


# ── Intent playbooks ───────────────────────────────────────────────────

PLAYBOOKS: dict[str, dict[str, Any]] = {
    "card_arrival": {
        "summary": "Standard card delivery is 3-5 working days after approval.",
        "steps": ["Confirm the delivery address on the account.",
                  "Check the dispatch tracking reference in the app.",
                  "If tracking has not moved for 48 hours, raise a card-dispatch enquiry."],
        "eta": "3-5 working days",
    },
    "card_not_working": {
        "summary": "A declined or unusable card is usually a PIN block or an activation issue.",
        "steps": ["Confirm the card has been activated in the app.",
                  "Ask whether the customer has entered the PIN three times.",
                  "If PIN-blocked, direct them to PIN reset and card replacement."],
        "eta": "PIN reset is instant; replacement is 3-5 working days",
    },
    "declined_card_payment": {
        "summary": "Card payments can decline for balance, limit, expiry or merchant category reasons.",
        "steps": ["Confirm available balance and daily limits.",
                  "Ask whether the merchant is card-present and in region.",
                  "If unexplained, request a fresh authorisation and note the decline code."],
        "eta": "Immediate if a limit change fixes it",
    },
    "lost_or_stolen_card": {
        "summary": "Lost or stolen cards must be frozen immediately and replaced.",
        "steps": ["Freeze the card in the app before any further discussion.",
                  "Confirm the last known transaction history.",
                  "Order a replacement and confirm the delivery address."],
        "eta": "Card frozen immediately; replacement 3-5 working days",
    },
    "cash_withdrawal_charge": {
        "summary": "ATM withdrawal fees depend on the plan and whether the operator is in network.",
        "steps": ["Check whether the ATM operator is in-network.",
                  "Explain the fee schedule for the customer's plan.",
                  "Escalate to fee review only if the charge matches a known operator glitch."],
        "eta": "Immediate",
    },
    "atm_support": {
        "summary": "ATM issues cover availability, cash limits and machines keeping cards.",
        "steps": ["Locate the nearest in-network ATM in the app.",
                  "Check daily cash limits against the customer's plan.",
                  "If a machine kept the card, start a card-swallow claim with the ATM reference."],
        "eta": "Limits change immediately; claims take 3-5 working days",
    },
    "pin_blocked": {
        "summary": "Three incorrect PIN entries lock the card until reset.",
        "steps": ["Confirm the PIN block is active in the app.",
                  "Guide a PIN reset or in-branch identity check.",
                  "Do not disclose the PIN or ask the customer to state it."],
        "eta": "Reset is immediate",
    },
    "change_pin": {
        "summary": "PINs can be changed in the app or at any in-branch cash machine.",
        "steps": ["Confirm identity before issuing a new PIN.",
                  "Direct to the app's security settings or an in-branch machine.",
                  "Confirm the new PIN takes effect immediately."],
        "eta": "Immediate",
    },
    "top_up_limits": {
        "summary": "Top-up limits are set by plan, verification level and payment source.",
        "steps": ["Check the customer's current limit in the app.",
                  "Explain that higher limits require additional verification.",
                  "Avoid committing to a specific figure before verification completes."],
        "eta": "Limits update after verification",
    },
    "default": {
        "summary": "Route to a human support agent with the conversation transcript.",
        "steps": ["Acknowledge the request and capture the customer's intent.",
                  "Do not speculate about account specifics.",
                  "Escalate with a priority tag and a 24-hour SLA."],
        "eta": "24 hours",
    },
}

RESPONDER_SYSTEM = (
    "You are a customer-support agent for a retail bank. "
    "Answer using only the playbook facts supplied. "
    "Never invent fees, limits, dates or account-specific balances. "
    "If the playbook does not cover the request, say the agent will escalate to a specialist. "
    "Write 2-4 short sentences in plain English, addressed to the customer. No markdown."
)


@dataclass
class AssistantOutput:
    """Everything one served request produced. This is the telemetry unit."""

    text: str
    intent: str
    confidence: float
    margin: float
    abstained: bool
    response: str
    app_latency_ms: float
    classifier_latency_ms: float = 0.0
    llm_latency_ms: float = 0.0
    model_version: str = ""
    request_id: str = ""
    top3: list[dict[str, Any]] = field(default_factory=list)

    def to_record(self, *, gold_intent: str | None, in_scope: bool) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "text_hash": stable_hash(self.text) % (10**12),
            "text_snippet": self.text[:240],
            "gold_intent": gold_intent,
            "in_scope": int(bool(in_scope)),
            "pred_intent": self.intent,
            "confidence": round(float(self.confidence), 4),
            "abstained": int(bool(self.abstained)),
            "response": self.response,
            "app_latency_ms": round(float(self.app_latency_ms), 3),
        }


# ── Classifier ─────────────────────────────────────────────────────────

class IntentClassifier:
    """Calibrated linear intent classifier with temperature scaling and abstention."""

    def __init__(self, pipeline=None, classes: Sequence[str] = (), version: str = "",
                 temperature: float = 1.0, abstain_threshold: float = 0.45):
        self.pipeline = pipeline
        self.classes: list[str] = list(classes)
        self.version = version or hashlib_short("unversioned")
        self.temperature = float(temperature)
        self.abstain_threshold = abstain_threshold

    # -- lifecycle ----------------------------------------------------
    @classmethod
    def train(cls, df: pd.DataFrame, *, label_col: str = "gold_intent",
              version: str | None = None, seed: int = 42,
              abstain_threshold: float = 0.45) -> IntentClassifier:
        from sklearn.calibration import CalibratedClassifierCV
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.pipeline import FeatureUnion, Pipeline
        from sklearn.svm import LinearSVC

        X = df["text"].astype(str)
        y = df[label_col].astype(str)
        union = FeatureUnion([
            ("word", TfidfVectorizer(analyzer="word", ngram_range=(1, 2),
                                     min_df=2, sublinear_tf=True, max_features=40_000)),
            ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5),
                                     min_df=3, sublinear_tf=True, max_features=60_000)),
        ])
        base = LinearSVC(C=0.8, random_state=seed)
        calibrated = CalibratedClassifierCV(base, method="sigmoid", cv=3)
        pipe = Pipeline([("features", union), ("clf", calibrated)])
        t0 = time.perf_counter()
        pipe.fit(X, y)
        elapsed = time.perf_counter() - t0

        classes = list(pipe.named_steps["clf"].classes_)
        model = cls(pipeline=pipe, classes=classes, version=version or f"clf_{hashlib_short(classes[0] + str(len(classes)))}")
        model.abstain_threshold = abstain_threshold
        logger.info("Trained intent classifier: %d classes in %.1fs (%s)",
                    len(classes), elapsed, model.version)
        return model

    # -- calibration --------------------------------------------------
    def fit_temperature(self, df: pd.DataFrame, *, label_col: str = "gold_intent",
                        text_col: str = "text", verbose: bool = True) -> float:
        """
        Fit a single temperature by minimising NLL on held-out labelled data.

        ``CalibratedClassifierCV(sigmoid)`` calibrates *within* the cross-validation
        folds, so its training-set fit is already better than anything the model
        will see at serving time — and it leaves the system noticeably
        overconfident (ECE ≈ 0.15 on this data). One temperature refit on a
        genuinely held-out split fixes most of that for the cost of a scalar.

        This matters beyond a nicer dashboard number: the abstention threshold,
        the confidence-weighted judge sampling and every calibration metric all
        read this score. An overconfident score makes abstention fire far too
        rarely, which is exactly the failure you cannot see without monitoring.
        """
        from scipy.optimize import minimize_scalar
        from sklearn.metrics import log_loss

        labeled = df[df[label_col].notna() & df[text_col].notna()]
        if len(labeled) < 60:
            logger.warning("Too few labelled rows (%d) to fit temperature; keeping T=1.", len(labeled))
            return self.temperature

        y_true = labeled[label_col].astype(str).to_numpy()
        # Select the gold class column positionally — validation rows may carry
        # intents the classifier was never trained on, and those must be mapped
        # to "no column", not to a wrong index.
        class_to_col = {c: i for i, c in enumerate(self.classes)}
        gold_cols = np.array([class_to_col.get(label, -1) for label in y_true])
        known = gold_cols >= 0
        if known.sum() < 60:
            logger.warning("Only %d labelled rows fall inside the model's scope; keeping T=1.",
                           int(known.sum()))
            return self.temperature
        labeled = labeled[known]
        y_true = y_true[known]
        gold_cols = gold_cols[known]

        probs = np.clip(self.pipeline.predict_proba(labeled[text_col].astype(str)), 1e-9, 1.0)

        # Temperature scaling on probabilities is p_i^(1/T) renormalised — the
        # softmax-over-logits form, recovered from the probabilities. It has to
        # be applied to the *whole* matrix: reconstructing a distribution from
        # the gold column alone and spreading the remainder evenly changes the
        # objective, and the optimiser happily runs the temperature to its bound
        # and produces a model that is uniformly unsure about everything.
        def nll(log_t: float) -> float:
            t = float(np.exp(log_t))
            p = probs ** (1.0 / t)
            p /= p.sum(axis=1, keepdims=True)
            return float(log_loss(y_true, p, labels=list(self.classes)))

        result = minimize_scalar(nll, bounds=(np.log(0.2), np.log(6.0)), method="bounded")
        new_t = float(np.exp(result.x))
        before, after = nll(np.log(1.0)), float(result.fun)
        self.temperature = new_t
        if verbose:
            logger.info("Temperature scaling: T=%.3f (validation NLL %.4f -> %.4f)", new_t, before, after)
        return new_t

    # -- inference ----------------------------------------------------
    def predict(self, texts: Sequence[str]) -> list[dict[str, Any]]:
        if not texts:
            return []
        probs = self.pipeline.predict_proba(list(texts))
        if abs(self.temperature - 1.0) > 1e-6:
            p = np.clip(probs, 1e-9, 1.0) ** (1.0 / self.temperature)
            probs = p / p.sum(axis=1, keepdims=True)
        classes = np.asarray(self.classes)
        order = np.argsort(-probs, axis=1)
        out = []
        for row, idx in zip(probs, order):
            top = float(row[idx[0]])
            second = float(row[idx[1]]) if len(idx) > 1 else 0.0
            out.append({
                "intent": str(classes[idx[0]]),
                "confidence": top,
                "margin": float(top - second),
                "abstained": bool(top < self.abstain_threshold),
                "top3": [{"intent": str(classes[i]), "p": float(row[i])} for i in idx[:3]],
            })
        return out

    def predict_frame(self, df: pd.DataFrame, text_col: str = "text") -> pd.DataFrame:
        preds = self.predict(df[text_col].astype(str).tolist())
        margin = [p["margin"] for p in preds]
        return pd.DataFrame({
            "pred_intent": [p["intent"] for p in preds],
            "confidence": [p["confidence"] for p in preds],
            "margin": margin,
            "abstained": [p["abstained"] for p in preds],
        })

    # -- persistence --------------------------------------------------
    def save(self, path: Path = MODEL_PATH) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as fh:
            pickle.dump({"pipeline": self.pipeline, "classes": self.classes,
                         "version": self.version, "abstain_threshold": self.abstain_threshold,
                         "temperature": self.temperature}, fh)
        META_PATH.write_text(json.dumps({
            "version": self.version, "classes": self.classes,
            "abstain_threshold": self.abstain_threshold,
            "temperature": self.temperature,
        }, indent=2), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: Path = MODEL_PATH) -> IntentClassifier:
        if not Path(path).exists():
            raise FileNotFoundError(f"No classifier at {path}. Run scripts/bootstrap.py.")
        with Path(path).open("rb") as fh:
            blob = pickle.load(fh)
        model = cls(blob["pipeline"], blob["classes"], blob["version"],
                    temperature=blob.get("temperature", 1.0))
        return model._with_threshold(blob.get("abstain_threshold", 0.45))

    def _with_threshold(self, value: float) -> IntentClassifier:
        self.abstain_threshold = value
        return self

    @property
    def signature(self) -> str:
        return f"{self.version}:{len(self.classes)}intents:T{self.temperature:.2f}"


def hashlib_short(text: str, n: int = 8) -> str:
    return f"{stable_hash(text) % (16 ** n):0{n}x}"


# ── Assistant ──────────────────────────────────────────────────────────

class SupportAssistant:
    """The application under observation: classify, look up a playbook, respond."""

    def __init__(self, classifier: IntentClassifier, llm: BaseLLM | None = None,
                 *, use_llm: bool = True, name: str = "support-assistant-v1"):
        self.classifier = classifier
        self.llm = llm if llm is not None else build_llm()
        self.use_llm = use_llm
        self.name = name
        self._responder_log: list[str] = []

    def build_responder_prompt(self, text: str, intent: str, playbook: dict[str, Any],
                               confidence: float) -> str:
        steps = "\n".join(f"- {s}" for s in playbook.get("steps", []))
        return (
            f"Intent: {humanise(intent)}\n"
            f"Playbook summary: {playbook.get('summary', '')}\n"
            f"Expected resolution: {playbook.get('eta', '')}\n"
            f"Steps:\n{steps}\n\n"
            f"Classifier confidence: {confidence:.2f}\n\n"
            f"Customer message: {text}\n\n"
            "Write the reply now."
        )

    def handle(self, text: str) -> AssistantOutput:
        t0 = time.perf_counter()
        pred = self.classifier.predict([text])[0]
        t_class = time.perf_counter()

        playbook = PLAYBOOKS.get(pred["intent"], PLAYBOOKS["default"])
        if pred["abstained"]:
            playbook = PLAYBOOKS["default"]

        t_llm = time.perf_counter()
        if self.use_llm:
            prompt = self.build_responder_prompt(text, pred["intent"], playbook, pred["confidence"])
            try:
                reply = self.llm.generate(prompt, system=RESPONDER_SYSTEM).text
            except Exception as exc:  # noqa: BLE001 - a serving path must not crash on the LLM
                logger.warning("LLM generation failed (%s); serving templated fallback.", exc)
                reply = templated_response(pred["intent"], playbook)
        else:
            reply = templated_response(pred["intent"], playbook)
        t_end = time.perf_counter()

        return AssistantOutput(
            text=text,
            intent=pred["intent"],
            confidence=float(pred["confidence"]),
            margin=float(pred["margin"]),
            abstained=bool(pred["abstained"]),
            response=reply,
            app_latency_ms=(t_end - t0) * 1000,
            classifier_latency_ms=(t_class - t0) * 1000,
            llm_latency_ms=(t_end - t_llm) * 1000,
            model_version=self.classifier.version,
            request_id=f"req_{stable_hash(text + str(time.time_ns())) % 10**12}",
            top3=pred["top3"],
        )

    def handle_frame(self, df: pd.DataFrame, text_col: str = "text",
                     progress_every: int = 25) -> list[AssistantOutput]:
        outputs: list[AssistantOutput] = []
        for i, text in enumerate(df[text_col].astype(str).tolist(), start=1):
            outputs.append(self.handle(text))
            if progress_every and i % progress_every == 0:
                logger.info("  served %d/%d", i, len(df))
        return outputs

    def classify_frame(self, df: pd.DataFrame, text_col: str = "text") -> pd.DataFrame:
        """Classify without generating — used by fast bulk labelling and tests."""
        return self.classifier.predict_frame(df, text_col)


def templated_response(intent: str, playbook: dict[str, Any]) -> str:
    """Deterministic reply used when the LLM is disabled or unavailable."""
    first = playbook.get("steps", ["A specialist will be in touch."])[0]
    eta = playbook.get("eta", "within 24 hours")
    return (
        f"Thanks for contacting us about {humanise(intent).lower()}. {first} "
        f"Expected resolution: {eta}. Is there anything else I can help you with?"
    )