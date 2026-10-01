"""
Central configuration
--------------------
Every threshold, path and backend toggle lives here so that a monitoring
policy is reviewable in one place. Overridable through environment variables
(``.env`` is loaded automatically) using the ``LDM_`` prefix.

Design note: monitoring thresholds are *policy*, not implementation details.
Keeping them out of the detection code means you can argue about alerting
policy separately from statistics — which is exactly what an interview panel
is probing when they ask "how do you decide when to page someone?".
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent


def _env(name: str, default: str) -> str:
    return os.getenv(f"LDM_{name}", default)


def _env_float(name: str, default: float) -> float:
    return float(os.getenv(f"LDM_{name}", default))


def _env_int(name: str, default: int) -> int:
    return int(os.getenv(f"LDM_{name}", default))


def _env_bool(name: str, default: bool) -> bool:
    return os.getenv(f"LDM_{name}", str(default)).strip().lower() in {"1", "true", "yes", "on"}


# ── Paths ────────────────────────────────────────────────────────────────
DATA_DIR = Path(_env("DATA_DIR", str(BASE_DIR / "data")))
RAW_DIR = DATA_DIR / "raw"
CACHE_DIR = DATA_DIR / "cache"
REFERENCE_DIR = DATA_DIR / "reference"
RUNS_DIR = DATA_DIR / "runs"
ARTIFACT_DIR = Path(_env("ARTIFACT_DIR", str(BASE_DIR / "artifacts")))
REPORT_DIR = DATA_DIR / "reports"
TELEMETRY_DB = DATA_DIR / "telemetry.db"
REFERENCE_SNAPSHOT = ARTIFACT_DIR / "reference_snapshot.npz"
REGISTRY_DIR = ARTIFACT_DIR / "registry"
CONFIG_DIR = BASE_DIR / "config"

for _d in (DATA_DIR, RAW_DIR, CACHE_DIR, REFERENCE_DIR, RUNS_DIR, ARTIFACT_DIR,
           REPORT_DIR, REGISTRY_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ── Dataset ──────────────────────────────────────────────────────────────
DATASET_NAME = _env("DATASET", "banking77")
DATASET_URL = _env(
    "DATASET_URL",
    "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data/train.csv",
)
DATASET_URL_TEST = _env(
    "DATASET_URL_TEST",
    "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data/test.csv",
)
# The assistant "launched" for retail card + ATM + top-up support. Production
# traffic later escapes this scope when two new products ship — the single most
# common real-world drift, and the one that actually degrades users.
LAUNCH_INTENTS = tuple(
    s.strip() for s in _env(
        "LAUNCH_INTENTS",
        ",".join([
            # cards
            "card_arrival", "card_delivery_estimate", "card_linking", "card_not_working",
            "card_swallowed", "card_about_to_expire", "card_acceptance", "compromised_card",
            "contactless_not_working", "declined_card_payment", "lost_or_stolen_card",
            "getting_spare_card", "order_physical_card",
            # PIN
            "pin_blocked", "change_pin", "passcode_forgotten",
            # ATMs and cash
            "atm_support", "cash_withdrawal_charge", "cash_withdrawal_not_recognised",
            "pending_cash_withdrawal", "wrong_amount_of_cash_received",
            "wrong_exchange_rate_for_cash_withdrawal",
            # top-ups
            "top_up_by_card_charge", "top_up_limits", "verify_top_up",
            "verify_source_of_funds", "automatic_top_up",
        ]),
    ).split(",") if s.strip()
)

# Intents that only appear in production *after* the shift. Used by the
# simulator to manufacture out-of-scope traffic and to measure the accuracy
# gap between in-scope and out-of-scope slices.
OUT_OF_SCOPE_INTENTS = tuple(
    s.strip() for s in _env(
        "OUT_OF_SCOPE_INTENTS",
        ",".join([
            # international remittance
            "transfer_not_received_by_recipient", "declined_transfer", "transfer_timing",
            "exchange_via_app", "exchange_charge", "fiat_currency_support", "country_support",
            "visa_or_mastercard", "supported_cards_and_currencies",
            # business / virtual cards
            "getting_virtual_card", "virtual_card_not_working", "get_disposable_virtual_card",
            "get_physical_card", "beneficiary_not_allowed", "receiving_money",
            # identity + account admin
            "verify_my_identity", "unable_to_verify_identity", "why_verify_identity",
            "edit_personal_details", "terminate_account", "age_limit",
            # billing
            "Refund_not_showing_up", "request_refund", "extra_charge_on_statement",
            "transaction_charged_twice", "top_up_failed",
        ]),
    ).split(",") if s.strip()
)

# ── Embeddings ───────────────────────────────────────────────────────────
# "auto" tries sentence-transformers, falls back to a deterministic hashing
# encoder so the project stays runnable with zero downloads and zero network.
EMBEDDING_BACKEND = _env("EMBEDDING_BACKEND", "auto")   # auto | sentence-transformers | hashing | ollama
EMBEDDING_MODEL = _env("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
EMBEDDING_DIM = _env_int("EMBEDDING_DIM", 128)
HASHING_NGRAM = _env_int("HASHING_NGRAM", 2)
HASHING_FEATURES = _env_int("HASHING_FEATURES", 2 ** 16)
OLLAMA_HOST = _env("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_EMBED_MODEL = _env("OLLAMA_EMBED_MODEL", "qwen3:8b")

# ── LLM (application under observation) ──────────────────────────────────
LLM_PROVIDER = _env("LLM_PROVIDER", "ollama")           # ollama | openai | anthropic | templated
LLM_MODEL = _env("LLM_MODEL", "qwen3:8b")
LLM_TEMPERATURE = _env_float("LLM_TEMPERATURE", 0.0)
LLM_MAX_TOKENS = _env_int("LLM_MAX_TOKENS", 220)
LLM_TIMEOUT_S = _env_float("LLM_TIMEOUT_S", 60.0)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")

# ── LLM-as-judge ──────────────────────────────────────────────────────────
JUDGE_PROVIDER = _env("JUDGE_PROVIDER", "auto")          # auto | ollama | openai | anthropic | rubric
JUDGE_MODEL = _env("JUDGE_MODEL", "qwen3:8b")
JUDGE_TEMPERATURE = _env_float("JUDGE_TEMPERATURE", 0.0)
JUDGE_MAX_TOKENS = _env_int("JUDGE_MAX_TOKENS", 320)
JUDGE_RUBRIC_VERSION = _env("JUDGE_RUBRIC_VERSION", "v1.0.0")
JUDGE_SAMPLE_SIZE = _env_int("JUDGE_SAMPLE_SIZE", 24)
JUDGE_REPEATS_FOR_SELF_CONSISTENCY = _env_int("JUDGE_REPEATS", 2)
JUDGE_MAX_WORKERS = _env_int("JUDGE_MAX_WORKERS", 4)
# Responses are generated by the LLM for a sample of each window, not for every
# request — the same sample-based evaluation real systems use, because judging
# every completion costs more than the completion.
LLM_RESPONSE_SAMPLE = _env_int("LLM_RESPONSE_SAMPLE", 40)
# Regressions smaller than this many standard errors are treated as noise.
JUDGE_REGRESSION_Z = _env_float("JUDGE_REGRESSION_Z", 2.0)
# Minimum Cohen's kappa vs. ground-truth labels before we trust the judge.
JUDGE_MIN_AGREEMENT_KAPPA = _env_float("JUDGE_MIN_KAPPA", 0.30)

# ── Drift thresholds ──────────────────────────────────────────────────────
PSI_MODERATE = _env_float("PSI_MODERATE", 0.10)
PSI_SEVERE = _env_float("PSI_SEVERE", 0.25)
KS_P_THRESHOLD = _env_float("KS_P_THRESHOLD", 0.01)
MMD_MODERATE = _env_float("MMD_MODERATE", 0.0030)
MMD_SEVERE = _env_float("MMD_SEVERE", 0.0150)
SWD_MODERATE = _env_float("SWD_MODERATE", 0.0055)
SWD_SEVERE = _env_float("SWD_SEVERE", 0.0120)
CENTROID_COS_MODERATE = _env_float("CENTROID_COS_MODERATE", 0.030)
CENTROID_COS_SEVERE = _env_float("CENTROID_COS_SEVERE", 0.070)
DOMAIN_AUC_MODERATE = _env_float("DOMAIN_AUC_MODERATE", 0.62)
DOMAIN_AUC_SEVERE = _env_float("DOMAIN_AUC_SEVERE", 0.78)
# Normalised Frechet distance (raw FID divided by the mean covariance trace).
FRECHET_MODERATE = _env_float("FRECHET_MODERATE", 0.08)
FRECHET_SEVERE = _env_float("FRECHET_SEVERE", 0.25)
NOVELTY_MODERATE = _env_float("NOVELTY_MODERATE", 0.25)
NOVELTY_SEVERE = _env_float("NOVELTY_SEVERE", 0.50)
EMBEDDING_DRIFT_VOTES = _env_int("EMBEDDING_DRIFT_VOTES", 2)

# ── Quality thresholds ─────────────────────────────────────────────────────
ACC_DROP_WARN = _env_float("ACC_DROP_WARN", 0.03)     # absolute F1 drop vs. reference
ACC_DROP_CRITICAL = _env_float("ACC_DROP_CRITICAL", 0.08)
ECE_WARN = _env_float("ECE_WARN", 0.06)
ECE_CRITICAL = _env_float("ECE_CRITICAL", 0.12)
BRIER_WARN = _env_float("BRIER_WARN", 0.15)
BRIER_CRITICAL = _env_float("BRIER_CRITICAL", 0.22)
JUDGE_SCORE_DROP_WARN = _env_float("JUDGE_SCORE_DROP_WARN", 0.10)
JUDGE_SCORE_DROP_CRITICAL = _env_float("JUDGE_SCORE_DROP_CRITICAL", 0.25)
ABSTENTION_JUMP = _env_float("ABSTENTION_JUMP", 0.15)

# ── Monitor cadence ────────────────────────────────────────────────────────
WINDOW_SIZE = _env_int("WINDOW_SIZE", 250)            # traffic per evaluation window
MIN_WINDOW = _env_int("MIN_WINDOW", 40)
REFERENCE_SIZE = _env_int("REFERENCE_SIZE", 1500)
DELAYED_LABEL_LAG_DAYS = _env_int("LABEL_LAG_DAYS", 2)
MMD_PERMUTATIONS = _env_int("MMD_PERMUTATIONS", 200)
DOMAIN_CV_FOLDS = _env_int("DOMAIN_CV_FOLDS", 5)
WASSERSTEIN_PROJECTIONS = _env_int("WASSERSTEIN_PROJECTIONS", 256)
DRIFT_N_BINS = _env_int("DRIFT_N_BINS", 10)

# ── MLflow ──────────────────────────────────────────────────────────────────
MLFLOW_TRACKING_URI = _env("MLFLOW_TRACKING_URI", f"sqlite:///{(BASE_DIR / 'mlruns' / 'mlflow.db').as_posix()}")
MLFLOW_EXPERIMENT = _env("MLFLOW_EXPERIMENT", "llm-drift-monitor")
MLFLOW_MODEL_NAME = _env("MLFLOW_MODEL_NAME", "support-intent-classifier")
PROMOTION_MARGIN = _env_float("PROMOTION_MARGIN", 0.005)

# ── Serving ─────────────────────────────────────────────────────────────────
API_HOST = _env("API_HOST", "127.0.0.1")
API_PORT = _env_int("API_PORT", 8000)
DASHBOARD_REFRESH_S = _env_int("DASHBOARD_REFRESH_S", 15)
LOG_LEVEL = _env("LOG_LEVEL", "INFO")

# ── Serving policy thresholds consumed by the API / orchestrator ────────────
AUTO_RETRAIN_ENABLED = _env_bool("AUTO_RETRAIN", False)


@dataclass(frozen=True)
class Thresholds:
    """Frozen snapshot of the alerting policy, stamped onto every decision."""

    embedding: dict[str, float] = field(default_factory=lambda: {
        "mmd_moderate": MMD_MODERATE,
        "mmd_severe": MMD_SEVERE,
        "swd_moderate": SWD_MODERATE,
        "swd_severe": SWD_SEVERE,
        "centroid_cos_moderate": CENTROID_COS_MODERATE,
        "centroid_cos_severe": CENTROID_COS_SEVERE,
        "domain_auc_moderate": DOMAIN_AUC_MODERATE,
        "domain_auc_severe": DOMAIN_AUC_SEVERE,
        "frechet_moderate": FRECHET_MODERATE,
        "frechet_severe": FRECHET_SEVERE,
        "novelty_moderate": NOVELTY_MODERATE,
        "novelty_severe": NOVELTY_SEVERE,
    })
    quality: dict[str, float] = field(default_factory=lambda: {
        "acc_drop_warn": ACC_DROP_WARN,
        "acc_drop_critical": ACC_DROP_CRITICAL,
        "ece_warn": ECE_WARN,
        "ece_critical": ECE_CRITICAL,
        "brier_warn": BRIER_WARN,
        "brier_critical": BRIER_CRITICAL,
    })
    judge: dict[str, float] = field(default_factory=lambda: {
        "score_drop_warn": JUDGE_SCORE_DROP_WARN,
        "score_drop_critical": JUDGE_SCORE_DROP_CRITICAL,
        "min_kappa": JUDGE_MIN_AGREEMENT_KAPPA,
    })

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


THRESHOLDS = Thresholds()
