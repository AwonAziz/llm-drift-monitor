"""
Telemetry store
---------------
A single SQLite file is the system of record for every window the monitor has
observed. It is deliberately boring: append-only tables, explicit schema, no
ORM magic. If you are going to claim you can reconstruct "what did the model
look like on 3 March at 14:00 UTC", this table is the proof.

Schema
------
``runs``        one row per monitoring execution
``windows``     one row per evaluation window (hour, day, batch)
``traffic``     one row per served request, with prediction + judge outcome
``metrics``     long-format metric time series (category, name, value, severity)
``judge_scores``one row per judged sample x rubric dimension
``decisions``   the triage action taken per window, with the signals behind it
``incidents``   an incident spans windows; opened, updated and closed
"""

from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from collections.abc import Iterable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from config import settings
from src.utils.logging import get_logger, set_log_context
from src.utils.stats import json_safe

logger = get_logger(__name__)

SCHEMA = """
PRAGMA journal_mode=WAL;

CREATE TABLE IF NOT EXISTS runs (
    run_id           TEXT PRIMARY KEY,
    started_at       TEXT NOT NULL,
    finished_at      TEXT,
    status           TEXT NOT NULL DEFAULT 'running',
    dataset_source   TEXT,
    encoder          TEXT,
    judge            TEXT,
    app_model        TEXT,
    notes            TEXT,
    config_snapshot  TEXT
);

CREATE TABLE IF NOT EXISTS windows (
    window_id    TEXT PRIMARY KEY,
    run_id       TEXT NOT NULL REFERENCES runs(run_id),
    window_index INTEGER NOT NULL,
    label        TEXT,
    started_at   TEXT NOT NULL,
    n_traffic    INTEGER DEFAULT 0,
    n_labeled    INTEGER DEFAULT 0,
    notes        TEXT
);
CREATE INDEX IF NOT EXISTS idx_windows_run ON windows(run_id, window_index);

CREATE TABLE IF NOT EXISTS traffic (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    window_id       TEXT NOT NULL REFERENCES windows(window_id),
    run_id          TEXT NOT NULL,
    ts              TEXT NOT NULL,
    request_id      TEXT,
    text_hash       TEXT,
    text_snippet    TEXT,
    in_scope        INTEGER,
    gold_intent     TEXT,
    pred_intent     TEXT,
    confidence      REAL,
    abstained       INTEGER,
    response        TEXT,
    app_latency_ms  REAL,
    judge_total     REAL,
    judge_flags     TEXT
);
CREATE INDEX IF NOT EXISTS idx_traffic_window ON traffic(window_id);
CREATE INDEX IF NOT EXISTS idx_traffic_run ON traffic(run_id);

CREATE TABLE IF NOT EXISTS metrics (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    window_id   TEXT NOT NULL REFERENCES windows(window_id),
    run_id      TEXT NOT NULL,
    window_index INTEGER NOT NULL,
    ts          TEXT NOT NULL,
    category    TEXT NOT NULL,      -- embedding | tabular | quality | judge | volume
    name        TEXT NOT NULL,
    value       REAL,
    baseline    REAL,
    threshold_moderate REAL,
    threshold_severe   REAL,
    severity    TEXT DEFAULT 'none',
    unit        TEXT,
    extra       TEXT
);
CREATE INDEX IF NOT EXISTS idx_metrics_series ON metrics(run_id, category, name, window_index);
CREATE UNIQUE INDEX IF NOT EXISTS idx_metrics_unique ON metrics(run_id, window_index, category, name);

CREATE TABLE IF NOT EXISTS judge_scores (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    window_id     TEXT NOT NULL REFERENCES windows(window_id),
    run_id        TEXT NOT NULL,
    sample_hash   TEXT,
    sample_snippet TEXT,
    dimension     TEXT NOT NULL,
    score         REAL NOT NULL,
    verdict       TEXT,
    rationale     TEXT,
    judge_model   TEXT,
    rubric_version TEXT,
    in_scope      INTEGER,
    agreement_with_gold REAL,
    extra         TEXT
);
CREATE INDEX IF NOT EXISTS idx_judge_window ON judge_scores(window_id, dimension);

CREATE TABLE IF NOT EXISTS decisions (
    window_id   TEXT PRIMARY KEY REFERENCES windows(window_id),
    run_id      TEXT NOT NULL,
    window_index INTEGER NOT NULL,
    ts          TEXT NOT NULL,
    action      TEXT NOT NULL,       -- noop | investigate | retrain | rollback
    severity    TEXT NOT NULL,
    health      REAL,
    confidence  REAL,
    signals     TEXT,
    rationale   TEXT
);

CREATE TABLE IF NOT EXISTS incidents (
    incident_id  TEXT PRIMARY KEY,
    run_id       TEXT NOT NULL,
    window_id    TEXT,
    opened_at    TEXT NOT NULL,
    updated_at   TEXT,
    closed_at    TEXT,
    severity     TEXT NOT NULL,
    title        TEXT,
    signals      TEXT,
    status       TEXT NOT NULL DEFAULT 'open',
    timeline     TEXT
);
CREATE INDEX IF NOT EXISTS idx_incidents_run ON incidents(run_id, status);
"""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class WindowHandle:
    window_id: str
    run_id: str
    window_index: int
    label: str = ""

    def __str__(self) -> str:
        return f"{self.window_id}#{self.window_index}({self.label or 'unlabelled'})"


@dataclass
class MetricPoint:
    category: str
    name: str
    value: float | None
    baseline: float | None = None
    threshold_moderate: float | None = None
    threshold_severe: float | None = None
    severity: str = "none"
    unit: str = ""
    extra: dict[str, Any] = field(default_factory=dict)


class TelemetryStore:
    """Thread-safe append-only telemetry store."""

    def __init__(self, path: Path | str | None = None):
        # Resolved at call time, not as a default argument. A default is bound
        # once at import, which makes the database path impossible to override
        # after startup and impossible to redirect in a test.
        self.path = Path(path if path is not None else settings.TELEMETRY_DB)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self._lock = threading.RLock()
        with self._connect() as conn:
            conn.executescript(SCHEMA)
            self._migrate(conn)
        logger.info("Telemetry store ready at %s", self.path)

    # ── connection handling ──────────────────────────────────────────
    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(self.path, timeout=30.0)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA foreign_keys=ON")
            self._local.conn = conn
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise

    # ── migrations ──────────────────────────────────────────────────
    #: Columns added after v1. ``CREATE TABLE IF NOT EXISTS`` silently leaves an
    #: existing table alone, so a dev database from an earlier run would fail on
    #: insert. This keeps those databases working instead of demanding a wipe.
    ADDED_COLUMNS: dict[str, dict[str, str]] = {
        "decisions": {"confidence": "REAL"},
        "metrics": {"unit": "TEXT"},
        "windows": {"n_labeled": "INTEGER DEFAULT 0"},
    }

    def _migrate(self, conn: sqlite3.Connection) -> None:
        for table, columns in self.ADDED_COLUMNS.items():
            existing = {row["name"] for row in conn.execute(f"PRAGMA table_info({table})")}
            if not existing:
                continue
            for column, ddl in columns.items():
                if column not in existing:
                    conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {ddl}")
                    logger.info("Migrated %s: added column %s", table, column)

    def close(self) -> None:
        conn = getattr(self._local, "conn", None)
        if conn is not None:
            conn.close()
            self._local.conn = None

    # ── runs ─────────────────────────────────────────────────────────
    def start_run(self, *, encoder: str = "", judge: str = "", app_model: str = "",
                  dataset_source: str = "", notes: str = "",
                  config_snapshot: dict | None = None) -> str:
        run_id = f"run_{datetime.now(timezone.utc):%Y%m%dT%H%M%S}_{uuid.uuid4().hex[:6]}"
        with self._lock, self._connect() as conn:
            conn.execute(
                "INSERT INTO runs (run_id, started_at, status, dataset_source, encoder, judge, "
                "app_model, notes, config_snapshot) VALUES (?,?,?,?,?,?,?,?,?)",
                (run_id, utc_now(), "running", dataset_source, encoder, judge, app_model,
                 notes, json.dumps(json_safe(config_snapshot or {}))),
            )
        logger.info("Run started: %s", run_id)
        return run_id

    def finish_run(self, run_id: str, status: str = "complete", notes: str = "") -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                "UPDATE runs SET finished_at=?, status=?, notes=COALESCE(NULLIF(?,''), notes) WHERE run_id=?",
                (utc_now(), status, notes, run_id),
            )
        logger.info("Run finished: %s (%s)", run_id, status)

    def list_runs(self, limit: int = 20) -> pd.DataFrame:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM runs ORDER BY started_at DESC LIMIT ?", (limit,)
            ).fetchall()
        return pd.DataFrame([dict(r) for r in rows]) if rows else pd.DataFrame()

    def get_run(self, run_id: str) -> dict | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM runs WHERE run_id=?", (run_id,)).fetchone()
        return dict(row) if row else None

    def latest_run_id(self, finished_only: bool = False) -> str | None:
        q = "SELECT run_id FROM runs"
        if finished_only:
            q += " WHERE status='complete'"
        q += " ORDER BY started_at DESC LIMIT 1"
        with self._connect() as conn:
            row = conn.execute(q).fetchone()
        return row["run_id"] if row else None

    # ── windows ──────────────────────────────────────────────────────
    def open_window(self, run_id: str, window_index: int, label: str = "", notes: str = "") -> WindowHandle:
        wid = f"{run_id}_w{window_index:03d}"
        with self._lock, self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO windows (window_id, run_id, window_index, label, started_at, notes) "
                "VALUES (?,?,?,?,?,?)",
                (wid, run_id, window_index, label, utc_now(), notes),
            )
        set_log_context(window=label or f"w{window_index:03d}")
        return WindowHandle(wid, run_id, window_index, label)

    def close_window(self, handle: WindowHandle, n_traffic: int, n_labeled: int) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                "UPDATE windows SET n_traffic=?, n_labeled=? WHERE window_id=?",
                (n_traffic, n_labeled, handle.window_id),
            )

    def get_window(self, run_id: str, index: int) -> dict | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM windows WHERE run_id=? AND window_index=?", (run_id, index)
            ).fetchone()
        return dict(row) if row else None

    def list_windows(self, run_id: str) -> pd.DataFrame:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM windows WHERE run_id=? ORDER BY window_index", (run_id,)
            ).fetchall()
        return pd.DataFrame([dict(r) for r in rows])

    # ── traffic ──────────────────────────────────────────────────────
    def log_traffic(self, handle: WindowHandle, records: Sequence[dict[str, Any]]) -> None:
        if not records:
            return
        ts = utc_now()
        rows = [
            (handle.window_id, handle.run_id, ts, r.get("request_id"), r.get("text_hash"),
             r.get("text_snippet"), r.get("in_scope"), r.get("gold_intent"), r.get("pred_intent"),
             r.get("confidence"), r.get("abstained"), r.get("response"), r.get("app_latency_ms"),
             json.dumps(json_safe(r.get("judge_flags", {}))))
            for r in records
        ]
        with self._lock, self._connect() as conn:
            conn.executemany(
                "INSERT INTO traffic (window_id, run_id, ts, request_id, text_hash, text_snippet, "
                "in_scope, gold_intent, pred_intent, confidence, abstained, response, app_latency_ms, judge_flags) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)", rows,
            )
            # Recount rather than increment. A window is written from more than
            # one place and `close_window` also sets the count, so an
            # increment drifts out of sync the moment the call order changes —
            # silently, and the dashboard is the only place it shows up.
            conn.execute(
                "UPDATE windows SET n_traffic=(SELECT COUNT(*) FROM traffic WHERE window_id=?) "
                "WHERE window_id=?", (handle.window_id, handle.window_id),
            )

    def traffic_for_window(self, window_id: str) -> pd.DataFrame:
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM traffic WHERE window_id=? ORDER BY id", (window_id,)).fetchall()
        return pd.DataFrame([dict(r) for r in rows])

    def traffic_for_run(self, run_id: str) -> pd.DataFrame:
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM traffic WHERE run_id=? ORDER BY id", (run_id,)).fetchall()
        return pd.DataFrame([dict(r) for r in rows])

    # ── metrics ──────────────────────────────────────────────────────
    def log_metrics(self, handle: WindowHandle, points: Iterable[MetricPoint]) -> None:
        points = list(points)
        if not points:
            return
        ts = utc_now()
        rows = [
            (handle.window_id, handle.run_id, handle.window_index, ts, p.category, p.name,
             _f(p.value), _f(p.baseline), _f(p.threshold_moderate), _f(p.threshold_severe),
             p.severity, p.unit, json.dumps(json_safe(p.extra)))
            for p in points
        ]
        with self._lock, self._connect() as conn:
            conn.executemany(
                "INSERT INTO metrics (window_id, run_id, window_index, ts, category, name, value, "
                "baseline, threshold_moderate, threshold_severe, severity, unit, extra) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?) "
                "ON CONFLICT(run_id, window_index, category, name) DO UPDATE SET "
                "value=excluded.value, baseline=excluded.baseline, severity=excluded.severity, "
                "threshold_moderate=excluded.threshold_moderate, threshold_severe=excluded.threshold_severe, "
                "extra=excluded.extra, ts=excluded.ts",
                rows,
            )

    def metric_series(self, run_id: str, category: str | None = None,
                      name: str | None = None) -> pd.DataFrame:
        q = "SELECT * FROM metrics WHERE run_id=?"
        args: list[Any] = [run_id]
        if category:
            q += " AND category=?"
            args.append(category)
        if name:
            q += " AND name=?"
            args.append(name)
        q += " ORDER BY window_index"
        with self._connect() as conn:
            rows = conn.execute(q, args).fetchall()
        return pd.DataFrame([dict(r) for r in rows])

    def metric_pivot(self, run_id: str) -> pd.DataFrame:
        df = self.metric_series(run_id)
        if df.empty:
            return df
        wide = df.pivot_table(index="window_index", columns="name", values="value", aggfunc="last")
        labels = self.list_windows(run_id).set_index("window_index")["label"] if not self.list_windows(run_id).empty else None
        wide = wide.join(labels) if labels is not None else wide
        return wide.reset_index()

    # ── judge ────────────────────────────────────────────────────────
    def log_judge_scores(self, handle: WindowHandle, records: Sequence[dict[str, Any]]) -> None:
        if not records:
            return
        rows = [
            (handle.window_id, handle.run_id, r.get("sample_hash"), r.get("sample_snippet"),
             r["dimension"], float(r["score"]), r.get("verdict"), r.get("rationale"),
             r.get("judge_model"), r.get("rubric_version"), r.get("in_scope"),
             _f(r.get("agreement_with_gold")), json.dumps(json_safe(r.get("extra", {}))))
            for r in records
        ]
        with self._lock, self._connect() as conn:
            conn.executemany(
                "INSERT INTO judge_scores (window_id, run_id, sample_hash, sample_snippet, dimension, "
                "score, verdict, rationale, judge_model, rubric_version, in_scope, agreement_with_gold, extra) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)", rows,
            )

    def judge_scores(self, run_id: str) -> pd.DataFrame:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT j.*, w.window_index, w.label FROM judge_scores j "
                "JOIN windows w ON w.window_id=j.window_id WHERE j.run_id=? ORDER BY w.window_index",
                (run_id,),
            ).fetchall()
        return pd.DataFrame([dict(r) for r in rows])

    # ── decisions + incidents ────────────────────────────────────────
    def log_decision(self, handle: WindowHandle, action: str, severity: str,
                     health: float, signals: dict[str, Any], rationale: str,
                     confidence: float = 0.0) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO decisions (window_id, run_id, window_index, ts, action, "
                "severity, health, confidence, signals, rationale) VALUES (?,?,?,?,?,?,?,?,?,?)",
                (handle.window_id, handle.run_id, handle.window_index, utc_now(), action, severity,
                 _f(health), _f(confidence), json.dumps(json_safe(signals)), rationale),
            )

    def decisions(self, run_id: str) -> pd.DataFrame:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM decisions WHERE run_id=? ORDER BY window_index", (run_id,)
            ).fetchall()
        return pd.DataFrame([dict(r) for r in rows])

    def open_incident(self, run_id: str, window_id: str, severity: str, title: str,
                      signals: dict[str, Any]) -> str:
        inc_id = f"inc_{uuid.uuid4().hex[:10]}"
        now = utc_now()
        with self._lock, self._connect() as conn:
            conn.execute(
                "INSERT INTO incidents (incident_id, run_id, window_id, opened_at, updated_at, "
                "severity, title, signals, status, timeline) VALUES (?,?,?,?,?,?,?,?,'open',?)",
                (inc_id, run_id, window_id, now, now, severity, title,
                 json.dumps(json_safe(signals)), json.dumps([{"at": now, "event": "opened"}])),
            )
        logger.warning("INCIDENT %s [%s] %s", inc_id, severity.upper(), title)
        return inc_id

    def append_incident_timeline(self, incident_id: str, event: str, data: dict | None = None) -> None:
        with self._lock, self._connect() as conn:
            row = conn.execute("SELECT timeline FROM incidents WHERE incident_id=?", (incident_id,)).fetchone()
            if row is None:
                return
            timeline = json.loads(row["timeline"] or "[]")
            timeline.append({"at": utc_now(), "event": event, "data": json_safe(data or {})})
            conn.execute(
                "UPDATE incidents SET timeline=?, updated_at=? WHERE incident_id=?",
                (json.dumps(timeline), utc_now(), incident_id),
            )

    def close_incident(self, incident_id: str, resolution: str) -> None:
        with self._lock, self._connect() as conn:
            row = conn.execute("SELECT timeline FROM incidents WHERE incident_id=?", (incident_id,)).fetchone()
            if row is None:
                return
            timeline = json.loads(row["timeline"] or "[]")
            timeline.append({"at": utc_now(), "event": "closed", "data": {"resolution": resolution}})
            conn.execute(
                "UPDATE incidents SET status='closed', closed_at=?, updated_at=?, timeline=?, "
                "title=COALESCE(title,'')||'' WHERE incident_id=?",
                (utc_now(), utc_now(), json.dumps(timeline), incident_id),
            )
        logger.info("INCIDENT %s closed: %s", incident_id, resolution)

    def incidents(self, run_id: str | None = None, status: str | None = None) -> pd.DataFrame:
        q, args = "SELECT * FROM incidents WHERE 1=1", []
        if run_id:
            q += " AND run_id=?"
            args.append(run_id)
        if status:
            q += " AND status=?"
            args.append(status)
        q += " ORDER BY opened_at DESC"
        with self._connect() as conn:
            rows = conn.execute(q, args).fetchall()
        return pd.DataFrame([dict(r) for r in rows])

    def open_incidents_for_run(self, run_id: str) -> list[dict]:
        df = self.incidents(run_id, status="open")
        return df.to_dict("records") if not df.empty else []

    # ── convenience ──────────────────────────────────────────────────
    def reset(self) -> None:
        """Drop all telemetry. Used by tests and by `--reset` runs."""
        with self._lock, self._connect() as conn:
            for table in ("traffic", "metrics", "judge_scores", "decisions",
                          "incidents", "windows", "runs"):
                conn.execute(f"DELETE FROM {table}")
        logger.warning("Telemetry store reset.")


def _f(v: Any) -> float | None:
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if f != f else f  # NaN -> None
