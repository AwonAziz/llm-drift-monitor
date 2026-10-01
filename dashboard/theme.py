"""Shared dashboard helpers: theme, data access and chart primitives."""

from __future__ import annotations

import json
import math
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.graph_objects as go

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ── theme ──────────────────────────────────────────────────────────────
BG = "#0d1117"
PANEL = "#161b22"
BORDER = "#21262d"
TEXT = "#c9d1d9"
MUTED = "#8b949e"
BLUE = "#58a6ff"
GREEN = "#3fb950"
YELLOW = "#d29922"
RED = "#f85149"
PURPLE = "#a371f7"
ORANGE = "#f0883e"

SEVERITY_COLOR = {"none": GREEN, "moderate": YELLOW, "severe": RED}
ACTION_COLOR = {"noop": GREEN, "investigate": YELLOW, "retrain": ORANGE, "rollback": RED}

REGIME_COLORS = {
    "baseline": GREEN,
    "volume_spike": BLUE,
    "style_shift": PURPLE,
    "new_intents": ORANGE,
    "mixed_crisis": RED,
    "recovery": BLUE,
}


def plotly_layout(title: str = "", height: int = 300, showlegend: bool = True) -> dict[str, Any]:
    return dict(
        title=dict(text=title, font=dict(size=14, color=TEXT)) if title else None,
        height=height,
        paper_bgcolor=PANEL,
        plot_bgcolor=PANEL,
        font=dict(color=TEXT, size=11),
        margin=dict(l=52, r=18, t=34 if title else 12, b=36),
        xaxis=dict(gridcolor=BORDER, zerolinecolor=BORDER, tickfont=dict(color=MUTED, size=10)),
        yaxis=dict(gridcolor=BORDER, zerolinecolor=BORDER, tickfont=dict(color=MUTED, size=10)),
        hoverlabel=dict(bgcolor=PANEL, font_size=12),
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1,
                    bgcolor="rgba(0,0,0,0)"),
        showlegend=showlegend,
    )


def line_with_bands(labels: Sequence[str], values: Sequence[float],
                     moderate: float | None = None, severe: float | None = None,
                     color: str = BLUE, name: str = "", connect: bool = True) -> go.Scatter:
    """A metric line with its alert thresholds drawn in, so the chart states the policy."""
    fig = go.Figure()
    y = [None if v is None or (isinstance(v, float) and math.isnan(v)) else float(v) for v in values]
    if connect and any(v is not None for v in y):
        # Bridge gaps so a window with no labels does not break the line.
        filled: list[float | None] = []
        last = None
        for v in y:
            if v is not None:
                last = v
            filled.append(v if v is not None else last)
        y = filled
    fig.add_trace(go.Scatter(
        x=list(labels), y=y, mode="lines+markers", name=name,
        line=dict(color=color, width=2), marker=dict(size=6),
        hovertemplate="w%{x}<br>%{y:.4f}<extra></extra>"))
    if moderate is not None:
        fig.add_hline(y=moderate, line_dash="dash", line_color=YELLOW,
                      annotation_text="moderate", annotation_font=dict(size=9, color=YELLOW))
    if severe is not None:
        fig.add_hline(y=severe, line_dash="dot", line_color=RED,
                      annotation_text="severe", annotation_font=dict(size=9, color=RED))
    return fig


def regime_bands(fig: go.Figure, labels: Sequence[str]) -> go.Figure:
    """Shade each regime run so a change in the background is visible on every chart."""
    groups: list[tuple[str, int]] = []
    for i, label in enumerate(labels):
        if groups and groups[-1][0] == label:
            continue
        groups.append((str(label), i))
    colors = {name: REGIME_COLORS.get(name, MUTED) for name, _ in groups}
    for idx, (name, start) in enumerate(groups):
        end = groups[idx + 1][1] - 0.5 if idx + 1 < len(groups) else len(labels) - 0.5
        fig.add_vrect(x0=start - 0.5, x1=end, fillcolor=colors[name],
                      opacity=0.07, line_width=0, layer="below")
        if idx > 0:
            fig.add_vline(x=start - 0.5, line_dash="dot", line_color=BORDER, line_width=1)
    return fig


# ── data access ────────────────────────────────────────────────────────

def load_store() -> Any:
    from src.storage.telemetry import TelemetryStore

    return TelemetryStore()


def resolve_run(store: Any, run_id: str | None) -> str | None:
    return run_id or store.latest_run_id()


def series(store: Any, run_id: str, category: str, name: str) -> pd.DataFrame:
    df = store.metric_series(run_id, category, name)
    if df.empty:
        return df
    return df.sort_values("window_index")


def pivot(store: Any, run_id: str) -> pd.DataFrame:
    df = store.metric_series(run_id)
    if df.empty:
        return df
    windows = store.list_windows(run_id)
    wide = df.pivot_table(index="window_index", columns="name", values="value", aggfunc="last")
    sev = df.pivot_table(index="window_index", columns="name", values="severity", aggfunc="last")
    wide = wide.join(sev, rsuffix="_sev")
    if not windows.empty:
        wide = wide.join(windows.set_index("window_index")["label"])
    return wide.reset_index()


def fmt(v: Any, n: int = 3, dash: str = "—") -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return dash
    try:
        f = float(v)
    except (TypeError, ValueError):
        return str(v)
    if not math.isfinite(f):
        return dash
    return f"{f:.{n}f}"


def severity_badge(severity: str) -> str:
    colour = SEVERITY_COLOR.get(severity, MUTED)
    return (f"<span style='background:{colour}22;color:{colour};padding:2px 10px;"
            f"border-radius:11px;font-size:11px;font-weight:700'>{severity.upper()}</span>")


def kpi(label: str, value: str, delta: str = "", delta_colour: str = MUTED) -> str:
    return (
        f"<div style='background:{PANEL};border:1px solid {BORDER};border-radius:9px;padding:11px 13px'>"
        f"<div style='font-size:10px;color:{MUTED};text-transform:uppercase;letter-spacing:.07em'>{label}</div>"
        f"<div style='font-size:22px;font-weight:700;color:{TEXT};margin-top:3px'>{value}</div>"
        f"<div style='font-size:11px;color:{delta_colour}'>{delta}</div></div>"
    )


def decode_json(value: Any, default: Any) -> Any:
    if isinstance(value, str) and value:
        try:
            return json.loads(value)
        except Exception:  # noqa: BLE001
            return default
    return default