"""
LLM Drift & Quality Dashboard
-----------------------------
Five tabs, ordered the way an on-call engineer actually reads an incident:

1. **Overview** — health, the decision the policy made and why, open incidents.
2. **Embedding drift** — the input-distribution detectors, over time and as a
   projection, with the reference cloud behind them.
3. **Output quality** — accuracy, calibration (with a reliability curve),
   abstention and per-intent damage.
4. **LLM-as-judge** — rubric scores, per-dimension movement, vetoes, and the
   judge regression suite.
5. **Run detail** — every window, every metric, the raw decisions.

Run: ``streamlit run dashboard/app.py``
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from dashboard.theme import (  # noqa: E402
    ACTION_COLOR,
    BLUE,
    BORDER,
    GREEN,
    MUTED,
    ORANGE,
    PANEL,
    PURPLE,
    RED,
    SEVERITY_COLOR,
    TEXT,
    YELLOW,
    decode_json,
    fmt,
    kpi,
    line_with_bands,
    load_store,
    pivot,
    plotly_layout,
    regime_bands,
    series,
    severity_badge,
)

st.set_page_config(page_title="LLM Drift Monitor", page_icon="🛰️", layout="wide",
                   initial_sidebar_state="expanded")

st.markdown(f"""
<style>
  .block-container {{ padding-top: 1.4rem; padding-bottom: 3rem; max-width: 1600px; }}
  h1, h2, h3 {{ color: {TEXT}; }}
  .section {{ font-size:12px; font-weight:700; color:{MUTED}; text-transform:uppercase;
              letter-spacing:.08em; margin: 1.2rem 0 .45rem; }}
  .narrative {{ background:{PANEL}; border:1px solid {BORDER}; border-left:3px solid {BLUE};
                border-radius:7px; padding:11px 15px; color:{TEXT}; font-size:13.5px;
                line-height:1.6; margin-bottom:.7rem; }}
  .narrative.severe {{ border-left-color:{RED}; }}
  .narrative.moderate {{ border-left-color:{YELLOW}; }}
  .narrative.none {{ border-left-color:{GREEN}; }}
  div[data-testid="stMetricValue"] {{ color:{TEXT}; }}
  .stTabs [data-baseweb="tab-list"] {{ gap: .35rem; }}
</style>
""", unsafe_allow_html=True)

store = load_store()

# ── Sidebar ────────────────────────────────────────────────────────────
runs = store.list_runs(20)
with st.sidebar:
    st.markdown("### 🛰️ LLM Drift Monitor")
    st.caption("Embedding drift · output quality · LLM-as-judge")
    if runs.empty:
        st.error("No monitoring runs found.\n\nRun `python scripts/demo_shift.py` first.")
        st.stop()
    labels = [f"{r['run_id'][4:]}  ·  {str(r['status'])}" for _, r in runs.iterrows()]
    selected = st.selectbox("Run", labels, index=0)
    run_id = runs.iloc[labels.index(selected)]["run_id"]
    run = runs[runs["run_id"] == run_id].iloc[0].to_dict()

    st.markdown("---")
    st.caption(f"**encoder**  `{run.get('encoder', '?')}`")
    st.caption(f"**judge**  `{run.get('judge', '?')}`")
    st.caption(f"**model**  `{run.get('app_model', '?')}`")
    st.caption(f"**dataset**  `{run.get('dataset_source', '?')}`")

    st.markdown("---")
    if st.button("▶ Re-judge latest window", use_container_width=True):
        with st.spinner("Judging…"):
            from src.app.assistant import PLAYBOOKS
            from src.judge import LLMBasedJudge
            windows = store.list_windows(run_id)
            wid = windows.sort_values("window_index").iloc[-1]["window_id"]
            traffic = store.traffic_for_window(wid)
            if traffic.empty:
                st.warning("No traffic in the latest window.")
            else:
                frame = traffic.rename(columns={"text_snippet": "text"})
                result = LLMBasedJudge().evaluate_window(frame, n=12, playbooks=PLAYBOOKS)
                st.success(f"Ad-hoc judge: {fmt(result.mean_score)} over {result.n_judged} samples")
    st.markdown("---")
    st.markdown("**Artifacts**")
    for name in ("report",):
        for p in sorted(Path("data/reports").glob(f"{name}_*.html"))[-1:]:
            st.caption(str(p))

# ── Data ───────────────────────────────────────────────────────────────
windows_df = store.list_windows(run_id)
decisions_df = store.decisions(run_id)
incidents_df = store.incidents(run_id)
metrics_df = store.metric_series(run_id)
wide = pivot(run_id)

if windows_df.empty:
    st.warning("This run has no windows.")
    st.stop()

order = windows_df.sort_values("window_index")
wlabels = [str(label) for label in order["label"].tolist()]
widx = order["window_index"].tolist()
sev_lookup: dict[str, dict[int, str]] = {}
if not metrics_df.empty:
    for (cat, name), grp in metrics_df.groupby(["category", "name"]):
        sev_lookup.setdefault(cat, {})[name] = dict(
            zip(grp.sort_values("window_index")["window_index"], grp["severity"]))


def vals(category: str, name: str) -> list:
    return series(store, run_id, category, name)["value"].tolist()


def thr(category: str, name: str) -> tuple[float | None, float | None]:
    df = series(store, run_id, category, name)
    if df.empty:
        return None, None
    row = df.iloc[0]
    mod = row["threshold_moderate"]
    sev = row["threshold_severe"]
    try:
        return (float(mod) if mod is not None else None,
                float(sev) if sev is not None else None)
    except (TypeError, ValueError):
        return None, None


def plot_metric(category: str, name: str, title: str, colour: str = BLUE,
                height: int = 260, connect: bool = True) -> go.Figure | None:
    df = series(store, run_id, category, name)
    if df.empty:
        return None
    mod, sev = thr(category, name)
    line = line_with_bands(widx, df["value"].tolist(), mod, sev, colour, name, connect)
    fig = go.Figure([line])
    fig.update_layout(**plotly_layout(title, height))
    return regime_bands(fig, wlabels)

# ── Header ─────────────────────────────────────────────────────────────
last_decision = decisions_df.sort_values("window_index").iloc[-1].to_dict() if not decisions_df.empty else {}
last_signals = decode_json(last_decision.get("signals"), {})
health = float(last_decision.get("health") or 100.0)
health_colour = GREEN if health >= 85 else (YELLOW if health >= 60 else RED)

st.markdown("# 🛰️ LLM Drift & Quality Monitor")
st.caption(f"Run `{run_id}` · {len(windows_df)} windows · "
           f"{int(windows_df['n_traffic'].sum()):,} requests · "
           f"{int(windows_df['n_labeled'].sum()):,} labelled · started {run.get('started_at')}")

k1, k2, k3, k4, k5, k6 = st.columns(6)
with k1:
    st.metric("Health", f"{health:.0f}")
with k2:
    action = str(last_decision.get("action", "?"))
    st.markdown(kpi("Decision", action.upper(), "", ACTION_COLOR.get(action, MUTED)))
with k3:
    auc = vals("embedding", "domain_classifier_auc")
    last_auc = auc[-1] if auc else None
    st.metric("Domain AUC", fmt(last_auc))
with k4:
    ood = vals("embedding", "ood_rate")
    st.metric("OOD rate", fmt(ood[-1] if ood else None, 3))
with k5:
    acc = vals("quality", "accuracy")
    st.metric("Accuracy", fmt(acc[-1] if acc else None))
with k6:
    js = vals("judge", "judge_score")
    base = float(last_signals.get("baseline_score", 0) or 0)
    delta = (js[-1] - base) if js and js[-1] is not None else None
    st.metric("Judge", fmt(js[-1] if js else None),
              f"{delta:+.3f} vs baseline" if delta is not None else "")

st.markdown("")

# ── Tabs ───────────────────────────────────────────────────────────────
tab_overview, tab_emb, tab_qual, tab_judge, tab_detail = st.tabs(
    ["Overview", "Embedding drift", "Output quality", "LLM-as-judge", "Run detail"])

# ══ Overview ═══════════════════════════════════════════════════════════
with tab_overview:
    st.markdown("<div class='section'>Latest decision</div>", unsafe_allow_html=True)
    rationale = str(last_decision.get("rationale", ""))
    cls = str(last_decision.get("severity", "none"))
    st.markdown(f"<div class='narrative {cls}'><b>{action.upper()}</b> &nbsp;{severity_badge(cls)}"
                f" &nbsp;confidence {fmt(last_decision.get('confidence'), 2)}<br>{rationale}</div>",
                unsafe_allow_html=True)

    fired = last_signals.get("fired", [])
    if fired:
        st.markdown("**Fired signals**")
        for f in fired:
            parts = str(f).split(":")
            colour = SEVERITY_COLOR.get(parts[-1], BLUE)
            st.markdown(f"- <span style='color:{colour}'>`{f}`</span>", unsafe_allow_html=True)

    st.markdown("<div class='section'>Signal timeline</div>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        fig = plot_metric("embedding", "domain_classifier_auc", "Domain classifier AUC", PURPLE)
        if fig is not None:
            st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = plot_metric("quality", "accuracy", "Accuracy (labelled traffic)", GREEN)
        if fig is not None:
            st.plotly_chart(fig, use_container_width=True)
    c3, c4 = st.columns(2)
    with c3:
        fig = plot_metric("judge", "judge_score", "LLM-as-judge weighted score", BLUE)
        if fig is not None:
            st.plotly_chart(fig, use_container_width=True)
    with c4:
        fig = plot_metric("quality", "ece", "Expected calibration error", ORANGE)
        if fig is not None:
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("<div class='section'>Incidents</div>", unsafe_allow_html=True)
    if incidents_df.empty:
        st.info("No incidents raised in this run.")
    else:
        for _, inc in incidents_df.iterrows():
            sev = str(inc["severity"])
            state = "OPEN" if inc["status"] == "open" else "CLOSED"
            colour = SEVERITY_COLOR.get(sev, BLUE)
            timeline = decode_json(inc.get("timeline"), [])
            with st.expander(f"{inc['incident_id']} · {inc['title']} · {state}",
                             expanded=inc["status"] == "open"):
                st.markdown(f"{severity_badge(sev)} · opened {inc['opened_at']}"
                            + (f" · closed {inc['closed_at']}" if inc.get("closed_at") else ""))
                if timeline:
                    st.dataframe(pd.DataFrame(timeline), use_container_width=True, hide_index=True)
                sig = decode_json(inc.get("signals"), {})
                if sig.get("fired"):
                    st.caption("Signals: " + ", ".join(sig["fired"]))

    st.markdown("<div class='section'>Traffic served per window</div>", unsafe_allow_html=True)
    fig = go.Figure(go.Bar(x=widx, y=windows_df.sort_values("window_index")["n_traffic"].tolist(),
                           marker_color=[ORANGE if name == "volume_spike" else BLUE
                                          for name in wlabels]))
    fig.update_layout(**plotly_layout("", 220, False))
    fig.update_yaxes(title_text="requests")
    st.plotly_chart(regime_bands(fig, wlabels), use_container_width=True)

# ══ Embedding drift ════════════════════════════════════════════════════
with tab_emb:
    st.markdown("<div class='section'>Five views of the same distribution</div>",
                unsafe_allow_html=True)
    st.caption(
        "Each detector fails differently. MMD catches shape changes the marginals miss; "
        "sliced Wasserstein is cheap and interpretable; the domain classifier answers the "
        "question a stakeholder actually asks — *could a model tell these apart?*; "
        "normalised Frechet catches variance drift that a centroid comparison is blind to; "
        "and the OOD rate converts all of it into a percentage of traffic.")

    grid = [
        ("mmd2", "MMD² (RBF, permutation-calibrated)", ORANGE),
        ("swd", "Sliced Wasserstein", BLUE),
        ("centroid_cosine_shift", "Centroid cosine shift", PURPLE),
        ("frechet_normalised", "Normalised Frechet distance", MUTED),
        ("domain_classifier_auc", "Domain classifier AUC", YELLOW),
        ("ood_rate", "Out-of-distribution rate", RED),
    ]
    for i in range(0, len(grid), 2):
        cols = st.columns(2)
        for c, (name, title, colour) in zip(cols, grid[i : i + 2]):
            with c:
                fig = plot_metric("embedding", name, title, colour)
                if fig is None:
                    st.info(f"{title}: no data.")
                else:
                    st.plotly_chart(fig, use_container_width=True)

    st.markdown("<div class='section'>Detector votes per window</div>", unsafe_allow_html=True)
    if not metrics_df.empty:
        emb = metrics_df[metrics_df["category"] == "embedding"]
        vote_tbl = (emb.pivot_table(index="window_index", columns="name", values="severity",
                                    aggfunc="last")
                    .reset_index())
        vote_tbl.insert(1, "regime", [wlabels[i] if i < len(wlabels) else ""
                                      for i in vote_tbl["window_index"] - (vote_tbl["window_index"].min())])
        order_cols = ["window_index", "regime"] + [c for c in vote_tbl.columns
                                                   if c not in ("window_index", "regime")]
        st.dataframe(vote_tbl[order_cols], use_container_width=True, hide_index=True,
                     column_config={c: st.column_config.TextColumn(c.replace("_", " ")) for c in order_cols[2:]})

    st.markdown("<div class='section'>Per-detector thresholds</div>", unsafe_allow_html=True)
    thr_rows = []
    for name, title, _ in grid:
        mod, sev = thr("embedding", name)
        thr_rows.append({"detector": title, "moderate": mod, "severe": sev})
    st.dataframe(pd.DataFrame(thr_rows), use_container_width=True, hide_index=True)

    st.markdown("<div class='section'>Reference vs latest window (PCA)</div>", unsafe_allow_html=True)
    try:
        from src.storage.reference import ReferenceSnapshot

        snap = ReferenceSnapshot.load()
        latest_wid = order.iloc[-1]["window_id"]
        traffic = store.traffic_for_window(latest_wid)
        if traffic.empty:
            st.info("No traffic stored for the latest window.")
        else:
            from sklearn.decomposition import PCA

            from src.embeddings import build_encoder

            enc = build_encoder()
            cur = enc.encode(traffic["text_snippet"].astype(str).tolist())
            ref = snap.embeddings
            n = min(700, len(ref), len(cur))
            ref_idx = np.arange(n)
            pooled = np.vstack([ref[ref_idx], cur[:n]])
            pca = PCA(n_components=2, random_state=0).fit(pooled)
            rp, cp = pca.transform(ref[ref_idx]), pca.transform(cur[:n])
            fig = go.Figure()
            fig.add_trace(go.Scattergl(x=rp[:, 0], y=rp[:, 1], mode="markers", name="reference",
                                       marker=dict(size=4, color=GREEN, opacity=0.45)))
            fig.add_trace(go.Scattergl(x=cp[:, 0], y=cp[:, 1], mode="markers", name="latest window",
                                       marker=dict(size=4, color=RED, opacity=0.5)))
            fig.update_layout(**plotly_layout(
                f"Reference vs latest window ({len(rp)} / {len(cp)} vectors, 2D PCA)", 460))
            st.plotly_chart(fig, use_container_width=True)
    except Exception as exc:  # noqa: BLE001
        st.warning(f"Projection unavailable: {exc}")

# ══ Output quality ═════════════════════════════════════════════════════
with tab_qual:
    st.markdown("<div class='section'>Performance against the frozen champion baseline</div>",
                unsafe_allow_html=True)
    qa, qb = st.columns(2)
    with qa:
        fig = plot_metric("quality", "accuracy", "Accuracy", GREEN)
        if fig is not None:
            st.plotly_chart(fig, use_container_width=True)
    with qb:
        fig = plot_metric("quality", "macro_f1", "Macro F1", BLUE)
        if fig is not None:
            st.plotly_chart(fig, use_container_width=True)
    qc, qd = st.columns(2)
    with qc:
        fig = plot_metric("quality", "ece", "Expected calibration error", ORANGE)
        if fig is not None:
            st.plotly_chart(fig, use_container_width=True)
    with qd:
        fig = plot_metric("quality", "brier", "Brier score", YELLOW)
        if fig is not None:
            st.plotly_chart(fig, use_container_width=True)
    qe, qf = st.columns(2)
    with qe:
        fig = plot_metric("quality", "abstention_rate", "Abstention rate", MUTED)
        if fig is not None:
            st.plotly_chart(fig, use_container_width=True)
    with qf:
        fig = plot_metric("quality", "label_coverage", "Label coverage", MUTED)
        if fig is not None:
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("<div class='section'>In-scope vs out-of-scope</div>", unsafe_allow_html=True)
    fig = go.Figure()
    for name, colour, label in (("accuracy_in_scope", GREEN, "in scope"),
                                ("accuracy_out_of_scope", RED, "out of scope")):
        s = series(store, run_id, "quality", name)
        if s.empty:
            continue
        fig.add_trace(go.Scatter(x=s["window_index"], y=s["value"], mode="lines+markers",
                                 name=label, line=dict(color=colour, width=2)))
    fig.update_layout(**plotly_layout("", 280))
    st.plotly_chart(regime_bands(fig, wlabels), use_container_width=True)

    st.markdown("<div class='section'>Reliability (latest window)</div>", unsafe_allow_html=True)
    latest_wid = order.iloc[-1]["window_id"]
    traffic = store.traffic_for_window(latest_wid)
    if traffic.empty:
        st.info("No traffic stored for the latest window.")
    else:
        from sklearn.metrics import calibration_curve

        lab = traffic[traffic["gold_intent"].notna()]
        if lab.empty:
            st.info("No labels in the latest window — reliability needs ground truth.")
        else:
            frac, mean_conf = calibration_curve(
                (lab["pred_intent"].to_numpy() == lab["gold_intent"].to_numpy()).astype(float),
                lab["confidence"].to_numpy(dtype=float), n_bins=10, strategy="uniform")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="perfect calibration",
                                     line=dict(color=MUTED, dash="dash")))
            fig.add_trace(go.Scatter(x=mean_conf, y=frac, mode="lines+markers", name="model",
                                     line=dict(color=ORANGE, width=2)))
            fig.update_layout(**plotly_layout("", 320))
            fig.update_xaxes(title_text="predicted confidence")
            fig.update_yaxes(title_text="observed accuracy")
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("<div class='section'>Worst-performing intents</div>", unsafe_allow_html=True)
    if traffic.empty or traffic["gold_intent"].isna().all():
        st.info("No labelled traffic in the latest window.")
    else:
        agg = (traffic.dropna(subset=["gold_intent"])
               .assign(ok=lambda d: d["pred_intent"] == d["gold_intent"])
               .groupby("gold_intent")["ok"].agg(["mean", "count"])
               .sort_values("mean"))
        agg = agg.rename(columns={"mean": "accuracy", "count": "n"})
        st.bar_chart(agg, height=280, color="#58a6ff")
        st.dataframe(agg.reset_index().head(15), use_container_width=True, hide_index=True)

# ══ LLM-as-judge ═══════════════════════════════════════════════════════
with tab_judge:
    st.markdown("<div class='section'>Rubric score over time</div>", unsafe_allow_html=True)
    js = series(store, run_id, "judge", "judge_score")
    if js.empty:
        st.info("No judge results in this run.")
    else:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=js["window_index"], y=js["value"], mode="lines+markers",
                                 name="judge score", line=dict(color=BLUE, width=2),
                                 marker=dict(size=7)))
        baseline = js["baseline"].iloc[0]
        if baseline is not None and not pd.isna(baseline):
            fig.add_hline(y=float(baseline), line_dash="dash", line_color=GREEN,
                          annotation_text=f"baseline {float(baseline):.3f}",
                          annotation_font=dict(size=10, color=GREEN))
        fig.update_layout(**plotly_layout("Weighted rubric score vs baseline", 300))
        st.plotly_chart(regime_bands(fig, wlabels), use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        fig = plot_metric("judge", "judge_correctness_rate", "Judge 'correct' rate", GREEN)
        if fig is not None:
            st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = plot_metric("judge", "judge_veto_rate", "Veto rate (safety / groundedness)", RED)
        if fig is not None:
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("<div class='section'>Per-dimension movement</div>", unsafe_allow_html=True)
    dims = sorted(n for n in metrics_df[metrics_df["category"] == "judge"]["name"].unique()
                  if n.startswith("judge_dim::"))
    if dims:
        rows = []
        base_dims = {}
        for d in dims:
            s = series(store, run_id, "judge", d)
            if s.empty:
                continue
            short = d.split("::")[-1]
            first = s["value"].iloc[0]
            base_dims[short] = first
            rows.append({"dimension": short,
                         "baseline": first,
                         "latest": s["value"].iloc[-1],
                         "delta": (s["value"].iloc[-1] - first),
                         "min": s["value"].min(), "max": s["value"].max()})
        tbl = pd.DataFrame(rows).sort_values("delta")
        st.dataframe(tbl, use_container_width=True, hide_index=True,
                     column_config={"delta": st.column_config.NumberColumn("delta", format="%.3f"),
                                    "baseline": st.column_config.NumberColumn("baseline", format="%.2f"),
                                    "latest": st.column_config.NumberColumn("latest", format="%.2f")})
        for _, row in tbl.iterrows():
            d = row["dimension"]
            s = series(store, run_id, "judge", f"judge_dim::{d}")
            if s.empty:
                continue
            fig = line_with_bands(s["window_index"], s["value"].tolist(), None, None, PURPLE, d)
            fig.update_layout(**plotly_layout(d, 200, False))
            st.plotly_chart(regime_bands(fig, wlabels), use_container_width=True)
    else:
        st.info("No per-dimension data yet.")

    st.markdown("<div class='section'>Judge rationale samples</div>", unsafe_allow_html=True)
    js_rows = store.judge_scores(run_id)
    if js_rows.empty:
        st.info("No judge records.")
    else:
        sample = js_rows.drop_duplicates(subset=["sample_hash"]).tail(12)
        for _, r in sample.iterrows():
            with st.expander(f"{str(r['label'])} · w{int(r['window_index'])} · "
                             f"score {fmt(r['score'], 2)} · {str(r['dimension'])}"):
                st.markdown(f"**Input** — {str(r['sample_snippet'])[:220]}")
                st.markdown(f"**Why** — {str(r['rationale'])[:400]}")

    st.markdown("<div class='section'>Judge regression suite</div>", unsafe_allow_html=True)
    path = Path("artifacts/judge_regression.json")
    if path.exists():
        st.json(json.loads(path.read_text(encoding="utf-8")))
    else:
        st.info("Run `python scripts/bootstrap.py` to generate the golden-set baseline.")

# ══ Run detail ═════════════════════════════════════════════════════════
with tab_detail:
    st.markdown("<div class='section'>Decisions</div>", unsafe_allow_html=True)
    if decisions_df.empty:
        st.info("No decisions recorded.")
    else:
        disp = decisions_df.sort_values("window_index").copy()
        disp["regime"] = [wlabels[i] for i in disp["window_index"] - disp["window_index"].min()]
        disp["severity_badge"] = [severity_badge(s) for s in disp["severity"]]
        st.dataframe(disp[["window_index", "regime", "action", "severity_badge", "health",
                           "confidence", "rationale"]].rename(columns={
                               "window_index": "#", "severity_badge": "severity"}),
                     use_container_width=True, hide_index=True, height=320)

    st.markdown("<div class='section'>Every metric</div>", unsafe_allow_html=True)
    if wide.empty:
        st.info("No metrics.")
    else:
        st.dataframe(wide, use_container_width=True, hide_index=True, height=460)

    st.markdown("<div class='section'>Run manifest</div>", unsafe_allow_html=True)
    st.json(run)