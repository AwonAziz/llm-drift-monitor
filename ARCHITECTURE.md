# Architecture

## The one-sentence version

A monitored serving path, five drift detectors with complementary failure modes,
a policy engine that turns signals into one action, and an append-only telemetry
store that every other surface reads from.

## Data flow

```
                      ┌──────────────── frozen at bootstrap ───────────────┐
                      │  reference_snapshot.npz                          │
                      │    embeddings (n_ref × d)                        │
                      │    per-intent centroids                          │
                      │    k-NN novelty cutoff (95th pct, leave-one-out) │
                      │    quality baseline (in-scope validation)         │
                      └───────────────────────────────────────────────────┘
                                              │
  ┌────────────┐   ┌──────────────┐   ┌───────▼────────┐   ┌────────────────┐
  │ TrafficSim │──►│ SupportAssis │──►│ EmbeddingDrift │   │ OutputQuality  │
  │ ulator     │   │ tant (serve) │   │ Detector       │   │ Monitor        │
  │            │   │ classify →   │   │ 6 signals      │   │ ECE, Brier,    │
  │ 14 windows │   │ playbook →   │   │ + votes + p    │   │ scope gap,     │
  │ 6 regimes  │   │ LLM reply    │   │ + significance │   │ label proxies  │
  └────────────┘   └──────┬───────┘   └───────┬────────┘   └───────┬────────┘
                        │                   │                    │
                        │            ┌──────▼────────┐           │
                        │            │ LLMBasedJudge │           │
                        │            │ 5-dim rubric  │           │
                        │            │ veto cap      │           │
                        │            │ regression z  │           │
                        │            └──────┬────────┘           │
                        │                   │                    │
                        └─────────┬─────────┴────────────────────┘
                                  ▼
                        ┌───────────────────┐
                        │ DriftOrchestrator │
                        │ health · severity │
                        │ confidence · action│
                        │ incident lifecycle │
                        └─────────┬─────────┘
                                  ▼
                   ┌──────────────┴──────────────┐
                   ▼              ▼               ▼
            TelemetryStore   Streamlit      report_*.md/.json/.html
            (SQLite)         dashboard      FastAPI read model
```

## Module responsibilities

| Module | Owns | Deliberately does not own |
|---|---|---|
| `config/settings.py` | Every threshold and path | Any detection logic — policy is reviewable separately from statistics |
| `src/app/` | The system under observation | Any knowledge of the monitoring stack |
| `src/data/` | Dataset ingestion, reference construction, shift simulation | Anything about serving |
| `src/embeddings/` | Encoding + a stable `signature` per encoder | Drift statistics |
| `src/storage/reference.py` | The frozen healthy state | How it is compared |
| `src/storage/telemetry.py` | Append-only persistence, migrations | Interpretation |
| `src/drift/` | The detectors and their thresholds | The action to take |
| `src/quality/` | Quality metrics against a baseline | Alerting |
| `src/judge/` | Rubric, evaluation, judge self-validation | Whether to act |
| `src/monitoring/orchestrator.py` | The triage policy | Computing signals |
| `src/reporting/` | Rendering stored data | Storing it |

The dependency arrows point one way. Nothing in `drift/` imports from
`monitoring/`, and nothing in `judge/` imports from `api/`. This is what makes
the detectors unit-testable in isolation and what lets you swap the policy
without touching a statistic.

## Design decisions worth defending

**A frozen reference, always.** Every detector loads a versioned snapshot rather
than recomputing a baseline from whatever data is nearby. Without a frozen
reference you cannot answer "is this worse than normal?", because "normal" keeps
moving. The snapshot is a single `.npz` plus a JSON sidecar with the encoder
signature, the classifier signature and the quality baseline.

**The encoder signature is part of the data.** `encoder.signature` is stamped
onto every run manifest. If you swap `all-MiniLM-L6-v2` for a different model,
previously recorded drift numbers are not comparable, and the system says so
rather than drawing a trend line across the change.

**Bins are frozen at construction.** The tabular detector computes its quantile
edges once, from the reference. Recomputing per window makes every window look
identical by construction — the most common way a drift dashboard starts lying.

**A permutation test confirms, it does not escalate.** Severity comes from
effect size. Significance gates whether the window may raise drift at all. This
keeps a p-value threshold from turning into a false-alarm generator.

**The policy is data, not code paths.** `DriftOrchestrator.decide()` returns a
`Decision` with its rationale, its fired signals, its component penalties and a
confidence score, and all of it is persisted. "Why did it retrain?" is answered
by a query, not by reading the source.

**Incidents are objects, not log lines.** An incident has a timeline of
observations, so "how long was this degraded?" is answerable.

## Data model

```sql
runs(run_id, started_at, finished_at, status, dataset_source, encoder, judge,
     app_model, notes, config_snapshot)
windows(window_id, run_id, window_index, label, started_at, n_traffic, n_labeled, notes)
traffic(window_id, run_id, ts, request_id, text_hash, text_snippet, in_scope,
        gold_intent, pred_intent, confidence, abstained, response, app_latency_ms)
metrics(window_id, run_id, window_index, ts, category, name, value, baseline,
        threshold_moderate, threshold_severe, severity, unit, extra)
judge_scores(window_id, run_id, sample_hash, sample_snippet, dimension, score,
             verdict, rationale, judge_model, rubric_version, in_scope, extra)
decisions(window_id, run_id, window_index, ts, action, severity, health,
          confidence, signals, rationale)
incidents(incident_id, run_id, window_id, opened_at, updated_at, closed_at,
          severity, title, signals, status, timeline)
```

`metrics` is long-format with a unique index on
`(run_id, window_index, category, name)` and an upsert, so a re-processed window
replaces its values instead of duplicating them. `n_traffic` is recounted from
the `traffic` table rather than incremented, so call order cannot corrupt it.

`added columns` are migrated on open — an existing development database keeps
working instead of demanding a wipe.

## Extension points

| To change | Do this | Not this |
|---|---|---|
| Swap the encoder | Implement `BaseEncoder` in `src/embeddings/encoder.py` | Patch the detectors |
| Change the alerting policy | Edit weights / gates in `DriftOrchestrator.__init__` | Add an `if` in a detector |
| Add a drift statistic | Add a function to `src/drift/embedding.py`, register it in `detect()`'s vote dict | Monkey-patch the result |
| Change the rubric | Edit `RUBRIC` and bump `RUBRIC_VERSION` | Edit the prompt inline |
| Use a hosted judge | Set `LDM_JUDGE_PROVIDER=openai` and `OPENAI_API_KEY` | Write a new evaluator |
| Add a metric | Emit a `MetricPoint` from `_metric_points()` | Write to SQLite from a script |
| Add a regime | Append a `Regime` to `default_schedule()` | Hand-edit generated data |
