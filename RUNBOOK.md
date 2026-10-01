# Runbook

Operational procedures for the monitoring platform. Every command assumes the
repository root as the working directory.

## First run

```bash
make setup          # venv + dependencies
ollama pull qwen3:8b   # optional; a real judge with no API key
make bootstrap      # data, champion, reference snapshot, judge baseline
make demo-fast      # prove the pipeline works (~50s, no model needed)
```

## Everyday

```bash
make demo           # full 14-window production shift with the real LLM
make dashboard      # Streamlit on :8501
make api            # FastAPI on :8000, docs at /docs
make report         # regenerate reports from the last run
```

## When an alert fires

The decision is already in the database; the question is what to do with it.

```bash
# 1. What did the policy decide, and why?
sqlite3 data/telemetry.db \
  "SELECT window_index, action, severity, health, confidence, rationale
     FROM decisions WHERE run_id='<RUN>' ORDER BY window_index DESC LIMIT 5;"

# 2. Which signals fired?
sqlite3 data/telemetry.db \
  "SELECT window_index, name, value, severity FROM metrics
    WHERE run_id='<RUN>' AND severity != 'none' ORDER BY window_index;"

# 3. Which incidents are open?
curl localhost:8000/monitoring/incidents?status=open | jq
```

Then follow the decision:

| Action | What to do |
|---|---|
| `noop` | Nothing. Confirm the detectors are not silent: check the null firing rate on recent healthy windows. |
| `investigate`, input drift only | Scope question. Do **not** retrain. Decide whether to expand the label set or route the new traffic to a specialist. |
| `investigate`, quality regression | Verify the served artifact hash and the dependency versions first. Then re-run the offline eval set before retraining. |
| `investigate`, judge regression | Run the golden suite against the current judge (`make eval-judge`). A judge change and an app change look identical on a dashboard. |
| `retrain` | Promoted automatically if `LDM_AUTO_RETRAIN=1`, otherwise a human runs `make bootstrap` with `--expanded`. |

## Retraining and promoting

```bash
# Widen scope to include the intents that arrived in production
python scripts/bootstrap.py --expanded

# Re-baseline the judge against the new champion
make eval-judge
```

After a promotion, the in-scope/out-of-scope split changes meaning. `build_artifacts`
derives scope from the classifier's own class list precisely so the post-promotion
validation numbers stay honest.

## Rotating the judge

```bash
ollama pull qwen3:14b
LDM_JUDGE_MODEL=qwen3:14b make eval-judge
```

The suite refuses to compare across a signature change and says so explicitly.
That is deliberate: bump the rubric or the judge, read the "no comparable
baseline" message, and re-baseline consciously rather than discovering a week
later that a dashboard was comparing two different rulers.

## Threshold tuning

Never tune a threshold without first measuring the null firing rate.

```bash
# Run several baseline windows and count how often each detector fires
python scripts/demo_shift.py --regime baseline --windows 6
sqlite3 data/telemetry.db \
  "SELECT name, SUM(severity != 'none') AS fired, COUNT(*) AS windows
     FROM metrics WHERE run_id='<RUN>' AND category='embedding'
    GROUP BY name;"
```

A detector that fires on more than ~10% of healthy windows is a detector that
will get muted. Fix the statistic, not the threshold.

## Scaling the demo

| Knob | Flag | Default |
|---|---|---|
| Requests per window | `--window-size` | 250 |
| LLM replies generated per window | `--response-sample` | 40 |
| Samples judged per window | `--judge-sample` | 24 |
| Judge concurrency | `--max-workers` | 4 |
| Minimum window for detection | `LDM_MIN_WINDOW` | 40 |
| MMD permutations | `LDM_MMD_PERMUTATIONS` | 200 |

Judging dominates wall-clock on CPU. `--fast` swaps in the deterministic mock
judge for structural work and CI; the full run is for the narrative.

## Troubleshooting

**"No classifier artifact" / "No reference snapshot"**
Run `make bootstrap`. The API returns 503 with that message rather than starting
half-configured.

**"Reference snapshot not comparable"**
The encoder signature changed. Re-run `make bootstrap` to refit, and treat the
run boundary as a break in the series.

**Judge calls are slow**
`/api/chat` with `think: false` is already used — a reasoning model on
`/api/generate` returns an empty string at low `num_predict` because the token
budget goes into a hidden trace. Remaining latency is the model. Lower
`--judge-sample`, raise `--max-workers`, or use `qwen3:1.7b`.

**Detectors fire on every window**
Almost always a mis-calibrated null. Check the permutation p-values in the
`metrics.extra` column; if they are all `< 0.01` on in-distribution traffic, the
observed statistic and the null are not on the same footing.

**Dashboard says "no monitoring runs"**
Run `make demo-fast`. The dashboard reads the same SQLite store the monitor
writes; it does not recompute anything.

**Everything is very slow**
Confirm the encoder: `curl localhost:8000/health` reports which one is loaded.
`hashing` is milliseconds; `sentence-transformers` on CPU is a few seconds per
window for embedding plus ~2s for the detectors.
