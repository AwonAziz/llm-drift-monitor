# LLM Drift & Quality Monitor

**Drift monitoring and LLM-as-judge regression tracking for production LLM systems.**

Most monitoring demos compare two histograms. This one watches a real LLM
application — a banking support assistant — across a scripted production shift,
and answers the question a platform team actually has to answer: *did the system
get worse, and what should we do about it?*

Built on **[Banking77](https://github.com/PolyAI-LDN/task-specific-datasets)**
(Casanueva et al., 2020) — 10k real customer-service queries across 77 banking
intents. An intent classifier is launched for 27 of them. Two products then ship
and 45% of production traffic falls outside the scope it was trained on.

```
                 ┌──────────────────────────────────────────────┐
   customer ───► │  intent classifier  ──► intent playbook      │
   message      │  (TF-IDF + calibrated LogReg, T=0.65)  ──► LLM│
                 └──────────────────────────────────────────────┘
                                        │
                 ┌──────────────────────┴───────────────────────┐
                 ▼                                              ▼
        EMBEDDING DRIFT                                 OUTPUT QUALITY
        MMD ��  sliced Wasserstein                        accuracy · macro-F1
        domain-classifier AUC                             ECE · Brier · abstention
        normalised Fréchet                                in-scope vs out-of-scope
        k-NN novelty rate                                 label-free proxies
                 │                                              │
                 └──────────────┬───────────────────────────────┘
                                ▼
                     LLM-AS-JUDGE  (qwen3:8b, local)
                     5-dimension anchored rubric · veto cap
                     pairwise win-rate · golden regression suite
                                │
                                ▼
                     DRIFT ORCHESTRATOR
                     health · severity · confidence · action
                     noop │ investigate │ retrain │ rollback
                                │
                 ┌──────────────┼──────────────┐
                 ▼              ▼              ▼
          SQLite telemetry   Streamlit     Markdown / JSON / HTML
          (every window)     dashboard      incident report
```

---

## What was built here, in one table

| Layer | Files | What it does |
|---|---|---|
| **Config** | `config/settings.py` | 40+ named thresholds, paths and backend toggles, all `LDM_*` overridable. Policy is reviewable separately from statistics. |
| **System under observation** | `src/app/assistant.py` | Calibrated + temperature-scaled intent classifier, intent playbooks, LLM responder, abstention. Produces the same shapes a real serving path does. |
| **Data** | `src/data/datasets.py`, `src/data/shifts.py` | Banking77 loader with parquet cache and a bundled offline corpus. 6-regime shift simulator with 5 deterministic text transforms. |
| **Embeddings** | `src/embeddings/encoder.py` | Three backends behind one interface, each with a stable `signature` used as a comparability guard. |
| **Reference** | `src/storage/reference.py` | Frozen snapshot: embeddings, per-intent centroids, leave-one-out k-NN novelty cutoff, reliability curve, quality baseline. |
| **Embedding drift** | `src/drift/embedding.py` | MMD (permutation-calibrated), sliced Wasserstein, domain-classifier AUC, normalised Fréchet, k-NN novelty, concept gap. |
| **Tabular drift** | `src/drift/tabular.py` | PSI / KS / chi-square / JS with bin edges frozen at construction. |
| **Output quality** | `src/quality/outputs.py` | ECE, MCE, adaptive ECE, Brier, AUC, reliability curve, per-intent damage, delayed-label attribution, label-free proxies, separate in-scope/out-of-scope baselines. |
| **LLM-as-judge** | `src/judge/` | Anchored 5-dimension rubric with veto cap, stratified sampling, regression test with z-gating, pairwise win-rate, self-consistency, and a 8-case golden regression suite. |
| **Policy** | `src/monitoring/orchestrator.py` | Health score, severity, confidence, confirmation windows, 4 actions, incident lifecycle with a timeline. |
| **Monitor** | `src/monitoring/monitor.py` | The per-window loop: serve → log → embed → detect → measure → judge → decide → persist. |
| **Persistence** | `src/storage/telemetry.py` | 7-table SQLite time-series store with upserts, schema migration, incident timelines. |
| **Serving** | `src/api/server.py` | 17 endpoints: inference (`/predict`, `/predict/batch`) + a monitoring read model + Prometheus exposition. |
| **Dashboard** | `dashboard/` | 5 tabs: Overview, Embedding drift, Output quality, LLM-as-judge, Run detail. |
| **Reports** | `src/reporting/` | Markdown, JSON and a self-contained HTML page with inline SVG and no CDN. |
| **Tests** | `tests/` | 184 tests, fully offline, 32 seconds. |
| **Ops** | `Dockerfile`, `docker-compose.yml`, `Makefile`, `.github/workflows/ci.yml` | 4 CI jobs: tests, judge gate, demo smoke, docker build. |

### Bugs found and fixed along the way

These are in the code and pinned by regression tests, because they are the
interesting part:

1. **The domain-classifier null was 0.63, not 0.50.** A logistic regression on
   raw 384-dim embeddings separates two samples of the same distribution at AUC
   0.63 by exploiting sampling noise. Every window looked like drift. Fixed with
   reference-fitted PCA plus dropping rows present in both sets.
2. **The MMD permutation null was computed on a different sample size than the
   observation**, so the p-value was significant everywhere. Fixed by computing
   both on the same bounded subsample.
3. **The k-NN novelty cutoff was computed without leave-one-out**, so a reference
   point's nearest neighbour was itself at distance 0 and the cutoff collapsed.
   The out-of-scope rate read 100% for genuinely in-distribution traffic.
4. **Accuracy was tautologically 1.0.** `accuracy_score(binary, correct)` where
   `binary` was derived from `correct` always returns 1.0. Caught by a test that
   asserts accuracy equals the hand-computed fraction.
5. **The judge had no veto cap.** A reply that asked for a customer's PIN and
   scored 5/5 elsewhere averaged 0.84 and passed. Caught by the golden suite.
6. **The quality baseline was averaged over out-of-scope intents**, dragging the
   bar down until no regression could ever be detected. Split into two baselines.
7. **`severity_from(..., higher_is_worse=False)` inverted the comparison**, so
   *improving* accuracy was reported as a severe regression.
8. **`n_traffic` was incremented**, so calling `close_window` before
   `log_traffic` silently corrupted the count. Now recounted from the table.
9. **The judge baseline had no variance gate**, so a 0.4-point wobble in a
   200-sample window paged someone. Now requires 2 standard errors.
10. **Quality metric thresholds were computed and then dropped** — the loop built
    them and never passed them to the record. The dashboard had no bands to draw.
11. **`/api/generate` with a reasoning model returns an empty string** at low
    `num_predict` because the budget goes into a hidden trace. Switched to
    `/api/chat` with `think: false`: 31s → 9.5s per judge call, and it returns
    the answer at all.
12. **Ollama embeddings are not a thing for small instruct models.** The `auto`
    chain probed `/api/embeddings`, got a 400, and now skips straight to
    sentence-transformers with the offline hashing encoder as the last resort.

---

## Quick start

```bash
git clone https://github.com/AwonAziz/llm-drift-monitor.git
cd llm-drift-monitor

python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS / Linux

pip install -r requirements.txt
pip install sentence-transformers          # semantic embeddings (optional)

# Optional but recommended: a real LLM judge, no API key, nothing leaves your machine
ollama pull qwen3:8b

make bootstrap        # fetch data, train the champion, validate the judge
make demo             # the full simulated production shift
make report           # open data/reports/report_*.html
```

No model download, no GPU, no network? Everything still runs:

```bash
make demo-fast        # 14 windows, offline encoder, deterministic mock judge (~50s)
```

---

## What the demo actually shows

`python scripts/demo_shift.py` replays 14 evaluation windows of a live assistant
and prints what the platform caught. Real output from a run with `qwen3:8b` as
both responder and judge, `all-MiniLM-L6-v2` embeddings and 2,758 requests:

| # | Regime | Domain AUC | OOD rate | Accuracy | ECE | Judge score | Decision |
|---|--------|-----------:|---------:|---------:|----:|------------:|----------|
| 0 | baseline | 0.58 | 0.06 | 0.907 | 0.059 | 0.580 | investigate (judge) |
| 1 | baseline | 0.55 | 0.04 | 0.954 | 0.044 | 0.450 | investigate (judge) |
| 2 | baseline | 0.51 | 0.02 | 0.960 | 0.048 | 0.385 | investigate (judge) |
| 3 | volume_spike | 0.59 | 0.03 | 0.971 | 0.026 | 0.320 | investigate (judge) |
| 4 | style_shift | **0.65** | 0.05 | 0.945 | 0.023 | 0.465 | investigate (input drift) |
| 5 | style_shift | 0.65 | 0.04 | 0.926 | 0.048 | 0.385 | investigate (input drift) |
| 6 | style_shift | 0.65 | 0.05 | 0.970 | 0.049 | 0.435 | investigate (input drift) |
| 7 | new_intents | 0.65 | **0.40** | **0.583** | **0.213** | 0.385 | investigate · **SEVERE** |
| 8 | new_intents | 0.69 | 0.43 | 0.524 | 0.238 | 0.400 | investigate · **SEVERE** |
| 9 | new_intents | 0.68 | 0.40 | 0.545 | 0.270 | 0.370 | investigate · **SEVERE** |
| 10 | mixed_crisis | **0.71** | **0.64** | **0.227** | **0.517** | 0.400 | investigate · **SEVERE** |
| 11 | mixed_crisis | 0.74 | 0.64 | 0.236 | 0.457 | 0.385 | **→ retrain, promote v2** |
| 12 | recovery | 0.61 | 0.40 | **0.946** | 0.060 | 0.400 | investigate |
| 13 | recovery | 0.68 | 0.41 | **0.973** | 0.059 | 0.400 | investigate |

Three things in that table are worth arguing about:

1. **Window 3 is a 2× traffic spike with identical mix, and nothing fires.**
   Every detector is sample-size aware and the p-values are calibrated, so a
   larger window does not manufacture significance. A detector that cries wolf on
   traffic volume gets muted within a week.

2. **Windows 4–6 move the input distribution and accuracy does not move with
   it.** A new partner channel sends terse, lowercase, emoji-laden text for the
   same intents. The embedding space has genuinely shifted (AUC 0.58 → 0.65, MMD
   2× baseline) while the model is still 94% correct. The correct action is to
   widen coverage and check the judge — **not** to retrain. The orchestrator
   encodes that distinction explicitly.

3. **Windows 7–11 are the real incident, and input drift under-rates it.** OOD
   rate goes to 64% and accuracy collapses from 0.96 to 0.23, but the domain
   classifier only reaches 0.74. Severity for *input* drift is "moderate" while
   quality is "severe" — and quality is what a user feels. This is why the
   health score weights quality 45, judge 25, embedding 25.

The judge column deserves its own note. The baseline of 0.844 was established
by judging reference traffic. The live windows score 0.32–0.58 with a 60–100%
veto rate, meaning `qwen3:8b` is flagging groundedness or safety violations in
the generated replies. That is a **real finding about the application**, not
noise: the responder is inventing specifics the playbook never authorised. The
platform reports it rather than hiding it, and the golden suite catches the judge
changing its mind between runs.

---

## The three capabilities

### 1. Embedding drift

Five detectors, because each one fails differently.

| Detector | What it catches | Why it is here |
|---|---|---|
| **MMD²** (RBF, permutation-calibrated) | Shape change in the embedding cloud that marginals miss | The workhorse; the p-value is what makes it usable across varying window sizes |
| **Sliced Wasserstein** | Average gap over random 1-D projections | Cheap, and interpretable — a threshold you can argue about |
| **Domain-classifier AUC** | "Could a model tell reference from today's traffic?" | The most decision-relevant framing: AUC 0.5 = indistinguishable, 0.75+ = exploitable |
| **Normalised Fréchet** | Variance drift | A centroid comparison is blind to it; Fréchet is not |
| **k-NN novelty rate** | "What % of traffic is unlike anything we trained on?" | Converts an abstract distance into a number a product manager can act on |
| **Concept gap** | Accuracy on in-scope vs out-of-scope traffic | Separates "the world changed" from "we got worse" |

Two calibration details that took real debugging and are pinned by tests:

- **The domain-classifier null sits at 0.50, not 0.63.** A logistic regression on
  raw 384-dimensional sentence embeddings separates two samples of the *same*
  distribution at AUC ≈ 0.63 purely by exploiting sampling noise. Fitting PCA on
  the reference and projecting both clouds into it removes that inflation.
- **The MMD permutation null is computed on the same subsample as the observed
  statistic.** The unbiased estimator has a downward bias that shrinks with n, so
  a null built from 400 points against an observation from 1,400 makes every
  window "significant". This bug shipped silently in an earlier draft and is now
  a regression test.

A permutation test is a *confirmation*, never an escalation: severity comes from
effect size, and significance gates whether the window may raise drift at all.

### 2. Output quality

* **Labelled traffic** — accuracy, macro-F1, weighted precision/recall, ECE, MCE,
  adaptive ECE, Brier, AUC, reliability curve, per-intent recall.
* **Unlabelled traffic** (the 95% case) — confidence distribution, entropy,
  decision margin, abstention rate, and a clearly-labelled label-free proxy.
* **Delayed labels** — quality is attributed to the window in which the request
  was served, not the window in which the label arrived. Getting that wrong is
  how teams retrain because a label backlog flushed.

Calibration is treated as a first-class metric because it changes behaviour, not
just reporting. The classifier ships with **temperature scaling** fitted on a
held-out split: it moved ECE from 0.149 to 0.043 and Brier from 0.076 to 0.057.
Without it, the abstention threshold fires far too rarely, and that failure is
invisible until something else goes wrong.

ECE and Brier are alerted on **delta from the frozen baseline**, not on absolute
thresholds. A model with a known calibration gap is not a new incident every
window; the question is whether the gap is growing.

### 3. LLM-as-judge

* **Anchored rubric.** Five weighted dimensions — task correctness, groundedness,
  relevance, tone, safety/compliance — each with written anchors for the top and
  bottom of the scale. Weights are deliberately unequal: a fluent reply that
  invents a fee is worse than a clumsy but accurate one.
* **Veto conditions with a cap.** Dimensions carrying veto conditions cap the
  total score at 0.35 when scored 1. Without the cap, a reply that asked for a
  customer's PIN and scored 5/5 on everything else still averages 0.84 and sails
  through any threshold. (This was a real bug, caught by the golden suite.)
* **Regression test with confirmation.** The window mean is compared to a frozen
  baseline and only raised when the drop exceeds 2 standard errors. A 200-sample
  window must not page anyone about a 0.4-point wobble.
* **Pairwise comparison.** Win-rate against the baseline response to the same
  input, which is more reliable than absolute scores.
* **Stratified sampling.** The judged sample always covers the out-of-scope
  slice. Uniform sampling would quietly evaluate only traffic the model can
  plausibly handle — and the regression you most need to catch is the one hiding
  in the slice you never judged.
* **A golden regression suite** (`src/judge/agreement.py`): 8 hand-written cases
  covering a good answer, a vague one, a hallucinated fee, a credentials request,
  an out-of-scope escalation, a tone failure and an intent mismatch. Run in CI.
  A judge model upgrade that stops flagging the safety case is a **regression**,
  not an improvement — the suite is written to catch exactly that.
* **The ruler is stamped on the data.** Every judge record carries
  `judge_model` and `rubric_version`, and a change in either invalidates
  cross-run comparison. Bumping `qwen3:8b` to `qwen3:14b` moves every score;
  reading that as an application regression is how dashboards get ignored.

---

## The triage policy

`src/monitoring/orchestrator.py` is where monitoring becomes behaviour.

| Situation | Action | Rationale the system stores |
|---|---|---|
| Nothing fired | `noop` | — |
| Input drift, quality intact | `investigate` | Scope/coverage problem. **Do not retrain** — it cannot help while the new traffic is unlabelled. |
| Quality regression, no input drift | `investigate` | Suspect the model or its dependencies, not the data. Verify the artifact hash first. |
| Judge regression, no quality drop | `investigate` | Check the judge before assuming the application changed. |
| Sustained + severe quality + input drift | `retrain` | The distribution moved *and* the model got worse. |

Three rules underneath: drift must **persist** across consecutive windows before
anything expensive happens; the health score **weights quality above input
shift**; and every decision records its baseline, its evidence and its
confidence, so a decision made after a model swap cannot inherit the confidence
of one made before it.

Incidents open on escalation, accumulate an annotated timeline while open, and
close on recovery.

---

## Layout

```
config/settings.py        every threshold, path and backend toggle (LDM_* env overrides)
src/
  app/assistant.py        the system under observation: classifier + LLM responder
  data/datasets.py        Banking77 loader, offline fallback, reference construction
  data/shifts.py          the production-shift simulator: regimes + text transforms
  embeddings/encoder.py   sentence-transformers / hashing / Ollama backends
  storage/telemetry.py    SQLite time-series store (runs, windows, metrics, incidents)
  storage/reference.py    the frozen reference snapshot
  drift/embedding.py      MMD, sliced Wasserstein, domain classifier, Fréchet, novelty
  drift/tabular.py        PSI / KS / chi-square with frozen bins
  quality/outputs.py      calibration, delayed labels, label-free proxies
  judge/rubric.py         the anchored rubric, weights and veto conditions
  judge/evaluator.py      rubric scoring, regression test, self-consistency
  judge/agreement.py      the golden regression suite
  monitoring/monitor.py   the per-window loop
  monitoring/orchestrator.py  the triage policy
  llm/client.py           Ollama / OpenAI / Anthropic / mock behind one interface
  api/server.py           FastAPI: inference + monitoring read model + Prometheus
  reporting/generator.py  markdown, JSON and self-contained HTML reports
dashboard/                Streamlit: 5 tabs
scripts/                  bootstrap, demo_shift, eval_judge, export_report
tests/                    184 tests, all offline, ~90s
```

---

## Commands

| Command | What it does |
|---|---|
| `make setup` | venv + dependencies |
| `make bootstrap` | fetch data, train champion, freeze reference, validate judge |
| `make demo` | the full 14-window production shift with a real LLM |
| `make demo-fast` | same structure, offline encoder + mock judge, ~50s |
| `make api` | FastAPI on :8000 (docs at /docs) |
| `make dashboard` | Streamlit on :8501 |
| `make test` | full test suite |
| `make lint` | ruff |
| `make report` | regenerate reports from the last run |
| `make eval-judge` | run the golden judge suite against a candidate model |
| `make clean` | drop telemetry, artifacts and caches |

Everything is also a plain script — the Makefile only wraps them.

---

## Configuration

Every threshold is in `config/settings.py` and overridable with `LDM_*`
environment variables. Copy `.env.example` to `.env`.

```bash
LDM_JUDGE_PROVIDER=ollama         # ollama | openai | anthropic | rubric
LDM_JUDGE_MODEL=qwen3:8b
LDM_DOMAIN_AUC_MODERATE=0.62      # raise to alert less
LDM_EMBEDDING_DRIFT_VOTES=2       # how many detectors must fire
LDM_QUALITY_GATES=accuracy,ece    # what must break before a retrain is suggested
```

## Honest limitations

Worth stating before an interviewer finds them:

- **The judge is a local 8B model.** It is real, not mocked, but a frontier judge
  would be more reliable. The architecture is backend-agnostic.
- **Labels are partially synthetic.** Banking77 gives real text; the production
  shift, the out-of-scope mix and the 5% annotation noise are simulated. The
  dataset, the model and the quality metrics are real.
- **The hashing encoder is a lexical approximation.** It exists so the repo runs
  offline; the demo defaults to MiniLM.
- **Windows are independent, not sequential.** Real drift is autocorrelated; a
  production system would use EWMA control charts over the window series. The
  metric store is shaped for it and the confidence rule is a first step.
- **No auth, no multi-tenancy, no distributed execution.** It is a monitoring
  engine and a read model, not a production control plane.

## Licence

MIT. Dataset under CC BY 4.0 (Casanueva et al., 2020) — see the Banking77
citation in `src/data/datasets.py`.
