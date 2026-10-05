# Interview playbook

How to talk about this project in a loop interview, in the order questions
actually arrive. The short version is at the top; the details are below.

---

## The 30-second version

> "I built a drift and quality monitoring platform for an LLM application. It
> watches a banking support assistant across a simulated production shift and
> makes one decision per window: no-op, investigate, retrain, or roll back. The
> core idea is that **input drift and output-quality drift are different
> incidents with different remedies** — a model can see traffic it has never seen
> and still be accurate, and it can keep seeing identical traffic and lose
> accuracy. Retraining fixes the first cause and does nothing for the second, so
> the policy weights quality far above input shift. The other thing I'd point at
> is that I treat the LLM judge as a component that itself needs monitoring: a
> golden evaluation set in CI, veto conditions that cap the score, and a rule
> that a judge model change invalidates historical comparisons."

---

## Likely questions, and what to say

### "What drift detection do you use, and why?"

Five detectors on sentence embeddings, deliberately overlapping, because each
fails differently:

- **MMD with an RBF kernel and a permutation-calibrated p-value.** Catches
  shape changes the marginals miss.
- **Sliced Wasserstein.** Cheap and interpretable; a threshold you can argue
  about in review.
- **Domain-classifier AUC.** Train a regularised logistic regression to tell
  reference from current traffic. AUC 0.5 means indistinguishable. This is the
  framing a stakeholder understands: *"could a model tell them apart?"*
- **Normalised Fréchet distance.** Catches variance drift that a centroid
  comparison is blind to.
- **k-NN novelty rate.** Turns an abstract distance into *"40% of today's traffic
  is unlike anything this model saw in training."*

**The story that matters** is the calibration work. My first version of the
domain classifier had a null AUC of 0.63 — it separated two samples of the
*same* distribution 63% of the time, purely by exploiting sampling noise in 384
dimensions. Fixing it (fit PCA on the reference, project both clouds into it,
drop rows that appear in both sets) moved the null to 0.50. I also found the MMD
p-value was mis-calibrated because I computed the observed statistic on the full
window and the permutation null on a subsample — the unbiased estimator's
downward bias shrinks with n, so every window came back "significant". Both are
now regression tests.

**If pressed on "why not just KS or PSI?"** — those are per-column. Text drift
lives in relationships and in topics you never enumerated. I keep PSI and KS on
cheap text-derived features alongside the embedding detectors, because a column
drift alert says *what* changed ("median message length halved, 30% lowercase")
while the embedding detectors say *that* something changed. Both are useful; they
answer different questions.

### "How do you know the drift detector isn't just noise?"

Three things, in order of importance:

1. **Effect size sets severity, significance only gates it.** A permutation test
   is a confirmation, not an escalation. If the effect is small, a significant
   p-value doesn't raise the severity.
2. **I measured the null.** Before tuning any threshold I ran each detector on
   in-distribution windows and checked the firing rate. If a detector fires on
   30% of healthy windows, it gets muted within a week and the whole dashboard
   loses credibility. In the demo, a 2× volume spike with identical traffic mix
   fires *nothing*.
3. **The volume-spike window is in the demo on purpose.** It's the control case:
   same distribution, twice the samples, no alert. Sample-size insensitivity is
   the single most common failure in drift dashboards, because PSI has no
   significance test built in.

### "How do you decide what to do about a drift alert?"

The orchestrator is explicit policy, not a pager and a shrug. The core rule:

| Situation | Action | Why |
|---|---|---|
| Input drift, quality intact | investigate | Scope problem. Retraining can't help while the new traffic is unlabelled. |
| Quality regression, no input drift | investigate | Suspect the model or its dependencies. Verify the artifact hash before retraining. |
| Judge regression, no quality drop | investigate | Check the judge before assuming the app changed. |
| Sustained + severe quality + input drift | retrain | The world moved *and* we got worse. |

Plus: **nothing expensive happens on a single window** — drift must persist for
a configurable number of consecutive windows. And the health score weights
quality 45, judge 25, embedding drift 25, volume 5, because a user feeling a
wrong answer costs more than the same-sized movement in the input distribution.

**Good follow-up to offer:** in the demo, window 7 has OOD rate 0.40 and accuracy
0.96 → 0.58, but the domain classifier only reaches 0.65. If you alerted on
input-drift severity alone you'd call that "moderate" and deprioritise it. The
incident is severe. That asymmetry is the argument for tracking quality
separately.

### "Tell me about the LLM-as-judge part."

Most candidates use an LLM judge as a vibe check. Three things I did that make it
defensible:

1. **Anchored rubric with veto conditions and a cap.** Five weighted dimensions
   with written anchors. The critical bit: dimensions carrying veto conditions
   *cap the total score* when they fail. Without that cap, a reply that asked for
   a customer's PIN and scored 5/5 on correctness, relevance and tone still
   averages 0.84 and passes any threshold you'd set. My first rubric had exactly
   this bug and the golden suite caught it.
2. **A golden evaluation set with a stored baseline and an explicit gate.** Eight
   hand-written cases: a good answer, a vague one, a hallucinated fee, a
   credentials request, an out-of-scope escalation, a tone failure, an
   intent mismatch. Run in CI. A judge upgrade that stops flagging the safety
   case is a *regression*, not an improvement.
3. **The judge is versioned like the app.** Every record carries the judge model
   and rubric version. Change either and cross-run comparison is refused with an
   explicit message. Bumping `qwen3:8b` to `qwen3:14b` moves every score, and
   reading that as an application regression is a mistake teams make constantly.

**On why pairwise at all**: the literature is clear that pairwise comparison is
more reliable than absolute scoring, so where a baseline response exists I
report win-rate against it rather than leaning on the absolute number.

**On cost**: judging every request isn't affordable, so evaluation is a
stratified sample per window — stratified on in-scope status specifically,
because uniform sampling evaluates only traffic the model can plausibly handle,
and the regression you most need to catch is the one hiding in the slice you
never judged.

### "What about when you don't have labels?"

Which is 95% of production traffic. Three regimes, and I'm explicit about which
is which:

- **Labelled** — full metrics. Ground truth; trust it.
- **Unlabelled** — confidence distribution, entropy, decision margin,
  abstention rate, and a label-free proxy. The proxy is deliberately *biased*
  and the dashboard labels it "proxy". I would never report it as accuracy.
- **Delayed labels** — the real production shape. Quality is attributed to the
  window in which the request was *served*, not the window in which the label
  arrived. Getting that attribution wrong is how teams retrain because a label
  backlog flushed.

### "What would you build next?"

In priority order, and I'd defend the ordering:

1. **A sequential change-point detector** (CUSUM or Page-Hinkley) over the
   window series. My windows are independent today; real drift is autocorrelated
   and a control chart catches gradual erosion that a per-window threshold walks
   past.
2. **Shadow scoring** — run the challenger on live traffic without serving it, so
   a retrain decision is backed by an A/B on the actual distribution.
3. **Slice-aware alerting** — the health score is global; a 5% slice can be fully
   broken while the aggregate looks fine. That's the most common way a
   production incident hides.
4. **Judge agreement at scale** — periodic Cohen's kappa against a stratified
   human-labelled sample, with a validity gate that suppresses judge-sourced
   alerts when the judge degrades.
5. **Streaming reference updates** — a frozen reference is correct but ages. A
   reference that adapts too fast never sees drift; one that never adapts needs
   manual refreshes.

### Questions *I* would ask if I were interviewing for this

Use these to steer — they show you understand the problem, not just the code.

- "Your domain-classifier AUC thresholds — how did you pick them, and what does
  the detector do on a window that's in-distribution but larger than usual?"
- "What happens to your baselines when you change the embedding model?"
- "How do you know your judge is still right?"
- "Show me a case where your system said 'investigate' and you agreed, and one
  where you disagreed."
- "Your health score is a weighted sum. Why those weights, and what would make
  you change them?"

---

## Numbers to have ready

Measured on this repo, not estimated:

| Claim | Number |
|---|---|
| Intent classifier (27 intents, Banking77) | accuracy 0.905, macro-F1 0.866, AUC 0.935 |
| Before temperature scaling | ECE 0.149, Brier 0.076 |
| After temperature scaling (T = 0.646) | ECE **0.043**, Brier **0.057** |
| Out-of-scope accuracy before promotion | 0.000 |
| Out-of-scope accuracy after promotion (53 intents) | > 0.6 on held-out probe |
| Accuracy during the crisis windows | 0.227 (from 0.971) |
| OOD rate during the crisis windows | 0.639 (from 0.021) |
| Domain AUC, baseline → crisis | 0.51 → 0.74 |
| Concept gap (in-scope − out-of-scope accuracy) | 0.583 at peak |
| Test suite | 184 tests, offline, ~90s |
| Full demo | 14 windows, 2,758 requests, 140 judge calls, 37 min on CPU |
| Fast demo | 14 windows, 52s |

## The three-minute demo

> Live demo: https://awonaziz.github.io/llm-drift-monitor/
> Open this first if there is no terminal available. It is a self-contained
> report from the same run the numbers below come from.


1. `make demo-fast` — run it live if you have a terminal. The table at the end
   prints itself.
2. Open `data/reports/report_*.html` — self-contained, inline SVG, no server.
3. `make dashboard` — five tabs. Start on **Embedding drift** (domain AUC over
   time with regime bands), then **Output quality** (the accuracy cliff and the
   reliability curve).
4. `curl localhost:8000/monitoring/status` — the same numbers as an API, plus
   `/metrics/prometheus` if your target company runs Prometheus.

If they ask "what would break in production", the honest answer is in the README's
limitations section — say it before they find it.
