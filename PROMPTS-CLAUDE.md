PROMPTURILE:
# BreaQ 2026 · Task 2 · Prompts for Claude Code (Opus 5.5, extra high)

Run order:

| # | Task | Difficulty | Est. time | Can run in parallel with |
|---|---|---|---|---|
| 0 | Context (start of EVERY new session) | — | 1 min | — |
| 1 | Noise × shots for all models (answer to the mentor's hint) | high | 2–3h | 2 |
| 2 | R2: entangled VQC, warm start + noise-aware fine-tune | high | 2–3h | 1 |
| 3 | All 492 frauds (CV) + statistical significance | high | 2h, overnight | 1 |
| 4 | **EXTRA: adaptive shot triage + noise meter** | high | 2–3h | after 1 |
| 5 | Slide figures | medium | 1h | after 1–4 |
| 6 | Final end-to-end verification | medium | 1h | last |

One prompt per session. Never more than two heavy simulations at once.

---

## Prompt 0 — Context (paste at the start of every new session)

```
First read docs/PROJECT.md and CLAUDE.md, then qai/protocol.py, qai/moco.py and qai/kit_vqc.py.
We are working on Task 2 (BreaQ 2026). Our main method is MOCO. The baseline VQC and logistic regression
have already been run through the official protocol and reproduce the task statement
(0.580 ± 0.053 / 0.703 ± 0.010).

Fixed rules for everything you do in this session:
- The test set is touched only for the final evaluation. Every choice (configuration, epoch, threshold,
  shot count, hyperparameters) is made on train or validation. If anything seems to require the test set
  for a choice, stop and ask me.
- Do not modify the functions in qai/protocol.py that were copied from the kit. If you need something else,
  write a new function next to them.
- Seeds [0, 1, 2, 3, 4]. Report mean ± std. Numbers from _fast runs are never reported.
- Every experiment states its hypothesis before the code, in the docstring and in docs/LOG.md.
- Results go to results/task2/ (CSV + .npz with scores), cached per seed and per p so runs can resume.
- Every circuit execution on hardware goes into the cost ledger, with its formula and a qml.Tracker check
  on a small sample.
- When done: update docs/PROJECT.md (the number, its source, and one changelog line), keeping the document
  in Romanian to match its existing style; run make task2-test; commit on the branch.
Confirm you have read the document, tell me in 5 lines what you understand MOCO is and what is currently
queued, then wait for the task.
```

---

## Prompt 1 — Noise × shots for all models (answer to the mentor's hint)

```
Task: qai/experiments/exp_noise_shots.py. This is our answer to the mentor's hint "increase the noise and
see what happens", so it is one of the most important slides.

Hypothesis (write it in the docstring before any code): the protocol uses exact expectation values, i.e.
infinite shots. On hardware, noise compresses the signal. For MOCO the compression is uniform
(λ², λ = 1 − 4p/3), so it only costs a shot factor of about λ⁻⁴. For the baseline the compression is large
and non-uniform: it breaks the ranking even with infinite shots, and with finite shots it collapses much earlier.

Models: R0 baseline (trained weights from cache), R3 MOCO single 1:10, and R2 if it is ready
(otherwise leave it behind a --models flag).
Grid: p ∈ {0, 0.001, 0.01, 0.05, 0.1, 0.2, 0.3, 0.5}, S ∈ {64, 256, 1024, 4096, 16384}.
Everything on the full official test set, 5 seeds.

Shot simulation must be EXACT for each model:
- Baseline (reads only Z₀): the noisy expectation is ⟨Z₀⟩_p from noisy_predict, so shots = Binomial(S, (1+⟨Z₀⟩)/2).
- MOCO (product state, H = Σ w_q Z_q): qubits are independent, so one binomial per qubit with mean λ²·⟨Z_q⟩.
  Check on 200 rows that the mean of the binomials converges to the kit's noisy_predict.
- R2 (entangled, H = Σ w_q Z_q): qubits are NOT independent. Use qml.probs on default.mixed with the kit's
  noise model (same insertion as noisy_qnode), sample multinomially from the joint 16-outcome distribution,
  then compute Σ w_q z_q per shot.
Noisy expectations are computed once per (model, seed, p) and cached. Only the sampling is repeated per S.
For reproducibility: rng = np.random.default_rng(stable hash of (model, seed, p, S)).

For each (model, p, S) compute: AUPRC, AUROC, Spearman ρ against the exact p = 0 scores, plus F1 and MCC
with the threshold chosen on validation at the SAME (p, S). Validation is sampled the same way, with shots,
not exactly.

Also compute:
1. Effective compression: median |s_p − ½| / |s_0 − ½|. For MOCO it must equal λ² (add a test for this).
2. Shot tax: the smallest S in the grid for which mean AUPRC ≥ AUPRC(p = 0, exact) − 0.01, per model and per p.
   If none exists, write "> 16384". For MOCO, compare with the theoretical prediction S(0)·λ⁻⁴.

Outputs:
- results/task2/noise_shots.csv (one row per model, seed, p, S);
- results/task2/shots_needed.csv;
- results/task2/noise_shots_heatmap.png: one AUPRC(p, S) heatmap per model, same colour scale;
- results/task2/noise_shots_curves.png: AUPRC vs p at S = 1024, all models on one plot, ± std bands,
  and a horizontal line for logistic regression.

Checks before reporting:
(a) at S = 16384 and p = 0, each model's AUPRC is within 0.01 of its exact value;
(b) for MOCO at p = 0.05, the effective compression equals λ² with error < 1e-6.
Add the target to make task2, update PROJECT.md (sections 2, 4 and 5, the "shots under noise" ledger column)
and LOG.md. At the end give me the "shots needed" table and 3 sentences on what it shows.
```

---

## Prompt 2 — R2: entangled VQC (answer to "it's not a real VQC")

```
Task: qai/experiments/exp_entangled.py (R2). It answers the objection "MOCO is not a real VQC / has no
entanglement" and honestly measures what entanglement costs here. It does not need to beat MOCO; it needs
to be correct.

R2 circuit on 4 qubits: RY(x_q) → Rot(α_q) → linear CNOT chain (0→1, 1→2, 2→3) → Rot(β_q) → readout
H = Σ_q w_q Z_q, with p_fraud = (1 − ⟨H⟩)/2 and w_q ≥ 0, Σw = 1 (parametrise with a softmax).
Report native gates, CNOTs and noise_locations using the functions in qai/protocol.py.
Same data as MOCO: the kit's encoder, 1:10 undersampling, same seed.

Three variants, each on 5 seeds:
- R2a, random init: the kit's initialisation, Adam, BCE, like the kit's train_vqc but on the R2 circuit.
- R2b, MOCO warm start: first run MOCO with the `linear_cnot` feature map (the Heisenberg reparametrisation
  is exact with CNOTs before the final Rots, so β and w come from the convex fit), α = 0, then fine-tune
  with Adam without noise.
- R2c, warm start + noise-aware fine-tune: like R2b, then a few epochs on the kit's noisy_qnode at p_train.
  p_train ∈ {0.01, 0.05}, chosen on validation AUPRC at p = 0.05 (on a stratified 20% validation subset).
The final epoch is chosen on validation (same subset), with early stopping. The test set is not touched.

Ledger for each variant, as in the kit: parameter-shift = 1 + 2·(number of parameters that actually reach
the observable), checked with qml.Tracker on 5 examples. Plus validation subset × number of evaluations,
plus 3 bases × rows for the MOCO warm start in R2b and R2c.
Evaluation: evaluate_under_noise from qai/protocol.py on the full validation and test sets,
p ∈ {0, 0.001, 0.01, 0.05}.

Optional, only if time remains after everything above: ZNE on R2c (fold factors 1, 3, 5, Richardson),
with qml.noise.mitigate_with_zne or manual folding, cross-checked against the manual version on 200 rows.
ZNE executions (×3 per score) go into the ledger. Hypothesis, already seen on a subset: ZNE raises ρ on
shallow circuits but cannot work on the kit's deep circuit, where the amplified signal is ~0.

Outputs: results/task2/r2_*.csv, one row per (variant, seed, p), all protocol columns plus the ledger.
Update the method ladder and the ledger in PROJECT.md. At the end give me the table
R0 / R2a / R2b / R2c / R3 / LogReg at p = 0 and p = 0.05, and tell me whether the warm start helped
compared with random init.
```

---

## Prompt 3 — All 492 frauds + statistical significance

```
Task: qai/experiments/exp_allfrauds_cv.py. The official test set has only 98 frauds, so the difference
between MOCO (0.716) and logistic regression may be statistical noise. We want to know honestly whether it
is significant, and to answer the mentor's request to run on all frauds.

Part A, CV on the full dataset: StratifiedKFold(5, shuffle=True, random_state=42) over all 284,807 rows.
In each fold, the non-test part is split 75/25 stratified (random_state=42) into train and validation,
exactly like the protocol. Everything is fitted on the fold's train only: undersampling, encoder.
Models: MOCO single 1:10, LogReg 1:10, LogReg 1:1, baseline R0 (30 epochs, as in the kit).
The MOCO configuration is fixed in advance. State honestly in the document that it was chosen on the
official validation set, which partially overlaps the test folds: a small limitation, not something we hide.
For each fold: AUPRC and AUROC at p = 0 and p = 0.05 (baseline via noisy_predict, MOCO via the λ² formula,
checked on 500 rows against noisy_qnode), plus F1 and MCC with the threshold chosen on the fold's validation
at the same p. Every fraud ends up in a test fold exactly once; verify this with an assert.

Reporting: mean ± std over folds, and PAIRED per-fold differences (MOCO − LogReg 1:10, MOCO − baseline),
plus a Wilcoxon test on the 5 differences. State clearly that with n = 5 the test has low power.

Part B, bootstrap on the official test set: for seed 0 (and then averaged over the 5 seeds),
2,000 stratified resamples of the test set, using the SAME indices for all models (paired).
Report a 95% CI for AUPRC per model and for the differences MOCO − LogReg 1:10 and MOCO − baseline,
at p = 0 and p = 0.05.

Hypotheses, written before running:
H1: MOCO − baseline > 0, significant at p = 0.05.
H2: MOCO − LogReg 1:10 is not significantly different from 0 (we match the classical reference, we do not beat it).
Report the result whichever way it comes out.

Outputs: results/task2/allfrauds_cv.csv, results/task2/bootstrap_ci.csv and a forest plot (differences with
95% CIs). Run in the background, cached per fold. Update PROJECT.md (section 4, and the answer to
"Why do you believe the difference is real?" in section 7).
```

---

## Prompt 4 — EXTRA: adaptive shot triage + noise meter (our differentiator)

Why this sets us apart: everyone reports a fixed number of shots per transaction. We show the noise tax is
paid only where it is needed. 99.8% of transactions are obviously legitimate, and 16–32 shots are enough to
decide them; only ambiguous cases get more shots. This is possible only because of the λ² theorem: we know
exactly where the threshold moves at any p, so we can stop measuring early with guarantees. The baseline
cannot do this, because its scores all collapse to ~0.500 (its threshold at p = 0.05 is 0.500 ± 0.001).

```
Task: qai/experiments/exp_adaptive_shots.py. This is the extra that sets us apart from the other teams,
so I want maximum rigour and a careful check of the statistical guarantees.

Idea: on hardware, each MOCO score is estimated from shots. Each shot yields 4 bits at once, so a per-shot
estimator h = Σ_q w_q z_q ∈ [−1, 1]. With a fixed number of shots (1024) we pay the same for every
transaction, although 99.8% are far from the threshold. Adaptive triage: measure in batches of b = 16 shots
and stop as soon as an anytime-valid confidence interval for the mean of h excludes the threshold t_h(p),
with a cap S_max. The threshold at p comes analytically from the p = 0 threshold (transfer_threshold),
so no noisy validation pass is needed.

Part A, the triage:
1. Anytime-valid CI for a mean bounded in [−1, 1]: Hoeffding with α_k = α·6/(π²k²) per batch k (union bound),
   or empirical-Bernstein, with α = 0.01. Implement it separately, with a test: over 10,000 simulated
   sequences, the frequency with which the interval misses the true mean must be ≤ α.
2. Rule: at batch k, if the CI excludes t_h, stop (fraud / legit decision); otherwise continue up to S_max.
   The score used for AUPRC is the running mean at stopping time. Also report the variant
   "triage for the decision + refine to S_fixed only for transactions flagged as fraud", in case AUPRC drops too much.
3. Exact simulation: product state, so independent binomials per qubit with mean λ²·⟨Z_q⟩. Check on 500 rows
   that, with stopping disabled, you reproduce the fixed-S version.
4. Evaluation on the full test set, 5 seeds, p ∈ {0, 0.01, 0.05, 0.2}, S_max ∈ {1024, 4096}:
   mean shots per transaction (separately for legit and fraud), F1, MCC, recall and AUPRC.
   The fair comparison: MOCO with a fixed S EQUAL to the adaptive mean shot count (same budget), plus fixed S = 1024.
   Hyperparameters (b, α, S_max) are chosen on validation, not on test.
5. The same procedure on the baseline (Z₀ readout, binomial), to show that triage cannot work there:
   scores are compressed towards ½, so almost every transaction hits S_max.

Part B, noise meter (bonus, ~30 min): from the theorem, median |s_p − ½| / |s_0 − ½| = λ², where s_0 is
classically computable (product state). From N_cal calibration transactions measured with S shots,
estimate λ² and then p̂ = (3/4)(1 − √λ²). Report the error of p̂ against p as a function of
N_cal ∈ {50, 200, 1000} and S ∈ {256, 1024}, at p ∈ {0.01, 0.05, 0.2}. Show that using p̂ in
transfer_threshold keeps the F1 of the threshold calibrated on the full noisy validation set.
Ledger conclusion: threshold calibration costs N_cal executions instead of 56,962.

Outputs: results/task2/adaptive_shots.csv, noise_meter.csv and two plots:
(a) histogram of shots per transaction, legit vs fraud, at p = 0.05;
(b) F1 vs mean shots, adaptive vs fixed, MOCO vs baseline.
Add the ledger column "inference shots per transaction" (fixed vs adaptive) and the line
"threshold calibration with the noise meter".
Update PROJECT.md: a new subsection in section 2 ("We pay the tax only where it is needed"), the results,
and a new question in section 7: "Why would this matter on real hardware?".
```

---

## Prompt 5 — Slide figures

```
Task: qai/experiments/task2_figures.py. Generate all figures for the presentation in one consistent style,
from the CSVs in results/task2/. Do not run any new experiment: if a CSV is missing, report what is missing
and move on.
Style: same font, fixed colours per model (baseline, R2, MOCO, LogReg), PNG at 200 dpi plus SVG,
readable on a projector (text ≥ 14 pt), titles that state the conclusion, not just what the plot shows.
Figures:
1. Test AUPRC vs p (official protocol), ± std bands, all models.
2. Spearman ρ vs p.
3. Ledger: log-scale bars for executions and shots, baseline vs MOCO, percentage written on each bar.
4. AUPRC(p, S) heatmaps for baseline and MOCO, side by side.
5. AUPRC vs p at S = 1024, up to p = 0.5 (answer to the hint).
6. Circuit diagram: MOCO (4 lines, RY → Rot) vs baseline, with gate, CNOT and noise-location counts.
7. Ablations: AUPRC at p = 0.05 and ρ for single / pair / linear_cnot / kit_zz.
8. Forest plot of the differences with 95% CIs (from Prompt 3).
9. Adaptive triage: shot histogram and F1 vs budget (from Prompt 4).
10. The validation selection table, as a clean image.
Write results/task2/figures/INDEX.md: for each figure, the slide it goes on and one sentence to say
during the presentation.
```

---

## Prompt 6 — Final verification (Sunday morning, before 05:00)

```
Task: final verification before submission. Do not add new experiments.
1. From a clean repo (git clone into a temporary folder): make install, make data, make task2-test, then
   python -m qai.experiments.task2_report. Everything must work from cache or rerun with fixed seeds.
2. Check every number in docs/PROJECT.md and results/task2/TABLES.md against the CSV it comes from.
   List any mismatch. Mark any number without a source.
3. Check the rules: no file chooses anything on the test set (search for y_test / test inside selection loops),
   every fit is on train, every table has ≥ 5 seeds, nothing _fast is reported.
4. Check the ledger: every line has its formula and its Tracker check.
5. Write README_TASK2.md: one command for everything, what each file produces, run times, library versions.
6. Commit and push. Give me the final list of problems, if any, ordered by severity.
```
