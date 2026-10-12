# Preregistration — s12: Culture-conditioned irrationality infusion

Date: 2026-10-08. Written BEFORE any model runs. Author: Prabhu Rangaraju (Independent Researcher).

## 1. Research question

Does calibrating an LLM's irrationality levers to *region-specific, measured preference
distributions* reproduce real cross-country heterogeneity in economic decisions better
than a single generic-human calibration or no calibration?

## 2. Falsifiable hypothesis

**H1:** Region-conditioned calibration yields a significantly smaller distributional
distance to real country-level choice distributions than generic-human calibration,
while preserving within-country dispersion.

**Falsification:** If the primary test (Sec. 7) fails (p >= 0.05 or wrong direction),
H1 is rejected and reported as rejected. No post-hoc metric shopping: the primary
metric, test, and decision rule are fixed below.

## 3. Ground truth (foundation verification, 2026-10-08)

- **GPS (Falk et al. 2018): NOT publicly downloadable.** The dataset requires a
  data-request form submission at gps.econ.uni-bonn.de (an external send; prohibited
  for this track) and its terms forbid redistribution to third parties. Verified live
  2026-10-08.
- **WVS trend file: requires registration.** Not accessible from this sandbox.
- **FALLBACK USED: Rieger, Wang & Hens (2015) INTRA survey, 53-country risk
  preferences.** The INFORMS supplemental files are stated as publicly available in
  the published abstract (doi:10.1287/mnsc.2013.1869) but are Cloudflare-blocked from
  this sandbox. The published per-country CPT parameter estimates from the INTRA
  program (Rieger, Wang & Hens 2017, *Theory and Decision*; numerical values as
  tabulated open-access in Xie, Hwang & Pantelous 2017, *APJFS*, who report their
  estimates as closely matching the INTRA-published values) ARE accessible and form
  the numerical ground truth. All values below are transcribed from that published
  table (Loss Aversion / Risk Aversion / Subjective Weighting with SEs, 27 countries).

### 3.1 Country-level parameters (published)

| Country   | lambda (loss aversion) | alpha (curvature) | delta (prob. weight) |
|-----------|----------------------|-------------------|----------------------|
| Iceland   | -0.112 | 1.441 | 1.551 |
| Greece    |  0.331 | 1.460 | 0.903 |
| Brazil    |  0.603 | 1.494 | 0.807 |
| Hungary   |  0.925 | 1.008 | 0.534 |
| Pakistan  |  1.198 | 1.501 | 0.946 |
| Australia |  1.651 | 1.210 | 0.771 |
| Germany   |  1.903 | 1.431 | 0.717 |
| Japan     |  1.932 | 1.441 | 0.714 |
| Denmark   |  2.205 | 1.504 | 0.861 |
| Italy     |  3.348 | 1.415 | 0.703 |

### 3.2 Country selection (preregistered, from data not stereotypes)

10 countries, stratified across the empirical loss-aversion distribution (lambda is the
PT parameter shown by Wang, Rieger & Hens 2017 to vary most with culture):
Iceland, Greece, Brazil, Hungary, Pakistan, Australia, Germany, Japan, Denmark, Italy.
lambda spans -0.11 to 3.35; alpha spans 1.01 to 1.50; delta spans 0.53 to 1.55.
Dropped interior/redundant countries (Mexico, Sweden, Poland, Canada, etc.) to keep
compute modest. Selection uses ONLY the published numbers above.

### 3.3 Within-country distributions (reconstruction, disclosed)

Published work reports per-country SEs, not individual-level SDs. Within-country SDs
are reconstructed as SD = SE * sqrt(n) with n = 150 (median INTRA per-country n;
~7000 subjects / 53 countries), winsorized: lambda SD in [0.05, 0.50], alpha SD in
[0.03, 0.25], delta SD in [0.05, 0.40]. Per-country SEs used: lambda ~0.010 typical
(Iceland 0.075), alpha ~0.005, delta ~0.014. Individual parameters drawn i.i.d. from
truncated normals (lambda truncated at 0.02; Iceland's published negative mean becomes
small-positive draws — disclosed). Sensitivity analysis varies all within-SDs x0.5/x1.5.
This reconstruction affects absolute fit levels, NOT the region-vs-generic comparison
(both conditions use the same reconstructed family); the hypothesis test is on the
*relative* comparison.

## 4. Model

Qwen3-1.7B-GGUF, Q4_K_M quantization, CPU-only via llama-cpp-python
(n_ctx=2048, n_threads=2, temp=0.7, top_p=0.8, top_k=20 — same as the companion
s9 track's paper params). sha256 verified. /no_think suffix to suppress reasoning
traces. max_tokens=64.

## 5. Conditions (prompt levers; inference-only)

For each simulated individual i, draw (alpha_i, lambda_i, delta_i):

- **REGION:** drawn from that individual's country's reconstructed distribution.
  Profile embedded in a third-person forecasting prompt (Wedgwood et al. 2026):
  > "You are forecasting the decision of one specific survey respondent. In a
  > validated economic survey this respondent stated: they would pay at most $CEg
  > for a 50/50 chance to win $100; they would need at least $WTAl to accept a
  > 50/50 chance of losing $100. For this respondent, losses loom about LAMEx as
  > large as same-sized gains. Now forecast: which option would this respondent
  > choose? Reply with only A or B."
  where CEg = round(100 * 0.5^phi_i)  [solves ce^(1/phi) = 0.5*100^(1/phi)],
        WTAl = max(1, round(100 * (0.5*lambda_i)^phi_i)),
        LAME = round(lambda_i,1).
  (phi = published "risk aversion"; phi > 1 = concave value function, matching the
  published universal gain-risk-aversion finding. Objective 0.5 used in the profile
  description — disclosed assumption.)
- **GENERIC:** identical prompt, but (alpha_i, lambda_i, delta_i) drawn from the
  POOLED 27-country distribution (one generic human profile).
- **UNCALIBRATED:** no profile. "Choose the option you think is best. Reply with
  only A or B."

A/B option order randomized per (individual, task) with master seed 20261008.
**No country name, ethnicity term, or trait inventory appears in ANY prompt**
(scripted audit in analysis). Country labels exist ONLY as analysis grouping keys.

## 6. Tasks (held out from calibration)

8 binary lotteries with stakes/probabilities NOT used in the profile elicitation,
selected 2026-10-08 by grid search for maximal between-country variance in the
deterministic CPT prediction (2 gain, 2 loss, 4 mixed; mixed lotteries are the only
ones where loss aversion lambda affects choice, so they carry the cultural signal):

- G_a: $35 sure vs 50% $100
- G_b: $55 sure vs 80% $100
- L_a: lose $40 sure vs 50% lose $100
- L_b: lose $45 sure vs 50% lose $100
- M_a: $0 sure vs 50/50 +$100/-$50
- M_b: $0 sure vs 50/50 +$120/-$70
- M_c: $10 sure vs 50/50 +$100/-$70
- M_d: -$10 sure vs 50/50 +$100/-$100

**Real-data counterpart:** deterministic CPT-predicted choice per individual:
choose B iff CPT(B) > CPT(A), with v(x) = x^alpha (x>=0), -lambda*(-x)^alpha (x<0),
and Tversky-Kahneman one-parameter weighting w(p) = p^delta/(p^delta+(1-p)^delta)^(1/delta).
(Exact weighting form used by Wang et al. 2017 is not public; TK-1992 form disclosed
as our assumption.)
**EV-optimal choice:** higher expected value (lower expected loss). Used for the
baseline rationality-gap replication.

## 7. Metrics, tests, decision rules

- **PRIMARY:** per country, Wasserstein-1 distance between the simulated
  per-individual %risky distribution (16 values in {0/8..8/8}) and the
  CPT-predicted distribution; mean over 10 countries. Paired permutation test
  (10 country pairs), one-sided H1: W1(region) < W1(generic), alpha = 0.05.
- **SECONDARY (descriptive, 95% bootstrap CIs):**
  (a) within-country dispersion preservation: mean |log(sim SD / real SD)|;
  (b) between-country heterogeneity: Spearman rho of country-mean %risky, sim vs real;
  (c) EV-alignment gap: mean |%-EV-optimal_sim - %-EV-optimal_real| (baseline check).
- **BASELINE REPLICATION (replicate-then-new, runs first):** uncalibrated model on
  the 8 tasks; expect its %EV-optimal > CPT-human %EV-optimal (the Liu et al. 2025
  rationality gap: LLMs more EV-rational than humans). Report as check, not gated.
- **DECISION RULE:** H1 SUPPORTED iff primary test p < 0.05 in the predicted
  direction. Otherwise H1 REJECTED. Secondaries never overturn the primary.

## 8. Robustness / strengtheners (preregistered)

R1. Prompt paraphrase (reworded profile template), 3 countries x 16 indiv x 8 tasks,
    region condition only.
R2. Seed variant (master seed 777), same subset.
R3. Temperature 0.3, same subset.
R4. Lambda-only ablation: profile carries only the loss-aversion sentence (no CE numbers).
R5. Within-SD sensitivity: x0.5 and x1.5 (recomputed in analysis, no new generations).
R6. Scripted no-country-label audit of all prompts + manual stereotype spot-check of
    50 generations (expect none; profiles are pure numbers).

## 9. Anti-essentialism design rules (HARD CONSTRAINT, author's kill criterion)

1. Calibrate ONLY to measured preference distributions with within-country variance.
2. NEVER fixed national personas, NEVER race/ethnicity categories, NEVER
   essentialized trait inventories (no Hofstede dimensions as inputs).
3. Culture is situated/contextual (Oyserman & Lee 2008): the prompt frames ONE
   specific respondent's measured survey answers, not a national character.
   The manuscript states this explicitly.
4. Any sub-analysis that cannot be done without stereotyping is killed on sight.

## 10. Sample sizes and output completeness (test 27)

- Baseline (uncalibrated): 8 replicates x 8 tasks = 64 generations (country-free;
  reused as the per-country uncalibrated distribution for all 10 countries).
- Main: 10 countries x 8 individuals x 2 conditioned arms x 8 tasks = 1280.
- Robustness: 4 variants x (2 countries x 8 indiv x 8 tasks) = 512.
- Expected total: 1856 JSONL records. Analysis asserts exact counts before any
  metric is computed; any shortfall is reported, not silently patched.

*Amendment 2026-10-08 (before main runs; 12 baseline generations already
collected and retained — job list is deterministic so they are reused):*
*N_INDIV 16 -> 8 and robustness subset 3 -> 2 countries (Germany, Italy),
purely to bound CPU time on the 2-CPU sandbox (measured ~21 s/generation under
load; the original 4224-generation plan projected >24 h). The primary test is a
paired permutation test over the 10 countries, whose validity does not depend on
the per-country individual count; per-country W1 estimates are noisier and this
is reflected in wider reported CIs. No other design element changed.*

*Amendment 2, 2026-10-08 (BEFORE any main-run generations; all v1/v2 pilot
records discarded as invalid — see below). Elicitation redesign, empirically
motivated. Diagnostics on 2026-10-08 showed: (a) the chat template produces
degenerate label bias (12/12 "B" with system message, 8/8 "A" without; P(label)
~ 0.95-1.0 regardless of profile); (b) raw completion removes the degeneracy
but binary A/B choices are content-insensitive to numerical CE/WTA profiles
(first-token P(A) ~ 0.34-0.39 identical for Italy-like vs Hungary-like profiles);
(c) the model cannot reliably compare a price to a stated numerical CE
(7/10 on adjacent comparisons, systematic errors); (d) qualitative persona
sentences DO move choices monotonically: gains P(yes) = 0.53/0.31/0.00 across
3 risk levels; losses P(yes) = 0.33/0.28/0.00 across 3 loss levels
(diagnostics /tmp/lp10.out, /tmp/lp11.out).*

*New elicitation (v3): raw completion (no chat template), second-person
qualitative persona + Yes/No willingness question. For each simulated
individual, draw (lambda_i, phi_i) as before (Sec. 3.3). Map to persona level
via ABSOLUTE pooled-distribution tertiles (phi: 1.394/1.490; lambda:
1.801/2.007 — computed from the reconstructed 27-country individual mixture,
seed 0), so country level-mixtures differ and between-country heterogeneity is
preserved. Gain tasks use the risk-aversion sentence for risk_level(phi_i);
loss tasks use the loss-aversion sentence for loss_level(lambda_i). Sentences
are fixed 3-level templates (see code/ground_truth.py RISK_SENTENCES /
LOSS_SENTENCES); no country name, ethnicity term, or trait inventory appears.
Personas are individual-level summaries of measured choice dispositions, vary
within country, and are a deterministic function of the drawn (lambda_i,
phi_i) — the distribution calibration of Sec. 3.3 is unchanged.*

*Task battery (v3): 6 Yes/No tasks (prices fixed across individuals):
G_25/G_45/G_65 "Would you pay $Y for a 50/50 chance to win $100?";
L_40/L_90/L_140 "Would you accept a 50/50 chance of losing $100 if paid $Y?".
Real-data counterpart: deterministic CPT Yes/No per individual
(cpt_yes: gain yes iff Y <= 100*0.5^phi_i; loss yes iff Y >= 100*(0.5*lambda_i)^phi_i).
EV-optimal: gain yes iff Y <= 50; loss yes iff Y >= 50.
Primary metric becomes per-country W1 between the simulated per-individual %yes
distribution (8 values in {0/6..6/6}) and the CPT-predicted distribution; mean
over 10 countries; paired permutation test, one-sided, alpha = 0.05 — otherwise
unchanged (Sec. 7). Hypothesis H1, decision rule, and secondaries (with "%risky"
read as "%yes") are unchanged.*

*Sample sizes (v3): baseline 8 x 6 = 48; main 10 x 8 x 2 x 6 = 960;
robustness 4 x (2 x 8 x 6) = 384; expected total 1392 JSONL records.
R4 lambda-only ablation now uses the loss-aversion sentence alone on all tasks.
All v1/v2 pilot records (smoke/baseline, A/B format) are discarded; results/
is rewritten fresh by the v3 runner (resume-skip keyed on the new schema).*

## 11. Data licensing

No restricted dataset is downloaded or redistributed. Ground truth = published
summary statistics (facts, not copyrightable expression) from open-access sources.
The reproduction package documents the transcription source; it ships NO
third-party data files beyond the transcribed CSV with full citation.

## 12. What is NOT done

No GPS form submission, no WVS registration, no INFORMS download attempts beyond
public pages, no arXiv/EasyChair/email submission, no live-browser steps.

*Post-data-collection correction, 2026-10-12 (AFTER main runs; documented here,
not a preregistered amendment). Independent review identified that the generic
arm as run reused the same 8 personas identically across all 10 countries
(draw_key = 5000 + ii, country=None), so generic per-country W1 estimates shared
one simulated distribution by construction. The generic arm was therefore
re-run with 8 fresh independent draws from the pooled 27-country mixture PER
COUNTRY (code/run_generic_v2.py; per-country draw keys 5000 + country_idx*100
+ ii and per-country generation seeds; all other v3 parameters identical).
This implements the preregistered Sec-5 specification ("drawn from the POOLED
27-country distribution") per simulated individual. All reported generic-arm
results use the corrected data; the original shared-persona records are
retained unmodified in results/main.jsonl as the audit trail. Hypotheses,
metrics, and the decision rule are unchanged.*
