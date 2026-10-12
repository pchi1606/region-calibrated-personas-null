# Reproduction package — Region-calibrated personas do not significantly improve fit to CPT-imputed country-level choice distributions

Author: Prabhu Rangaraju (Independent Researcher).

This package reproduces the headline results of the paper
"Region-Calibrated Personas Do Not Significantly Improve Fit to
CPT-Imputed Country-Level Choice Distributions" (CogSci 2027) from the
bundled generation outputs and the published ground-truth parameters.
One command:

```bash
bash reproduce.sh
```

## What it does

1. Checks Python 3.10+ and installs the pinned requirements
   (`numpy`, `scipy`).
2. Runs the corrected analysis (`code/analyze_v2.py`) on the bundled model
   generations (`results/*.jsonl`: 48 baseline + 480 region + 480 corrected
   generic + 384 robustness generations with Qwen3-1.7B on CPU),
   recomputing the prospect-theory ground-truth predictions from the
   published per-country INTRA parameters.
3. Runs the verifier (`scripts/verify.py`), which asserts the headline
   numbers in `expected/metrics_v2.json` match the recomputed values
   exactly (Wasserstein-1 distances 0.304 region / 0.285 generic /
   0.294 uncalibrated, one-sided permutation p = 0.631, H1 rejected).

The default path needs no model download and finishes in under two minutes.

**Reviewer correction:** the generic arm was re-run with 8 fresh independent
draws from the pooled mixture per country (`code/run_generic_v2.py`),
replacing the original shared-persona implementation. The packaged
`code/run.py` incorporates this correction, so `--full` reproduces the
corrected data. See `PREREGISTRATION.md` for the dated correction note.

The default path needs no model download and finishes in under two minutes.

## Full generation rerun (optional)

To regenerate every model output from scratch (Qwen3-1.7B Q4_K_M on CPU,
about 11 hours on 2 CPUs):

```bash
bash reproduce.sh --full
```

This downloads the model weights (~1.1 GB, public GGUF), re-runs all 1,392
generations with the preregistered seeds, then runs the analysis and verifier.

## Ground truth provenance

`data/intra_country_parameters.csv` transcribes published per-country
cumulative-prospect-theory estimates from the INTRA survey program
(Rieger, Wang & Hens 2015, 2017), numerical values as tabulated open-access in
Xie, Hwang & Pantelous (2018), who report their estimates as closely matching
the INTRA-published values. See `data/PROVENANCE.md`. No restricted microdata
is included or required.

## Files

- `reproduce.sh` — one-command entry point.
- `code/ground_truth.py`, `code/run.py`, `code/analyze.py` — experiment code.
- `results/*.jsonl` — the 1,392 model generations (bundled).
- `data/intra_country_parameters.csv` — published per-country parameters.
- `data/PROVENANCE.md` — data provenance and licensing notes.
- `scripts/verify.py` — headline-number verifier.
- `scripts/fetch_model.sh` — model download (used only with `--full`).
- `expected/metrics.json` — expected headline metrics.
- `paper/paper.pdf` — the manuscript.
- `PREREGISTRATION.md` — the preregistration (frozen before any runs).
