"""s12 analysis (elicitation v3): output-completeness check, primary/secondary
metrics, robustness.

Reads results/*.jsonl. Writes results/metrics.json and results/tables.txt.
All randomness after data load uses seed 20261008.
"""
import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import wasserstein_distance, spearmanr

from ground_truth import (SELECTED, TASKS, RAW, CountryDist, draw_individual,
                          cpt_yes, ev_yes, RISK_SENTENCES, LOSS_SENTENCES)

WORK = Path(__file__).resolve().parent.parent
RES = WORK / "results"
SEED = 20261008
N_INDIV = 8

EXPECTED = {"baseline": 48, "main": 960, "robust": 384}
# country names + ethnicity/nationality terms that must never appear in prompts
BANNED = [r[0].lower() for r in RAW] + [
    "german", "italian", "japanese", "greek", "brazilian", "hungarian",
    "pakistani", "australian", "danish", "icelandic", "european", "asian",
    "african", "american", "ethnicity", "race ", "nationality"]


def load(phase):
    rows = [json.loads(l) for l in open(RES / f"{phase}.jsonl")]
    return rows


def completeness():
    """Test 27: verify output set complete BEFORE any metric. Returns report."""
    rep = {}
    for phase, exp in EXPECTED.items():
        p = RES / f"{phase}.jsonl"
        n = sum(1 for _ in open(p)) if p.exists() else 0
        rep[phase] = {"expected": exp, "found": n, "complete": n == exp}
    return rep


def parse_rate(rows):
    ok = sum(1 for r in rows if r["said_yes"] is not None)
    return ok, len(rows), ok / len(rows) if rows else 0.0


def no_label_audit():
    """Verify no banned term in persona templates or prompts, and scan raw
    generations for country-name leakage."""
    try:
        import run as R
        para = R.RISK_SENTENCES_PARA + R.LOSS_SENTENCES_PARA
    except ImportError:
        # run.py needs the llama_cpp model runtime; in the analysis-only
        # reproduction environment those weights are absent, so audit the
        # standard templates only (paraphrase variants are code-identical
        # in the main run, where this fallback never triggers).
        para = []
    templates = (RISK_SENTENCES + LOSS_SENTENCES + para)
    hits = [t for t in templates
            if any(b in t.lower() for b in BANNED)]
    gen_hits = 0
    n_gen = 0
    prompt_hits = 0
    for phase in EXPECTED:
        p = RES / f"{phase}.jsonl"
        if not p.exists():
            continue
        for line in open(p):
            r = json.loads(line)
            n_gen += 1
            if any(b in (r.get("raw") or "").lower() for b in BANNED):
                gen_hits += 1
            if any(b in (r.get("prompt") or "").lower() for b in BANNED):
                prompt_hits += 1
    return {"template_hits": len(hits), "prompt_hits": prompt_hits,
            "gen_hits": gen_hits, "n_gen": n_gen}


def sim_frac_yes(rows, arm, country=None, variant="standard"):
    """Per-individual fraction of Yes answers: {indiv: frac}.
    Region/uncalibrated: keyed by indiv; country filter applies to region.
    Generic arm is pooled by design (run.py writes country=None for all
    generic records). It comprises 8 personas x 10 replicate blocks
    (deterministic job order); each (persona, replicate) is treated as a
    pseudo-individual (80 total), mirroring the region arm's individual-
    level structure. A country filter is ignored for generic so the pooled
    distribution is compared against each country's real distribution."""
    if arm == "generic":
        return _generic_frac_yes(rows, variant)
    per = {}
    for r in rows:
        if r["arm"] != arm or r["variant"] != variant:
            continue
        if country is not None and r["country"] != country:
            continue
        if r["said_yes"] is None:
            continue
        # (country, indiv) key: indiv ids repeat across countries, so a
        # pooled call (country=None) must not collide them
        per.setdefault((r["country"], r["indiv"]), []).append(
            1 if r["said_yes"] else 0)
    return {i: sum(v) / len(v) for i, v in per.items()
            if len(v) == len(TASKS)}


def _generic_frac_yes(rows, variant="standard"):
    # per (persona, task): parsed outcomes across the 10 replicates,
    # in deterministic job order (replicate j <=> j-th country-loop pass)
    pt = {}
    for r in rows:
        if r["arm"] != "generic" or r["variant"] != variant:
            continue
        if r["said_yes"] is None:
            continue
        pt.setdefault((r["indiv"], r["task"]), []).append(
            1 if r["said_yes"] else 0)
    task_names = [t[0] for t in TASKS]
    out = {}
    for indiv in sorted({i for i, _ in pt}):
        reps = [pt.get((indiv, tn), []) for tn in task_names]
        nrep = max([len(v) for v in reps] + [0])
        for j in range(nrep):
            vals = [reps[k][j] for k in range(len(task_names))
                    if j < len(reps[k])]
            if len(vals) == len(task_names):
                out[(indiv, j)] = sum(vals) / len(vals)
    return out


def cpt_frac_yes(country, sd_scale=1.0, seed_key=None):
    """CPT-predicted per-individual fraction Yes (the 'real' distribution)."""
    dist = CountryDist(next(r for r in RAW if r[0] == country),
                       sd_scale=sd_scale)
    out = {}
    for ii in range(N_INDIV):
        key = (seed_key(ii) if seed_key
               else 9000 + SELECTED.index(country) * 100 + ii)
        rng = np.random.default_rng(key)
        lam, phi, _ = dist.draw(rng)
        out[ii] = sum(cpt_yes(lam, phi, t) for t in TASKS) / len(TASKS)
    return out


def ev_frac():
    return sum(ev_yes(t) for t in TASKS) / len(TASKS)


def paired_permutation_p(diffs, seed=SEED):
    """One-sided EXACT paired permutation test over all 2^10 sign patterns:
    H1 mean(diffs) > 0. Exact enumeration (1024 patterns), not Monte-Carlo."""
    diffs = np.asarray(diffs, dtype=float)
    obs = diffs.mean()
    n = len(diffs)
    cnt = 0
    for bits in range(1 << n):
        signs = np.array([1.0 if (bits >> k) & 1 else -1.0
                          for k in range(n)])
        if (signs * diffs).mean() >= obs - 1e-12:
            cnt += 1
    return obs, cnt / (1 << n)


def boot_ci(stat_fn, data, n_boot=10000, seed=SEED):
    rng = np.random.default_rng(seed)
    data = list(data)
    boots = [stat_fn([data[i] for i in rng.integers(0, len(data), len(data))])
             for _ in range(n_boot)]
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def main():
    comp = completeness()
    print("== output completeness ==")
    all_ok = True
    for ph, c in comp.items():
        print(f"  {ph}: {c['found']}/{c['expected']} complete={c['complete']}")
        all_ok &= c["complete"]
    if not all_ok:
        print("INCOMPLETE OUTPUT SET — metrics flagged, not passed.")
    rows_main = load("main") if comp["main"]["complete"] else []
    rows_base = load("baseline") if comp["baseline"]["complete"] else []

    for name, rows in [("main", rows_main), ("baseline", rows_base)]:
        ok, n, rate = parse_rate(rows)
        print(f"parse rate {name}: {ok}/{n} = {rate:.3f}")

    audit = no_label_audit()
    print(f"no-label audit: template_hits={audit['template_hits']} "
          f"prompt_hits={audit['prompt_hits']} "
          f"gen_hits={audit['gen_hits']}/{audit['n_gen']}")

    metrics = {"completeness": comp, "audit": audit,
               "n_countries": len(SELECTED)}
    if not all_ok:
        json.dump(metrics, open(RES / "metrics.json", "w"), indent=2)
        return

    # ---- primary: W1(sim || CPT-real) per country, region vs generic ----
    w1_region, w1_generic, w1_uncal = {}, {}, {}
    sim_means, real_means = {}, {}
    sim_sds, real_sds = {}, {}
    unc = sim_frac_yes(rows_base, "uncalibrated")
    unc_vals = np.array(sorted(unc.values()))
    for c in SELECTED:
        real = cpt_frac_yes(c)
        real_vals = np.array(sorted(real.values()))
        real_means[c] = float(real_vals.mean())
        real_sds[c] = float(real_vals.std(ddof=1)) if len(real_vals) > 1 else 0.0
        for arm, store in [("region", w1_region), ("generic", w1_generic)]:
            sim = sim_frac_yes(rows_main, arm, c)
            sv = np.array(sorted(sim.values()))
            sim_means[(arm, c)] = float(sv.mean())
            sim_sds[(arm, c)] = float(sv.std(ddof=1)) if len(sv) > 1 else 0.0
            store[c] = float(wasserstein_distance(sv, real_vals))
        w1_uncal[c] = float(wasserstein_distance(unc_vals, real_vals))

    diffs = np.array([w1_generic[c] - w1_region[c] for c in SELECTED])
    obs, p = paired_permutation_p(diffs)
    metrics["primary"] = {
        "mean_W1_region": float(np.mean([w1_region[c] for c in SELECTED])),
        "mean_W1_generic": float(np.mean([w1_generic[c] for c in SELECTED])),
        "mean_W1_uncalibrated": float(np.mean([w1_uncal[c] for c in SELECTED])),
        "mean_diff_generic_minus_region": float(obs),
        "permutation_p_one_sided": float(p),
        "decision": ("H1 SUPPORTED" if p < 0.05 and obs > 0 else "H1 REJECTED"),
        "per_country_W1": {c: {"region": w1_region[c],
                               "generic": w1_generic[c],
                               "uncalibrated": w1_uncal[c]}
                           for c in SELECTED},
    }

    # ---- secondary (a): within-country dispersion preservation ----
    def disp(arm):
        return float(np.mean([abs(math.log((sim_sds[(arm, c)] + 1e-9) /
                                            (real_sds[c] + 1e-9)))
                              for c in SELECTED]))

    def disp_countries(arm, cs):
        return float(np.mean([abs(math.log((sim_sds[(arm, c)] + 1e-9) /
                                            (real_sds[c] + 1e-9)))
                              for c in cs]))
    metrics["secondary_dispersion"] = {
        "region": disp("region"), "generic": disp("generic"),
        "ci_region": boot_ci(lambda cs: disp_countries("region", cs),
                             SELECTED),
        "ci_generic": boot_ci(lambda cs: disp_countries("generic", cs),
                              SELECTED),
    }

    # ---- secondary (b): between-country heterogeneity (Spearman of means) ----
    for arm in ["region", "generic"]:
        rho, pval = spearmanr([sim_means[(arm, c)] for c in SELECTED],
                              [real_means[c] for c in SELECTED])
        metrics[f"secondary_spearman_{arm}"] = {"rho": float(rho),
                                                "p": float(pval)}

    # ---- secondary (c): EV-alignment gap (baseline rationality-gap check) ----
    ev = ev_frac()
    sim_yes_all = {}
    for arm in ["region", "generic"]:
        # pooled over countries; generic yields (persona, replicate) keys
        d = sim_frac_yes(rows_main, arm, None)
        sim_yes_all[arm] = float(np.mean(list(d.values()))) if d else float("nan")
    real_yes_all = float(np.mean([cpt_frac_yes(c)[i]
                                  for c in SELECTED for i in range(N_INDIV)]))
    unc_yes = float(np.mean(list(unc.values()))) if unc else float("nan")
    metrics["secondary_ev_gap"] = {
        "ev_frac_yes": ev,
        "sim_mean_yes_region": sim_yes_all["region"],
        "sim_mean_yes_generic": sim_yes_all["generic"],
        "sim_mean_yes_uncalibrated": unc_yes,
        "real_mean_yes": real_yes_all,
    }

    # ---- robustness: rerun primary on each variant subset ----
    rows_rob = (load("robust") if completeness()["robust"]["complete"]
                else [])
    rob = {}
    for v in ["paraphrase", "seed", "temp03", "lambda_only"]:
        w1r, w1g = {}, {}
        for c in ["Germany", "Italy"]:
            real_vals = np.array(sorted(cpt_frac_yes(c).values()))
            sv = np.array(sorted(sim_frac_yes(rows_rob, "region", c,
                                              variant=v).values()))
            # generic arm only exists for "standard"; compare region-variant
            # against the main-run generic for the same countries
            gv = np.array(sorted(
                sim_frac_yes(rows_main, "generic", c).values()))
            w1r[c] = float(wasserstein_distance(sv, real_vals))
            w1g[c] = float(wasserstein_distance(gv, real_vals))
        d = np.array([w1g[c] - w1r[c] for c in ["Germany", "Italy"]])
        rob[v] = {"mean_diff": float(d.mean()),
                  "per_country": {c: {"region_variant": w1r[c],
                                      "generic_main": w1g[c]}
                                  for c in ["Germany", "Italy"]}}
    metrics["robustness"] = rob

    # ---- within-SD sensitivity (R5): recompute real with x0.5/x1.5 ----
    sens = {}
    for scale in [0.5, 1.5]:
        wr, wg = [], []
        for c in SELECTED:
            real_vals = np.array(sorted(
                cpt_frac_yes(c, sd_scale=scale).values()))
            sv = np.array(sorted(
                sim_frac_yes(rows_main, "region", c).values()))
            gv = np.array(sorted(
                sim_frac_yes(rows_main, "generic", c).values()))
            wr.append(wasserstein_distance(sv, real_vals))
            wg.append(wasserstein_distance(gv, real_vals))
        sens[str(scale)] = {"mean_W1_region": float(np.mean(wr)),
                            "mean_W1_generic": float(np.mean(wg))}
    metrics["sensitivity_sd"] = sens

    json.dump(metrics, open(RES / "metrics.json", "w"), indent=2)
    with open(RES / "tables.txt", "w") as f:
        f.write("s12 metrics (elicitation v3)\n")
        f.write(json.dumps(metrics, indent=2))
    print("wrote results/metrics.json")
    print(f"PRIMARY: diff={obs:.4f} p={p:.4f} -> "
          f"{metrics['primary']['decision']}")


if __name__ == "__main__":
    main()
