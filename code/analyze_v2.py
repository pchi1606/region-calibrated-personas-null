"""s12 corrected analysis (elicitation v3, generic arm v2).

Implements the independent-review correction: the generic main arm is the
re-run with per-country INDEPENDENT draws (results/main_generic_v2.jsonl,
8 fresh pooled-mixture individuals per country), replacing the original
shared-persona generic records in results/main.jsonl. The region arm,
baseline arm, and robust arm are unchanged.

What differs from code/analyze.py:
  - Generic per-country W1 uses the STANDARD per-individual path keyed by
    (country, indiv) -- no pseudo-individual special-casing, because v2
    generic records carry their country and are independent per country.
  - Completeness/audit cover the CORRECTED dataset (baseline 48 + main
    480 region + 480 v2-generic + robust 384 = 1392). The original
    shared-persona generic records are excluded from all metrics (kept in
    results/main.jsonl as the audit trail).
  - Adds the reviewer-requested sensitivity: primary test re-run excluding
    the thinnest cells (Japan, Denmark, Pakistan), labeled exploratory.

Writes results/metrics_v2.json and results/tables_v2.txt.
All randomness after data load uses seed 20261008.
"""
import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import wasserstein_distance, spearmanr

from ground_truth import SELECTED, TASKS, RISK_SENTENCES, LOSS_SENTENCES
import analyze as A

WORK = Path(__file__).resolve().parent.parent
RES = WORK / "results"
SEED = 20261008
N_INDIV = 8

# Corrected dataset: baseline + main(region + v2 generic) + robust
EXPECTED_V2 = {"baseline": 48, "main_region": 480, "main_generic_v2": 480,
               "robust": 384}


def load_v2():
    """Rows for the corrected analysis.

    Region rows come from results/main.jsonl (arm == 'region').
    Generic rows: prefer per-country generic from main.jsonl (present after
    a --full rerun with the corrected run.py); otherwise fall back to the
    bundled results/main_generic_v2.jsonl (quick mode).
    """
    rows_main = [json.loads(l) for l in open(RES / "main.jsonl")]
    rows_region = [r for r in rows_main if r["arm"] == "region"]
    v2_in_main = [r for r in rows_main
                  if r["arm"] == "generic" and r["country"] is not None]
    if v2_in_main:
        rows_generic = v2_in_main
    else:
        rows_generic = [json.loads(l)
                        for l in open(RES / "main_generic_v2.jsonl")]
    rows_base = [json.loads(l) for l in open(RES / "baseline.jsonl")]
    rows_rob = [json.loads(l) for l in open(RES / "robust.jsonl")]
    return rows_region, rows_generic, rows_base, rows_rob


def completeness_v2():
    rep = {}
    rows_main = [json.loads(l) for l in open(RES / "main.jsonl")]
    n_region = sum(1 for r in rows_main if r["arm"] == "region")
    v2_in_main = sum(1 for r in rows_main
                     if r["arm"] == "generic" and r["country"] is not None)
    if v2_in_main:
        n_generic_v2 = v2_in_main
    else:
        n_generic_v2 = (sum(1 for _ in open(RES / "main_generic_v2.jsonl"))
                        if (RES / "main_generic_v2.jsonl").exists() else 0)
    counts = {
        "baseline": sum(1 for _ in open(RES / "baseline.jsonl")),
        "main_region": n_region,
        "main_generic_v2": n_generic_v2,
        "robust": sum(1 for _ in open(RES / "robust.jsonl")),
    }
    for phase, exp in EXPECTED_V2.items():
        n = counts[phase]
        rep[phase] = {"expected": exp, "found": n, "complete": n == exp}
    return rep


def sim_frac_yes_std(rows, arm, country=None, variant="standard"):
    """Per-individual fraction of Yes: {(country, indiv): frac}.

    Standard path (same as analyze.sim_frac_yes for non-generic arms).
    v2 generic records carry their country, so the country filter applies
    and each country's generic distribution is an independent estimate.
    """
    per = {}
    for r in rows:
        if r["arm"] != arm or r["variant"] != variant:
            continue
        if country is not None and r["country"] != country:
            continue
        if r["said_yes"] is None:
            continue
        per.setdefault((r["country"], r["indiv"]), []).append(
            1 if r["said_yes"] else 0)
    return {i: sum(v) / len(v) for i, v in per.items()
            if len(v) == len(TASKS)}


def no_label_audit_v2(rows_all):
    """Banned-term audit over the corrected dataset's prompts/generations."""
    hits = [t for t in (RISK_SENTENCES + LOSS_SENTENCES)
            if any(b in t.lower() for b in A.BANNED)]
    gen_hits = sum(1 for r in rows_all
                   if any(b in (r.get("raw") or "").lower()
                          for b in A.BANNED))
    prompt_hits = sum(1 for r in rows_all
                      if any(b in (r.get("prompt") or "").lower()
                             for b in A.BANNED))
    return {"template_hits": len(hits), "prompt_hits": prompt_hits,
            "gen_hits": gen_hits, "n_gen": len(rows_all)}


def parse_failure_analysis(rows_all):
    """Reviewer-requested: are parse failures non-random?

    Reports parse (failure) rates by arm, by task, by domain, and by
    persona level (risk/loss sentence index, extracted from the prompt).
    If failures correlate with extreme personas, the missingness is
    informative and cell thinning is not ignorable.
    """
    from ground_truth import RISK_SENTENCES as RS, LOSS_SENTENCES as LS
    by_arm, by_task, by_domain, by_persona = {}, {}, {}, {}
    for r in rows_all:
        ok = r["said_yes"] is not None
        for d, k in [(by_arm, r["arm"]),
                     (by_task, (r["arm"], r["task"])),
                     (by_domain, (r["arm"], r["domain"]))]:
            d.setdefault(k, [0, 0])
            d[k][0] += ok
            d[k][1] += 1
        # persona level from prompt text
        pr = (r.get("prompt") or "")
        lvl = None
        for i, s in enumerate(RS):
            if s[:40] in pr:
                lvl = ("risk", i)
        for i, s in enumerate(LS):
            if s[:40] in pr:
                lvl = ("loss", i)
        key = (r["arm"], lvl if lvl else "uncalibrated")
        by_persona.setdefault(key, [0, 0])
        by_persona[key][0] += ok
        by_persona[key][1] += 1

    def rates(d):
        return {str(k): {"parse_rate": v[0] / v[1], "n": v[1]}
                for k, v in sorted(d.items(), key=lambda x: str(x[0]))}
    return {"by_arm": rates(by_arm), "by_task": rates(by_task),
            "by_domain": rates(by_domain), "by_persona": rates(by_persona)}


def main():
    comp = completeness_v2()
    print("== v2 output completeness ==")
    all_ok = True
    for ph, c in comp.items():
        print(f"  {ph}: {c['found']}/{c['expected']} complete={c['complete']}")
        all_ok &= c["complete"]
    if not all_ok:
        print("INCOMPLETE V2 OUTPUT SET -- metrics flagged, not passed.")

    rows_region, rows_generic, rows_base, rows_rob = load_v2()
    rows_main_v2 = rows_region + rows_generic

    for name, rows in [("main_region", rows_region),
                       ("main_generic_v2", rows_generic),
                       ("baseline", rows_base)]:
        ok, n, rate = A.parse_rate(rows)
        print(f"parse rate {name}: {ok}/{n} = {rate:.3f}")

    audit = no_label_audit_v2(rows_main_v2 + rows_base + rows_rob)
    print(f"v2 no-label audit: template_hits={audit['template_hits']} "
          f"prompt_hits={audit['prompt_hits']} "
          f"gen_hits={audit['gen_hits']}/{audit['n_gen']}")

    pf = parse_failure_analysis(rows_main_v2 + rows_base + rows_rob)
    print("parse-failure analysis: by_arm =",
          {k: round(v["parse_rate"], 3) for k, v in pf["by_arm"].items()})

    metrics = {"completeness": comp, "audit": audit,
               "n_countries": len(SELECTED),
               "parse_failure_analysis": pf,
               "design_note": (
                   "generic arm v2: 8 fresh independent pooled-mixture "
                   "draws per country (reviewer correction); original "
                   "shared-persona generic records excluded from metrics")}
    if not all_ok:
        json.dump(metrics, open(RES / "metrics_v2.json", "w"), indent=2)
        return

    # ---- primary: W1(sim || CPT-real) per country, region vs generic(v2) ----
    w1_region, w1_generic, w1_uncal = {}, {}, {}
    sim_means, real_means = {}, {}
    sim_sds, real_sds = {}, {}
    unc = sim_frac_yes_std(rows_base, "uncalibrated")
    unc_vals = np.array(sorted(unc.values()))
    for c in SELECTED:
        real = A.cpt_frac_yes(c)
        real_vals = np.array(sorted(real.values()))
        real_means[c] = float(real_vals.mean())
        real_sds[c] = float(real_vals.std(ddof=1)) if len(real_vals) > 1 else 0.0
        for arm, rows, store in [("region", rows_region, w1_region),
                                 ("generic", rows_generic, w1_generic)]:
            sim = sim_frac_yes_std(rows, arm, c)
            sv = np.array(sorted(sim.values()))
            sim_means[(arm, c)] = float(sv.mean())
            sim_sds[(arm, c)] = float(sv.std(ddof=1)) if len(sv) > 1 else 0.0
            store[c] = float(wasserstein_distance(sv, real_vals))
        w1_uncal[c] = float(wasserstein_distance(unc_vals, real_vals))

    diffs = np.array([w1_generic[c] - w1_region[c] for c in SELECTED])
    obs, p = A.paired_permutation_p(diffs)
    metrics["primary"] = {
        "mean_W1_region": float(np.mean([w1_region[c] for c in SELECTED])),
        "mean_W1_generic": float(np.mean([w1_generic[c] for c in SELECTED])),
        "mean_W1_uncalibrated": float(np.mean([w1_uncal[c] for c in SELECTED])),
        "mean_diff_generic_minus_region": float(obs),
        "permutation_p_one_sided": float(p),
        "n_favor_region": int((diffs > 0).sum()),
        "decision": ("H1 SUPPORTED" if p < 0.05 and obs > 0
                     else "H1 REJECTED"),
        "per_country_W1": {c: {"region": w1_region[c],
                               "generic": w1_generic[c],
                               "uncalibrated": w1_uncal[c],
                               "n_region": len(sim_frac_yes_std(
                                   rows_region, "region", c)),
                               "n_generic": len(sim_frac_yes_std(
                                   rows_generic, "generic", c))}
                            for c in SELECTED},
    }

    # ---- reviewer-requested sensitivity: drop thinnest cells (exploratory) --
    thin = {"Japan", "Denmark", "Pakistan"}
    keep = [c for c in SELECTED if c not in thin]
    d_thin = np.array([w1_generic[c] - w1_region[c] for c in keep])
    obs_t, p_t = A.paired_permutation_p(d_thin)
    metrics["sensitivity_drop_thin_cells"] = {
        "dropped": sorted(thin), "kept": keep,
        "mean_diff": float(obs_t), "p_one_sided": float(p_t),
        "note": "exploratory; not preregistered",
    }

    # ---- secondary (a): dispersion ----
    def disp(arm):
        rows = rows_region if arm == "region" else rows_generic
        return float(np.mean([abs(math.log((sim_sds[(arm, c)] + 1e-9) /
                                            (real_sds[c] + 1e-9)))
                              for c in SELECTED]))

    def disp_countries(arm, cs):
        return float(np.mean([abs(math.log((sim_sds[(arm, c)] + 1e-9) /
                                            (real_sds[c] + 1e-9)))
                              for c in cs]))
    metrics["secondary_dispersion"] = {
        "region": disp("region"), "generic": disp("generic"),
        "ci_region": A.boot_ci(lambda cs: disp_countries("region", cs),
                               SELECTED),
        "ci_generic": A.boot_ci(lambda cs: disp_countries("generic", cs),
                                SELECTED),
    }

    # ---- secondary (b): between-country heterogeneity (Spearman of means) --
    # v2 generic now has genuine between-country variation (independent
    # per-country draws), so this comparison is fair.
    for arm in ["region", "generic"]:
        rho, pval = spearmanr([sim_means[(arm, c)] for c in SELECTED],
                              [real_means[c] for c in SELECTED])
        metrics[f"secondary_spearman_{arm}"] = {"rho": float(rho),
                                                "p": float(pval)}

    # ---- secondary (c): EV-alignment gap ----
    ev = A.ev_frac()
    sim_yes_all = {}
    for arm, rows in [("region", rows_region), ("generic", rows_generic)]:
        d = sim_frac_yes_std(rows, arm, None)
        sim_yes_all[arm] = (float(np.mean(list(d.values()))) if d
                            else float("nan"))
    real_yes_all = float(np.mean([A.cpt_frac_yes(c)[i]
                                  for c in SELECTED for i in range(N_INDIV)]))
    unc_yes = float(np.mean(list(unc.values()))) if unc else float("nan")
    metrics["secondary_ev_gap"] = {
        "ev_frac_yes": ev,
        "sim_mean_yes_region": sim_yes_all["region"],
        "sim_mean_yes_generic": sim_yes_all["generic"],
        "sim_mean_yes_uncalibrated": unc_yes,
        "real_mean_yes": real_yes_all,
    }

    # ---- robustness: region-variant vs v2 generic_main (Germany/Italy) ----
    rob = {}
    for v in ["paraphrase", "seed", "temp03", "lambda_only"]:
        w1r, w1g = {}, {}
        for c in ["Germany", "Italy"]:
            real_vals = np.array(sorted(A.cpt_frac_yes(c).values()))
            sv = np.array(sorted(sim_frac_yes_std(
                rows_rob, "region", c, variant=v).values()))
            gv = np.array(sorted(sim_frac_yes_std(
                rows_generic, "generic", c).values()))
            w1r[c] = float(wasserstein_distance(sv, real_vals))
            w1g[c] = float(wasserstein_distance(gv, real_vals))
        d = np.array([w1g[c] - w1r[c] for c in ["Germany", "Italy"]])
        rob[v] = {"mean_diff": float(d.mean()),
                  "per_country": {c: {"region_variant": w1r[c],
                                      "generic_main_v2": w1g[c]}
                                  for c in ["Germany", "Italy"]}}
    metrics["robustness"] = rob

    # ---- within-SD sensitivity: recompute real with x0.5/x1.5 ----
    sens = {}
    for scale in [0.5, 1.5]:
        wr, wg = [], []
        for c in SELECTED:
            real_vals = np.array(sorted(
                A.cpt_frac_yes(c, sd_scale=scale).values()))
            sv = np.array(sorted(sim_frac_yes_std(
                rows_region, "region", c).values()))
            gv = np.array(sorted(sim_frac_yes_std(
                rows_generic, "generic", c).values()))
            wr.append(wasserstein_distance(sv, real_vals))
            wg.append(wasserstein_distance(gv, real_vals))
        sens[str(scale)] = {"mean_W1_region": float(np.mean(wr)),
                            "mean_W1_generic": float(np.mean(wg)),
                            "mean_diff": float(np.mean(wg) - np.mean(wr))}
    metrics["sensitivity_sd"] = sens

    json.dump(metrics, open(RES / "metrics_v2.json", "w"), indent=2)
    with open(RES / "tables_v2.txt", "w") as f:
        f.write("s12 metrics v2 (elicitation v3, generic arm corrected)\n")
        f.write(json.dumps(metrics, indent=2))
    print("wrote results/metrics_v2.json")
    print(f"PRIMARY v2: diff={obs:.4f} p={p:.4f} -> "
          f"{metrics['primary']['decision']}")
    print(f"THIN-CELL SENSITIVITY: diff={obs_t:.4f} p={p_t:.4f} "
          f"(n={len(keep)} countries)")


if __name__ == "__main__":
    main()
