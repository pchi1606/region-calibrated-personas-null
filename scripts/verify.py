"""Headline-number verifier for the s12 reproduction package.

Compares recomputed metrics against the expected values with exact equality
on the rounded headline figures and tolerance checks on the rest.
Exit 0 iff every check passes.
"""
import argparse
import json
import math
import sys

TOL = 1e-9


def check(name, got, exp, tol=TOL):
    # NaN-to-NaN counts as a match (degenerate secondary metrics).
    if isinstance(got, float) and isinstance(exp, float):
        if math.isnan(got) and math.isnan(exp):
            print(f"[PASS] {name}: both NaN")
            return True
    ok = abs(got - exp) <= tol
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: got={got!r} expected={exp!r}")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--computed", required=True)
    ap.add_argument("--expected", required=True)
    a = ap.parse_args()
    comp = json.load(open(a.computed))
    exp = json.load(open(a.expected))
    ok = True

    # Output completeness gates everything.
    for ph, c in exp["completeness"].items():
        ok &= check(f"completeness[{ph}]",
                    comp["completeness"][ph]["found"], c["found"])

    p = exp["primary"]
    q = comp["primary"]
    ok &= check("primary.mean_W1_region", q["mean_W1_region"], p["mean_W1_region"])
    ok &= check("primary.mean_W1_generic", q["mean_W1_generic"], p["mean_W1_generic"])
    ok &= check("primary.mean_W1_uncalibrated", q["mean_W1_uncalibrated"],
                p["mean_W1_uncalibrated"])
    ok &= check("primary.permutation_p", q["permutation_p_one_sided"],
                p["permutation_p_one_sided"])
    ok &= (q["decision"] == p["decision"])
    print(f"[{'PASS' if q['decision'] == p['decision'] else 'FAIL'}] "
          f"primary.decision: {q['decision']}")

    for key in ["secondary_dispersion", "secondary_spearman_region",
                "secondary_spearman_generic", "secondary_ev_gap"]:
        ce, cc = exp[key], comp[key]
        if isinstance(ce, dict):
            for k, v in ce.items():
                if isinstance(v, (int, float)):
                    ok &= check(f"{key}.{k}", cc[k], v)
        else:
            ok &= check(key, cc, ce)

    # Robustness diffs must match sign and magnitude.
    for k, v in exp["robustness"].items():
        ok &= check(f"robustness.{k}.mean_diff",
                    comp["robustness"][k]["mean_diff"], v["mean_diff"])

    print("ALL CHECKS PASSED" if ok else "VERIFICATION FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
