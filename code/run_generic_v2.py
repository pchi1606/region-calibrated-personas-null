"""Regenerate the GENERIC main arm with per-country INDEPENDENT draws (v2).

Reviewer correction (independent review, revise-major): the original
main-run generic arm reused 8 personas x 10 replicates identically across
all 10 countries (run.py draw_key = 5000 + ii, country=None in every
record). Consequences: (a) generic per-country W1 estimates share the same
simulated distribution, correlating the paired differences by construction
and weakening the permutation test's exchangeability; (b) the generic arm
had zero between-country variation by design, making secondary (b)
(heterogeneity Spearman) an unfair comparison; (c) the generic arm enjoyed
an asymmetric precision advantage (80 pseudo-individuals vs 8 region
individuals per country).

The preregistration (Sec 5) specifies the generic arm as
"(alpha_i, lambda_i, delta_i) drawn from the POOLED 27-country distribution"
per simulated individual. This script implements that reading: 8 FRESH
independent draws from the pooled mixture PER COUNTRY (80 individuals
total, 8 per country, country set on each record), so each country's
generic W1 is an independent estimate with the same n as the region arm.

Everything else is IDENTICAL to the original v3 main run: same TASKS, same
raw-completion elicitation, same qualitative 3-level personas, same model
(Qwen3-1.7B Q4_K_M), temp 0.7 / top_p 0.8 / top_k 20 / max_tokens 24 /
n_ctx 1024, same /no_think prefix. Only two things change, both documented:
  1. persona-draw key: 5000 + country_idx*100 + ii (was 5000 + ii);
  2. generation seed: (MASTER_SEED + country_idx*1000 + ii*131 + ti*17)
     (was MASTER_SEED + ii*131 + ti*17), so sampling RNG is also
     independent per country.

Output: results/main_generic_v2.jsonl (480 records).
The original results/main.jsonl is NOT modified (kept as the audit trail).
The no-label audit must be re-run over the v2 generations (see analyze_v2).
"""
import json
import sys
import time
from pathlib import Path

import run as runmod
from run import (MODEL, MASTER_SEED, N_INDIV, TASKS, parse_yesno)

WORK = Path(__file__).resolve().parent.parent
OUT = WORK / "results"
OUT_FILE = OUT / "main_generic_v2.jsonl"

SELECTED = runmod.SELECTED


def draw_key_v2(arm, country, ii):
    """Per-country keys for the generic v2 rerun."""
    assert arm == "generic" and country is not None
    return 5000 + SELECTED.index(country) * 100 + ii


def main():
    # Patch run.draw_key BEFORE build_prompt is called: build_prompt looks up
    # draw_key in run's module globals at call time.
    runmod.draw_key = draw_key_v2
    from llama_cpp import Llama

    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    llm = Llama(model_path=str(MODEL), n_ctx=1024, n_threads=2,
                verbose=False)
    print(f"model loaded in {time.time() - t0:.1f}s", flush=True)

    jobs = []
    for country in SELECTED:
        ci = SELECTED.index(country)
        for ii in range(N_INDIV):
            for ti in range(len(TASKS)):
                jobs.append((country, ii, ti, ci))

    seen = set()
    if OUT_FILE.exists():
        for line in open(OUT_FILE):
            try:
                r = json.loads(line)
                seen.add((r["country"], r["indiv"], r["task"]))
            except Exception:
                pass
    jobs = [j for j in jobs if (j[0], j[1], TASKS[j[2]][0]) not in seen]
    print(f"generic_v2 jobs={len(jobs)} (skipped {len(seen)} done)",
          flush=True)

    fresh = not seen
    done = 0
    t_start = time.time()
    with open(OUT_FILE, "w" if fresh else "a") as f:
        for country, ii, ti, ci in jobs:
            prompt, domain = runmod.build_prompt("generic", country, ii, ti,
                                                 "standard")
            g0 = time.time()
            seed = (MASTER_SEED + ci * 1000 + ii * 131 + ti * 17) % 2**31
            out = llm.create_completion(
                prompt=prompt, temperature=0.7, top_p=0.8, top_k=20,
                max_tokens=24, seed=seed)
            text = out["choices"][0]["text"]
            yn = parse_yesno(text)
            rec = {"phase": "main_generic_v2", "arm": "generic",
                   "country": country, "indiv": ii, "task": TASKS[ti][0],
                   "domain": domain, "variant": "standard", "temp": 0.7,
                   "seed": MASTER_SEED, "gen_seed": seed,
                   "draw_key": draw_key_v2("generic", country, ii),
                   "choice": yn,
                   "said_yes": yn == "yes" if yn else None,
                   "gen_s": round(time.time() - g0, 2),
                   "prompt": prompt, "raw": text}
            f.write(json.dumps(rec) + "\n")
            f.flush()
            done += 1
            if done % 50 == 0:
                el = time.time() - t_start
                print(f"  {done}/{done + len(seen)} gen "
                      f"({el / max(done, 1):.1f}s/gen)", flush=True)
    print(f"generic_v2 done: {done} new, file={OUT_FILE}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
