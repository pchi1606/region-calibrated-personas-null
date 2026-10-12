"""s12 generation runner (elicitation v3): baseline, main, robustness.

Writes JSONL incrementally. Uses raw completion (NOT the chat template: the
template produced degenerate label bias) with qualitative 3-level personas
and Yes/No willingness questions (validated monotone in persona intensity).

Usage: code/venv/bin/python code/run.py --phase smoke|baseline|main|robust
"""
import argparse
import json
import re
import time
from pathlib import Path

import numpy as np
from llama_cpp import Llama

from ground_truth import (SELECTED, TASKS, draw_individual, risk_level,
                          loss_level, RISK_SENTENCES, LOSS_SENTENCES)

WORK = Path(__file__).resolve().parent.parent
MODEL = WORK / "data" / "Qwen3-1.7B-Q4_K_M.gguf"
OUT = WORK / "results"

MASTER_SEED = 20261008
N_INDIV = 8

# Paraphrased persona sentences for R1 (same 3 levels, reworded).
RISK_SENTENCES_PARA = [
    "Uncertainty does not bother you. You generally favor a gamble instead of a guaranteed sum.",
    "You are fairly careful with money. Guaranteed sums appeal to you, though you do take gambles at times.",
    "You are extremely careful with money. You would much rather have a guaranteed amount than gamble.",
]
LOSS_SENTENCES_PARA = [
    "Risking a loss is fine with you. You take gambles that might lose money when the pay is reasonable.",
    "You do not like losing money. Taking a gamble that could lose requires solid compensation for you.",
    "You will not risk losing money. Only an enormous payment could make you accept a gamble with a possible loss.",
]


def build_prompt(arm, country, ii, ti, variant):
    """Return (prompt_text, domain). Raw-completion format, no chat template."""
    _, domain, price = TASKS[ti]
    risk_s = RISK_SENTENCES_PARA if variant == "paraphrase" else RISK_SENTENCES
    loss_s = LOSS_SENTENCES_PARA if variant == "paraphrase" else LOSS_SENTENCES
    if arm == "uncalibrated":
        persona = ""
    else:
        draw_rng = np.random.default_rng(draw_key(arm, country, ii))
        lam, phi, _ = draw_individual(
            draw_rng, country if arm == "region" else None)
        if variant == "lambda_only":
            # Ablation: loss-aversion persona only, applied to all tasks.
            persona = LOSS_SENTENCES[loss_level(lam)] + " "
        elif domain == "gain":
            persona = risk_s[risk_level(phi)] + " "
        else:
            persona = loss_s[loss_level(lam)] + " "
    if domain == "gain":
        q = (f"Would you pay ${price} for a 50/50 chance to win $100? "
             f"Answer:")
    else:
        q = (f"Would you accept a 50/50 chance of losing $100 if paid "
             f"${price}? Answer:")
    return persona + q, domain


def draw_key(arm, country, ii):
    if arm == "region":
        return 9000 + SELECTED.index(country) * 100 + ii
    # generic: pooled mixture, per-country independent draws (reviewer
    # correction; see code/run_generic_v2.py in the full repo). Each country
    # gets 8 fresh pooled-mixture individuals.
    return 5000 + SELECTED.index(country) * 100 + ii


def parse_yesno(text):
    """First Yes/No token in the output; None if absent."""
    m = re.search(r"\b(yes|no)\b", (text or "").lower())
    return m.group(1) if m else None


def build_jobs(phase, master_seed):
    jobs = []
    if phase == "smoke":
        combos = [("region", "Germany", 0), ("generic", "Germany", 0),
                  ("uncalibrated", None, 0), ("region", "Italy", 1)]
        for arm, country, ii in combos:
            jobs.append((arm, country, ii, 0, master_seed, "standard", 0.7))
        return jobs
    if phase == "baseline":
        for ii in range(N_INDIV):
            for ti in range(len(TASKS)):
                jobs.append(("uncalibrated", None, ii, ti, master_seed,
                             "standard", 0.7))
        return jobs
    if phase == "main":
        for country in SELECTED:
            for ii in range(N_INDIV):
                for ti in range(len(TASKS)):
                    jobs.append(("region", country, ii, ti, master_seed,
                                 "standard", 0.7))
                    # generic: per-country independent draws (reviewer
                    # correction); country is set, not None.
                    jobs.append(("generic", country, ii, ti, master_seed,
                                 "standard", 0.7))
        return jobs
    if phase == "robust":
        subset = ["Germany", "Italy"]
        variants = [("paraphrase", 0.7, master_seed), ("seed", 0.7, 777),
                    ("temp03", 0.3, master_seed), ("lambda_only", 0.7,
                                                   master_seed)]
        for vname, temp, mseed in variants:
            for country in subset:
                for ii in range(N_INDIV):
                    for ti in range(len(TASKS)):
                        jobs.append(("region", country, ii, ti, mseed, vname,
                                     temp))
        return jobs
    raise ValueError(f"unknown phase {phase}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", required=True,
                    choices=["smoke", "baseline", "main", "robust"])
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    out_path = OUT / f"{args.phase}.jsonl"

    t0 = time.time()
    llm = Llama(model_path=str(MODEL), n_ctx=1024, n_threads=2, verbose=False)
    print(f"model loaded in {time.time() - t0:.1f}s", flush=True)

    jobs = build_jobs(args.phase, MASTER_SEED)
    # Resume: skip jobs already present (deterministic job list).
    seen = set()
    if out_path.exists():
        for line in open(out_path):
            try:
                r = json.loads(line)
                seen.add((r["arm"], r["country"], r["indiv"], r["task"],
                          r["variant"], r["temp"], r["seed"]))
            except Exception:
                pass
    jobs = [j for j in jobs
            if (j[0], j[1], j[2], TASKS[j[3]][0], j[5], j[6], j[4]) not in seen]
    print(f"phase={args.phase} jobs={len(jobs)} (skipped {len(seen)} done)",
          flush=True)
    done = 0
    t_start = time.time()
    # NOTE: v3 changed the elicitation; old v1/v2 records are invalid.
    # The file is rewritten fresh (not appended) on the first v3 run.
    fresh = not seen
    with open(out_path, "w" if fresh else "a") as f:
        for arm, country, ii, ti, mseed, variant, temp in jobs:
            prompt, domain = build_prompt(arm, country, ii, ti, variant)
            g0 = time.time()
            out = llm.create_completion(
                prompt=prompt, temperature=temp, top_p=0.8, top_k=20,
                max_tokens=24, seed=(mseed + ii * 131 + ti * 17) % 2**31)
            text = out["choices"][0]["text"]
            yn = parse_yesno(text)
            rec = {"phase": args.phase, "arm": arm, "country": country,
                   "indiv": ii, "task": TASKS[ti][0], "domain": domain,
                   "variant": variant, "temp": temp, "seed": mseed,
                   "choice": yn,
                   "said_yes": yn == "yes" if yn else None,
                   "gen_s": round(time.time() - g0, 2),
                   "prompt": prompt, "raw": text}
            f.write(json.dumps(rec) + "\n")
            f.flush()
            done += 1
            if done % 50 == 0:
                el = time.time() - t_start
                print(f"  {done}/{len(jobs)}  {el / done:.2f}s/gen  "
                      f"elapsed {el / 60:.1f} min", flush=True)
    print(f"done: {done} records -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
