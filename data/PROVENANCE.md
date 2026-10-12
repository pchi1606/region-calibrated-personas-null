# Data provenance and licensing

## `intra_country_parameters.csv`

Transcribed 2026-10-08 by the author from the open-access table in:

Xie, Y., Hwang, S., & Pantelous, A. A. (2018). Loss aversion around the
world: Empirical evidence from pension funds. *Journal of Banking & Finance,
88*, 52-62. https://doi.org/10.1016/j.jbankfin.2017.11.007
(Numerical values transcribed from the open-access table hosted at
https://apjfs.org/file/download/6487?view=1)

The table reports per-country estimates of loss aversion (lambda), risk
aversion / curvature (phi), and subjective probability weighting (delta) with
standard errors, and the paper states these estimates closely match the
INTRA-published values of:

- Rieger, M. O., Wang, M., & Hens, T. (2015). Risk preferences around the
  world. *Management Science, 61*(3), 637-648.
- Rieger, M. O., Wang, M., & Hens, T. (2017). Estimating cumulative prospect
  theory parameters from an international survey. *Theory and Decision, 82*(4),
  567-596.
- Wang, M., Rieger, M. O., & Hens, T. (2017). The impact of culture on loss
  aversion. *Journal of Behavioral Decision Making, 30*(2), 270-281.

Only published summary statistics (numerical facts) are transcribed; no
copyrighted expression is reproduced. No restricted microdata (GPS, WVS, or
INTRA individual-level files) is included, downloaded, or required.

## `results/*.jsonl`

Model outputs generated for this paper with Qwen3-1.7B (Q4_K_M GGUF) via
llama.cpp on CPU, using the preregistered seeds and prompts in `code/run.py`.
Each record carries the arm, country grouping key, individual index, task,
generation parameters, parsed choice, and raw text.

## License of this package's original content

Code and documentation in this package are the author's original work.
The transcribed parameter values remain the intellectual property of their
original publishers and are used here as facts under fair use for research
reproduction, with full citation (see CITATION.md).
