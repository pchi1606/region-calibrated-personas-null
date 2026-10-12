#!/usr/bin/env bash
# One-command reproduction for the s12 paper.
# Default: run the analysis on the bundled generations, verify headline numbers.
# No model download needed (under 2 minutes).
#   bash reproduce.sh --full
# additionally re-runs all 1,392 generations on CPU (~11 h on 2 CPUs).
set -euo pipefail
cd "$(dirname "$0")"

MODE="${1:-quick}"
PY=python3

echo "==> checking Python"
$PY -c "import sys; assert sys.version_info >= (3,10)" || {
  echo "Python 3.10+ required"; exit 1; }

echo "==> setting up an isolated virtual environment"
if [ ! -x ".venv/bin/python" ]; then
  $PY -m venv .venv
fi
VENV_PY=.venv/bin/python

echo "==> installing pinned requirements"
$VENV_PY -m pip install --quiet -r scripts/requirements.txt
PY=$VENV_PY

if [ "$MODE" = "--full" ]; then
  echo "==> fetching model weights (~1.1 GB public GGUF)"
  bash scripts/fetch_model.sh data
  echo "==> running all generations (preregistered seeds + reviewer correction)"
  $PY code/run.py --phase baseline
  $PY code/run.py --phase main
  $PY code/run.py --phase robust
fi

echo "==> running corrected analysis (generic arm v2: per-country draws)"
$PY code/analyze_v2.py

echo "==> verifying headline numbers"
$PY scripts/verify.py --computed results/metrics_v2.json \
  --expected expected/metrics_v2.json

echo "REPRODUCTION OK"
