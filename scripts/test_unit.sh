#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
python_bin="${PYTHON_BIN:-.venv/bin/python}"
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export MKL_NUM_THREADS=2
export MPLCONFIGDIR="$PWD/.cache/matplotlib"
"$python_bin" -m pytest -q \
  tests/test_data.py tests/test_evaluate.py tests/test_compare.py \
  tests/test_views.py tests/test_smiles_invariance.py tests/test_source_audit.py \
  tests/test_source_quarantine.py tests/test_error_analysis.py tests/test_consistency.py \
  tests/test_query_label_isolation.py tests/test_release_wrappers.py "$@"
