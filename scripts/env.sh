#!/usr/bin/env bash
# Source from the project root. Never changes system Python or shell startup files.
export PYTHONPATH="$(pwd)/src${PYTHONPATH:+:$PYTHONPATH}"
export UV_CACHE_DIR="$(pwd)/.cache/uv"
export HF_HOME="$(pwd)/.cache/huggingface"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export HF_HUB_DISABLE_TELEMETRY=1
export PIP_CACHE_DIR="$(pwd)/.cache/pip"
export TMPDIR="$(pwd)/.tmp"
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false
export MPLCONFIGDIR="$(pwd)/.cache/matplotlib"
export XDG_CACHE_HOME="$(pwd)/.cache"
export UV_PYTHON_INSTALL_DIR="$(pwd)/.tools/python"
export TORCHINDUCTOR_COMPILE_THREADS=4
