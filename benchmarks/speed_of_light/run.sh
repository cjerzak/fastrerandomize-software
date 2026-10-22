#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
export SOL_ROOT="$PWD"
export SOL_CONDA_ENV="${SOL_CONDA_ENV:-preference_fm_cuda_20261009}"
export RETICULATE_CONDA="${RETICULATE_CONDA:-/home/cjerzak/miniconda3/bin/conda}"
export RETICULATE_PYTHON="${RETICULATE_PYTHON:-/home/cjerzak/miniconda3/envs/$SOL_CONDA_ENV/bin/python}"
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
exec Rscript benchmarks/speed_of_light/benchmark.R "$@"
