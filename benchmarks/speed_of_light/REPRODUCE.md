# Reproduce the Speed of Light comparison

The comparison uses baseline source `4489e28e5f2ee1dddc34415afc4de23bfd31796d` and patched source `171777d1838e7c2a16bca89c6cec6081f3bf23bf`. Both are measured with the same revised harness. The original exploratory study is preserved at commit `8eb9eda` and in the existing top-level `results/` captures; fresh runs live under `results/baseline/` and `results/optimized/`.

## Prepare isolated snapshots

Use worktrees under `/Users/cjerzak/Documents`. The recorded implementation worktree is `/Users/cjerzak/Documents/fastrerandomize-diagonal-prep-20261022`; the original analysis worktree contains the baseline package. The recorded remote study root is `/home/cjerzak/Documents/fastrerandomize-diagonal-prep-20261022`. Use a fresh remote directory for another study so the original captures are preserved.

The remote host already has R, reticulate, testthat, jsonlite, CUDA 13.4, Nsight Systems 2026.3.2, Nsight Compute 2026.3.1, and the `preference_fm_cuda_20261009` Python environment. No Python packages, system R installation, GPU clocks, power policy, or driver permissions were changed.

From the implementation worktree, create the remote layout and copy the two source snapshots:

```bash
ssh cjerzak-rog 'mkdir -p /home/cjerzak/Documents/fastrerandomize-diagonal-prep-20261022/{baseline,optimized}/{source,rlib,results,benchmarks}'
rsync -az /Users/cjerzak/Documents/fastrerandomize-sol-20261022/fastrerandomize/ cjerzak-rog:/home/cjerzak/Documents/fastrerandomize-diagonal-prep-20261022/baseline/source/fastrerandomize/
rsync -az fastrerandomize/ cjerzak-rog:/home/cjerzak/Documents/fastrerandomize-diagonal-prep-20261022/optimized/source/fastrerandomize/
for revision in baseline optimized; do
  rsync -az --exclude=results --exclude='*.png' --exclude='*.pdf' --exclude='*.svg' benchmarks/ "cjerzak-rog:/home/cjerzak/Documents/fastrerandomize-diagonal-prep-20261022/$revision/benchmarks/"
done
```

On the remote machine, install into separate private libraries:

```bash
cd /home/cjerzak/Documents/fastrerandomize-diagonal-prep-20261022
R CMD INSTALL --library=baseline/rlib baseline/source/fastrerandomize
R CMD INSTALL --library=optimized/rlib optimized/source/fastrerandomize
```

`run.sh` selects the existing CUDA interpreter, disables JAX preallocation, and sets OMP/OpenBLAS threads to one. `SOL_CONDA_ENV`, `RETICULATE_CONDA`, and `RETICULATE_PYTHON` can override its defaults. In the recorded runs there were no custom XLA flags; R used the system reference BLAS. Each revision runs in a separate R process and uses its own installed package.

## Validate before measuring

The tests request an environment named `fastrerandomize_env`. Expose the existing interpreter through a private directory, without modifying the existing environment. Copy its conda history as well: an empty history is insufficient for this reticulate version.

```bash
cd /home/cjerzak/Documents/fastrerandomize-diagonal-prep-20261022
mkdir -p envs/fastrerandomize_env/{bin,conda-meta}
ln -s /home/cjerzak/miniconda3/envs/preference_fm_cuda_20261009/bin/python envs/fastrerandomize_env/bin/python
cp /home/cjerzak/miniconda3/envs/preference_fm_cuda_20261009/conda-meta/history envs/fastrerandomize_env/conda-meta/history
export CONDA_ENVS_PATH="$PWD/envs"
export RETICULATE_CONDA=/home/cjerzak/miniconda3/bin/conda
export RETICULATE_PYTHON="$CONDA_ENVS_PATH/fastrerandomize_env/bin/python"
export XLA_PYTHON_CLIENT_PREALLOCATE=false OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
cd optimized
JAX_PLATFORMS=cuda Rscript benchmarks/speed_of_light/validate.R > results/tests_cuda.log 2>&1
JAX_PLATFORMS=cpu SOL_TEST_FILTER='kernels|pure-r' Rscript benchmarks/speed_of_light/validate.R > results/tests_cpu.log 2>&1
```

`validate.R` requires a working backend before running tests, records the device, and exports assertion totals. The separate `diagnose-custom-tail.R` reproduces the legacy custom-callback limitation described in the report; its expected baseline failure is not part of the passing regression suite.

Local package checks used:

```bash
R CMD build --no-build-vignettes fastrerandomize
R CMD check --no-manual --ignore-vignettes fastrerandomize_0.4.tar.gz
```

Vignettes and the PDF manual were excluded. Local JAX tests skipped because the default conda environment was absent; CUDA and CPU coverage above ran explicitly without skips.

## Run measurements sequentially

Use a fresh shell after validation so the test environment overrides do not leak into benchmarking. Do not overlap GPU jobs. From the remote study root:

```bash
cd /home/cjerzak/Documents/fastrerandomize-diagonal-prep-20261022
bash optimized/benchmarks/speed_of_light/run-comparison.sh > comparison.log 2>&1
```

The runner builds calibration tools, then measures baseline followed by optimized. For each revision it runs `api`, `hostprep`, `focus`, `pipeline`, and `sweep`, then captures three full-pipeline and three single-batch calls with Nsight Systems. GPU timings wait on `jax.block_until_ready`. The primary workload explicitly sets `SOL_P=2048`; other primary defaults are 4,096 units, batch 4,096, and 1,048,576 draws. Data seed is 20261022, assignment seed 77, and acceptance probability 0.001. The automatic package batch size would be 1,024.

Sampling:

- Complete API: one first call after backend/input setup, then seven warmed calls returning R objects. Validation and saving comparison outputs are outside the timers.
- Production preparation: one Rprof capture and five unprofiled calls. `production_design()` requests the same internal shortcut as the default package caller, while retaining compatibility with the baseline helper.
- Resident balance and generated batch: 100 synchronized calls after 300 and 150 warmups, respectively.
- Complete fused GPU pipeline: 15 synchronized calls after three warmups.
- Nearby shapes: the same seven cases, 30 synchronized calls after five warmups.
- Nsight Systems: three synchronized calls after five warmups for each capture target. Profiling timings remain separate from headline unprofiled timings.

CUDA-event calibration uses 300 warmup launches and 31 samples, each averaging five kernel launches. Its DRAM triad, FP32 FMA, and strict-FP32 square SGEMMs provide empirical references, not universal throughput ceilings.

The runner attempts Nsight Compute `SpeedOfLight` on the production batch using existing permissions and `--clock-control none`. If it fails with `ERR_NVGPUCTRPERM`, keep that log and retain the analytical roofline label. Do not substitute instrumented timings for the unprofiled results.

## Collect and verify evidence

From the local implementation worktree:

```bash
for revision in baseline optimized; do
  rsync -az "cjerzak-rog:/home/cjerzak/Documents/fastrerandomize-diagonal-prep-20261022/$revision/results/" "benchmarks/speed_of_light/results/$revision/"
done
```

`compare.py` rebuilds each summary, verifies decoded R assignments, keys, and balance scores with `identical()`, and writes `results/comparison.json`. It requires R/jsonlite and the exported Nsight SQLite files. The figure requires NumPy and Matplotlib. The original results remain untouched.

Record and verify source and measurement-harness hashes before comparing; the
comparison also uses these manifests to normalize source paths in compiled HLO:

```bash
python3 benchmarks/speed_of_light/manifest.py baseline --source-root /Users/cjerzak/Documents/fastrerandomize-sol-20261022 --commit 4489e28e5f2ee1dddc34415afc4de23bfd31796d --remote-root /home/cjerzak/Documents/fastrerandomize-diagonal-prep-20261022/baseline
python3 benchmarks/speed_of_light/manifest.py optimized --source-root /Users/cjerzak/Documents/fastrerandomize-diagonal-prep-20261022 --commit 171777d1838e7c2a16bca89c6cec6081f3bf23bf --remote-root /home/cjerzak/Documents/fastrerandomize-diagonal-prep-20261022/optimized
```

Then rebuild the comparison and figure:

```bash
python3 benchmarks/speed_of_light/compare.py
/Users/cjerzak/miniconda3/bin/python3 benchmarks/speed_of_light/plot.py
```

Raw `.nsys-rep`, `.sqlite`, `.hlo.txt`, `.Rprof`, and saved `.rds` outputs remain in the worktree and on the remote host, excluded from Git. Git contains source manifests, individual timing samples, correctness results, telemetry, readable profiler summaries, the comparison, and the report figure.
