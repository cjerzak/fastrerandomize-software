# Reproduce the Speed of Light analysis

The recorded measurements use production source `171777d1838e7c2a16bca89c6cec6081f3bf23bf`, collected October 22, 2026. Results are stored in `results/`. The [source manifest](results/manifest.json) preserves the measured source and harness hashes and the remote capture location. The commands below measure a single checkout of the current package.

## Prepare an isolated snapshot

Use a worktree under `/Users/cjerzak/Documents` and a fresh remote directory for each study. From the repository root:

```bash
git worktree add --detach /Users/cjerzak/Documents/fastrerandomize-sol-current main
cd /Users/cjerzak/Documents/fastrerandomize-sol-current
ssh cjerzak-rog 'mkdir -p /home/cjerzak/Documents/fastrerandomize-sol-current/{source,rlib,results,benchmarks}'
rsync -az fastrerandomize/ cjerzak-rog:/home/cjerzak/Documents/fastrerandomize-sol-current/source/fastrerandomize/
rsync -az --exclude=results --exclude='*.png' --exclude='*.pdf' --exclude='*.svg' benchmarks/ cjerzak-rog:/home/cjerzak/Documents/fastrerandomize-sol-current/benchmarks/
```

The remote host has R, reticulate, testthat, jsonlite, CUDA 13.4, Nsight Systems 2026.3.2, Nsight Compute 2026.3.1, and the `preference_fm_cuda_20261009` Python environment. No Python packages, system R installation, GPU clocks, power policy, or driver permissions need to change.

On the remote machine, install into a private library:

```bash
cd /home/cjerzak/Documents/fastrerandomize-sol-current
R CMD INSTALL --library=rlib source/fastrerandomize > results/install.log 2>&1
```

`run.sh` selects the existing CUDA interpreter, disables JAX preallocation, and sets OMP/OpenBLAS threads to one. `SOL_CONDA_ENV`, `RETICULATE_CONDA`, and `RETICULATE_PYTHON` can override its defaults. In the recorded run there were no custom XLA flags; R used the system reference BLAS.

## Validate before measuring

The tests request an environment named `fastrerandomize_env`. Expose the existing interpreter through a private directory, without modifying the existing environment. Copy its conda history as well: an empty history is insufficient for this reticulate version.

```bash
cd /home/cjerzak/Documents/fastrerandomize-sol-current
mkdir -p envs/fastrerandomize_env/{bin,conda-meta}
ln -s /home/cjerzak/miniconda3/envs/preference_fm_cuda_20261009/bin/python envs/fastrerandomize_env/bin/python
cp /home/cjerzak/miniconda3/envs/preference_fm_cuda_20261009/conda-meta/history envs/fastrerandomize_env/conda-meta/history
export CONDA_ENVS_PATH="$PWD/envs"
export RETICULATE_CONDA=/home/cjerzak/miniconda3/bin/conda
export RETICULATE_PYTHON="$CONDA_ENVS_PATH/fastrerandomize_env/bin/python"
export XLA_PYTHON_CLIENT_PREALLOCATE=false OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
JAX_PLATFORMS=cuda Rscript benchmarks/speed_of_light/validate.R > results/tests_cuda.log 2>&1
JAX_PLATFORMS=cpu SOL_TEST_FILTER='kernels|pure-r' Rscript benchmarks/speed_of_light/validate.R > results/tests_cpu.log 2>&1
```

`validate.R` requires a working backend before running tests, records the device, and exports assertion totals. The separate `diagnose-custom-tail.R` reproduces the custom-callback limitation described in the report; it is not part of the passing regression suite.

Local package checks used:

```bash
R CMD build --no-build-vignettes fastrerandomize
R CMD check --no-manual --ignore-vignettes fastrerandomize_0.4.tar.gz
```

Vignettes and the PDF manual were excluded. Local JAX tests skipped because the default conda environment was absent; CUDA and CPU coverage above ran explicitly without skips.

## Run measurements sequentially

Use a fresh shell after validation so the test environment overrides do not leak into benchmarking. Do not overlap GPU jobs. On the remote host:

```bash
cd /home/cjerzak/Documents/fastrerandomize-sol-current
bash benchmarks/speed_of_light/run-analysis.sh > results/analysis.log 2>&1
```

The runner builds calibration tools, runs `api`, `hostprep`, `focus`, `pipeline`, and `sweep`, then captures three full-pipeline and three single-batch calls with Nsight Systems. GPU timings wait on `jax.block_until_ready`. The primary workload explicitly sets `SOL_P=2048`; other primary defaults are 4,096 units, batch 4,096, and 1,048,576 draws. Data seed is 20261022, assignment seed 77, and acceptance probability 0.001. The automatic package batch size would be 1,024.

Sampling:

- Complete API: one first call after backend/input setup, then seven warmed calls returning R objects. Validation and saving outputs are outside the timers.
- Production preparation: one Rprof capture and five unprofiled calls. `production_design()` requests the same diagonal-only preparation as the default package caller.
- Resident balance and generated batch: 100 synchronized calls after 300 and 150 warmups, respectively.
- Complete fused GPU pipeline: 15 synchronized calls after three warmups.
- Nearby shapes: seven cases, 30 synchronized calls after five warmups.
- Nsight Systems: three synchronized calls after five warmups for each capture target. Profiling timings remain separate from headline unprofiled timings.

CUDA-event calibration uses 300 warmup launches and 31 samples, each averaging five kernel launches. Its DRAM triad, FP32 FMA, and strict-FP32 square SGEMMs provide empirical references, not universal throughput ceilings.

The runner attempts Nsight Compute `SpeedOfLight` on the production batch using existing permissions and `--clock-control none`. If it fails with `ERR_NVGPUCTRPERM`, keep that log and retain the analytical roofline label. Do not substitute instrumented timings for the unprofiled results.

## Collect and verify evidence

From the local study worktree, collect results and verify source and measurement-harness hashes:

```bash
rsync -az cjerzak-rog:/home/cjerzak/Documents/fastrerandomize-sol-current/results/ benchmarks/speed_of_light/results/
python3 benchmarks/speed_of_light/manifest.py --source-root "$PWD" --commit "$(git rev-parse HEAD)" --remote-root /home/cjerzak/Documents/fastrerandomize-sol-current
python3 benchmarks/speed_of_light/analyze.py
python3 benchmarks/speed_of_light/metrics.py
/Users/cjerzak/miniconda3/bin/python3 benchmarks/speed_of_light/plot.py
```

`analyze.py` builds `results/summary.json` and `results/sweep_summary.csv` from the timing samples and exported Nsight SQLite files. `metrics.py` derives `results/metrics.json` from that summary and the captured telemetry, including repeated-call reproducibility and the independent diagonal-reference checks recorded by the benchmark. `plot.py` requires NumPy and Matplotlib and renders the PNG, PDF, and SVG figures.

To regenerate only the metrics and figures from the committed measurements, run `metrics.py` and `plot.py`; no GPU run or raw profiler files are needed. The measured run's manifest retains its recorded paths and hashes as provenance; a new run creates its own manifest.

Raw `.nsys-rep`, `.sqlite`, `.hlo.txt`, `.Rprof`, and saved `.rds` outputs remain in the worktree and on the remote host, excluded from Git. Git contains the source manifest, individual timing samples, correctness results, telemetry, readable profiler summaries, derived metrics, and the report figure.
