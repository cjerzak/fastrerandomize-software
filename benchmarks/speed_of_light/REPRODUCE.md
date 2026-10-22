# Reproduce the Speed of Light analysis

The production snapshot is commit `4489e28e5f2ee1dddc34415afc4de23bfd31796d`. Run experiments in an isolated worktree and copy its package into a private directory on `cjerzak-rog`. The study's remote directory is `/home/cjerzak/Documents/fastrerandomize-sol-20261022`, outside Dropbox. Reusing that directory overwrites benchmark results; use a new directory to preserve the original capture.

The remote host already has R, reticulate, jsonlite, the `preference_fm_cuda_20261009` conda environment, JAX/JAXlib 0.11.1, CUDA 13.4, and Nsight Systems/Compute. The harness installs the package only in a private R library. It does not install or upgrade Python packages. `run.sh` sets CUDA selection, disables JAX preallocation, and sets OMP/OpenBLAS threads to one; it records any inherited `XLA_FLAGS`. The recorded study had no custom XLA flags.

## Prepare the remote directory

From the analysis worktree on the local machine:

```bash
ssh cjerzak-rog 'mkdir -p /home/cjerzak/Documents/fastrerandomize-sol-20261022/source /home/cjerzak/Documents/fastrerandomize-sol-20261022/rlib /home/cjerzak/Documents/fastrerandomize-sol-20261022/results'
rsync -az fastrerandomize/ cjerzak-rog:/home/cjerzak/Documents/fastrerandomize-sol-20261022/source/fastrerandomize/
rsync -az --exclude=results --exclude='*.png' --exclude='*.pdf' --exclude='*.svg' benchmarks/ cjerzak-rog:/home/cjerzak/Documents/fastrerandomize-sol-20261022/benchmarks/
```

On `cjerzak-rog`:

```bash
cd /home/cjerzak/Documents/fastrerandomize-sol-20261022
R CMD INSTALL --library="$PWD/rlib" source/fastrerandomize
/usr/local/cuda/bin/nvcc -O3 -arch=sm_120 benchmarks/speed_of_light/calibrate.cu -lcublas -o benchmarks/speed_of_light/calibrate
/usr/local/cuda/bin/nvcc -O2 -shared -Xcompiler=-fPIC benchmarks/speed_of_light/markers.cu -o benchmarks/speed_of_light/markers.so
```

`run.sh` defaults to the existing conda environment and interpreter. Override `SOL_CONDA_ENV`, `RETICULATE_CONDA`, and `RETICULATE_PYTHON` if necessary. The private library is looked up relative to `SOL_ROOT`, which the runner sets to its repository root.

## Run measurements sequentially

Do not overlap GPU jobs; close other compute workloads first. The current high-dimensional R API benchmark takes several minutes because it repeats the existing host preparation.

```bash
benchmarks/speed_of_light/calibrate > results/calibration.jsonl
bash benchmarks/speed_of_light/run.sh sweep > results/sweep.log 2>&1
SOL_P=2048 bash benchmarks/speed_of_light/run.sh focus > results/focus.log 2>&1
SOL_P=2048 bash benchmarks/speed_of_light/run.sh pipeline > results/pipeline.log 2>&1
SOL_P=2048 bash benchmarks/speed_of_light/run.sh api > results/api.log 2>&1
SOL_P=2048 bash benchmarks/speed_of_light/run.sh hostprep > results/hostprep.log 2>&1
```

`focus`, `pipeline`, `api`, `profile`, and `hostprep` accept `SOL_N`, `SOL_P`, `SOL_BATCH`, and `SOL_DRAWS`. The primary case is `4096, 2048, 4096, 1048576`. The normal default for `SOL_P` is 1024, so explicitly set 2048 as above. The generator and balance kernels come from the installed production package. `hostprep` is a separate preparation experiment and does not modify that package.

The helpers save unprofiled individual timings, compilation times, validation results, compiler HLO, environment metadata, and 100 ms `nvidia-smi` telemetry. Remote telemetry timestamps use America/Chicago; `measurement_start_epoch` and `measurement_end_epoch` are Unix timestamps. The initial exploratory sweep and pipeline were recorded before epoch-window logging was added; their sample timings are unchanged.

## Capture GPU execution

```bash
SOL_P=2048 SOL_TARGET=all SOL_OUTPUT_TAG=profile_all nsys profile \
  --trace=cuda,nvtx --sample=none --cpuctxsw=none \
  --cuda-graph-trace=node --capture-range=cudaProfilerApi \
  --capture-range-end=stop --force-overwrite=true \
  --output=results/profile_all \
  bash benchmarks/speed_of_light/run.sh profile > results/profile_all.log 2>&1

nsys stats --report cuda_gpu_kern_sum,cuda_gpu_mem_time_sum,nvtx_sum \
  --format csv --force-export=true results/profile_all.nsys-rep \
  > results/profile_all_stats.txt 2>&1
```

For the supplementary single-batch trace, replace `SOL_TARGET=all` with `SOL_TARGET=batch`, `SOL_OUTPUT_TAG=profile_all` with `SOL_OUTPUT_TAG=profile_batch`, and the output filenames with `profile_batch`. Each capture follows five warmups and records three synchronized calls. The output tag keeps each capture's telemetry separate. The initial full-pipeline capture used the default metadata directory `results/profile/`. The NVTX labels distinguish the timed regions. Nsight's kernel union duration, rather than `nvidia-smi` GPU utilization, is used to check dispatch gaps.

The original attempt to collect actual Nsight Compute counters was:

```bash
/usr/local/cuda/bin/ncu --section SpeedOfLight --launch-count 1 \
  benchmarks/speed_of_light/calibrate > results/ncu_attempt.log 2>&1
```

It reported `ERR_NVGPUCTRPERM`. The tool continued running the calibration executable; those instrumented calibration timings were not used. No privilege or driver setting was changed. A future counter-enabled study should profile the production `SOL_TARGET=batch` capture with `--profile-from-start off`, using the existing CUDA profiler start/stop markers, and should compare profiler timing against the unprofiled run before interpreting SOL counters.

## Rebuild the summary and figure

From the local analysis worktree, copy the raw outputs back:

```bash
rsync -az cjerzak-rog:/home/cjerzak/Documents/fastrerandomize-sol-20261022/results/ benchmarks/speed_of_light/results/
python3 benchmarks/speed_of_light/analyze.py
python3 benchmarks/speed_of_light/plot.py
```

`analyze.py` uses only the standard library and needs the exported Nsight SQLite files. It normalizes trailing whitespace and line endings in readable log exports; original raw exports remain on the remote host. `plot.py` requires NumPy and Matplotlib and writes PNG, PDF, and SVG figures. The original local figure used `/Users/cjerzak/miniconda3/bin/python3`, which already had those packages.

Raw `.nsys-rep`, `.sqlite`, `.hlo.txt`, and `.Rprof` files remain in the study worktree and on the remote machine but are excluded from Git. Summary JSON, individual timing samples, readable profiler summaries, validation results, and the source manifest are included in the analysis branch. No production source, main-branch documentation, system environment, or GPU policy needs to change to reproduce the measurements.
