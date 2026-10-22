#!/usr/bin/env bash
# Run from the parent of baseline/ and optimized/ after installing both sources.
set -euo pipefail
study_root="$PWD"
for revision in baseline optimized; do
  cd "$study_root/$revision"
  /usr/local/cuda/bin/nvcc -O3 -arch=sm_120 benchmarks/speed_of_light/calibrate.cu -lcublas -o benchmarks/speed_of_light/calibrate
  /usr/local/cuda/bin/nvcc -O2 -shared -Xcompiler=-fPIC benchmarks/speed_of_light/markers.cu -o benchmarks/speed_of_light/markers.so
  benchmarks/speed_of_light/calibrate > results/calibration.jsonl
  for mode in api hostprep focus pipeline sweep; do
    printf '%s %s %s\n' "$(date -Is)" "$revision" "$mode"
    SOL_P=2048 bash benchmarks/speed_of_light/run.sh "$mode" > "results/$mode.log" 2>&1
  done
  for target in all batch; do
    printf '%s %s profile_%s\n' "$(date -Is)" "$revision" "$target"
    SOL_P=2048 SOL_TARGET="$target" SOL_OUTPUT_TAG="profile_$target" nsys profile \
      --trace=cuda,nvtx --sample=none --cpuctxsw=none --cuda-graph-trace=node \
      --capture-range=cudaProfilerApi --capture-range-end=stop --force-overwrite=true \
      --output="results/profile_$target" \
      bash benchmarks/speed_of_light/run.sh profile > "results/profile_$target.log" 2>&1
    nsys stats --report cuda_gpu_kern_sum,cuda_gpu_mem_time_sum,nvtx_sum \
      --format csv --force-export=true "results/profile_$target.nsys-rep" \
      > "results/profile_${target}_stats.txt" 2>&1
  done
done
cd "$study_root/optimized"
# Profiler failures are recorded separately; instrumented timings are not used.
SOL_P=2048 SOL_TARGET=batch SOL_OUTPUT_TAG=ncu_attempt /usr/local/cuda/bin/ncu \
  --clock-control none --profile-from-start off --section SpeedOfLight --launch-count 1 \
  --target-processes all bash benchmarks/speed_of_light/run.sh profile \
  > results/ncu_attempt.log 2>&1 || printf 'Nsight Compute exited with status %s\n' "$?"
