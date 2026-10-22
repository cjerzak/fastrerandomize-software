# Speed of Light Analysis

Monte Carlo rerandomization has substantial remaining performance headroom, even in a deliberately favorable GPU domain. On `cjerzak-rog`, the balance matrix multiplication sustains **19.1 TFLOP/s** in the production pipeline. The warmed GPU scoring and ranking stage processes **1,048,576 draws in 2.90 seconds**. The complete R call takes **26.15 seconds**, of which separately measured covariate preparation accounts for about **23.21 seconds**. These results support a precise kernel-throughput statement; they do not support describing the whole operation as near its hardware limit.

This analysis measures production source at commit `4489e28e5f2ee1dddc34415afc4de23bfd31796d`, dated October 22, 2026. Production R files and the main checkout are unchanged. The analysis branch contains only benchmark and report artifacts.

![Roofline estimate, GPU time breakdown, and R API timing](speed_of_light.png)

[PDF figure](speed_of_light.pdf) · [SVG figure](speed_of_light.svg) · [Machine-readable results](results/summary.json) · [Reproduction instructions](REPRODUCE.md)

## Workload and scope

The selected task is `generate_randomizations()` using Monte Carlo sampling: generate fixed-treatment-count assignments, evaluate covariate balance, rank the scores, and regenerate the accepted assignments. It is central to the package's purpose and its large-randomization claims. The measurements call the package's actual R-created JAX functions, rather than a separately translated implementation.

The primary case has 4,096 experimental units, 2,048 treated units, 2,048 independent normal covariates, 1,048,576 candidate draws, batches of 4,096, and acceptance probability 0.001, returning 1,049 assignments. It uses the default balance statistic and `approximate_inv = TRUE`. R data generation uses seed 20261022; the JAX assignment stream uses seed 77.

This is intentionally favorable to dense GPU arithmetic: many covariates, large aligned matrices, repeated batches, and a small accepted output. Batch size is explicitly 4,096; the package's automatic size at this number of units is 1,024. The precision is unchanged: float32 arrays with `Precision.HIGHEST`. The compiled program calls cuBLASLt, and the trace identifies a SIMT SGEMM kernel, `cutlass_80_simt_sgemm_256x128_8x4_nn_align1`. Reduced-precision Tensor Core TOPS are therefore not the appropriate comparison.

The host is an Intel Core Ultra 9 285H with an NVIDIA GeForce RTX 5080 **Laptop** GPU: 60 SMs, 7,680 CUDA cores, 16 GB GDDR7, and 48 MiB L2. The run uses driver 615.71.10, CUDA toolkit 13.4, JAX/JAXlib 0.11.1, NumPy 2.5.3, reticulate, and R 4.5.2. R uses the system reference BLAS, `/usr/lib/x86_64-linux-gnu/blas/libblas.so.3.12.1`. Host timings are specific to that configuration. Detailed versions are in [hardware and software](results/hardware_and_software.txt), and [the manifest](results/manifest.json) records matching local and remote hashes for all production R source files, DESCRIPTION, and NAMESPACE.

## Roofline model

NVIDIA's Speed of Light sections compare achieved throughput with the maximum of each hardware resource; a roofline model bounds throughput using arithmetic intensity and compute/memory ceilings. The resource model here is analytical. Nsight Compute 2026.3.1 refused hardware-counter collection with `ERR_NVGPUCTRPERM`; [the recorded attempt](results/ncu_attempt.log) preserves that result. Nsight Systems 2026.3.2 successfully measured CUDA kernel durations and the pipeline timeline. These estimates are **not counter-measured Nsight Compute SOL percentages**. [NVIDIA profiling guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#sections-and-rules)

For the dominant balance product, `W[B,n] @ Xw[n,p]`, let a fused multiply-add count as two floating-point operations:

```text
F = 2 B n p                         = 68,719,476,736 FLOP per batch
D = 4 (B n + n p + B p)            = 134,217,728 bytes of minimum GEMM traffic
I = F / D                          = 512 FLOP/byte
P_ref = 7,680 × 2 × 2.287 GHz      = 35.12832 TFLOP/s
BW_ref                             = 896.064 GB/s
t_ref = max(F / P_ref, D / BW_ref) = max(1.956 ms, 0.150 ms) = 1.956 ms
```

The compute reference uses NVIDIA's listed upper laptop boost clock, 2,287 MHz. The bandwidth value comes from CUDA device information and agrees with NVIDIA's rounded 896 GB/s specification. It is a reference at that frequency, not a measurement of sustainable power-limited laptop performance. The model's arithmetic intensity makes this GEMM compute-bound even under the optimistic traffic assumption. [NVIDIA laptop specifications](https://www.nvidia.com/en-us/geforce/laptops/50-series/)

Across 768 GEMM launches inside three complete GPU pipeline captures, the median kernel duration is **3.605 ms**, giving **19.06 TFLOP/s**, **54.3% of the upper-boost FP32 reference**, or **1.84 times its idealized compute time**. This counts useful dense-product FLOPs and excludes permutation generation and other kernels. The sustained full-pipeline trace, rather than a brief standalone capture, is the basis of the reported 19.1 TFLOP/s.

Clocks were not locked. The sustained scoring run logged approximately 1.47–2.20 GHz during high GPU activity, and memory reached 14,001 MHz. Consequently, the percentage relative to an instantaneous, clock-adjusted ceiling would differ from 54.3%. The 100 ms telemetry cannot establish the exact clock of every 3–4 ms GEMM; it is not a substitute for hardware counters. Peak-frequency and measured-clock denominators must not be mixed silently.

Independent CUDA-event calibrations measured **770.2 GB/s** for a large DRAM triad and **31.43 TFLOP/s** for a register-resident FP32 FMA workload. Strict-FP32 cuBLAS SGEMM controls reached 16.22 and 17.15 TFLOP/s at square sizes 4,096 and 8,192. These controls have different shapes, duty cycles, and clock behavior from the package pipeline; their rates are empirical baselines, not universal hardware ceilings. [Calibration samples](results/calibration.jsonl)

For 1,048,576 draws, the dense balance arithmetic alone has a reference floor of **0.501 seconds**. Dividing that by the whole 2.90-second GPU pipeline would give about 17%, but that would omit necessary integer random-number generation, permutation sorting, and validation work. It is therefore **not an end-to-end SOL utilization estimate**. Actual DRAM traffic, cache utilization, and integer/shared-memory pipeline ceilings remain unmeasured because hardware counters were unavailable.

## Measured timing

| Scope | Median | Variation and sampling |
|---|---:|---|
| Balance GEMM inside the complete GPU pipeline | 3.605 ms per batch | 768 kernel launches in three Nsight Systems captures |
| Complete `WhitenedT2Jit` on resident inputs | 4.364 ms per batch | 100 synchronized calls; 10th–90th percentile 4.241–4.506 ms |
| Generate, validate, and score one batch | 11.348 ms per batch | 100 synchronized calls; 10th–90th percentile 10.984–11.940 ms |
| Fused generation, validation, scoring, and ranking of 1,048,576 draws | 2.896 s | 15 synchronized calls; 10th–90th percentile 2.870–2.925 s |
| Complete warmed R API, including preparation and R output | 26.148 s | Seven calls; range 25.349–27.799 s |
| Existing covariate preparation alone | 23.212 s | Three calls; range 23.088–23.322 s |

The first API call after backend setup took 34.869 seconds. This includes additional compilation but is not a fresh-process startup measurement. Input generation and backend initialization precede the API timers. The warmed API times include input validation, host preparation, transfers, scoring, ranking, accepted-assignment regeneration and checks, and materialization of the R return value. Harness validation is outside the timed interval.

Device timings wait for completion using `jax.block_until_ready`; compilation and warmup are outside the measurements. The focused balance and batch measurements use 300 and 150 warmups, respectively; the complete GPU pipeline uses three warmups. GPU workloads run sequentially. No clocks, power policy, existing Python environment, or system R installation was changed. [JAX benchmarking guidance](https://docs.jax.dev/en/latest/benchmarking.html)

## Where time goes

The complete GPU trace, averaged over three captures, attributes **60.9% of kernel time to assignment permutation sorting**, **31.8% to the balance GEMM**, and **7.3% to other kernels**. The means per million-draw call are 1.779, 0.928, and 0.213 seconds, respectively. The assignment path launches twelve segmented radix-sort passes per batch. Final ranking of the balance scores is negligible in comparison. [Trace summary](results/profile_all_stats.txt)

CUDA kernels occupy about 99.6% of each captured pipeline interval. That measures time spent executing kernels, not closeness to peak throughput. Dispatch gaps are small in the fused pipeline. Even making its balance GEMM instantaneous would improve GPU-stage latency by only about 1.47 times under this decomposition; permutation generation is the larger GPU optimization target.

For the full R call, separately measured preparation is about **89%** of the warmed latency. [`.balance_design()`](../../fastrerandomize/R/FRR_HelperFxns.R) computes `stats::cov(X)`, an eigendecomposition, and full matrix products even when the default diagonal approximation only consumes centered, variance-scaled `Xw`. An R profile of this preparation found 42.0% of samples in covariance computation, 34.5% in matrix multiplication, and 20.9% in `eigen`. [Preparation profile summary](results/hostprep/n4096_p2048_b4096_d1048576.json)

A **preparation-only experiment** in the harness computes the required diagonal inputs directly. Its median is **0.344 seconds**, with bit-for-bit identical `Xw` and inverse-variance vectors for this case. The production function was not patched, and no full-call speedup for this experiment is claimed. A future implementation should take this shortcut only for the default diagonal balance path; full-covariance scoring and custom threshold functions can require the other design matrices. This is the strongest immediate optimization candidate. GPU permutation work follows it in priority.

## Nearby domains and correctness

The initial exploratory sweep used 30 synchronized calls per function and five warmups. It includes all seven tested shapes, rather than only the best result. The longer primary measurements above supersede the sweep's timing for the selected case. Smaller kernels show more timing variability, so these are descriptive comparisons, not controlled scaling exponents.

| Units | Covariates | Draws per batch | Balance only | Generate and score batch |
|---:|---:|---:|---:|---:|
| 1,024 | 16 | 4,096 | 0.286 ms | 2.414 ms |
| 1,024 | 256 | 4,096 | 0.478 ms | 2.463 ms |
| 1,024 | 1,024 | 4,096 | 0.839 ms | 2.896 ms |
| 1,024 | 2,048 | 4,096 | 1.435 ms | 3.543 ms |
| 4,096 | 1,024 | 1,024 | 0.798 ms | 2.797 ms |
| 4,096 | 1,024 | 4,096 | 2.521 ms | 9.473 ms |
| 4,096 | 2,048 | 4,096 | 4.433 ms | 11.161 ms |

The narrow-covariate case spends very little time on balance arithmetic. A low FLOP/s number there would not, by itself, diagnose poor resource utilization. [Sweep data](results/sweep_summary.csv)

All generated assignments in checked batches were binary with the prescribed treatment count. The production pipeline reported zero invalid assignments; its accepted-draw regeneration checks passed. Returned assignments and balance vectors were identical across seven repeated calls. A double-precision R reference checked 64 accepted scores, with maximum relative error **9.07 × 10^-8**; 64 resident-batch scores had maximum relative error **1.13 × 10^-7**. The isolated scorer and the batch generation/scoring path agreed. The diagonal-only preparation experiment checked exact equality of both required design outputs. These checks validate this benchmark and do not establish arbitrary-data or cross-backend guarantees.

## Publication recommendation

Defer a claim that the package runs near the GPU's Speed of Light. The clearest result is useful FP32 throughput combined with substantial, identified optimization opportunities. No README paragraph has been added. If a measured-results paragraph is desired, the following wording preserves the scope:

> [Speed of Light Analysis](benchmarks/speed_of_light/REPORT.md). On an NVIDIA RTX 5080 Laptop GPU, the warmed Monte Carlo GPU pipeline scored and ranked 1,048,576 assignments for 4,096 units and 2,048 covariates in 2.90 seconds. Its FP32 balance matrix multiplication sustained 19.1 TFLOP/s, about 54% of an upper-boost FP32 roofline reference. Assignment generation dominated GPU time, while covariate preparation dominated the complete 26.15-second R call, identifying substantial scope for further improvement.

This paragraph is a draft for placement in the repository-root README; its relative report link is written for that destination. It is not a claim of counter-measured NVIDIA SOL utilization.
