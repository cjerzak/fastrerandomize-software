# fastrerandomize (development version)

## Correctness and Reliability

* Monte Carlo permutations now use an int32 payload. The previous int8 payload
  produced assignments with the wrong number of treated units on some GPU
  backends (observed on ROCm for `n_units >= 256`). Every generated assignment,
  Monte Carlo and exact, is now checked to contain exactly `n_treated` treated
  units, with an informative error otherwise.
* Balance inputs are built in double precision on the host: covariates are
  centered, then scaled (diagonal approximation) or whitened (full inverse,
  using a pseudo-inverse for singular covariance). This removes float32
  cancellation for uncentered covariates (relative error about 1e-3 down to
  about 1e-6) and the CPU round trip for matrix inversion on Metal. Balance
  values can differ from earlier versions in the last digits, so draws at the
  acceptance cutoff may change.
* Constant covariates are dropped with a warning (they previously produced
  `NaN` balance), and non-finite covariates are rejected.
* `randomization_test()` computes `tau_obs` in double precision and
  re-evaluates candidates that tie it up to float32 error, so exact ties (such
  as the observed assignment and, with equal group sizes, its complement) are
  counted consistently. Candidates must be 0/1.
* Matrix products in the new kernels use full float32 precision, so TF32 on
  NVIDIA GPUs does not reduce accuracy.
* Fixed the cosine distance kernel for JAX versions without the `a_min`/`a_max`
  arguments of `jnp.clip`.

## Performance

* Monte Carlo generation runs every batch in one compiled program. Keys are
  derived per draw index rather than materialized, and the compiled kernels are
  reused across calls. Results, including the returned keys, are unchanged for
  a given seed and do not depend on `batch_size`, which now defaults to `NULL`
  (sized automatically). Set `options(fastrerandomize.mc_fused = FALSE)` to
  dispatch batches from R (the default on Metal).
* The fiducial-interval search in `randomization_test()` uses the closed form
  `a_w + tau * (c_w - b_w)` of each candidate's statistic, computed in one pass
  over the candidates. It runs the same stochastic search and grid as before,
  10 to 1000 times faster.
* Euclidean, squared Euclidean and Mahalanobis `fast_distance()` metrics use a
  matrix-product formulation on centered (and scaled or whitened) inputs:
  about 5x faster for Euclidean and 20-60x for full Mahalanobis on GPU.
* `check_jax_availability()` caches a successful check for the session, removing
  about 0.7 s of conda overhead from every call.

## Interface

* When `randomization_test()` receives only `candidate_randomizations_array`,
  the array is no longer copied back to R; the result stores it in
  `candidate_randomizations_array` and sets `candidate_randomizations` to `NULL`.
* Documented the `threshold_func` call signature.

# fastrerandomize 0.4

## Correctness and Reliability

* Fixed full-inverse Mahalanobis distance calculation on the Metal backend.
* Corrected two-sided fiducial interval inversion to use `alpha / 2`, return
  typed missing bounds when no grid point is accepted, and handle negative or
  near-zero effects symmetrically in both JAX and pure-R implementations.
* Made `c_initial` a functional fiducial-search step-size control while
  preserving the existing default scale.
* Exact enumeration now supports tiny designs and warns before very large
  allocations; Monte Carlo sampling can draw with replacement beyond the
  number of unique assignments.
* Added an optional full-range integer `seed` for Monte Carlo JAX streams.
* File output now preserves balance measures in a named CSV without row names.
* JAX tests now skip correctly when the backend availability check returns
  `NULL`.

## Maintenance

* Core JAX kernels are initialized once, and broken or unused internal kernels
  have been removed.
* Improved required-input validation and diagnostic printing when `R2` is
  supplied without `sigma`.

## Testing Infrastructure

* Migrated test suite to testthat framework in standard R package location
  (`tests/testthat/`).
* Tests now run automatically during `R CMD check`.
* Added `skip_if_no_jax()` helper for graceful skipping when JAX is unavailable.
* Test files organized by category:
    - `test-pure-r.R`: Pure R implementation tests (always run)
    - `test-jax-integration.R`: JAX-accelerated function tests
    - `test-distance.R`: Distance metric tests
    - `test-edge-cases.R`: Boundary condition tests

# fastrerandomize 0.3

## New Features

* Added `diagnose_rerandomization()` function for pre-analysis evaluation of
  rerandomization designs. This function helps researchers determine optimal
  acceptance thresholds by computing:
    - Expected number of acceptable randomizations
    - Minimum balance criterion values
    - Power analysis for different threshold choices

* Added `fast_distance()` function for hardware-accelerated pairwise distance
  computation supporting multiple metrics: Euclidean, Manhattan, Mahalanobis,
  cosine, and correlation-based distances.

* Added S3 methods (`print`, `summary`, `plot`) for `fastrerandomize_diagnostic`
  class to visualize and summarize diagnostic results.

## Improvements

* Updated JAX backend to support CUDA 12 and CUDA 13.
* Improved documentation throughout the package.
* Various bug fixes and performance improvements.

# fastrerandomize 0.2

## Initial CRAN Release (2025-01-14)

First release on CRAN with core functionality:

* `generate_randomizations()`: Generate pools of acceptable randomizations
  based on covariate balance.
* `generate_randomizations_exact()`: Exact enumeration for small experiments.
* `generate_randomizations_mc()`: Monte Carlo sampling for larger experiments.
* `randomization_test()`: Permutation-based inference with optional fiducial
  intervals.
* `build_backend()`: Create conda environment with JAX and GPU support.
* `check_jax_availability()`: Verify JAX backend availability.
* Pure R fallback implementations (`_R` suffix functions) for environments
  without JAX.
* Support for CPU, CUDA, and METAL hardware acceleration frameworks.
* S3 classes with `print`, `summary`, and `plot` methods for results objects.
* Included datasets: `QJEData` and `YOPData`.
