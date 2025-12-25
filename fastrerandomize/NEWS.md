# fastrerandomize 0.4

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
