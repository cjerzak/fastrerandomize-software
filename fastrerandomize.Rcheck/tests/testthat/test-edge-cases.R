# Edge Cases Tests
# Boundary conditions and parameter combinations

# Skip all tests if JAX is not available
skip_if_no_jax()

test_that("small n exact enumeration (n=6, k=3) works", {
  set.seed(100)
  X_small <- matrix(rnorm(6 * 2), 6, 2)
  RandomizationSet <- fastrerandomize::generate_randomizations(
    n_units = 6,
    n_treated = 3,
    X = X_small,
    randomization_accept_prob = 1.0,  # Keep all
    randomization_type = "exact",
    verbose = FALSE
  )
  # Should have choose(6,3) = 20 randomizations
  expect_equal(nrow(RandomizationSet$randomizations), 20)
})

test_that("exact and monte_carlo give similar balance distributions", {
  set.seed(101)
  # Use larger n to allow meaningful Monte Carlo comparison
  # choose(12, 6) = 924, so max_draws can be reasonably large
  X_small <- matrix(rnorm(12 * 2), 12, 2)

  res_exact <- fastrerandomize::generate_randomizations(
    n_units = 12,
    n_treated = 6,
    X = X_small,
    randomization_accept_prob = 1.0,
    randomization_type = "exact",
    verbose = FALSE
  )

  res_mc <- fastrerandomize::generate_randomizations(
    n_units = 12,
    n_treated = 6,
    X = X_small,
    randomization_accept_prob = 1.0,
    randomization_type = "monte_carlo",
    verbose = FALSE,
    max_draws = 900L,  # Must be <= choose(12, 6) = 924
    batch_size = 100L
  )

  # Monte Carlo should produce similar mean balance (roughly)
  mean_exact <- mean(res_exact$balance)
  mean_mc <- mean(res_mc$balance)
  # Allow 50% tolerance due to sampling variance
  expect_lt(abs(mean_exact - mean_mc) / mean_exact, 0.5)
})

test_that("single covariate works", {
  set.seed(102)
  X_single <- matrix(rnorm(20), 20, 1)
  RandomizationSet <- fastrerandomize::generate_randomizations(
    n_units = 20,
    n_treated = 10,
    X = X_single,
    randomization_accept_prob = 0.5,
    randomization_type = "monte_carlo",
    verbose = FALSE,
    max_draws = 1000L,
    batch_size = 100L
  )
  expect_gt(nrow(RandomizationSet$randomizations), 0)
})

test_that("many covariates work with diagonal approximation", {
  set.seed(103)
  X_many <- matrix(rnorm(20 * 20), 20, 20)
  RandomizationSet <- fastrerandomize::generate_randomizations(
    n_units = 20,
    n_treated = 10,
    X = X_many,
    randomization_accept_prob = 0.5,
    randomization_type = "monte_carlo",
    approximate_inv = TRUE,  # Use diagonal approx for stability
    verbose = FALSE,
    max_draws = 1000L,
    batch_size = 100L
  )
  expect_gt(nrow(RandomizationSet$randomizations), 0)
})
