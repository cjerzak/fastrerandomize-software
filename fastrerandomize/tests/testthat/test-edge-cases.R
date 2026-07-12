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

test_that("tiny unfiltered exact enumeration returns every assignment", {
  RandomizationSet <- fastrerandomize::generate_randomizations(
    n_units = 4,
    n_treated = 2,
    X = NULL,
    randomization_accept_prob = 1,
    randomization_type = "exact",
    verbose = FALSE
  )

  expect_equal(nrow(RandomizationSet$randomizations), 6)
  expect_true(all(rowSums(RandomizationSet$randomizations) == 2))
  expect_null(RandomizationSet$balance)
})

test_that("small filtered exact pools warn but still return assignments", {
  set.seed(106)
  expect_warning(
    result <- fastrerandomize::generate_randomizations(
      n_units = 4,
      n_treated = 2,
      X = matrix(rnorm(8), 4, 2),
      randomization_accept_prob = 0.5,
      randomization_type = "exact",
      verbose = FALSE
    ),
    "nominally retains fewer than 10"
  )
  expect_gt(nrow(result$randomizations), 0)
})

test_that("monte_carlo samples with replacement and supports one partial batch", {
  set.seed(104)
  X_small <- matrix(rnorm(6 * 2), 6, 2)

  result_a <- fastrerandomize::generate_randomizations(
    n_units = 6,
    n_treated = 3,
    X = X_small,
    randomization_accept_prob = 0.2,
    randomization_type = "monte_carlo",
    max_draws = 50L,
    batch_size = 100L,
    seed = 9876L,
    verbose = FALSE
  )
  result_b <- fastrerandomize::generate_randomizations(
    n_units = 6,
    n_treated = 3,
    X = X_small,
    randomization_accept_prob = 0.2,
    randomization_type = "monte_carlo",
    max_draws = 50L,
    batch_size = 100L,
    seed = 9876L,
    verbose = FALSE
  )

  expect_equal(nrow(result_a$randomizations), 10)
  expect_true(all(rowSums(result_a$randomizations) == 3))
  expect_equal(result_a$randomizations, result_b$randomizations)
  expect_equal(result_a$balance, result_b$balance)
})

test_that("monte_carlo low-acceptance warning describes the expected count", {
  set.seed(107)
  expect_warning(
    result <- fastrerandomize::generate_randomizations(
      n_units = 6,
      n_treated = 3,
      X = matrix(rnorm(12), 6, 2),
      randomization_accept_prob = 0.01,
      randomization_type = "monte_carlo",
      max_draws = 50L,
      batch_size = 100L,
      seed = 2468L,
      verbose = FALSE
    ),
    "max_draws \\* randomization_accept_prob is below 1"
  )
  expect_equal(nrow(result$randomizations), 1)
})

test_that("file output contains named assignments and optional balance", {
  set.seed(105)
  output_file <- tempfile(fileext = ".csv")
  on.exit(unlink(output_file), add = TRUE)

  returned_path <- fastrerandomize::generate_randomizations(
    n_units = 6,
    n_treated = 3,
    X = matrix(rnorm(12), 6, 2),
    randomization_accept_prob = 1,
    randomization_type = "exact",
    file = output_file,
    verbose = FALSE
  )
  written <- utils::read.csv(output_file, check.names = FALSE)

  expect_equal(returned_path, normalizePath(output_file))
  expect_equal(names(written), c(paste0("W", 1:6), "balance"))
  expect_equal(nrow(written), choose(6, 3))

  unfiltered_file <- tempfile(fileext = ".csv")
  on.exit(unlink(unfiltered_file), add = TRUE)
  fastrerandomize::generate_randomizations(
    n_units = 4,
    n_treated = 2,
    X = NULL,
    randomization_accept_prob = 1,
    randomization_type = "exact",
    file = unfiltered_file,
    verbose = FALSE
  )
  unfiltered <- utils::read.csv(unfiltered_file, check.names = FALSE)
  expect_equal(names(unfiltered), paste0("W", 1:4))
})

test_that("initialize_jax reuses retained kernels and omits dead kernels", {
  fastrerandomize:::initialize_jax()
  env <- fastrerandomize:::fastrr_env
  core_kernel <- env$FastDiffInMeans
  fastrerandomize:::initialize_jax()

  expect_identical(core_kernel, env$FastDiffInMeans)
  expect_false(any(c(
    "BatchedVectorizedFastHotel2T2",
    "VectorizedTakeAxis0",
    "Potential2Obs",
    "Y_VectorizedFastDiffInMeans",
    "YW_VectorizedFastDiffInMeans",
    "WVectorizedFastDiffInMeans",
    "GreaterEqualMagCompare"
  ) %in% ls(envir = env)))
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
