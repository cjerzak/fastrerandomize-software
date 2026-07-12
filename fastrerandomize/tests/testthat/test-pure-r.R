# Pure R Unit Tests (no JAX required)
# Tests for base R implementations with _R suffix

# hotellingT2_R tests
test_that("hotellingT2_R basic calculation works", {
  X <- matrix(c(1, 2, 3, 4, 5, 6, 7, 8), nrow = 4)
  W <- c(1, 0, 1, 0)
  result <- hotellingT2_R(X, W)
  expect_equal(result, 1.2, tolerance = 1e-6)
})

test_that("hotellingT2_R returns NA for invalid assignment", {
  X <- matrix(c(1, 2, 3, 4, 5, 6, 7, 8), nrow = 4)
  W_all_treat <- c(1, 1, 1, 1)
  result <- hotellingT2_R(X, W_all_treat)
  expect_true(is.na(result))
})

# diff_in_means_R tests
test_that("diff_in_means_R basic calculation works", {
  Y <- 1:4
  W <- c(1, 0, 1, 0)
  result <- diff_in_means_R(Y, W)
  expect_equal(result, -1)
})

test_that("diff_in_means_R with positive effect works", {
  Y <- c(10, 1, 12, 2)
  W <- c(1, 0, 1, 0)
  result <- diff_in_means_R(Y, W)
  # mean(10,12) - mean(1,2) = 11 - 1.5 = 9.5
  expect_equal(result, 9.5)
})

# compute_diff_at_tau_for_oneW_R tests
test_that("compute_diff_at_tau_for_oneW_R basic calculation works", {
  Wprime <- c(0, 1, 0, 1)
  obsY <- c(1, 2, 3, 4)
  obsW <- c(1, 0, 1, 0)
  result <- compute_diff_at_tau_for_oneW_R(Wprime, obsY, obsW, 1)
  expect_equal(result, 3)
})

test_that("compute_diff_at_tau_for_oneW_R with zero tau works", {
  Wprime <- c(1, 0, 1, 0)
  obsY <- c(5, 2, 5, 2)
  obsW <- c(1, 0, 1, 0)
  result <- compute_diff_at_tau_for_oneW_R(Wprime, obsY, obsW, 0)
  # With tau=0, Y0 = obsY, then reassigned same way
  # mean(5,5) - mean(2,2) = 3
  expect_equal(result, 3)
})

# randomization_test_R tests
test_that("randomization_test_R returns correct p-value and tau_obs", {
  obsW <- c(1, 1, 0, 0)
  obsY <- c(2, 2, 1, 0)
  comb <- t(combn(4, 2))
  allW <- matrix(0, nrow = nrow(comb), ncol = 4)
  for (i in seq_len(nrow(comb))) allW[i, comb[i, ]] <- 1

  res <- randomization_test_R(obsW, obsY, allW, findFI = FALSE)
  expect_equal(res$p_value, 1/3, tolerance = 1e-6)
  expect_equal(res$tau_obs, 1.5)
})

test_that("randomization_test_R with no effect has p-value of 1", {
  obsW <- c(1, 1, 0, 0)
  obsY <- c(1, 1, 1, 1)  # No difference between groups
  comb <- t(combn(4, 2))
  allW <- matrix(0, nrow = nrow(comb), ncol = 4)
  for (i in seq_len(nrow(comb))) allW[i, comb[i, ]] <- 1

  res <- randomization_test_R(obsW, obsY, allW, findFI = FALSE)
  expect_equal(res$tau_obs, 0)
  expect_equal(res$p_value, 1)  # All permutations give same result
})

# generate_randomizations_R tests
test_that("generate_randomizations_R monte_carlo output structure is valid", {
  set.seed(1)
  X <- matrix(rnorm(8), 4, 2)
  res <- generate_randomizations_R(4, 2, X, 1, "monte_carlo",
                                   max_draws = 6, batch_size = 2)
  expect_equal(ncol(res$randomizations), 4)
  expect_equal(length(res$balance), nrow(res$randomizations))
})

test_that("generate_randomizations_R exact output structure is valid", {
  set.seed(2)
  X <- matrix(rnorm(8), 4, 2)
  res <- generate_randomizations_R(4, 2, X, 1, "exact",
                                   max_draws = 100, batch_size = 10)
  expect_equal(ncol(res$randomizations), 4)
  # All 6 combinations: choose(4, 2) = 6

  expect_equal(nrow(res$randomizations), choose(4, 2))
  # Each row has exactly 2 treated

  expect_true(all(rowSums(res$randomizations) == 2))
})

test_that("generate_randomizations_R accept_prob filtering works", {
  set.seed(3)
  X <- matrix(rnorm(20), 10, 2)
  res_all <- generate_randomizations_R(10, 5, X, 1.0, "monte_carlo",
                                       max_draws = 100, batch_size = 50)
  res_half <- generate_randomizations_R(10, 5, X, 0.5, "monte_carlo",
                                        max_draws = 100, batch_size = 50)
  # With 50% acceptance, should have roughly half as many (allow some variance)
  expect_lte(nrow(res_half$randomizations), nrow(res_all$randomizations))
})

test_that("backend and exact-enumeration helpers handle boundary cases", {
  expect_true(fastrerandomize:::.is_metal_backend("METAL:0"))
  expect_true(fastrerandomize:::.is_metal_backend("Metal device"))
  expect_false(fastrerandomize:::.is_metal_backend("TFRT_CPU_0"))

  expect_no_warning(fastrerandomize:::.warn_if_large_exact(1e6))
  expect_warning(
    fastrerandomize:::.warn_if_large_exact(1000001),
    "1,000,001 combinations"
  )
  expect_null(fastrerandomize:::output2output(NULL, "R"))
})

test_that("JAX availability helper treats NULL and errors as unavailable", {
  expect_false(testthat::with_mocked_bindings(
    jax_is_available(),
    check_jax_availability = function(...) NULL,
    .package = "fastrerandomize"
  ))
  expect_false(testthat::with_mocked_bindings(
    jax_is_available(),
    check_jax_availability = function(...) stop("backend failure"),
    .package = "fastrerandomize"
  ))
})

test_that("fiducial helpers use symmetric bounds and alpha over two", {
  expect_equal(fastrerandomize:::.fi_initial_bounds(-2), c(-8, 4))
  expect_equal(fastrerandomize:::.fi_initial_bounds(0), c(-1, 1))
  expect_equal(fastrerandomize:::.fi_step_multiplier(2), 1)
  expect_equal(fastrerandomize:::.fi_step_multiplier(1), 0.5)
  expect_error(fastrerandomize:::.fi_step_multiplier(0), "finite positive")

  grid <- fastrerandomize:::.fi_search_grid(c(2, -2), tau_obs = 0, length.out = 5)
  expect_equal(grid, seq(-4, 4, length.out = 5))

  tau_seq <- c(1, 2)
  tail_probs <- c(0.03, 0.06)
  expect_equal(
    fastrerandomize:::.fi_accepted_range(tau_seq, tail_probs, alpha = 0.05),
    c(1, 2)
  )
  expect_equal(
    fastrerandomize:::.fi_accepted_range(tau_seq, tail_probs, alpha = 0.10),
    c(2, 2)
  )
  expect_no_warning(
    empty <- fastrerandomize:::.fi_accepted_range(tau_seq, c(0, 0), alpha = 0.05)
  )
  expect_equal(empty, c(NA_real_, NA_real_))
})

test_that("base-R fiducial search handles negative effects and c_initial", {
  combinations <- t(combn(8, 4))
  allW <- matrix(0, nrow = nrow(combinations), ncol = 8)
  for (i in seq_len(nrow(combinations))) allW[i, combinations[i, ]] <- 1
  obsW <- allW[1, ]

  set.seed(100)
  obsY <- rnorm(8, sd = 0.5) - 2 * obsW
  tau_obs <- diff_in_means_R(obsY, obsW)

  set.seed(11)
  interval_small_step <- find_fiducial_interval_R(
    obsW, obsY, allW, tau_obs, c_initial = 1, n_search_attempts = 100
  )
  set.seed(11)
  interval_large_step <- find_fiducial_interval_R(
    obsW, obsY, allW, tau_obs, c_initial = 4, n_search_attempts = 100
  )

  expect_true(all(is.finite(interval_small_step)))
  expect_lte(interval_small_step[1], interval_small_step[2])
  expect_false(isTRUE(all.equal(interval_small_step, interval_large_step)))
})

test_that("JAX seeds use the full R integer-compatible range", {
  expect_equal(fastrerandomize:::.resolve_jax_seed(0), 0L)
  expect_equal(
    fastrerandomize:::.resolve_jax_seed(.Machine$integer.max),
    .Machine$integer.max
  )
  expect_error(fastrerandomize:::.resolve_jax_seed(-1), "whole number")
  expect_error(fastrerandomize:::.resolve_jax_seed(1.5), "whole number")

  set.seed(2026)
  seed_a <- fastrerandomize:::.resolve_jax_seed(NULL)
  set.seed(2026)
  seed_b <- fastrerandomize:::.resolve_jax_seed(NULL)
  expect_equal(seed_a, seed_b)
})

test_that("diagnostic print retains R2 when sigma is absent", {
  diagnostic <- diagnose_rerandomization(
    smd = c(0.1, 0.2), n_T = 10, n_C = 10, R2 = 0.4
  )
  printed <- capture.output(print(diagnostic))
  expect_true(any(grepl("sigma = not supplied, R\\^2 = 0.4", printed)))
})

test_that("randomization_test validates required R inputs before JAX", {
  candidates <- matrix(c(1, 0), nrow = 1)
  expect_error(
    randomization_test(
      obsW = NULL,
      obsY = c(1, 2),
      candidate_randomizations = candidates
    ),
    "'obsW' is required"
  )
  expect_error(
    randomization_test(
      obsW = c(1, 0),
      obsY = NULL,
      candidate_randomizations = candidates
    ),
    "'obsY' is required"
  )
  expect_error(
    randomization_test(
      obsW = c(1, 0),
      obsY = c(1, 2),
      alpha = 1,
      candidate_randomizations = candidates
    ),
    "strictly between 0 and 1"
  )
})
