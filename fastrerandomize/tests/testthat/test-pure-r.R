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

# ---------------------------------------------------------------------------
# Host-side helpers for the GEMM-form kernels
# ---------------------------------------------------------------------------
whitened_t2 <- function(Xw, W) {
  n1 <- sum(W)
  n0 <- length(W) - n1
  d <- colSums(Xw[W == 1, , drop = FALSE]) * (1 / n1 + 1 / n0) - colSums(Xw) / n0
  n0 * n1 / (n0 + n1) * sum(d^2)
}

test_that("whitened balance design reproduces Hotelling T^2 in double precision", {
  set.seed(11)
  for (X in list(matrix(rnorm(40 * 3), 40) + 100, matrix(rnorm(40), 40))) {
    W <- sample(rep(c(1, 0), each = 20))
    full <- fastrerandomize:::.balance_design(X, approximate_inv = FALSE)
    expect_equal(whitened_t2(full$Xw, W), hotellingT2_R(X, W), tolerance = 1e-10)

    d <- colMeans(X[W == 1, , drop = FALSE]) - colMeans(X[W == 0, , drop = FALSE])
    diag_ref <- 10 * sum(d^2 / colMeans(sweep(X, 2, colMeans(X))^2))
    approx <- fastrerandomize:::.balance_design(X, approximate_inv = TRUE)
    expect_equal(whitened_t2(approx$Xw, W), diag_ref, tolerance = 1e-10)
  }
})

test_that("diagonal-only preparation preserves the existing design exactly", {
  set.seed(14)
  X <- matrix(rnorm(40 * 3), 40)
  cases <- list(X, X + 1e9, X[, 1, drop = FALSE],
                cbind(X, X[, 1] + X[, 2]), matrix(rnorm(6 * 10), 6),
                suppressWarnings(fastrerandomize:::.prepare_X(cbind(X, 7), 40L)),
                suppressWarnings(fastrerandomize:::.prepare_X(matrix(7, 8, 2), 8L)))
  for (x in cases) {
    legacy <- fastrerandomize:::.balance_design(x, TRUE)
    fast <- fastrerandomize:::.balance_design(x, TRUE, need_full_inverse = FALSE)
    expect_identical(fast$Xw, legacy$Xw)
    expect_identical(fast$S_inv_approx, legacy$S_inv_approx)
    expect_null(fast$S_inv)
    expect_true(is.matrix(legacy$S_inv))
    expect_identical(fastrerandomize:::.balance_design(x, FALSE, need_full_inverse = FALSE),
                     fastrerandomize:::.balance_design(x, FALSE))
  }
})

test_that("diagonal-only preparation does not construct or decompose covariance", {
  testthat::local_mocked_bindings(
    cov = function(...) stop("unexpected covariance computation"), .package = "stats"
  )
  testthat::local_mocked_bindings(
    .psd_inv_sqrt = function(...) stop("unexpected covariance decomposition"),
    .package = "fastrerandomize"
  )
  X <- matrix(seq_len(24), 8)
  result <- fastrerandomize:::.balance_design(X, TRUE, need_full_inverse = FALSE)
  expect_equal(colMeans(result$Xw), rep(0, 3), tolerance = 1e-15)
  expect_equal(colMeans(result$Xw^2), rep(1, 3), tolerance = 1e-15)
})

test_that("full balance design uses a pseudo-inverse for collinear covariates", {
  set.seed(13)
  X2 <- matrix(rnorm(30 * 2), 30)
  X3 <- cbind(X2, X2[, 1] + X2[, 2])
  W <- sample(rep(c(1, 0), each = 15))
  expect_equal(
    whitened_t2(fastrerandomize:::.balance_design(X3, FALSE)$Xw, W),
    whitened_t2(fastrerandomize:::.balance_design(X2, FALSE)$Xw, W),
    tolerance = 1e-8
  )
})

test_that("covariate preparation drops constant columns and rejects non-finite values", {
  X <- cbind(1:6, 5, c(2, 4, 1, 3, 6, 5))
  expect_warning(prepared <- fastrerandomize:::.prepare_X(X, 6L), "Dropping 1 constant")
  expect_equal(prepared, X[, c(1, 3)])
  expect_warning(all_constant <- fastrerandomize:::.prepare_X(matrix(2, 6, 2), 6L),
                 "All covariates")
  expect_equal(all_constant, matrix(0, 6, 1))
  expect_error(fastrerandomize:::.prepare_X(replace(X, 2, NA), 6L), "finite")
  expect_error(fastrerandomize:::.prepare_X(X, 5L), "one row per unit")
})

test_that("closed-form fiducial statistic matches direct construction", {
  set.seed(12)
  obsW <- sample(rep(c(1, 0), c(12, 18)))
  Y <- rnorm(30) + 2 * obsW
  cand <- rbind(t(replicate(25, sample(obsW))), c(rep(1, 11), rep(0, 19)))  # last row: 11 treated
  G <- cbind(cand %*% (Y - mean(Y)), cand %*% obsW, rowSums(cand))
  fi <- fastrerandomize:::.fi_components(G, Y, obsW, n0 = 18, n1 = 12)
  for (tau in c(-1.5, 0, 0.7)) {
    direct <- apply(cand, 1, function(w) {
      Y0 <- Y - obsW * tau
      Yo <- (Y0 + w * tau) * w + Y0 * (1 - w)
      sum(Yo * w) / 12 - sum(Yo * (1 - w)) / 18
    })
    expect_equal(fi$a + tau * fi$slope, direct, tolerance = 1e-12)
  }
  expect_equal(fi$r, rowSums(cand))
  # The complement (18 treated) gives the exactly negated statistic
  expect_identical(fastrerandomize:::.dim64(Y, 1 - obsW, 12, 18),
                   -fastrerandomize:::.dim64(Y, obsW, 18, 12))
})

test_that("row-count invariant and batch sizing helpers behave", {
  expect_error(fastrerandomize:::.assert_valid_rows(3, 10L), "without exactly 10 treated units")
  expect_true(fastrerandomize:::.assert_valid_rows(0, 10L))
  expect_error(fastrerandomize:::.assert_reproducible(2, 40L), "different assignments for the same random key")
  expect_true(fastrerandomize:::.assert_reproducible(0, 40L))
  expect_equal(fastrerandomize:::.resolve_batch_size(NULL, 100L, 1e6), 32768L)
  expect_equal(fastrerandomize:::.resolve_batch_size(NULL, 1000L, 1e6), 4096L)
  expect_equal(fastrerandomize:::.resolve_batch_size(NULL, 1e6, 1e6), 1024L)
  expect_equal(fastrerandomize:::.resolve_batch_size(NULL, 100L, 50L), 50L)
  expect_equal(fastrerandomize:::.resolve_batch_size(777, 100L, 1e6), 777L)
})

test_that("JAX availability check is cached per conda environment", {
  env <- fastrerandomize:::fastrr_env
  old <- env$jax_checked_env
  on.exit(assign("jax_checked_env", old, envir = env), add = TRUE)
  env$jax_checked_env <- "cached_env_for_test"
  testthat::local_mocked_bindings(
    py_available = function(...) TRUE,
    use_condaenv = function(...) stop("conda should not be consulted"),
    .package = "reticulate"
  )
  expect_true(fastrerandomize::check_jax_availability("cached_env_for_test"))
  expect_message(
    expect_null(fastrerandomize::check_jax_availability("other_env")),
    "conda environment is not available"
  )
})
