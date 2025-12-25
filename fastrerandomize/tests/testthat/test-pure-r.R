# Pure R Unit Tests (no JAX required)
# Tests for base R implementations with _R suffix

# Source R files directly for unit testing of internal functions
local({
  r_dir <- system.file("R", package = "fastrerandomize")
  if (r_dir == "") {
    # Try alternate path for development
    r_dir <- file.path(getwd(), "R")
  }
  if (dir.exists(r_dir)) {
    r_files <- list.files(r_dir, pattern = "\\.R$", full.names = TRUE)
    for (f in r_files) {
      tryCatch(source(f, local = TRUE), error = function(e) NULL)
    }
  }
})

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
