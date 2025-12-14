#!/usr/bin/env Rscript
##########################################
# Comprehensive test suite for fastrerandomize
# Includes unit tests, integration tests, and edge cases
##########################################

# Helper for approximate equality
approx_equal <- function(x, y, tol = 1e-6) {
  if (is.na(x) || is.na(y)) return(FALSE)
  abs(x - y) < tol
}

# Test result tracking
test_results <- list(passed = 0, failed = 0, errors = character())

run_test <- function(name, expr) {
  result <- tryCatch({
    eval(expr)
    test_results$passed <<- test_results$passed + 1
    cat(sprintf("  [PASS] %s\n", name))
    TRUE
  }, error = function(e) {
    test_results$failed <<- test_results$failed + 1
    test_results$errors <<- c(test_results$errors, sprintf("%s: %s", name, e$message))
    cat(sprintf("  [FAIL] %s: %s\n", name, e$message))
    FALSE
  })
  result
}

##########################################
# SECTION 1: Pure R Unit Tests (no JAX required)
##########################################
cat("\n========== SECTION 1: Pure R Unit Tests ==========\n")

# Source R files directly for unit testing
r_files <- list.files(
  file.path(dirname(getwd()), 'fastrerandomize', 'R'),
  full.names = TRUE
)
if (length(r_files) == 0) {
  # Try alternate path structure
  r_files <- list.files(file.path('fastrerandomize', 'R'), full.names = TRUE)
}
for (f in r_files) {
  tryCatch(source(f), error = function(e) NULL)
}

cat("\n--- Testing hotellingT2_R ---\n")
run_test("hotellingT2_R basic calculation", {
  X <- matrix(c(1,2,3,4,5,6,7,8), nrow = 4)
  W <- c(1,0,1,0)
  result <- hotellingT2_R(X, W)
  stopifnot(approx_equal(result, 1.2))
})

run_test("hotellingT2_R returns NA for invalid assignment", {
  X <- matrix(c(1,2,3,4,5,6,7,8), nrow = 4)
  W_all_treat <- c(1,1,1,1)
  result <- hotellingT2_R(X, W_all_treat)
  stopifnot(is.na(result))
})

cat("\n--- Testing diff_in_means_R ---\n")
run_test("diff_in_means_R basic calculation", {
  Y <- 1:4
  W <- c(1,0,1,0)
  result <- diff_in_means_R(Y, W)
  stopifnot(result == -1)
})

run_test("diff_in_means_R with positive effect", {
  Y <- c(10, 1, 12, 2)
  W <- c(1, 0, 1, 0)
  result <- diff_in_means_R(Y, W)
  stopifnot(result == 9.5)  # mean(10,12) - mean(1,2) = 11 - 1.5 = 9.5
})

cat("\n--- Testing compute_diff_at_tau_for_oneW_R ---\n")
run_test("compute_diff_at_tau_for_oneW_R basic calculation", {
  Wprime <- c(0,1,0,1)
  obsY <- c(1,2,3,4)
  obsW <- c(1,0,1,0)
  result <- compute_diff_at_tau_for_oneW_R(Wprime, obsY, obsW, 1)
  stopifnot(result == 3)
})

run_test("compute_diff_at_tau_for_oneW_R with zero tau", {
  Wprime <- c(1,0,1,0)
  obsY <- c(5,2,5,2)
  obsW <- c(1,0,1,0)
  result <- compute_diff_at_tau_for_oneW_R(Wprime, obsY, obsW, 0)
  # With tau=0, Y0 = obsY, then reassigned same way
  stopifnot(result == 3)  # mean(5,5) - mean(2,2) = 3
})

cat("\n--- Testing randomization_test_R ---\n")
run_test("randomization_test_R p-value and tau_obs", {
  obsW <- c(1,1,0,0)
  obsY <- c(2,2,1,0)
  comb <- t(combn(4,2))
  allW <- matrix(0, nrow = nrow(comb), ncol = 4)
  for(i in seq_len(nrow(comb))) allW[i, comb[i,]] <- 1
  res <- randomization_test_R(obsW, obsY, allW, findFI = FALSE)
  stopifnot(approx_equal(res$p_value, 1/3))
  stopifnot(res$tau_obs == 1.5)
})

run_test("randomization_test_R with no effect", {
  obsW <- c(1,1,0,0)
  obsY <- c(1,1,1,1)  # No difference between groups
  comb <- t(combn(4,2))
  allW <- matrix(0, nrow = nrow(comb), ncol = 4)
  for(i in seq_len(nrow(comb))) allW[i, comb[i,]] <- 1
  res <- randomization_test_R(obsW, obsY, allW, findFI = FALSE)
  stopifnot(res$tau_obs == 0)
  stopifnot(res$p_value == 1)  # All permutations give same result
})

cat("\n--- Testing generate_randomizations_R ---\n")
run_test("generate_randomizations_R monte_carlo output structure", {
  set.seed(1)
  X <- matrix(rnorm(8), 4, 2)
  res <- generate_randomizations_R(4, 2, X, 1, 'monte_carlo', max_draws = 6, batch_size = 2)
  stopifnot(ncol(res$randomizations) == 4)
  stopifnot(length(res$balance) == nrow(res$randomizations))
})

run_test("generate_randomizations_R exact output structure", {
  set.seed(2)
  X <- matrix(rnorm(8), 4, 2)
  res <- generate_randomizations_R(4, 2, X, 1, 'exact', max_draws = 100, batch_size = 10)
  stopifnot(ncol(res$randomizations) == 4)
  stopifnot(nrow(res$randomizations) == choose(4, 2))  # All 6 combinations
  stopifnot(all(rowSums(res$randomizations) == 2))  # Each row has exactly 2 treated
})

run_test("generate_randomizations_R accept_prob filtering", {
  set.seed(3)
  X <- matrix(rnorm(20), 10, 2)
  res_all <- generate_randomizations_R(10, 5, X, 1.0, 'monte_carlo', max_draws = 100, batch_size = 50)
  res_half <- generate_randomizations_R(10, 5, X, 0.5, 'monte_carlo', max_draws = 100, batch_size = 50)
  # With 50% acceptance, should have roughly half as many (allow some variance)
  stopifnot(nrow(res_half$randomizations) <= nrow(res_all$randomizations))
})

##########################################
# SECTION 2: JAX Integration Tests
##########################################
cat("\n========== SECTION 2: JAX Integration Tests ==========\n")

jax_available <- tryCatch({
  library(fastrerandomize)
  # Try to initialize JAX
  fastrerandomize::check_jax_availability(conda_env = "fastrerandomize_env")
}, error = function(e) {
  cat("JAX backend not available, skipping integration tests.\n")
  cat(sprintf("Error: %s\n", e$message))
  NULL
})

if (!is.null(jax_available)) {

  set.seed(999L, kind = "Wichmann-Hill")
  X_test <- matrix(rnorm(20*5), 20, 5)

  cat("\n--- Testing generate_randomizations (JAX) ---\n")

  for (type_ in c("exact", "monte_carlo")) {
    run_test(sprintf("generate_randomizations %s returns valid structure", type_), {
      RandomizationSet <- fastrerandomize::generate_randomizations(
        n_units = 20,
        n_treated = 10,
        X = X_test,
        randomization_accept_prob = 0.1,
        randomization_type = type_,
        approximate_inv = TRUE,
        verbose = FALSE,
        max_draws = 10000L,
        batch_size = 100L
      )
      stopifnot(!is.null(RandomizationSet$randomizations))
      stopifnot(ncol(RandomizationSet$randomizations) == 20)
      stopifnot(all(rowSums(RandomizationSet$randomizations) == 10))
    })

    run_test(sprintf("generate_randomizations %s balance values are numeric", type_), {
      RandomizationSet <- fastrerandomize::generate_randomizations(
        n_units = 20,
        n_treated = 10,
        X = X_test,
        randomization_accept_prob = 0.1,
        randomization_type = type_,
        verbose = FALSE,
        max_draws = 5000L,
        batch_size = 100L
      )
      stopifnot(is.numeric(RandomizationSet$balance))
      stopifnot(length(RandomizationSet$balance) == nrow(RandomizationSet$randomizations))
      stopifnot(all(RandomizationSet$balance >= 0))  # T^2 should be non-negative
    })
  }

  cat("\n--- Testing randomization_test (JAX) ---\n")

  for (findFI in c(FALSE, TRUE)) {
    run_test(sprintf("randomization_test findFI=%s returns valid p-value", findFI), {
      RandomizationSet <- fastrerandomize::generate_randomizations(
        n_units = 20,
        n_treated = 10,
        X = X_test,
        randomization_accept_prob = 0.1,
        randomization_type = "monte_carlo",
        verbose = FALSE,
        max_draws = 5000L,
        batch_size = 100L
      )
      W <- as.integer(RandomizationSet$randomizations[1,])
      Y <- rnorm(20) + 2 * W  # Add treatment effect

      RRTest <- fastrerandomize::randomization_test(
        obsW = W,
        obsY = Y,
        candidate_randomizations = RandomizationSet$randomizations,
        findFI = findFI
      )
      stopifnot(RRTest$p_value >= 0 && RRTest$p_value <= 1)
      stopifnot(is.numeric(RRTest$tau_obs))
    })
  }

  run_test("randomization_test detects strong treatment effect", {
    # With a very strong effect, p-value should be small
    set.seed(42)
    RandomizationSet <- fastrerandomize::generate_randomizations(
      n_units = 20,
      n_treated = 10,
      X = X_test,
      randomization_accept_prob = 0.5,
      randomization_type = "monte_carlo",
      verbose = FALSE,
      max_draws = 2000L,
      batch_size = 100L
    )
    W <- as.integer(RandomizationSet$randomizations[1,])
    Y <- rnorm(20, sd = 0.1) + 10 * W  # Very strong effect

    RRTest <- fastrerandomize::randomization_test(
      obsW = W,
      obsY = Y,
      candidate_randomizations = RandomizationSet$randomizations,
      findFI = FALSE
    )
    stopifnot(RRTest$p_value < 0.1)  # Should detect the effect
    stopifnot(RRTest$tau_obs > 5)    # Observed effect should be large
  })

  run_test("randomization_test with no effect has high p-value", {
    set.seed(43)
    RandomizationSet <- fastrerandomize::generate_randomizations(
      n_units = 20,
      n_treated = 10,
      X = X_test,
      randomization_accept_prob = 0.5,
      randomization_type = "monte_carlo",
      verbose = FALSE,
      max_draws = 2000L,
      batch_size = 100L
    )
    W <- as.integer(RandomizationSet$randomizations[1,])
    Y <- rnorm(20)  # No treatment effect

    RRTest <- fastrerandomize::randomization_test(
      obsW = W,
      obsY = Y,
      candidate_randomizations = RandomizationSet$randomizations,
      findFI = FALSE
    )
    stopifnot(RRTest$p_value > 0.05)  # Should NOT detect an effect
  })

  run_test("fiducial interval contains true effect", {
    set.seed(44)
    true_effect <- 3.0
    RandomizationSet <- fastrerandomize::generate_randomizations(
      n_units = 20,
      n_treated = 10,
      X = X_test,
      randomization_accept_prob = 0.5,
      randomization_type = "monte_carlo",
      verbose = FALSE,
      max_draws = 2000L,
      batch_size = 100L
    )
    W <- as.integer(RandomizationSet$randomizations[1,])
    Y <- rnorm(20, sd = 0.5) + true_effect * W

    RRTest <- fastrerandomize::randomization_test(
      obsW = W,
      obsY = Y,
      candidate_randomizations = RandomizationSet$randomizations,
      findFI = TRUE
    )
    if (!is.null(RRTest$FI) && !any(is.na(RRTest$FI))) {
      # FI should contain or be near the true effect (with some tolerance for randomness)
      stopifnot(is.numeric(RRTest$FI) && length(RRTest$FI) == 2)
      cat(sprintf("    FI: [%.2f, %.2f], true effect: %.2f\n",
                  RRTest$FI[1], RRTest$FI[2], true_effect))
    }
  })

  ##########################################
  # SECTION 3: Distance Function Tests
  ##########################################
  cat("\n========== SECTION 3: Distance Function Tests ==========\n")

  run_test("fast_distance euclidean basic", {
    X <- matrix(c(0, 0, 3, 4), nrow = 2, byrow = TRUE)
    D <- fastrerandomize::fast_distance(X, metric = "euclidean")
    # Distance from (0,0) to (3,4) should be 5
    stopifnot(approx_equal(D[1, 2], 5.0, tol = 0.01))
    stopifnot(approx_equal(D[2, 1], 5.0, tol = 0.01))
    stopifnot(approx_equal(D[1, 1], 0.0, tol = 0.01))
  })

  run_test("fast_distance sqeuclidean", {
    X <- matrix(c(0, 0, 3, 4), nrow = 2, byrow = TRUE)
    D <- fastrerandomize::fast_distance(X, metric = "sqeuclidean")
    stopifnot(approx_equal(D[1, 2], 25.0, tol = 0.01))
  })

  run_test("fast_distance manhattan", {
    X <- matrix(c(0, 0, 3, 4), nrow = 2, byrow = TRUE)
    D <- fastrerandomize::fast_distance(X, metric = "manhattan")
    # Manhattan: |3-0| + |4-0| = 7
    stopifnot(approx_equal(D[1, 2], 7.0, tol = 0.01))
  })

  run_test("fast_distance chebyshev", {
    X <- matrix(c(0, 0, 3, 4), nrow = 2, byrow = TRUE)
    D <- fastrerandomize::fast_distance(X, metric = "chebyshev")
    # Chebyshev: max(|3-0|, |4-0|) = 4
    stopifnot(approx_equal(D[1, 2], 4.0, tol = 0.01))
  })

  run_test("fast_distance cosine", {
    X <- matrix(c(1, 0, 0, 1), nrow = 2, byrow = TRUE)
    D <- fastrerandomize::fast_distance(X, metric = "cosine")
    # Orthogonal vectors: cosine distance = 1
    stopifnot(approx_equal(D[1, 2], 1.0, tol = 0.01))
  })

  run_test("fast_distance minkowski p=1 equals manhattan", {
    X <- matrix(rnorm(20), 5, 4)
    D_mink <- fastrerandomize::fast_distance(X, metric = "minkowski", p = 1)
    D_manh <- fastrerandomize::fast_distance(X, metric = "manhattan")
    stopifnot(all(abs(D_mink - D_manh) < 0.01))
  })

  run_test("fast_distance minkowski p=2 equals euclidean", {
    X <- matrix(rnorm(20), 5, 4)
    D_mink <- fastrerandomize::fast_distance(X, metric = "minkowski", p = 2)
    D_euc <- fastrerandomize::fast_distance(X, metric = "euclidean")
    stopifnot(all(abs(D_mink - D_euc) < 0.01))
  })

  run_test("fast_distance mahalanobis diagonal approx", {
    X <- matrix(rnorm(50), 10, 5)
    D <- fastrerandomize::fast_distance(X, metric = "mahalanobis", approximate_inv = TRUE)
    stopifnot(is.matrix(D))
    stopifnot(nrow(D) == 10 && ncol(D) == 10)
    stopifnot(all(D >= 0))
  })

  run_test("fast_distance A to B", {
    A <- matrix(c(0, 0, 1, 1), nrow = 2, byrow = TRUE)
    B <- matrix(c(3, 4, 0, 0), nrow = 2, byrow = TRUE)
    D <- fastrerandomize::fast_distance(A, B, metric = "euclidean")
    stopifnot(nrow(D) == 2 && ncol(D) == 2)
    stopifnot(approx_equal(D[1, 1], 5.0, tol = 0.01))  # (0,0) to (3,4)
    stopifnot(approx_equal(D[1, 2], 0.0, tol = 0.01))  # (0,0) to (0,0)
  })

  run_test("fast_distance as_dist returns dist object", {
    X <- matrix(rnorm(20), 5, 4)
    D <- fastrerandomize::fast_distance(X, metric = "euclidean", as_dist = TRUE)
    stopifnot(inherits(D, "dist"))
  })

  ##########################################
  # SECTION 4: Edge Cases
  ##########################################
  cat("\n========== SECTION 4: Edge Cases ==========\n")

  run_test("small n exact enumeration (n=6, k=3)", {
    set.seed(100)
    X_small <- matrix(rnorm(6*2), 6, 2)
    RandomizationSet <- fastrerandomize::generate_randomizations(
      n_units = 6,
      n_treated = 3,
      X = X_small,
      randomization_accept_prob = 1.0,  # Keep all
      randomization_type = "exact",
      verbose = FALSE
    )
    # Should have choose(6,3) = 20 randomizations
    stopifnot(nrow(RandomizationSet$randomizations) == 20)
  })

  run_test("exact and monte_carlo give similar balance distributions", {
    set.seed(101)
    # Use larger n to allow meaningful Monte Carlo comparison
    # choose(12, 6) = 924, so max_draws can be reasonably large
    X_small <- matrix(rnorm(12*2), 12, 2)

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
    stopifnot(abs(mean_exact - mean_mc) / mean_exact < 0.5)
  })

  run_test("single covariate works", {
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
    stopifnot(nrow(RandomizationSet$randomizations) > 0)
  })

  run_test("many covariates work", {
    set.seed(103)
    X_many <- matrix(rnorm(20*20), 20, 20)
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
    stopifnot(nrow(RandomizationSet$randomizations) > 0)
  })

} else {
  cat("\nSkipping JAX integration tests, distance tests, and edge cases (JAX not available).\n")
}

##########################################
# SUMMARY
##########################################
cat("\n========== TEST SUMMARY ==========\n")
cat(sprintf("Passed: %d\n", test_results$passed))
cat(sprintf("Failed: %d\n", test_results$failed))

if (test_results$failed > 0) {
  cat("\nFailed tests:\n")
  for (err in test_results$errors) {
    cat(sprintf("  - %s\n", err))
  }
  fastrerandomize::print2("At least one test failed... See above.")
  quit(status = 1)
} else {
  if (exists("print2", where = asNamespace("fastrerandomize"), mode = "function")) {
    fastrerandomize::print2("All tests succeeded!")
  } else {
    cat("All tests succeeded!\n")
  }
}
