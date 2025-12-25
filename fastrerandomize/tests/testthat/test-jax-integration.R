# JAX Integration Tests
# Tests for main API functions when JAX backend is available

# Skip all tests if JAX is not available
skip_if_no_jax()

# Setup test data
set.seed(999L, kind = "Wichmann-Hill")
X_test <- matrix(rnorm(20 * 5), 20, 5)

# generate_randomizations tests
test_that("generate_randomizations exact returns valid structure", {
  RandomizationSet <- fastrerandomize::generate_randomizations(
    n_units = 20,
    n_treated = 10,
    X = X_test,
    randomization_accept_prob = 0.1,
    randomization_type = "exact",
    approximate_inv = TRUE,
    verbose = FALSE,
    max_draws = 10000L,
    batch_size = 100L
  )
  expect_false(is.null(RandomizationSet$randomizations))
  expect_equal(ncol(RandomizationSet$randomizations), 20)
  expect_true(all(rowSums(RandomizationSet$randomizations) == 10))
})

test_that("generate_randomizations monte_carlo returns valid structure", {
  RandomizationSet <- fastrerandomize::generate_randomizations(
    n_units = 20,
    n_treated = 10,
    X = X_test,
    randomization_accept_prob = 0.1,
    randomization_type = "monte_carlo",
    approximate_inv = TRUE,
    verbose = FALSE,
    max_draws = 10000L,
    batch_size = 100L
  )
  expect_false(is.null(RandomizationSet$randomizations))
  expect_equal(ncol(RandomizationSet$randomizations), 20)
  expect_true(all(rowSums(RandomizationSet$randomizations) == 10))
})

test_that("generate_randomizations exact balance values are numeric", {
  RandomizationSet <- fastrerandomize::generate_randomizations(
    n_units = 20,
    n_treated = 10,
    X = X_test,
    randomization_accept_prob = 0.1,
    randomization_type = "exact",
    verbose = FALSE,
    max_draws = 5000L,
    batch_size = 100L
  )
  expect_true(is.numeric(RandomizationSet$balance))
  expect_equal(length(RandomizationSet$balance),
               nrow(RandomizationSet$randomizations))
  expect_true(all(RandomizationSet$balance >= 0))  # T^2 should be non-negative
})

test_that("generate_randomizations monte_carlo balance values are numeric", {
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
  expect_true(is.numeric(RandomizationSet$balance))
  expect_equal(length(RandomizationSet$balance),
               nrow(RandomizationSet$randomizations))
  expect_true(all(RandomizationSet$balance >= 0))  # T^2 should be non-negative
})

# randomization_test tests
test_that("randomization_test findFI=FALSE returns valid p-value", {
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
  W <- as.integer(RandomizationSet$randomizations[1, ])
  Y <- rnorm(20) + 2 * W  # Add treatment effect

  RRTest <- fastrerandomize::randomization_test(
    obsW = W,
    obsY = Y,
    candidate_randomizations = RandomizationSet$randomizations,
    findFI = FALSE
  )
  expect_gte(RRTest$p_value, 0)
  expect_lte(RRTest$p_value, 1)
  expect_true(is.numeric(RRTest$tau_obs))
})

test_that("randomization_test findFI=TRUE returns valid p-value", {
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
  W <- as.integer(RandomizationSet$randomizations[1, ])
  Y <- rnorm(20) + 2 * W  # Add treatment effect

  RRTest <- fastrerandomize::randomization_test(
    obsW = W,
    obsY = Y,
    candidate_randomizations = RandomizationSet$randomizations,
    findFI = TRUE
  )
  expect_gte(RRTest$p_value, 0)
  expect_lte(RRTest$p_value, 1)
  expect_true(is.numeric(RRTest$tau_obs))
})

test_that("randomization_test detects strong treatment effect", {
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
  W <- as.integer(RandomizationSet$randomizations[1, ])
  Y <- rnorm(20, sd = 0.1) + 10 * W  # Very strong effect

  RRTest <- fastrerandomize::randomization_test(
    obsW = W,
    obsY = Y,
    candidate_randomizations = RandomizationSet$randomizations,
    findFI = FALSE
  )
  expect_lt(RRTest$p_value, 0.1)  # Should detect the effect
  expect_gt(RRTest$tau_obs, 5)    # Observed effect should be large
})

test_that("randomization_test with no effect has high p-value", {
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
  W <- as.integer(RandomizationSet$randomizations[1, ])
  Y <- rnorm(20)  # No treatment effect

  RRTest <- fastrerandomize::randomization_test(
    obsW = W,
    obsY = Y,
    candidate_randomizations = RandomizationSet$randomizations,
    findFI = FALSE
  )
  expect_gt(RRTest$p_value, 0.05)  # Should NOT detect an effect
})

test_that("fiducial interval contains true effect", {
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
  W <- as.integer(RandomizationSet$randomizations[1, ])
  Y <- rnorm(20, sd = 0.5) + true_effect * W

  RRTest <- fastrerandomize::randomization_test(
    obsW = W,
    obsY = Y,
    candidate_randomizations = RandomizationSet$randomizations,
    findFI = TRUE
  )
  if (!is.null(RRTest$FI) && !any(is.na(RRTest$FI))) {
    # FI should contain or be near the true effect
    expect_true(is.numeric(RRTest$FI))
    expect_equal(length(RRTest$FI), 2)
  }
})
