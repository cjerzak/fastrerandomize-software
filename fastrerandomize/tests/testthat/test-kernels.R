# GEMM-form kernel tests
# Monte Carlo reproducibility against the legacy key stream, batch invariance,
# closed-form fiducial search, and GEMM-form distances

skip_if_no_jax()

env <- function() fastrerandomize:::fastrr_env
to_r <- function(x) fastrerandomize:::.to_r(x)

# Legacy Monte Carlo stream: draw i permutes the base vector with key i of
# jax.random.split(PRNGKey(seed), max_draws); balance in double precision
legacy_mc <- function(X, n_treated, max_draws, seed, approximate_inv) {
  e <- env()
  invisible(fastrerandomize:::.ensure_backend("fastrerandomize_env", TRUE))
  n <- nrow(X)
  base <- e$jnp$array(c(rep(1L, n_treated), rep(0L, n - n_treated)), dtype = e$jnp$int32)
  keys <- e$jax$random$split(e$jax$random$PRNGKey(as.integer(seed)), as.integer(max_draws))
  W <- to_r(e$jax$jit(e$BatchPermutation)(keys, base))
  n0 <- n - n_treated
  diff <- (W %*% X) / n_treated - ((1 - W) %*% X) / n0
  S_inv <- if (approximate_inv) {
    diag(1 / colMeans(sweep(X, 2, colMeans(X))^2), nrow = ncol(X))
  } else {
    solve(stats::cov(X))
  }
  list(W = W, keys = to_r(keys), M = n0 * n_treated / n * rowSums((diff %*% S_inv) * diff))
}

key_id <- function(keys) apply(keys, 1, paste, collapse = "_")

test_that("counter-derived keys equal jax.random.split", {
  invisible(fastrerandomize:::.ensure_backend("fastrerandomize_env", TRUE))
  e <- env()
  expect_true(e$counter_keys_ok)
  key <- e$jax$random$PRNGKey(123L)
  expect_equal(
    to_r(e$CounterKeysJit(key, e$jnp$arange(1000L, dtype = e$jnp$uint32))),
    to_r(e$jax$random$split(key, 1000L))
  )
})

test_that("monte_carlo reproduces the legacy key stream and selection", {
  set.seed(301)
  for (n in c(255L, 256L, 300L)) {
    for (approximate_inv in c(TRUE, FALSE)) {
      X <- matrix(rnorm(n * 3), n) + 50
      res <- fastrerandomize::generate_randomizations_mc(
        n_units = n, n_treated = n %/% 2, X = X, randomization_accept_prob = 0.02,
        max_draws = 2000L, approximate_inv = approximate_inv, seed = 77L, verbose = FALSE
      )
      ref <- legacy_mc(X, n %/% 2, 2000L, 77L, approximate_inv)
      W <- to_r(res$candidate_randomizations)
      M <- as.numeric(to_r(res$M_candidate_randomizations))
      rows <- match(key_id(to_r(res$keys_candidate_randomizations)), key_id(ref$keys))

      expect_false(anyNA(rows))
      expect_true(all(rowSums(W) == n %/% 2))
      expect_equal(unname(W), unname(ref$W[rows, ]))
      expect_equal(M, ref$M[rows], tolerance = 1e-4)
      # Same accepted set as the legacy stable sort, up to float32 near-ties at the cutoff
      ref_rows <- order(ref$M)[seq_len(40L)]
      cutoff <- ref$M[ref_rows[40L]]
      differ <- union(setdiff(rows, ref_rows), setdiff(ref_rows, rows))
      expect_true(all(abs(ref$M[differ] - cutoff) <= 1e-4 * cutoff))
    }
  }
})

test_that("monte_carlo results do not depend on batch size or dispatch mode", {
  set.seed(302)
  X <- matrix(rnorm(60 * 4), 60)
  run <- function(batch_size) {
    fastrerandomize::generate_randomizations_mc(
      n_units = 60, n_treated = 25, X = X, randomization_accept_prob = 0.05,
      max_draws = 3000L, batch_size = batch_size, seed = 5L, verbose = FALSE
    )
  }
  base <- run(NULL)
  op <- options(fastrerandomize.mc_fused = FALSE)
  on.exit(options(op), add = TRUE)
  variants <- list(run(7L), run(64L), run(NULL))
  for (v in variants) {
    expect_identical(to_r(v$candidate_randomizations), to_r(base$candidate_randomizations))
    expect_identical(to_r(v$keys_candidate_randomizations), to_r(base$keys_candidate_randomizations))
    expect_equal(to_r(v$M_candidate_randomizations), to_r(base$M_candidate_randomizations),
                 tolerance = 1e-5)
  }
})

test_that("monte_carlo edge cases", {
  set.seed(303)
  X <- matrix(rnorm(20 * 2), 20)
  for (n_treated in c(1L, 19L)) {
    r <- fastrerandomize::generate_randomizations(
      n_units = 20, n_treated = n_treated, X = X, randomization_accept_prob = 0.1,
      max_draws = 500L, seed = 1L, verbose = FALSE
    )
    expect_true(all(rowSums(r$randomizations) == n_treated))
  }

  all_draws <- fastrerandomize::generate_randomizations(
    n_units = 20, n_treated = 10, X = X[, 1, drop = FALSE], randomization_accept_prob = 1,
    max_draws = 50L, batch_size = 1000L, seed = 2L, verbose = FALSE
  )
  expect_equal(nrow(all_draws$randomizations), 50)
  expect_false(is.unsorted(all_draws$balance))

  expect_warning(
    constant <- fastrerandomize::generate_randomizations(
      n_units = 20, n_treated = 10, X = matrix(3, 20, 2), randomization_accept_prob = 0.1,
      max_draws = 200L, seed = 3L, verbose = FALSE
    ),
    "All covariates"
  )
  expect_true(all(constant$balance == 0))

  expect_error(
    fastrerandomize::generate_randomizations(
      n_units = 20, n_treated = 10, X = replace(X, 3, NA), randomization_accept_prob = 0.1,
      max_draws = 200L, verbose = FALSE
    ),
    "finite"
  )
})

test_that("a custom threshold_func uses the legacy signature", {
  set.seed(304)
  X <- matrix(rnorm(40 * 3), 40)
  legacy <- function(X_, S_inv_, S_inv_approx_, W_, n0, n1, approximate_inv_) {
    env()$VectorizedFastHotel2T2(X_, S_inv_, S_inv_approx_, W_, n0, n1, approximate_inv_)
  }
  args <- list(n_units = 40, n_treated = 20, X = X, randomization_accept_prob = 0.05,
               max_draws = 2000L, seed = 9L, verbose = FALSE)
  default <- do.call(fastrerandomize::generate_randomizations, args)
  custom <- do.call(fastrerandomize::generate_randomizations, c(args, threshold_func = legacy))
  expect_true(all(rowSums(custom$randomizations) == 20))
  expect_gte(mean(key_id(custom$randomizations) %in% key_id(default$randomizations)), 0.95)
})

test_that("exact enumeration matches double-precision Hotelling T^2", {
  set.seed(305)
  X <- matrix(rnorm(10 * 3), 10) + 20
  for (approximate_inv in c(TRUE, FALSE)) {
    res <- fastrerandomize::generate_randomizations_exact(
      n_units = 10, n_treated = 5, X = X, randomization_accept_prob = 1,
      approximate_inv = approximate_inv, verbose = FALSE
    )
    W <- to_r(res$candidate_randomizations)
    expect_equal(nrow(W), choose(10, 5))
    expect_true(all(rowSums(W) == 5))
    ref <- if (approximate_inv) {
      v <- colMeans(sweep(X, 2, colMeans(X))^2)
      apply(W, 1, function(w) {
        d <- colMeans(X[w == 1, ]) - colMeans(X[w == 0, ])
        2.5 * sum(d^2 / v)
      })
    } else {
      apply(W, 1, function(w) hotellingT2_R(X, w))
    }
    expect_equal(as.numeric(to_r(res$M_candidate_randomizations)), ref, tolerance = 1e-4)
  }
})

# Translation of the pre-closed-form fiducial search (500 stochastic steps,
# then a 100-point grid), evaluated directly in double precision
legacy_fi <- function(obsW, obsY, cand, alpha = 0.05) {
  n0 <- sum(obsW == 0); n1 <- sum(obsW == 1)
  dim_ <- function(y, w) sum(y * w) / n1 - sum(y * (1 - w)) / n0
  stat <- function(w, tau) dim_(obsY - obsW * tau + w * tau, w)
  tau_obs <- dim_(obsY, obsW)
  bounds <- fastrerandomize:::.fi_initial_bounds(tau_obs)
  lo <- bounds[1]; hi <- bounds[2]
  z <- stats::qnorm(1 - alpha)
  k <- 2 / (z * (2 * pi)^(-1/2) * exp(-z^2 / 2))
  for (t in 1:500) {
    w <- cand[sample(1:nrow(cand), size = 1), ]
    s_lo <- stat(w, lo); c_lo <- k * (tau_obs - lo)
    lo <- if (s_lo < tau_obs) lo + c_lo * (alpha / 2) / t else lo - c_lo * (1 - alpha / 2) / t
    s_hi <- stat(w, hi); c_hi <- k * (hi - tau_obs)
    hi <- if (s_hi > tau_obs) hi - c_hi * (alpha / 2) / t else hi + c_hi * (1 - alpha / 2) / t
  }
  grid <- fastrerandomize:::.fi_search_grid(c(lo, hi), tau_obs, length.out = 100L)
  pv <- sapply(grid, function(tp) {
    s <- apply(cand, 1, stat, tau = tp)
    min(mean(tau_obs >= s), mean(tau_obs <= s))
  })
  list(FI = fastrerandomize:::.fi_accepted_range(grid, pv, alpha), step = diff(grid)[1])
}

test_that("closed-form fiducial interval matches the legacy search", {
  set.seed(306)
  n <- 40
  cand <- t(replicate(400, sample(rep(c(1, 0), each = n / 2))))
  obsW <- sample(rep(c(1, 0), each = n / 2))
  obsY <- rnorm(n) + 1.5 * obsW

  set.seed(99)
  res <- fastrerandomize::randomization_test(obsW = obsW, obsY = obsY,
                                             candidate_randomizations = cand, findFI = TRUE)
  set.seed(99)
  ref <- legacy_fi(obsW, obsY, cand)
  expect_true(all(abs(res$FI - ref$FI) <= 2 * ref$step))
})

test_that("p-values count exact ties like the double-precision implementation", {
  set.seed(307)
  n <- 30
  obsW <- sample(rep(c(1, 0), each = n / 2))
  obsY <- rnorm(n, mean = 100) + obsW
  cand <- rbind(obsW, 1 - obsW, t(replicate(200, sample(obsW))))
  res <- fastrerandomize::randomization_test(obsW = obsW, obsY = obsY,
                                             candidate_randomizations = cand)
  ref <- randomization_test_R(obsW, obsY, cand)
  expect_identical(res$p_value, ref$p_value)
  expect_identical(res$tau_obs, ref$tau_obs)

  # Device-array input is not copied back to R
  arr <- env()$jnp$array(cand, dtype = env()$jnp$int8)
  res_arr <- fastrerandomize::randomization_test(obsW = obsW, obsY = obsY,
                                                 candidate_randomizations_array = arr)
  expect_identical(res_arr$p_value, ref$p_value)
  expect_null(res_arr$candidate_randomizations)
  expect_false(is.null(res_arr$candidate_randomizations_array))

  expect_error(
    fastrerandomize::randomization_test(obsW = obsW, obsY = obsY,
                                        candidate_randomizations = cand * 0.5 + 0.25),
    "only 0 and 1"
  )
})

test_that("GEMM-form distances match double-precision references", {
  set.seed(308)
  A <- matrix(rnorm(30 * 4), 30) + 1000
  B <- matrix(rnorm(20 * 4), 20) + 1000
  ref_sq <- outer(seq_len(30), seq_len(20), Vectorize(function(i, j) sum((A[i, ] - B[j, ])^2)))

  expect_equal(fastrerandomize::fast_distance(A, B, metric = "sqeuclidean"), ref_sq,
               tolerance = 1e-5)
  expect_equal(fastrerandomize::fast_distance(A, B, metric = "euclidean", row_batch_size = 7L),
               sqrt(ref_sq + 1e-12), tolerance = 1e-5)

  D_self <- fastrerandomize::fast_distance(A, metric = "sqeuclidean")
  expect_equal(diag(D_self), rep(0, 30))
  expect_equal(D_self, as.matrix(stats::dist(A))^2, tolerance = 1e-5, ignore_attr = TRUE)

  pooled <- rbind(A, B)
  S_inv <- solve(stats::cov(pooled))
  ref_mahal <- outer(seq_len(30), seq_len(20), Vectorize(function(i, j) {
    d <- A[i, ] - B[j, ]
    drop(t(d) %*% S_inv %*% d)
  }))
  expect_equal(fastrerandomize::fast_distance(A, B, metric = "mahalanobis",
                                              approximate_inv = FALSE, squared = TRUE),
               ref_mahal, tolerance = 1e-4)
  expect_equal(fastrerandomize::fast_distance(A, B, metric = "mahalanobis", cov_inv = S_inv,
                                              approximate_inv = FALSE, squared = TRUE),
               ref_mahal, tolerance = 1e-4)

  # Indefinite user cov_inv keeps the direct quadratic form
  C <- diag(c(1, -1, 1, 1))
  ref_indef <- outer(seq_len(30), seq_len(20), Vectorize(function(i, j) {
    d <- A[i, ] - B[j, ]
    drop(t(d) %*% C %*% d)
  }))
  expect_equal(fastrerandomize::fast_distance(A, B, metric = "mahalanobis", cov_inv = C,
                                              approximate_inv = FALSE, squared = TRUE),
               ref_indef, tolerance = 1e-3)

  x1 <- matrix(c(0, 1, 3), ncol = 1)
  expect_equal(fastrerandomize::fast_distance(x1, metric = "sqeuclidean"),
               as.matrix(stats::dist(x1))^2, ignore_attr = TRUE, tolerance = 1e-6)
})
