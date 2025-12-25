# Distance Function Tests
# Tests for fast_distance() with various metrics

# Skip all tests if JAX is not available
skip_if_no_jax()

test_that("fast_distance euclidean basic works", {
  X <- matrix(c(0, 0, 3, 4), nrow = 2, byrow = TRUE)
  D <- fastrerandomize::fast_distance(X, metric = "euclidean")
  # Distance from (0,0) to (3,4) should be 5
  expect_equal(D[1, 2], 5.0, tolerance = 0.01)
  expect_equal(D[2, 1], 5.0, tolerance = 0.01)
  expect_equal(D[1, 1], 0.0, tolerance = 0.01)
})

test_that("fast_distance sqeuclidean works", {
  X <- matrix(c(0, 0, 3, 4), nrow = 2, byrow = TRUE)
  D <- fastrerandomize::fast_distance(X, metric = "sqeuclidean")
  expect_equal(D[1, 2], 25.0, tolerance = 0.01)
})

test_that("fast_distance manhattan works", {
  X <- matrix(c(0, 0, 3, 4), nrow = 2, byrow = TRUE)
  D <- fastrerandomize::fast_distance(X, metric = "manhattan")
  # Manhattan: |3-0| + |4-0| = 7
  expect_equal(D[1, 2], 7.0, tolerance = 0.01)
})

test_that("fast_distance chebyshev works", {
  X <- matrix(c(0, 0, 3, 4), nrow = 2, byrow = TRUE)
  D <- fastrerandomize::fast_distance(X, metric = "chebyshev")
  # Chebyshev: max(|3-0|, |4-0|) = 4
  expect_equal(D[1, 2], 4.0, tolerance = 0.01)
})

test_that("fast_distance cosine works", {
  X <- matrix(c(1, 0, 0, 1), nrow = 2, byrow = TRUE)
  D <- fastrerandomize::fast_distance(X, metric = "cosine")
  # Orthogonal vectors: cosine distance = 1
  expect_equal(D[1, 2], 1.0, tolerance = 0.01)
})

test_that("fast_distance minkowski p=1 equals manhattan", {
  X <- matrix(rnorm(20), 5, 4)
  D_mink <- fastrerandomize::fast_distance(X, metric = "minkowski", p = 1)
  D_manh <- fastrerandomize::fast_distance(X, metric = "manhattan")
  expect_equal(D_mink, D_manh, tolerance = 0.01)
})

test_that("fast_distance minkowski p=2 equals euclidean", {
  X <- matrix(rnorm(20), 5, 4)
  D_mink <- fastrerandomize::fast_distance(X, metric = "minkowski", p = 2)
  D_euc <- fastrerandomize::fast_distance(X, metric = "euclidean")
  expect_equal(D_mink, D_euc, tolerance = 0.01)
})

test_that("fast_distance mahalanobis diagonal approx works", {
  X <- matrix(rnorm(50), 10, 5)
  D <- fastrerandomize::fast_distance(X, metric = "mahalanobis",
                                      approximate_inv = TRUE)
  expect_true(is.matrix(D))
  expect_equal(nrow(D), 10)
  expect_equal(ncol(D), 10)
  expect_true(all(D >= 0))
})

test_that("fast_distance A to B works", {
  A <- matrix(c(0, 0, 1, 1), nrow = 2, byrow = TRUE)
  B <- matrix(c(3, 4, 0, 0), nrow = 2, byrow = TRUE)
  D <- fastrerandomize::fast_distance(A, B, metric = "euclidean")
  expect_equal(nrow(D), 2)
  expect_equal(ncol(D), 2)
  expect_equal(D[1, 1], 5.0, tolerance = 0.01)  # (0,0) to (3,4)
  expect_equal(D[1, 2], 0.0, tolerance = 0.01)  # (0,0) to (0,0)
})

test_that("fast_distance as_dist returns dist object", {
  X <- matrix(rnorm(20), 5, 4)
  D <- fastrerandomize::fast_distance(X, metric = "euclidean", as_dist = TRUE)
  expect_s3_class(D, "dist")
})
