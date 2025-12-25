# Helper functions for fastrerandomize tests
# This file is automatically loaded by testthat before running tests

#' Check if two values are approximately equal
#' @param x First value
#' @param y Second value
#' @param tol Tolerance for comparison (default 1e-6)
#' @return TRUE if values are within tolerance, FALSE otherwise
approx_equal <- function(x, y, tol = 1e-6) {
  if (is.na(x) || is.na(y)) return(FALSE)
  abs(x - y) < tol
}

#' Check if JAX backend is available
#' @return TRUE if JAX is available, FALSE otherwise
jax_is_available <- function() {
  result <- tryCatch({
    fastrerandomize::check_jax_availability(conda_env = "fastrerandomize_env")
    TRUE
  }, error = function(e) {
    FALSE
  })
  result
}

#' Skip test if JAX is not available
skip_if_no_jax <- function() {
  if (!jax_is_available()) {
    testthat::skip("JAX backend not available")
  }
}
