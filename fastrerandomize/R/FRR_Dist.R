#' JAX-accelerated distance calculations
#'
#' @description
#' Compute pairwise distances between the rows of one matrix \code{A} or two matrices
#' \code{A} and \code{B}, using JAX-backed, JIT-compiled kernels. Supports common metrics:
#' Euclidean, squared Euclidean, Manhattan, Chebyshev, Cosine, Minkowski (with optional
#' feature weights), and Mahalanobis (with full or diagonal inverse covariance).
#'
#' The function automatically batches computations to avoid excessive device memory use.
#'
#' @param A A numeric matrix with rows as observations and columns as features.
#' @param B Optional numeric matrix with the same number of columns as \code{A}.
#'   If \code{NULL}, distances are computed within \code{A} (i.e., \eqn{n \times n}).
#' @param metric Character; one of
#'   \code{"euclidean"}, \code{"sqeuclidean"}, \code{"manhattan"},
#'   \code{"chebyshev"}, \code{"cosine"}, \code{"minkowski"}, \code{"mahalanobis"}.
#'   Default is \code{"euclidean"}.
#' @param p Numeric order for Minkowski distance (must be \eqn{>0}). Default is \code{2}.
#' @param weights Optional numeric vector of length \code{ncol(A)} with nonnegative
#'   feature weights. Used for \code{"minkowski"} and \code{"manhattan"} (the latter is
#'   equivalent to Minkowski with \code{p = 1}).
#' @param cov_inv Optional inverse covariance matrix (p x p) for Mahalanobis (ignored
#'   if \code{approximate_inv = TRUE}). If not supplied and \code{approximate_inv = FALSE},
#'   the covariance of \code{rbind(A, B)} is estimated and (pseudo-)inverted in double
#'   precision.
#' @param approximate_inv Logical; if \code{TRUE} and \code{metric = "mahalanobis"},
#'   uses a diagonal inverse (reciprocal variances) for speed and robustness.
#'   Default \code{TRUE}.
#' @param squared Logical; if \code{TRUE}, return squared distances when supported
#'   (\code{"euclidean"} and \code{"mahalanobis"}). Ignored for other metrics.
#'   Default \code{FALSE}.
#' @param row_batch_size Optional integer; number of rows of \code{A} to process per batch.
#'   If \code{NULL}, a safe size is chosen automatically.
#' @param as_dist Logical; if \code{TRUE} and \code{B} is \code{NULL}, return a base
#'   \code{\link[stats]{dist}} object (for symmetric metrics). Default \code{FALSE}.
#' @param return_type Either \code{"R"} (convert to base R matrix/\code{dist}) or
#'   \code{"jax"} (return a JAX array). Default \code{"R"}.
#' @param verbose Logical; print batching progress. Default \code{FALSE}.
#' @param conda_env Character; conda environment name used by \code{reticulate}.
#'   Default \code{"fastrerandomize_env"}.
#' @param conda_env_required Logical; whether the specified conda environment must be
#'   used. Default \code{TRUE}.
#'
#' @return An \eqn{n \times m} distance matrix in the format specified by \code{return_type}.
#'   If \code{as_dist = TRUE} and \code{B = NULL} (symmetric case), returns a
#'   \code{\link[stats]{dist}} object.
#'
#' @details
#' - **Mahalanobis**: with \code{approximate_inv = TRUE}, the diagonal of the pooled
#'   covariance is used (variance stabilizer); otherwise a full inverse covariance is used.
#' - **Weighted distances**: supply \code{weights} (length \code{p}) for
#'   \code{"minkowski"} and \code{"manhattan"} (the latter uses \code{p = 1}).
#' - Euclidean, squared Euclidean and Mahalanobis distances are computed as
#'   \eqn{\|a\|^2 + \|b\|^2 - 2 a^\top b} (a matrix product) on inputs that are
#'   centered, and scaled or whitened for Mahalanobis, in double precision first.
#' - Computations run in float32 and are JIT-compiled with JAX; where applicable,
#'   GPU/Metal/CPU device selection follows your existing backend.
#'
#' @examples
#' \dontrun{
#' # Simple Euclidean within-matrix distances (returns an n x n matrix)
#' X <- matrix(rnorm(50 * 8), 50, 8)
#' D <- fast_distance(X, metric = "euclidean")
#'
#' # Cosine distance between two sets
#' A <- matrix(rnorm(100 * 16), 100, 16)
#' B <- matrix(rnorm(120 * 16), 120, 16)
#' Dcos <- fast_distance(A, B, metric = "cosine")
#'
#' # Minkowski with p = 3 and feature weights
#' w <- runif(ncol(A))
#' Dm3 <- fast_distance(A, B, metric = "minkowski", p = 3, weights = w)
#'
#' # Mahalanobis (diagonal approx, fast & robust)
#' Dmah_diag <- fast_distance(X, metric = "mahalanobis", approximate_inv = TRUE)
#'
#' # Mahalanobis with full inverse (computed internally)
#' Dmah_full <- fast_distance(X, metric = "mahalanobis", approximate_inv = FALSE)
#'
#' # Return a base R 'dist' object
#' D_dist <- fast_distance(X, metric = "euclidean", as_dist = TRUE)
#' }
#'
#' @seealso \code{\link[stats]{dist}}
#' @export
fast_distance <- function(
    A,
    B = NULL,
    metric = c("euclidean", "sqeuclidean", "manhattan", "chebyshev",
               "cosine", "minkowski", "mahalanobis"),
    p = 2,
    weights = NULL,
    cov_inv = NULL,
    approximate_inv = TRUE,
    squared = FALSE,
    row_batch_size = NULL,
    as_dist = FALSE,
    return_type = "R",
    verbose = FALSE,
    conda_env = "fastrerandomize_env",
    conda_env_required = TRUE
){
  # Ensure JAX is available and kernels are initialized
  if (is.null(.ensure_backend(conda_env, conda_env_required))){ return(NULL) } 
  jnp <- fastrr_env$jnp
  
  metric <- match.arg(metric)
  B_was_null <- is.null(B)
  if (B_was_null) B <- A
  
  A <- as.matrix(A)
  B <- as.matrix(B)
  assertthat::assert_that(ncol(A) == ncol(B),
                          msg = "A and B must have the same number of columns (features).")
  n <- nrow(A); m <- nrow(B); p_dim <- ncol(A)
  
  # Optional weights
  w_jax <- NULL
  if (!is.null(weights)) {
    assertthat::assert_that(length(weights) == p_dim,
                            msg = "length(weights) must equal ncol(A).")
    w_jax <- jnp$array(as.numeric(weights), dtype = jnp$float32)
  }
  
  # Euclidean-type and Mahalanobis distances: squared Euclidean distances of
  # inputs centered (limits cancellation in the GEMM form) and, for
  # Mahalanobis, scaled or whitened on the host in double precision
  gemm_metric <- metric %in% c("euclidean", "sqeuclidean", "mahalanobis")
  S_inv <- NULL
  if (gemm_metric) {
    pooled <- rbind(A, B)
    center <- colMeans(pooled)
    A <- sweep(A, 2L, center)
    B <- sweep(B, 2L, center)
    if (metric == "mahalanobis") {
      transform <- if (approximate_inv) {
        # Diagonal inverse of pooled covariance
        diag(sqrt(1 / (colMeans(sweep(pooled, 2L, center)^2) + 1e-8)), nrow = p_dim)
      } else if (is.null(cov_inv)) {
        .psd_inv_sqrt(stats::cov(pooled))
      } else {
        .psd_factor(as.matrix(cov_inv))
      }
      if (is.null(transform)) {
        # Indefinite user-supplied cov_inv: evaluate the quadratic form directly
        gemm_metric <- FALSE
        S_inv <- jnp$array(as.matrix(cov_inv), dtype = jnp$float32)
      } else {
        A <- A %*% transform
        B <- B %*% transform
      }
    }
  }
  
  # Cast to JAX (float32)
  A_jax <- jnp$array(A, dtype = jnp$float32)
  B_jax <- jnp$array(B, dtype = jnp$float32)
  
  # Choose a safe batch size if not provided: GEMM metrics are bounded by the
  # n x m output (<= 2^26 floats per batch); broadcast metrics by the
  # n x m x p difference tensor (<= 1e7 floats)
  if (is.null(row_batch_size)) {
    row_batch_size <- if (gemm_metric) {
      min(n, max(1L, floor(2^26 / m)))
    } else {
      max_elems <- 1e7
      est <- as.numeric(n) * as.numeric(m) * as.numeric(p_dim)
      if (est > max_elems) max(1L, floor(max_elems / (m * p_dim))) else n
    }
  }
  nb <- ceiling(n / row_batch_size)
  chunks <- vector("list", nb)
  
  if (verbose) message(sprintf("Computing %s distances in %d batch(es), batch size = %d",
                               metric, nb, row_batch_size))
  
  for (b in seq_len(nb)) {
    start <- (b - 1L) * row_batch_size + 1L
    end <- min(n, b * row_batch_size)
    if (verbose) message(sprintf("  batch %d/%d: rows [%d, %d]", b, nb, start, end))
    idx <- jnp$arange(as.integer(start - 1L), as.integer(end))
    A_chunk <- jnp$take(A_jax, idx, axis = 0L)
    
    chunk <- if (gemm_metric) {
      d2 <- fastrr_env$PairwiseSqEuclidGemm(A_chunk, B_jax, idx, B_was_null)
      if (metric == "sqeuclidean" || squared) d2 else jnp$sqrt(d2 + 1e-12)
    } else switch(
      metric,
      manhattan = if (!is.null(w_jax)) {
        # Weighted L1 is Minkowski with p=1
        fastrr_env$PairwiseWeightedMinkowski(A_chunk, B_jax, w_jax,
                                             jnp$array(1.0, dtype = jnp$float32))
      } else {
        fastrr_env$PairwiseManhattan(A_chunk, B_jax)
      },
      chebyshev = fastrr_env$PairwiseChebyshev(A_chunk, B_jax),
      cosine = fastrr_env$PairwiseCosine(A_chunk, B_jax),
      minkowski = {
        assertthat::assert_that(is.numeric(p) && p > 0, msg = "p must be > 0 for Minkowski.")
        if (!is.null(w_jax)) {
          fastrr_env$PairwiseWeightedMinkowski(
            A_chunk, B_jax, w_jax,
            jnp$array(as.numeric(p), dtype = jnp$float32)
          )
        } else {
          fastrr_env$PairwiseMinkowski(
            A_chunk, B_jax,
            jnp$array(as.numeric(p), dtype = jnp$float32)
          )
        }
      },
      mahalanobis = {
        d2 <- fastrr_env$PairwiseMahalanobisFull(A_chunk, B_jax, S_inv)
        if (squared) d2 else jnp$sqrt(d2 + 1e-12)
      },
      stop("Unknown metric: ", metric)
    )
    
    chunks[[b]] <- chunk
  }
  
  D_jax <- if (length(chunks) == 1L) chunks[[1]] else fastrr_env$jnp$concatenate(chunks, 0L)
  
  # Convert / shape output
  if (as_dist && B_was_null) {
    if (return_type == "R") {
      D_R <- as.matrix(output2output(D_jax, return_type = "R"))
      return(stats::as.dist(D_R))
    } else {
      warning("`as_dist = TRUE` ignored when return_type = 'jax'. Returning a JAX array.")
      return(D_jax)
    }
  } else {
    return(output2output(D_jax, return_type = return_type))
  }
}

# F with F %*% t(F) equal to symmetric PSD C (NULL if C is indefinite), so that
# (a - b)' C (a - b) = ||(a - b)' F||^2.
.psd_factor <- function(C) {
  C <- (C + t(C)) / 2
  eig <- eigen(C, symmetric = TRUE)
  tol <- max(abs(eig$values)) * max(dim(C)) * .Machine$double.eps
  if (any(eig$values < -tol)) { return(NULL) }
  eig$vectors %*% diag(sqrt(pmax(eig$values, 0)), nrow = length(eig$values))
}
