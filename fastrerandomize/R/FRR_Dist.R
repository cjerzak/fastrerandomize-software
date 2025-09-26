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
#'   it is estimated from \code{rbind(A, B)} and inverted in JAX.
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
  browser()
  if (is.null(check_jax_availability(conda_env = conda_env))){ return(NULL) } 
  if (!"PairwiseEuclidean" %in% ls(envir = fastrr_env)) {
    initialize_jax(conda_env = conda_env, conda_env_required = conda_env_required)
  }
  
  metric <- match.arg(metric)
  if (is.null(B)) B <- A
  
  A <- as.matrix(A)
  B <- as.matrix(B)
  assertthat::assert_that(ncol(A) == ncol(B),
                          msg = "A and B must have the same number of columns (features).")
  n <- nrow(A); m <- nrow(B); p_dim <- ncol(A)
  
  # Cast to JAX (float32)
  A_jax <- fastrr_env$jnp$array(A, dtype = fastrr_env$jnp$float32)
  B_jax <- fastrr_env$jnp$array(B, dtype = fastrr_env$jnp$float32)
  
  # Optional weights
  w_jax <- NULL
  if (!is.null(weights)) {
    assertthat::assert_that(length(weights) == p_dim,
                            msg = "length(weights) must equal ncol(A).")
    w_jax <- fastrr_env$jnp$array(as.numeric(weights), dtype = fastrr_env$jnp$float32)
  }
  
  # Mahalanobis setup
  diag_inv <- NULL
  S_inv <- NULL
  if (metric == "mahalanobis") {
    if (approximate_inv) {
      # Diagonal inverse of pooled covariance
      pooled <- fastrr_env$jnp$array(rbind(A, B), dtype = fastrr_env$jnp$float32)
      var_vec <- fastrr_env$jnp$var(pooled, axis = 0L)
      diag_inv <- fastrr_env$jnp$reciprocal(var_vec + 1e-8)
    } else {
      if (is.null(cov_inv)) {
        S_inv <- fastrr_env$jnp$cov(fastrr_env$jnp$array(rbind(A, B)), rowvar = FALSE)
        IS_METAL_BACKEND <- grepl(reticulate::py_str(fastrr_env$jax$devices()[[1]]), "METAL")
        if (IS_METAL_BACKEND) {
          S_inv <- S_inv$to_device(fastrr_env$jax$devices("cpu")[[1]])
        }
        S_inv <- fastrr_env$jnp$linalg$inv(S_inv)
        if (IS_METAL_BACKEND) {
          S_inv <- S_inv$to_device(fastrr_env$jax$devices("METAL")[[1]])
        }
      } else {
        S_inv <- fastrr_env$jnp$array(as.matrix(cov_inv), dtype = fastrr_env$jnp$float32)
      }
    }
  }
  
  # Choose a safe batch size if not provided (keeps ~<= 1e7 float elements in diff tensor)
  if (is.null(row_batch_size)) {
    max_elems <- 1e7
    est <- as.numeric(n) * as.numeric(m) * as.numeric(p_dim)
    row_batch_size <- if (est > max_elems) max(1L, floor(max_elems / (m * p_dim))) else n
  }
  nb <- ceiling(n / row_batch_size)
  chunks <- vector("list", nb)
  
  if (verbose) message(sprintf("Computing %s distances in %d batch(es), batch size = %d",
                               metric, nb, row_batch_size))
  
  for (b in seq_len(nb)) {
    start <- (b - 1L) * row_batch_size + 1L
    end <- min(n, b * row_batch_size)
    if (verbose) message(sprintf("  batch %d/%d: rows [%d, %d]", b, nb, start, end))
    idx <- fastrr_env$jnp$arange(as.integer(start - 1L), as.integer(end))
    A_chunk <- fastrr_env$jnp$take(A_jax, idx, axis = 0L)
    
    chunk <- switch(
      metric,
      euclidean = if (squared) {
        fastrr_env$PairwiseSqEuclidean(A_chunk, B_jax)
      } else {
        fastrr_env$PairwiseEuclidean(A_chunk, B_jax)
      },
      sqeuclidean = fastrr_env$PairwiseSqEuclidean(A_chunk, B_jax),
      manhattan = if (!is.null(w_jax)) {
        # Weighted L1 is Minkowski with p=1
        fastrr_env$PairwiseWeightedMinkowski(A_chunk, B_jax, w_jax,
                                             fastrr_env$jnp$array(1.0, dtype = fastrr_env$jnp$float32))
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
            fastrr_env$jnp$array(as.numeric(p), dtype = fastrr_env$jnp$float32)
          )
        } else {
          fastrr_env$PairwiseMinkowski(
            A_chunk, B_jax,
            fastrr_env$jnp$array(as.numeric(p), dtype = fastrr_env$jnp$float32)
          )
        }
      },
      mahalanobis = {
        d2 <- if (approximate_inv) {
          fastrr_env$PairwiseMahalanobisDiag(A_chunk, B_jax, diag_inv)
        } else {
          fastrr_env$PairwiseMahalanobisFull(A_chunk, B_jax, S_inv)
        }
        if (squared) d2 else fastrr_env$jnp$sqrt(d2 + 1e-12)
      },
      stop("Unknown metric: ", metric)
    )
    
    chunks[[b]] <- chunk
  }
  
  D_jax <- if (length(chunks) == 1L) chunks[[1]] else fastrr_env$jnp$concatenate(chunks, 0L)
  
  # Convert / shape output
  if (as_dist && is.null(B)) {
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
