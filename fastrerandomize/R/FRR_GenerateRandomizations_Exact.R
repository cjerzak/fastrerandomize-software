#' Generate Complete Randomizations with Optional Balance Constraints
#' 
#' @description
#' Generates all possible treatment assignments for a completely randomized experiment,
#' optionally filtering them based on covariate balance criteria. The function can
#' generate either all possible randomizations or a subset that meets specified
#' balance thresholds using Hotelling's T-squared statistic.
#'
#' @param n_units An integer specifying the total number of experimental units
#' @param n_treated An integer specifying the number of units to be assigned to treatment
#' @param X A numeric matrix of covariates where rows represent units and columns
#'   represent different covariates. Default is \code{NULL}, in which case all possible
#'   randomizations are returned without balance filtering.
#' @param randomization_accept_prob A numeric value between 0 and 1 specifying the
#'   quantile threshold for accepting randomizations based on balance statistics.
#'   Default is 1 (accept all randomizations).
#' @param approximate_inv A logical value indicating whether to use an approximate inverse 
#'   (diagonal of the covariance matrix) instead of the full matrix inverse when computing 
#'   balance metrics. This can speed up computations for high-dimensional covariates.
#'   Default is \code{TRUE}.
#' @param threshold_func A function that calculates balance statistics for candidate
#'   randomizations. Default is \code{VectorizedFastHotel2T2} which computes Hotelling's T-squared
#'   statistic.
#' @param verbose A logical value indicating whether to print progress information. Default is \code{TRUE}.
#' @param conda_env A character string specifying the name of the conda environment to use 
#'   via \code{reticulate}. Default is "fastrerandomize_env".
#' @param conda_env_required A logical indicating whether the specified conda environment 
#'   must be strictly used. If \code{TRUE}, an error is thrown if the environment is not found. 
#'   Default is TRUE.
#'
#' @return The function returns a \emph{list} with two elements:
#' \code{candidate_randomizations}: an array of randomization vectors
#' \code{M_candidate_randomizations}: an array of their balance measures. 
#'
#' @details
#' The function works in two main steps:
#' 1. Generates all possible combinations of treatment assignments given n_units
#'    and n_treated
#' 2. If covariates (X) are provided, filters these combinations based on balance
#'    criteria using the specified threshold function
#'
#' A warning is emitted before enumeration when the number of combinations is
#' greater than one million. Small exact designs are still enumerated; when
#' balance filtering nominally retains fewer than ten assignments, the function
#' warns rather than failing.
#'
#' The balance filtering process uses Hotelling's T-squared statistic by default to measure
#' multivariate balance between treatment and control groups. Randomizations are
#' accepted if their balance measure is below the specified quantile threshold.
#'
#' @examples
#' 
#' \dontrun{
#' # Generate synthetic data 
#' X <- matrix(rnorm(60), nrow = 10)  # 10 units, 6 covariates
#' 
#' # Generate balanced randomizations with covariates
#' BalancedRandomizations <- generate_randomizations_exact(
#'   n_units = 10,
#'   n_treated = 5,
#'   X = X,
#'   randomization_accept_prob = 0.25  # Keep top 25% most balanced
#' )
#' }
#'
#' @importFrom utils combn
#' @import reticulate
#'
#' @note
#' This function requires 'JAX' and 'NumPy' to be installed and accessible through
#' the reticulate package. 
#'
#' @references
#' Hotelling, H. (1931). The generalization of Student's ratio. 
#' The Annals of Mathematical Statistics, 2(3), 360-378.
#'
#' @seealso
#' \code{\link{generate_randomizations}} for full randomization generation function. 
#' \code{\link{generate_randomizations_mc}} for the Monte Carlo version. 
#'
#' @export
generate_randomizations_exact <- function(
                                   n_units, 
                                   n_treated,
                                   X = NULL,
                                   randomization_accept_prob = 1,
                                   approximate_inv = TRUE, 
                                   threshold_func = NULL,
                                   verbose = TRUE,
                                   conda_env = "fastrerandomize_env", 
                                   conda_env_required = TRUE){
  if (length(n_units) != 1L || !is.numeric(n_units) || !is.finite(n_units) ||
      n_units < 2 || n_units != floor(n_units)) {
    stop("'n_units' must be a whole number of at least 2.", call. = FALSE)
  }
  if (length(n_treated) != 1L || !is.numeric(n_treated) || !is.finite(n_treated) ||
      n_treated < 1 || n_treated >= n_units || n_treated != floor(n_treated)) {
    stop("'n_treated' must be a whole number between 1 and n_units - 1.", call. = FALSE)
  }
  if (length(randomization_accept_prob) != 1L ||
      !is.numeric(randomization_accept_prob) ||
      !is.finite(randomization_accept_prob) ||
      randomization_accept_prob < 0 || randomization_accept_prob > 1) {
    stop("'randomization_accept_prob' must be between 0 and 1.", call. = FALSE)
  }
  n_units <- as.integer(n_units)
  n_treated <- as.integer(n_treated)
  if (!is.null(X)) {
    X <- .prepare_X(X, n_units)
  }

  max_rand_num <- choose(n_units, n_treated)
  .warn_if_large_exact(max_rand_num)
  if (!is.null(X) && randomization_accept_prob < 1 &&
      max_rand_num * randomization_accept_prob < 10) {
    warning(
      paste0(
        "The requested balance filter nominally retains fewer than 10 randomizations. ",
        "The exact design will still be generated, but downstream inference may have low resolution."
      ),
      call. = FALSE
    )
  }

  if (is.null(.ensure_backend(conda_env, conda_env_required))) { return(NULL) }
  jnp <- fastrr_env$jnp

  # Get all combinations of positions to set to 1 (int32 rows, checked below)
  combinations <- jnp$array(  utils::combn(n_units, n_treated) - 1L )
  ZerosHolder <- jnp$zeros(as.integer(n_units), dtype = jnp$int32)
  candidate_randomizations <- fastrr_env$InsertOnesVectorized(combinations, ZerosHolder)
  .assert_valid_rows(
    .to_r(jnp$sum(jnp$not_equal(jnp$sum(candidate_randomizations, 1L), n_treated))),
    n_treated
  )
  candidate_randomizations <- candidate_randomizations$astype(jnp$uint16)

  M_results <- NULL; if(!is.null(X)){
    # Balance inputs are built in double precision on the host
    design <- .balance_design(X, approximate_inv)

    # Calculate balance measure (Hotelling T-squared) for each candidate randomization
    if (.is_default_threshold(threshold_func)) {
      M_results <- fastrr_env$WhitenedT2Jit(
        jnp$array(design$Xw, dtype = jnp$float32),
        candidate_randomizations,
        jnp$float32(n_units - n_treated),
        jnp$float32(n_treated)
      )
    } else {
      M_results <- jnp$reshape(threshold_func(
        jnp$array(X, dtype = jnp$float32),                        # Covariates
        jnp$array(design$S_inv, dtype = jnp$float32),             # matrix inverse
        jnp$array(design$S_inv_approx, dtype = jnp$float32),      # diagonal inverse
        jnp$array(candidate_randomizations, dtype = jnp$float32), # Possible assignments
        jnp$array(n_units - n_treated),
        jnp$array(n_treated),                                     # Sample sizes
        approximate_inv
      ), -1L)
    }
    
    # Find acceptance threshold based on specified quantile
    a_threshold <- jnp$quantile( 
      M_results,  
      jnp$array(randomization_accept_prob)
    )

    # Keep only randomizations with balance measure at or below threshold (use less_equal to include ties)
    candidate_randomizations <- jnp$take(
      candidate_randomizations,
      indices = (takeM_ <- jnp$where(jnp$less_equal(M_results, a_threshold))[[1]] ),
      axis = 0L
    )
    M_results <- jnp$take( M_results, indices = takeM_, axis = 0L )
  }
  
  return(list("candidate_randomizations"=candidate_randomizations,
              "M_candidate_randomizations"=M_results))
}
