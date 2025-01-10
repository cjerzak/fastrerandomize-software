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
#' @param seed An integer seed for random number generation, used when enumerating 
#'   or filtering exact randomizations with potentially randomized steps (e.g., 
#'   random draws in thresholding). Default is \code{NULL} (no fixed seed).
#' @param verbose A logical value indicating whether to print progress information. Default is \code{TRUE}.
#' @param conda_env A character string specifying the name of the conda environment to use 
#'   via \code{reticulate}. Default is "fastrerandomize".
#' @param conda_env_required A logical indicating whether the specified conda environment 
#'   must be strictly used. If \code{TRUE}, an error is thrown if the environment is not found. 
#'   Default is TRUE.
#' @param threshold_func A function that calculates balance statistics for candidate
#'   randomizations. Default is \code{VectorizedFastHotel2T2} which computes Hotelling's T-squared
#'   statistic.
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
generate_randomizations_exact <- function(n_units, n_treated,
                                   X = NULL,
                                   randomization_accept_prob = 1,
                                   approximate_inv = TRUE, 
                                   threshold_func = NULL,
                                   seed = NULL, 
                                   verbose = TRUE,
                                   conda_env = "fastrerandomize", 
                                   conda_env_required = TRUE){
  if(is.null(check_jax_availability(conda_env=conda_env))) { return(NULL) }
  
  if (!"VectorizedFastHotel2T2" %in% ls(envir = fastrr_env)) {
    initialize_jax(conda_env = conda_env, conda_env_required = conda_env_required)
  }
  if(is.null(threshold_func)){ threshold_func <- fastrr_env$VectorizedFastHotel2T2 }
  
  max_rand_num <- choose(n_units, n_treated)
  assertthat::assert_that( max_rand_num*min(randomization_accept_prob) > 10, 
              msg = "Value of min(randomization_accept_prob) indices less than 10 accepted randomizations. Increase min(randomization_accept_prob)!")
  
  # Get all combinations of positions to set to 1
  combinations <- fastrr_env$jnp$array(  utils::combn(n_units, n_treated) - 1L )
  ZerosHolder <- fastrr_env$jnp$zeros(as.integer(n_units), dtype=fastrr_env$jnp$uint16)
  candidate_randomizations <- fastrr_env$InsertOnesVectorized(combinations, ZerosHolder)

  M_results <- NULL; if(!is.null(X)){
    # Set up sample sizes for treatment/control
    n0_array <- fastrr_env$jnp$array(  (n_units - n_treated) )
    n1_array <- fastrr_env$jnp$array(  n_treated )
    
    # Calculate balance measure (Hotelling T-squared) for each candidate randomization
    M_results <-  threshold_func(
      fastrr_env$jnp$array( X ),                    # Covariates
      fastrr_env$jnp$array(candidate_randomizations, dtype = fastrr_env$jnp$float32),  # Possible assignments
      n0_array, 
      n1_array,                # Sample sizes
      approximate_inv
    )
    
    # Find acceptance threshold based on specified quantile
    a_threshold <- fastrr_env$jnp$quantile( 
      M_results,  
      fastrr_env$jnp$array(randomization_accept_prob)
    )

    # Keep only randomizations with balance measure below threshold
    candidate_randomizations <- fastrr_env$jnp$take(
      candidate_randomizations,
      indices = (takeM_ <- fastrr_env$jnp$where(fastrr_env$jnp$less(M_results,a_threshold))[[1]] ),
      axis = 0L
    )
    M_results <- fastrr_env$jnp$take( M_results, indices = takeM_, axis = 0L )
  }
  
  return(list("candidate_randomizations"=candidate_randomizations,
              "M_candidate_randomizations"=M_results))
}