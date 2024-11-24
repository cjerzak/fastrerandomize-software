#' Generate Complete Randomizations with Optional Balance Constraints
#' 
#' @description
#' Generates all possible treatment assignments for a completely randomized experiment,
#' optionally filtering them based on covariate balance criteria. The function can
#' generate either all possible randomizations or a subset that meets specified
#' balance thresholds using Hotelling's T² statistic.
#'
#' @param n_units An integer specifying the total number of experimental units
#' @param n_treated An integer specifying the number of units to be assigned to treatment
#' @param X A numeric matrix of covariates where rows represent units and columns
#'   represent different covariates. Default is NULL, in which case all possible
#'   randomizations are returned without balance filtering.
#' @param randomization_accept_prob A numeric value between 0 and 1 specifying the
#'   quantile threshold for accepting randomizations based on balance statistics.
#'   Default is 1 (accept all randomizations).
#' @param threshold_func A function that calculates balance statistics for candidate
#'   randomizations. Default is VectorizedFastHotel2T2 which computes Hotelling's T²
#'   statistic.
#'
#' @return A JAX NumPy array where each row represents a valid treatment assignment
#'   vector (binary: 1 for treated, 0 for control) that meets the balance criteria
#'   if specified.
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
#' # Generate all possible randomizations for 6 units with 3 treated
#' rand <- GenerateRandomizations(n_units = 6, n_treated = 3)
#'
#' # Generate balanced randomizations with covariates
#' X <- matrix(rnorm(60), nrow = 10)  # 10 units, 6 covariates
#' BalancedRandomizations <- GenerateRandomizations(
#'   n_units = 10,
#'   n_treated = 5,
#'   X = X,
#'   randomization_accept_prob = 0.25  # Keep top 25% most balanced
#' )
#'
#' @importFrom utils combn
#' @import reticulate
#'
#' @note
#' This function requires JAX and NumPy to be installed and accessible through
#' the reticulate package. The function assumes the existence of helper functions
#' InsertOnesVectorized and VectorizedFastHotel2T2.
#'
#' @references
#' Hotelling, H. (1931). The generalization of Student's ratio. 
#' The Annals of Mathematical Statistics, 2(3), 360-378.
#'
#' @seealso
#' \code{\link{VectorizedFastHotel2T2}} for details on the balance statistic calculation
#' \code{\link{InsertOnesVectorized}} for the treatment assignment generation
#'
#' @export
generate_randomizations_exact <- function(n_units, n_treated,
                                   X = NULL,
                                   randomization_accept_prob = 1,
                                   approximate_inv = TRUE, 
                                   threshold_func = VectorizedFastHotel2T2){
  # Get all combinations of positions to set to 1
  combinations <- jnp$array(  utils::combn(n_units, n_treated) - 1L )
  ZerosHolder <- jnp$zeros(as.integer(n_units), dtype=jnp$int32)
  candidate_randomizations <- InsertOnesVectorized(combinations,
                                                   ZerosHolder)
  
  if(!is.null(X)){
    # Set up sample sizes for treatment/control
    n0_array <- jnp$array(  (n_units - n_treated) )
    n1_array <- jnp$array(  n_treated )
    
    # Calculate balance measure (Hotelling T-squared) for each candidate randomization
    M_results <-  threshold_func(
      jnp$array( X ),                    # Covariates
      jnp$array(candidate_randomizations, dtype = jnp$float32),  # Possible assignments
      n0_array, 
      n1_array,                # Sample sizes
      approximate_inv
    )
    
    # Find acceptance threshold based on specified quantile
    a_threshold <- np$array(jnp$quantile( 
      M_results,  
      jnp$array(randomization_accept_prob)
    ))[[1]]
    
    # Convert to regular array
    M_results <- c(np$array( M_results ))
    
    # Keep only randomizations with balance measure below threshold
    candidate_randomizations <- jnp$take(
      candidate_randomizations,
      indices = jnp$array(which(M_results <= a_threshold)-1L),
      axis = 0L
    )
  }
  
  return( candidate_randomizations )
}