#' Fast randomization test
#' 
#' 
#'
#'
#' @param obsW A numeric vector where `0`'s correspond to control units and `1`'s to treated units.
#' @param obsY An optional numeric vector of observed outcomes. If not provided, the function assumes a NULL value.
#' @param X A numeric matrix of covariates.
#' @param alpha The significance level for the test. Default is `0.05`.
#' @param candidate_randomizations A numeric matrix of candidate randomizations.
#' @param candidate_randomizations_array An optional 'JAX' array of candidate randomizations. If not provided, the function coerces `candidate_randomizations` into a 'JAX' array.
#' @param n0_array An optional array specifying the number of control units.
#' @param n1_array An optional array specifying the number of treated units.
#' @param randomization_accept_prob An numeric scalar or vector of probabilities for accepting each randomization.
#' @param findFI A logical value indicating whether to find the fiducial interval. Default is FALSE.
#' @param c_initial A numeric value representing the initial criterion for the randomization. Default is `2`.
#' @param max_draws An integer specifying the maximum number of candidate randomizations 
#'   to generate (or to consider) for the test when \code{randomization_type = "monte_carlo"}. 
#'   Default is \code{1e6}.
#' @param batch_size An integer specifying the batch size for Monte Carlo sampling. 
#'   Batches are processed one at a time for memory efficiency. Default is \code{1e5}.
#' @param randomization_type A string specifying the type of randomization for the test. 
#'   Allowed values are "exact" or "monte_carlo". Default is "monte_carlo".
#' @param approximate_inv A logical value indicating whether to use an approximate inverse 
#'   (diagonal of the covariance matrix) instead of the full matrix inverse when computing 
#'   balance metrics. This can speed up computations for high-dimensional covariates.
#'   Default is \code{TRUE}.
#' @param file A character string specifying the path (including filename) where candidate 
#'   randomizations will be saved or loaded from. If \code{NULL}, randomizations 
#'   remain in memory. Default is NULL.
#' @param verbose A logical value indicating whether to print progress information. Default is \code{TRUE}.  
#' @param conda_env A character string specifying the name of the conda environment to use 
#'   via \code{reticulate}. Default is \code{"fastrerandomize"}.
#' @param conda_env_required A logical indicating whether the specified conda environment 
#'   must be strictly used. If \code{TRUE}, an error is thrown if the environment is not found. 
#'   Default is \code{TRUE}.
#'   
#' @return Returns an S3 object with slots: \itemize{
#'   \item `p_value` A numeric value or vector representing the p-value of the test (or the expected p-value under the prior structure specified in the function inputs).
#'   \item `FI` A numeric vector representing the fiducial interval if \code{findFI=TRUE}.
#'   \item `tau_obs` A numeric value or vector representing the estimated treatment effect(s).
#'   \item `fastrr_env` The fastrerandomize environment. 
#' }
#'
#' @section References:
#' \itemize{
#' \item Zhang, Y. and Zhao, Q., 2023. What is a randomization test?. Journal of the American Statistical Association, 118(544), pp.2928-2942.
#' }
#'
#' @examples
#' \dontrun{
#' # A small synthetic demonstration with 6 units, 3 treated and 3 controls:
#'
#' # Generate pre-treatment covariates
#' X <- matrix(rnorm(24*2), ncol = 2)
#'
#' # Generate candidate randomizations
#' RandomizationSet_MC <- generate_randomizations(
#'   n_units = nrow(X),
#'   n_treated = round(nrow(X)/2),
#'   X = X,
#'   randomization_accept_prob = 0.1,
#'   randomization_type = "monte_carlo",
#'   max_draws = 100000,
#'   batch_size = 1000
#' )
#'
#' # Generate outcome
#' W <- RandomizationSet_MC$randomizations[1,]
#' obsY <- rnorm(nrow(X), mean = 2 * W)
#'
#' # Perform randomization test
#' results_base <- randomization_test(
#'   obsW = W,
#'   obsY = obsY,
#'   X = X,
#'   candidate_randomizations = RandomizationSet_MC$randomizations,
#' )
#' print(results_base)
#'
#' # Perform randomization test
#' result_fi <- randomization_test(
#'   obsW = W,
#'   obsY = obsY,
#'   X = X,
#'   candidate_randomizations = RandomizationSet_MC$randomizations,
#'   findFI = TRUE
#' )
#' print(result_fi)
#' }
#'
#' @seealso
#' \code{\link{generate_randomizations}} for randomization generation function. 
#'
#' @export
#' @md

randomization_test <- function(obsW = NULL,
                               obsY = NULL,
                               X = NULL,
                               alpha = 0.05,
                               candidate_randomizations = NULL,
                               candidate_randomizations_array = NULL,
                               n0_array = NULL,
                               n1_array = NULL,
                               randomization_accept_prob = 1.,
                               findFI = FALSE,
                               c_initial = 2,
                               max_draws = 10^6, 
                               batch_size = 10^5, 
                               randomization_type = "monte_carlo", 
                               approximate_inv = TRUE,
                               file = NULL, 
                               verbose = TRUE, 
                               conda_env = "fastrerandomize", 
                               conda_env_required = TRUE
                               ){
  if( is.null(check_jax_availability(conda_env=conda_env)) ) { return(NULL) }

  tau_obs <- FI <- NULL
  if (!"VectorizedFastHotel2T2" %in% ls(envir = fastrr_env)) {
    initialize_jax(conda_env = conda_env, conda_env_required = conda_env_required)
  }

  if(is.null(n0_array)){ n0_array <- fastrr_env$jnp$array(sum(obsW == 0)) }
  if(is.null(n1_array)){ n1_array <- fastrr_env$jnp$array(sum(obsW == 1)) }

  if(!is.null(obsW)){obsW <- c(unlist(obsW))}
  if(!is.null(X)){X <- as.matrix(X)}
  if(!is.null(obsY)){obsY <- c(unlist(obsY))}

  #if(is.null(candidate_randomizations_array) & is.null(candidate_randomizations)){
  if(is.null(candidate_randomizations)){
      candidate_randomizations <- fastrr_env$np$array( candidate_randomizations_array )
  }
  if(is.null(candidate_randomizations_array)){
      candidate_randomizations_array <- fastrr_env$jnp$array(candidate_randomizations, 
                                                             dtype = fastrr_env$jnp$float32)
  }

  # perform randomization inference using input data
  {
    tau_obs <- c(fastrr_env$np$array( fastrr_env$FastDiffInMeans(
                                          fastrr_env$jnp$array(obsY), # 
                                          fastrr_env$jnp$array(obsW), # 
                                           n0_array, #
                                           n1_array # 
                                           ) ))
    tau_perm_null_0 <- fastrr_env$np$array(
      fastrr_env$W_VectorizedFastDiffInMeans(
          fastrr_env$jnp$array(obsY),  # y_ =
          candidate_randomizations_array, # t_ =
          n0_array, # n0 =
          n1_array # n1 =
    ))
    p_value <- mean( abs(tau_perm_null_0) >= abs(tau_obs) )

    if( findFI ){
      obsY_array <- fastrr_env$jnp$array( obsY )
      obsW_array <- fastrr_env$jnp$array( obsW )

      n_search_attempts <- 500
      exhaustive_search  <-  length(obsW) <= n_search_attempts
      bound_counter <- 0
      upperBound_storage_vec <- lowerBound_storage_vec <- rep(NA, n_search_attempts)
      {
        bound_counter <- bound_counter + 1
        lowerBound_estimate_step_t <- tau_obs-3*tau_obs
        upperBound_estimate_step_t <- tau_obs+3*tau_obs

        #setting optimal c
        c_step_t <- c_initial
        z_alpha <- stats::qnorm( p = (1-alpha) )
        k <- 2 / (  z_alpha *   (2 * pi)^(-1/2) * exp( -z_alpha^2 / 2)  )
        NAHolder <- rep(NA, length(obsW))
        for(step_t in 1:n_search_attempts){
          #initialize for next step
          permutation_treatment_vec <- candidate_randomizations[sample(1:nrow(candidate_randomizations), size=1),]
          lower_Y_0_under_null <- lower_Y_obs_perm <- NAHolder
          upper_Y_0_under_null <- upper_Y_obs_perm <- lower_Y_obs_perm

          #update lower
          {
            lower_Y_0_under_null[obsW == 0] <- obsY[obsW == 0]
            lower_Y_0_under_null[obsW == 1] <- obsY[obsW == 1] - lowerBound_estimate_step_t
            lower_Y_obs_perm[permutation_treatment_vec==0] <- lower_Y_0_under_null[permutation_treatment_vec==0]
            lower_Y_obs_perm[permutation_treatment_vec==1] <- lower_Y_0_under_null[permutation_treatment_vec==1] + lowerBound_estimate_step_t
            lower_tau_at_step_t <- fastrr_env$np$array( fastrr_env$FastDiffInMeans(
                                                             fastrr_env$jnp$array(lower_Y_obs_perm), 
                                                             fastrr_env$jnp$array(permutation_treatment_vec),
                                                             n0_array, n1_array) )

            c_step_t <-  k * (tau_obs  - lowerBound_estimate_step_t)
            if(lower_tau_at_step_t < tau_obs) {  lowerBound_estimate_step_t <- lowerBound_estimate_step_t + c_step_t * (alpha/2) / step_t  }
            if(lower_tau_at_step_t >= tau_obs) { lowerBound_estimate_step_t <- lowerBound_estimate_step_t - c_step_t * (1-alpha/2) / step_t }
          }

          #update upper
          {
            upper_Y_0_under_null[obsW == 0] <- obsY[obsW == 0]
            upper_Y_0_under_null[obsW == 1] <- obsY[obsW == 1] - upperBound_estimate_step_t
            upper_Y_obs_perm[permutation_treatment_vec==0] <- upper_Y_0_under_null[permutation_treatment_vec==0]
            upper_Y_obs_perm[permutation_treatment_vec==1] <- upper_Y_0_under_null[permutation_treatment_vec==1] + upperBound_estimate_step_t
            upper_tau_at_step_t <- fastrr_env$np$array( fastrr_env$FastDiffInMeans(fastrr_env$jnp$array(upper_Y_obs_perm), 
                                                                                   fastrr_env$jnp$array(permutation_treatment_vec), 
                                                                                   n0_array, n1_array) )

            c_step_t <- k * (upperBound_estimate_step_t  -  tau_obs)
            if(upper_tau_at_step_t > tau_obs) {  upperBound_estimate_step_t <- upperBound_estimate_step_t - c_step_t * (alpha/2) / step_t  }
            if(upper_tau_at_step_t <= tau_obs) { upperBound_estimate_step_t <- upperBound_estimate_step_t + c_step_t * (1-alpha/2) / step_t }
          }
          lowerBound_storage_vec[step_t] <- lowerBound_estimate_step_t
          upperBound_storage_vec[step_t] <- upperBound_estimate_step_t
        }
      }#for(bound_side in c("lower", "upper"))

      # save results
      FI <- c(lowerBound_storage_vec[length(lowerBound_storage_vec)],
              upperBound_storage_vec[length(upperBound_storage_vec)])

      # stage 2
      {
        tau_pseudo_seq <- seq(FI[1]-1, FI[2]*2,length.out=100)
        pvals_vec <- sapply(tau_pseudo_seq, function(tau_pseudo){
          stat_vec_at_tau_pseudo <- fastrr_env$np$array(     fastrr_env$vec1_get_stat_vec_at_tau_pseudo(
                                                                                  candidate_randomizations_array,# treatment_pseudo
                                                                                  obsY_array,# obsY_array
                                                                                  obsW_array, # obsW_array
                                                                                  tau_pseudo, # tau_pseudo
                                                                                  n0_array, # n0_array
                                                                                  n1_array #
                                                                                  )  )

          ret_ <- min(mean( tau_obs >= stat_vec_at_tau_pseudo),
                      mean( tau_obs <= stat_vec_at_tau_pseudo))
          return( ret_ )
        } )
        tau_pseudo_seq_AcceptNull <- tau_pseudo_seq[pvals_vec>0.05]
        FI <- c(min(tau_pseudo_seq_AcceptNull),
                max(tau_pseudo_seq_AcceptNull))
      }
    }
  }
  
  # -------------------------------------------------------------------
  # Wrap in an S3 constructor
  return(
    fastrerandomize_test(
      p_value = p_value,
      FI      = FI,
      tau_obs = tau_obs,
      candidate_randomizations = candidate_randomizations, 
      fastrr_env = fastrr_env, 
      call    = match.call()  
    )
  )
}
