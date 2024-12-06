#!/usr/bin/env Rscript
#' Fast randomization test
#'
#' @param obsW A numeric vector where `0`'s correspond to control units and `1`'s to treated units.
#' @param X A numeric matrix of covariates.
#' @param obsY An optional numeric vector of observed outcomes. If not provided, the function assumes a NULL value.
#' @param c_initial A numeric value representing the initial criterion for the randomization. Default is `2`.
#' @param alpha The significance level for the test. Default is `0.05`.
#' @param candidate_randomizations A numeric matrix of candidate randomizations.
#' @param candidate_randomizations_array An optional JAX array of candidate randomizations. If not provided, the function coerces `candidate_randomizations` into a JAX array.
#' @param n0_array An optional array specifying the number of control units.
#' @param n1_array An optional array specifying the number of treated units.
#' @param true_treatment_effect An optional numeric value specifying the true treatment effect. Default is NULL.
#' @param randomization_accept_prob An numeric scalar or vector of probabilities for accepting each randomization.
#' @param findFI A logical value indicating whether to find the fiducial interval. Default is FALSE.
#'
#' @return A list consisting of \itemize{
#'   \item `p_value` A numeric value or vector representing the p-value of the test (or the expected p-value under the prior structure specified in the function inputs).
#'   \item `FI` A numeric vector representing the fiducial interval if `findFI=T`.
#'   \item `tau_obs` A numeric value or vector representing the estimated treatment effect(s)
#' }

#'
#' @section References:
#' \itemize{
#' \item
#' }
#'
#' @examples
#' # For a tutorial, see
#' # github.com/cjerzak/fastrerandomization-software
#'
#' @export
#' @md

randomization_test <- function(
                               obsW = NULL,
                               obsY = NULL,
                               X = NULL,
                               alpha = 0.05,
                               candidate_randomizations = NULL,
                               candidate_randomizations_array = NULL,
                               n0_array = NULL,
                               n1_array = NULL,
                               randomization_accept_prob = 1.,
                               findFI = F,
                               c_initial = 2,
                               max_draws = 10^6, 
                               batch_size = 10^5, 
                               randomization_type = "monte_carlo", 
                               approximate_inv = TRUE,
                               file = NULL, 
                               conda_env = "fastrerandomize", conda_env_required = T
                               ){
  tau_obs <- FI <- covers_truth <- NULL
  if(!"VectorizedFastHotel2T2" %in% ls(envir = .GlobalEnv)){
    initialize_jax_code <- paste(deparse(initialize_jax),collapse="\n")
    initialize_jax_code <- gsub(initialize_jax_code,pattern="function \\(\\)",replace="")
    eval( parse( text = initialize_jax_code ), envir = environment() )
  }


  if(is.null(n0_array)){ n0_array <- jnp$array(sum(obsW == 0)) }
  if(is.null(n1_array)){ n1_array <- jnp$array(sum(obsW == 1)) }

  if(!is.null(obsW)){obsW <- c(unlist(obsW))}
  if(!is.null(X)){X <- as.matrix(X)}
  if(!is.null(obsY)){obsY <- c(unlist(obsY))}

  #if(is.null(candidate_randomizations_array) & is.null(candidate_randomizations)){
  if(is.null(candidate_randomizations)){
      candidate_randomizations <- np$array( candidate_randomizations_array )
  }
  if(is.null(candidate_randomizations_array)){
      candidate_randomizations_array <- jnp$array(candidate_randomizations, dtype = jnp$float32)
  }

  # perform randomization inference using input data
  {
    tau_obs <- c(np$array( FastDiffInMeans(jnp$array(obsY), # 
                                           jnp$array(obsW), # 
                                           n0_array, #
                                           n1_array # 
                                           ) ))
    tau_perm_null_0 <- np$array(
      W_VectorizedFastDiffInMeans(
          jnp$array(obsY),  # y_ =
          candidate_randomizations_array, # t_ =
          n0_array, # n0 =
          n1_array # n1 =
    ))
    p_value <- mean( abs(tau_perm_null_0) >= abs(tau_obs) )

    if( findFI ){
      obsY_array <- jnp$array( obsY )
      obsW_array <- jnp$array( obsW )

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
        z_alpha <- qnorm( p = (1-alpha) )
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
            lower_tau_at_step_t <- np$array( FastDiffInMeans(jnp$array(lower_Y_obs_perm), jnp$array(permutation_treatment_vec),
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
            #upper_tau_at_step_t <- mean(upper_Y_obs_perm[permutation_treatment_vec == 1]) - mean(upper_Y_obs_perm[permutation_treatment_vec == 0])
            upper_tau_at_step_t <- np$array( FastDiffInMeans(jnp$array(upper_Y_obs_perm), jnp$array(permutation_treatment_vec), n0_array, n1_array) )

            c_step_t <- k * (upperBound_estimate_step_t  -  tau_obs)
            #if(is.na(c_step_t)){
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
          stat_vec_at_tau_pseudo <- np$array(     vec1_get_stat_vec_at_tau_pseudo(candidate_randomizations_array,# treatment_pseudo
                                                                                  obsY_array,# obsY_array
                                                                                  obsW_array, # obsW_array
                                                                                  tau_pseudo, # tau_pseudo
                                                                                  n0_array, # n0_array
                                                                                  n1_array #
                                                                                  )  )

          ret_ <- min(mean( tau_obs >= stat_vec_at_tau_pseudo),
                      mean( tau_obs <= stat_vec_at_tau_pseudo))
          #ret_ <- reject_ <- tau_obs >= quantiles_[1] & tau_obs <= quantiles_[2]
          return( ret_ )
        } )
        tau_pseudo_seq_AcceptNull <- tau_pseudo_seq[pvals_vec>0.05]
        FI <- c(min(tau_pseudo_seq_AcceptNull),
                max(tau_pseudo_seq_AcceptNull))
      }
    }
  }

  return( list(p_value = p_value,
                 FI = FI,
                 tau_obs = tau_obs) )
}
