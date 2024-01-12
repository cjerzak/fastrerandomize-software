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
#' @param prior_treatment_effect_mean An optional numeric value for the prior mean of the treatment effect. Default is NULL.
#' @param prior_treatment_effect_SD An optional numeric value for the prior standard deviation of the treatment effect. Default is NULL.
#' @param true_treatment_effect An optional numeric value specifying the true treatment effect. Default is NULL.
#' @param simulate A logical value indicating whether to run `RandomizationTest` in simulation mode. Default is FALSE.
#' @param coef_prior An optional function generating coefficients on values of `X` for predicting `Y(0)`.
#' @param nSimulate_obsW A numeric value specifying the number of simulated values for obsW. Default is `50L`.
#' @param nSimulate_obsY A numeric value specifying the number of simulated values for obsY. Default is `50L`.
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

RandomizationTest <- function(
                               obsW = NULL,
                               obsY = NULL,
                               X = NULL,
                               alpha = 0.05,
                               candidate_randomizations = NULL,
                               candidate_randomizations_array = NULL,
                               n0_array = NULL,
                               n1_array = NULL,
                               prior_treatment_effect_mean = NULL,
                               prior_treatment_effect_SD = NULL,
                               true_treatment_effect = NULL,
                               simulate = F ,
                               coef_prior = NULL,
                               nSimulate_obsW = 50L,
                               nSimulate_obsY = 50L,
                               randomization_accept_prob = 1.,
                               findFI = F,
                               c_initial = 2){
  tau_obs <- FI <- covers_truth <- NULL

  if(!simulate){
    if(is.null(n0_array)){ n0_array <- jnp$array(sum(obsW == 0)) }
    if(is.null(n1_array)){ n1_array <- jnp$array(sum(obsW == 1)) }
  }
  if(simulate){
    if(is.null(n0_array)){ n0_array <- jnp$array(nrow(X)/2) }
    if(is.null(n1_array)){ n1_array <- jnp$array(nrow(X)/2) }
  }
  if(!is.null(obsW)){obsW <- c(unlist(obsW))}
  if(!is.null(X)){X <- as.matrix(X)}
  if(!is.null(obsY)){obsY <- c(unlist(obsY))}

  if(is.null(candidate_randomizations_array) & is.null(candidate_randomizations)){
    n_treated <- ( n_units <- nrow(X) ) / 2
    if(simulate == T){
      candidate_randomizations_array <- GenerateRandomizations(n_units, n_treated)
    }
    if(simulate == F){
      candidate_randomizations_array <- GenerateRandomizations(
                                            n_units = n_units,
                                            n_treated = n_treated,
                                            X = X,
                                            randomization_accept_prob = randomization_accept_prob)
    }
  }
  if(is.null(candidate_randomizations)){
      candidate_randomizations <- np$array( candidate_randomizations_array )
  }
  if(is.null(candidate_randomizations_array)){
      candidate_randomizations_array <- jnp$array(candidate_randomizations, dtype = jnp$float32)
  }

  # simulate generates new (synthetic values) of Y_obs
  if( simulate ){
    obsY1_array <- jnp$array( replicate(nSimulate_obsY, {
      prior_coef_draw <- coef_prior()
      Y_0 <- X %*% prior_coef_draw

      #tau_samp <- rnorm(n=n_units, mean=prior_treatment_effect_mean, sd = prior_treatment_effect_SD) # assumes tau_i
      tau_samp <- rnorm(n=1, mean=prior_treatment_effect_mean, sd = prior_treatment_effect_SD) # assumes one tau
      Y_1 <- Y_0 + tau_samp

      return(cbind(Y_0, Y_1))
    }) )
    obsY1_array <- jnp$transpose(obsY1_array, axes = c(2L,0L,1L))
    obsY0_array <- jnp$take(obsY1_array, 0L, axis = 2L)
    obsY1_array <- jnp$take(obsY1_array, 1L, axis = 2L)

    chi_squared_approx <- F
    M_results <- jnp$squeeze(VectorizedFastHotel2T2(jnp$array( X ),
                                                    candidate_randomizations_array,
                                                    n0_array, n1_array), 1L:2L)
    a_threshold_vec <- jnp$quantile(M_results, jnp$array(prob_accept_randomization_seq))
    M_results <- np$array( M_results )
    #if(chi_squared_approx==T){ a_threshold <- qchisq(p=prob_accept_randomization_seq[ii], df=k_covars) }

    GetPvals_vmapped <- jax$jit(jax$vmap( function( sampledIndex, acceptedWs_array ){
      print2("Jitting...")
      obsW_ <- VectorizedTakeAxis0_R(acceptedWs_array, sampledIndex)
      obsY_array <- Potential2Obs_R(obsY0_array, obsY1_array, obsW_)
      tau_obs <- Y_VectorizedFastDiffInMeans_R(obsY_array, obsW_, n0_array, n1_array)
      tau_perm_null_0 <-  YW_VectorizedFastDiffInMeans_R(
        obsY_array,  # y_ =
        acceptedWs_array, # w_ =
        n0_array, # n0 =
        n1_array) # n1 =
      p_value_inner <- GreaterEqualMagCompare_R(tau_perm_null_0, tau_obs) # pvals across yhat
      return( p_value_inner ) #  mean here is over Yhat
    }, in_axes = list(0L,NULL)))
    p_value <- sapply(np$array(a_threshold_vec),   function(a_){
      print(a_ / max(np$array(a_threshold_vec)))
      # a_ <- np$array(a_threshold_vec)[[30]]
      py_gc$collect()

      # select acceptable randomizations based on threshold
      acceptedWs_array <- candidate_randomizations_array[jnp$less_equal( M_results,  a_),]
      sampTheseIndices <- 0L:((AcceptedRandomizations <- acceptedWs_array$shape[[1]])  - 1L)
      sampledIndices <- jnp$expand_dims(jnp$array( as.integer(as.numeric(
                              sample(as.character(sampTheseIndices), nSimulate_obsW, replace = T) ))),1L)
      p_value_outer <-  jnp$mean(GetPvals_vmapped(sampledIndices, acceptedWs_array))
    })
    p_value <- np$array( p_value )
    suggested_randomization_accept_prob <- prob_accept_randomization_seq[ which.min(p_value)[1] ]
    plot( p_value )
  }

  # perform randomization inference using input data
  if( !simulate ){
    tau_obs <- c(np$array( FastDiffInMeans(jnp$array(obsY),
                                           jnp$array(obsW),
                                           n0_array,
                                           n1_array) ))
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
            #lower_tau_at_step_t <- mean(lower_Y_obs_perm[permutation_treatment_vec == 1]) - mean(lower_Y_obs_perm[permutation_treatment_vec == 0])
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

  if( !simulate ){
    return( list(p_value = p_value,
                 FI = FI,
                 tau_obs = tau_obs) )
  }
  if( simulate ){
    return( list(p_value = p_value,
                 suggested_randomization_accept_prob = suggested_randomization_accept_prob,
                 FI = FI,
                 tau_obs = tau_obs) )
  }
}




#!/usr/bin/env Rscript
#' Fast generation of all possible complete randomizations given target number of experimental units.
#'
#' @param n_units A integer specifying total number of experimental units.
#' @param n_treated An integer specifying total number of treated units.
#'
#' @return A JAX array containing all possible complete randomizations.
#' @examples
#' # For a tutorial, see
#' # github.com/cjerzak/fastrerandomization-software
#'
#' @export
#' @md

GenerateRandomizations <- function(n_units, n_treated,
                                   X = NULL,
                                   randomization_accept_prob = 1){
  # Get all combinations of positions to set to 1
  combinations <- jnp$array(  combn(n_units, n_treated) - 1L )
  ZerosHolder <- jnp$zeros(as.integer(n_units), dtype=jnp$int32)
  candidate_randomizations <- InsertOnesVectorized(combinations,
                                             ZerosHolder)

  if(!is.null(X)){
    n0_array <- jnp$array(  (n_units - n_treated) )
    n1_array <- jnp$array(  n_treated )
    # samp_ <- jnp$array( X ); w_ <- jnp$array(candidate_randomizations, dtype = jnp$float32); n0 <- n1 <- n0_array
    M_results <-  VectorizedFastHotel2T2(jnp$array( X ) ,
                                         jnp$array(candidate_randomizations, dtype = jnp$float32),
                                         n0_array, n1_array)
    a_threshold <- np$array(jnp$quantile( M_results,  jnp$array(randomization_accept_prob)))[[1]]
    M_results <- c(np$array( M_results ))
    candidate_randomizations <- jnp$take(candidate_randomizations,
                                         indices = jnp$array(which(M_results <= a_threshold)-1L),
                                         axis = 0L)
  }

  return( candidate_randomizations )
}
