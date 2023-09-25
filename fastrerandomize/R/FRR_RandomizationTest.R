#!/usr/bin/env Rscript
#' Fast randomization test
#'
#' @usage
#'
#' RandomizationTest(X, ...)
#'
#' @param obsW A numeric vector where `0`'s correspond to control units and `1`'s to treated units.
#' @param obsY A numeric vector containing observed outcomes.
#' @param X A numeric matrix of covariates.
#' @param obsY An optional numeric vector of observed outcomes. If not provided, the function assumes a NULL value.
#' @param obsW A numeric vector where `0`'s correspond to control units and `1`'s to treated units.
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
                               obsW,
                               obsY = NULL,
                               X = NULL,
                               alpha = 0.05,
                               candidate_randomizations,
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
                               randomization_accept_prob = NULL,
                               findFI = F,
                               c_initial = 2){
  tau_obs <- FI <- covers_truth <- NULL

  if(is.null(candidate_randomizations_array)){
    candidate_randomizations_array <- jnp$array( candidate_randomizations )
  }
  if(is.null(n0_array)){ n0_array <- jnp$array(sum(obsW == 0)) }
  if(is.null(n1_array)){ n0_array <- jnp$array(sum(obsW == 1)) }

  # simulate generates new (synthetic values) of Y_obs
  if(simulate==T){
    obsY1_array <- jnp$array( replicate(nSimulate_obsY, {
      prior_coef_draw <- coef_prior()
      Y_0 <- X %*% prior_coef_draw

      #tau_samp <- rnorm(n=n_units, mean=prior_treatment_effect_mean, sd = prior_treatment_effect_SD)
      tau_samp <- rnorm(n=1, mean=prior_treatment_effect_mean, sd = prior_treatment_effect_SD)
      Y_1 <- Y_0 + tau_samp

      return(cbind(Y_0, Y_1))
    }) )
    obsY1_array <- jnp$transpose(obsY1_array, axes = c(2L,0L,1L))
    obsY0_array <- jnp$take(obsY1_array,0L, axis = 2L)
    obsY1_array <- jnp$take(obsY1_array,1L, axis = 2L)

    chi_squared_approx <- F
    #candidate_randomizations <- candidate_randomizations[sample(1:nrow(candidate_randomizations), nrow(candidate_randomizations)),]
    if(is.null(candidate_randomizations_array)){
      candidate_randomizations <- np$array( candidate_randomizations_array )
    }
    if(is.null(candidate_randomizations_array)){
      candidate_randomizations_array <- jnp$array(candidate_randomizations, dtype = jnp$float32)
    }
    M_results <- jnp$squeeze(VectorizedFastHotel2T2(jnp$array( X ),
                                                    candidate_randomizations_array,
                                                    n0_array, n1_array), 1L:2L)
    a_threshold_vec <- jnp$quantile(M_results, jnp$array(prob_accept_randomization_seq))
    #a_threshold_vec <- quantile(np$array(M_results), prob_accept_randomization_seq)
    M_results <- np$array( M_results )
    #if(chi_squared_approx==T){ a_threshold <- qchisq(p=prob_accept_randomization_seq[ii], df=k_covars) }
    #if(chi_squared_approx==F){ a_threshold <- quantile(x=M_results, probs=prob_accept_randomization_seq[ii]) }

    #success_index <- which(M_results <= a_threshold)
    #permutations_accept_ii <- permutation_matrix[success_index, ]

    p_value <- sapply(np$array(a_threshold_vec), function(a_){
          acceptedWs_array <- jnp$take( candidate_randomizations_array,
                                        jnp$where( jnp$less_equal( M_results,  a_))[[1]], axis = 0L)
          AcceptedRandomizations <- acceptedWs_array$shape[[1]]
          sampTheseIndices <- 0L:(AcceptedRandomizations-1L)

          p_value_outer_vec <- c(replicate(nSimulate_obsW, {
            obsW_ <- VectorizedTakeAxis0(acceptedWs_array,
                                         jnp$array(sample(sampTheseIndices,1)))
            obsY_array <- Potentisl2Obs(obsY0_array, obsY1_array, obsW_)
            tau_obs <- Y_VectorizedFastDiffInMeans(obsY_array,
                                                   obsW_,
                                                   n0_array,
                                                   n1_array)

            tau_perm_null_0 <-  YW_VectorizedFastDiffInMeans(
                obsY_array,  # y_ =
                acceptedWs_array, # t_ =
                n0_array, # n0 =
                n1_array # n1 =
              )
            p_value_inner_vec <- np$array( GreaterEqualMagCompare(tau_perm_null_0, tau_obs) )
            mean( p_value_inner_vec )
          }))
      } )
    # plot(colMeans(p_value))
  }

  if(simulate == F){
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
  }

  if(findFI == T){
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
    FI <- c(lowerBound_storage_vec[length(lowerBound_storage_vec)], upperBound_storage_vec[length(upperBound_storage_vec)])

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

        #quantiles_ <- c(quantile(stat_vec_at_tau_pseudo,alpha/2), quantile(stat_vec_at_tau_pseudo,1-alpha/2))
        #quantiles_ <- np$array(jnp$stack( list(jnp$quantile(stat_vec_at_tau_pseudo, alpha/2), jnp$quantile(stat_vec_at_tau_pseudo, 1-alpha/2)), 0L))

        ret_ <- min(mean( tau_obs >= stat_vec_at_tau_pseudo),
                    mean( tau_obs <= stat_vec_at_tau_pseudo))
        #ret_ <- reject_ <- tau_obs >= quantiles_[1] & tau_obs <= quantiles_[2]
        return( ret_ )
      } )
      # checks
      #plot( tau_pseudo_seq,  pvals_vec );abline(h=0.05,col="gray",lty=2); abline(v=FI[1],lty=2); abline(v=FI[2],lty=2); abline(v=tau_obs)
      FI <- summary(tau_pseudo_seq[pvals_vec>0.05])[c(1,6)]
    }
  }

  return_list <- list(
                      p_value = p_value,
                      FI = FI,
                      tau_obs = tau_obs
                      )
  return(return_list)
}
