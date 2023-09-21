#!/usr/bin/env Rscript
#' Fast randomization test
#'
#' @usage
#'
#' randomization_test(X, ...)
#'
#' @param obsW A numeric vector where `0`'s correspond to control units and `1`'s to treated units.
#' @param obsY A numeric vector containing observed outcomes.

#' @return A list consiting of \itemize{
#'   \item `pval` A p-value.
#' }
#'
#' @section References:
#' \itemize{
#' \item
#' }
#'
#' @examples
#' # For a tutorial, see
#' # github.com/cjerzak/fastrerandomization
#'
#' @export
#' @md

randomization_test <- function(
                               X,
                               obsY = NULL,
                               obsW,
                               c_initial = 2,
                               alpha = 0.05,
                               input_permutation_matrix,
                               input_permutation_matrix_array,
                               n0_array,
                               n1_array,
                               prior_treatment_effect_mean = NULL,
                               prior_treatment_effect_SD = NULL,
                               true_treatment_effect = NULL,
                               simulate=F,
                               coef_prior = NULL,
                               nSimulate = NULL,
                               findCI = F){
  CI <- CI_width <- covers_truth <- zero_in_CI <- NULL

  # simulate generates new (synthetic values) of Y_obs
  if(simulate==T){
    if(ncol(input_permutation_matrix) == 1){ input_permutation_matrix <- t(input_permutation_matrix)}

    {
      prior_coef_draw <- coef_prior()
      Y_0 <- X %*% prior_coef_draw

      #tau_samp <- rnorm(n=n_units, mean=prior_treatment_effect_mean, sd = prior_treatment_effect_SD)
      tau_samp <- rnorm(n=1, mean=prior_treatment_effect_mean, sd = prior_treatment_effect_SD)
      Y_1 <- Y_0 + tau_samp
      #obsW <- input_permutation_matrix[sample(1:nrow(input_permutation_matrix),1),]
    }

    #observed_value considerations
    #data_matrix_original$treated[obsW == 1] <- 1
    #data_matrix_original$treated[obsW == 0] <- 0

    obsY[obsW == 0] <- Y_0[obsW == 0]
    obsY[obsW == 1] <- Y_1[obsW == 1]
  }

  #tau_obs <- mean(obsY[obsW == 1]) - mean(obsY[obsW == 0])
  tau_obs <- c(np$array( FastDiffInMeans(jnp$array(obsY),
                                         jnp$array(obsW),
                                         n0_array,
                                         n1_array) ))

  #fix if nrow = 0
  if( ncol(input_permutation_matrix)==1 ) { input_permutation_matrix <- t(input_permutation_matrix)}
  # if(T == F){tau_perm_null_0 <- apply( input_permutation_matrix, 1, function(perm_treat_vec) mean(obsY[perm_treat_vec == 1]) - mean(obsY[perm_treat_vec == 0]) ) }
  if(T == T){
    tau_perm_null_0 <- np$array(
      VectorizedFastDiffInMeans(
        jnp$array(obsY),  # y_ =
        input_permutation_matrix_array, # t_ =
        n0_array, # n0 =
        n1_array # n1 =
      ))
  }
  # print( st )
  p_value <- mean( abs(tau_perm_null_0) >= abs(tau_obs) )

  if(findCI == T){
    n_search_attempts <- 500
    exhaustive_search  <-  nrow(input_permutation_matrix) <= n_search_attempts
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
      for(step_t in 1:n_search_attempts)
      {
        #initialize for next step
        permutation_treatment_vec <- input_permutation_matrix[sample(1:nrow(input_permutation_matrix), size=1),]
        lower_Y_0_under_null <- lower_Y_obs_perm <- rep(NA, length(obsW))
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
          upper_tau_at_step_t <- mean(upper_Y_obs_perm[permutation_treatment_vec == 1]) - mean(upper_Y_obs_perm[permutation_treatment_vec == 0])

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
    CI <- c(lowerBound_storage_vec[length(lowerBound_storage_vec)], upperBound_storage_vec[length(upperBound_storage_vec)])
    CI_width <- abs( max(CI) - min(CI) )
    zero_in_CI <- 1 * ( min(CI) < 0  &  max(CI) > 0 )
    covers_truth <- 1 * ( min(CI) < true_treatment_effect  &  max(CI) > true_treatment_effect )

    if(T == T){
      tau_pseudo_seq <- seq(CI[1]-1, CI[2]*2,length.out=100)
      pvals_vec <- sapply(tau_pseudo_seq, function(tau_pseudo){
        stat_vec_at_tau_pseudo <- apply(input_permutation_matrix,1,function(treatment_pseudo){
          Y0_under_null <- obsY - obsW*tau_pseudo
          #Y0_under_null1 <- (obsY-tau_pseudo)*obsW + obsY*(1-obsW)
          Y1_under_null_pseudo <- Y0_under_null + treatment_pseudo*tau_pseudo
          Yobs_pseudo <- Y1_under_null_pseudo*treatment_pseudo + Y0_under_null * (1-treatment_pseudo)
          #stat_ <- mean(Y1_under_null_pseudo[treatment_pseudo == 1]) - mean(Y0_under_null[treatment_pseudo == 0])
          stat_ <- mean(Yobs_pseudo[treatment_pseudo == 1]) - mean(Yobs_pseudo[treatment_pseudo == 0])
        })
        quantiles_ <- c(quantile(stat_vec_at_tau_pseudo,alpha/2), quantile(stat_vec_at_tau_pseudo,1-alpha/2))
        #my_ecdf <- ecdf(stat_vec_at_tau_pseudo); ret_ <- min(my_ecdf(tau_obs),(1-my_ecdf(tau_obs))) * 2
        ret_ <- min(mean( tau_obs >= stat_vec_at_tau_pseudo),
                    mean( tau_obs <= stat_vec_at_tau_pseudo))
        #ret_ <- reject_ <- tau_obs >= quantiles_[1] & tau_obs <= quantiles_[2]
        return( ret_ )
      } )
      #plot( tau_pseudo_seq,  pvals_vec );abline(h=0.05,col="gray",lty=2); abline(v=CI[1],lty=2); abline(v=CI[2],lty=2); abline(v=tau_obs)

      #CI <- summary(tau_pseudo_seq[pvals_vec])[c(1,6)]
      CI <- summary(tau_pseudo_seq[pvals_vec>0.05])[c(1,6)]
      CI_width <- abs( max(CI) - min(CI) )
    }
  }

  return_list <- list(CI=CI,
                      p_value=p_value,
                      CI_width=CI_width,
                      covers_truth=covers_truth,
                      zero_in_CI=zero_in_CI,
                      true_effect=true_treatment_effect,
                      tau_obs=tau_obs)
  return(return_list)
}
