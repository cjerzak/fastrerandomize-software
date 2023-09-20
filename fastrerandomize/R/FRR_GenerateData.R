#!/usr/bin/env Rscript
#' Generate data
#'
#' @usage
#'
#' generate_data()
#'
#' @param

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

generate_data <- function(n_units, proportion_treated, k_covars, rho, SD_inherent,
                          treatment_effect_mean, treatment_effect_SD, covariates_SD){
  #Y_0_coefficients <- as.matrix(rep(true_treatment_effect/k_covars, times=k_covars)) #as.matrix( c(c(1:k_covars)/2) )
  Y_0_coefficients <- as.matrix(rnorm(k_covars, sd = 1/1:k_covars))
  Y_1_coefficients <- Y_0_coefficients #as.matrix( c(c(1:k_covars)/2)  )

  if( k_covars == 1 ){ Sigma_X <- covariates_SD^2 * diag(k_covars) }
  if( k_covars > 1 )
  {
    Sigma_X <- matrix(rho, nrow=k_covars, ncol=k_covars)
    diag(Sigma_X) <- diag( covariates_SD^2 * diag(k_covars) )
    if( any(eigen(Sigma_X)$values < 0) )
    {
      okay <- F
      while(okay==F)
      {
        Sigma_X <- matrix(rho, nrow=k_covars, ncol=k_covars)
        diag(Sigma_X) <- diag( covariates_SD^2 * diag(k_covars) )
        if( any(eigen(Sigma_X)$values < 0 ) ) { rho <- rho - 0.001; okay <- F}
        if( all(eigen(Sigma_X)$values >= 0) ) { okay <- T }
      }
    }
  }

  #draw data matrix
  data_matrix <- as.data.frame( mvtnorm::rmvnorm(n = n_units,
                                                 mean = rep(0, k_covars),
                                                 sigma = Sigma_X) )
  #setup outcome
  data_matrix <- as.matrix(data_matrix)
  Y_0 <- data_matrix[,1:k_covars] %*%  Y_0_coefficients + 0 + rnorm(n=n_units, 0, sd=SD_inherent)
  Y_1 <- data_matrix[,1:k_covars] %*%  Y_1_coefficients + rnorm(n=n_units, mean=treatment_effect_mean, sd=treatment_effect_SD) + rnorm(n=n_units, 0, sd=SD_inherent)
  data_matrix <- as.data.frame( cbind(as.data.frame(data_matrix), Y_0, Y_1) )
  return(list("data_matrix"=data_matrix,
              "Y_0_coefficients" = Y_0_coefficients,
              "Y_1_coefficients" = Y_1_coefficients))
}
