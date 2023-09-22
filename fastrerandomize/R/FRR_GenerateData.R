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
                          treatment_effect_mean, treatment_effect_SD, covariates_SD,
                          Y0_coefficients = NULL, Y1_coefficients = NULL){
  if(is.null(Y0_coefficients)){
    Y0_coefficients <- as.matrix(rnorm(k_covars, sd = 1/1:k_covars))
    Y1_coefficients <- Y0_coefficients #as.matrix( c(c(1:k_covars)/2)  )
  }

  if( k_covars == 1 ){ Sigma_X <- covariates_SD^2 * diag(k_covars) }
  if( k_covars > 1 ){
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

  # draw data matrix
  data_matrix <- as.data.frame( mvtnorm::rmvnorm(n = n_units,
                                                 mean = rep(0, k_covars),
                                                 sigma = Sigma_X) )
  # setup outcome
  data_matrix <- as.matrix(data_matrix)
  Y0 <- data_matrix[,1:k_covars] %*%  Y0_coefficients + 0 + rnorm(n=n_units, 0, sd=SD_inherent)
  Y1 <- data_matrix[,1:k_covars] %*%  Y1_coefficients + rnorm(n=n_units, mean=treatment_effect_mean, sd=treatment_effect_SD) + rnorm(n=n_units, 0, sd=SD_inherent)
  data_matrix <- as.data.frame( cbind(as.data.frame(data_matrix), Y0, Y1) )
  return(list("data_matrix"=data_matrix,
              "Y0_coefficients" = Y0_coefficients,
              "Y1_coefficients" = Y1_coefficients))
}
