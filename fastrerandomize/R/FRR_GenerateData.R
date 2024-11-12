#!/usr/bin/env Rscript
#' This function generates simulated causal data based on specified parameters. The functional form of the outcome models is:
#' \deqn{Y_0 = X \beta_0 + \epsilon_0}
#' \deqn{Y_1 = X \beta_1 + \tau + \epsilon_1}
#' where \eqn{\tau} is the treatment effect, which is drawn from a normal distribution with mean `treatment_effect_mean` and standard deviation `treatment_effect_SD`.
#' The dimension of \eqn{\beta_0} and \eqn{\beta_1} is `k_covars`.
#' The correlation coefficient of the covariates is `rho`.
#' Y0_coefficients and Y1_coefficients are optional arguments that can be provided to specify the coefficients for the control and treated outcome models, and they determine \eqn{\beta_0} and \eqn{\beta_1}.
#' If they are not provided, the function assumes a NULL value, and the coefficients are drawn from a normal distribution with decreasing variance.
#' Example usage: 
#' ```
#' GenerateCausalData(n_units = 100, proportion_treated = 0.5, k_covars = 3, rho = 0.5, SD_inherent = 1, treatment_effect_mean = 0, treatment_effect_SD = 1, covariates_SD = 1)
#' ```
#' 
#' @param n_units A numeric value specifying the total number of units in the sample.
#' @param proportion_treated A numeric value between 0 and 1 indicating the proportion of units that receive treatment.
#' @param k_covars A numeric value indicating the number of covariates to be generated.
#' @param rho A numeric value representing the correlation coefficient of the covariates.
#' @param SD_inherent A numeric value indicating the standard deviation inherent to the data.
#' @param treatment_effect_mean A numeric value representing the mean of the treatment effect.
#' @param treatment_effect_SD A numeric value indicating the standard deviation of the treatment effect.
#' @param covariates_SD A numeric value or vector specifying the standard deviation of the covariates.
#' @param Y0_coefficients An optional numeric vector specifying the coefficients for the control outcome model. If not provided, the function assumes a NULL value, and the coefficients are drawn from a normal distribution with decreasing variance.
#' @param Y1_coefficients An optional numeric vector specifying the coefficients for the treated outcome model. If not provided, the function assumes a NULL value, and the coefficients are drawn from a normal distribution with decreasing variance.
#'
#' @return A list consisting of \itemize{
#'   \item `data_matrix` A data frame containing the simulated covariates and outcomes for both control (Y0) and treatment (Y1) groups. Access them through `data_matrix$Y0` and `data_matrix$Y1`.
#'   \item `Y0_coefficients` A numeric vector representing the coefficients used for the control outcome model.
#'   \item `Y1_coefficients` A numeric vector representing the coefficients used for the treated outcome model.
#' }
#'
#'
#' @examples
#' # For a tutorial, see
#' # github.com/cjerzak/fastrerandomization-software
#'
#' @export
#' @md

# 
GenerateCausalData <- function(n_units, proportion_treated, k_covars, rho, SD_inherent,
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
  Y0 <- data_matrix[,1:k_covars] %*%  Y0_coefficients + rnorm(n=n_units, 0, sd=SD_inherent)
  Y1 <- data_matrix[,1:k_covars] %*%  Y1_coefficients + rnorm(n=n_units, mean=treatment_effect_mean, sd=treatment_effect_SD) + rnorm(n=n_units, 0, sd=SD_inherent)
  data_matrix <- as.data.frame( cbind(as.data.frame(data_matrix), Y0, Y1) )
  return(list("data_matrix"=data_matrix,
              "Y0_coefficients" = Y0_coefficients,
              "Y1_coefficients" = Y1_coefficients))
}



#' Perform sanity checks on synthetic data
#'
#' This function performs several sanity checks on synthetic data to ensure the quality
#' of the generated dataset and the strength of relationships between variables.
#'
#' @param synthetic_data A list containing:
#'   \itemize{
#'     \item data_matrix - Matrix containing the synthetic data
#'     \item Y0_coefficients - Coefficients for potential outcome Y0
#'     \item Y1_coefficients - Coefficients for potential outcome Y1
#'   }
#'
#' @return A list of 4 linear models:
#'   \itemize{
#'     \item lm_model_Y0 - Linear model for Y0 ~ X
#'     \item lm_model_Y1 - Linear model for Y1 ~ X
#'     \item lm_model_obsY - Linear model for observed Y ~ X
#'     \item lm_model_obsY_obsW - Linear model for treatment effect
#'   }
#'
#' @details
#' The function performs the following checks:
#' 1. Verifies R-squared > 0.01 for Y0 and Y1 regressed on X
#' 2. Checks out-of-sample R-squared > 0.01 for Y0 and Y1 predictions
#' 3. Confirms treatment effect is statistically significant (p < 0.05)
#'
#' @examples
#' \dontrun{
#' synthetic_data <- generate_synthetic_data()
#' models <- sanity_check_synthetic_data(synthetic_data)
#' }
#' @md
#' @export
sanity_check_synthetic_data <- function(synthetic_data, InSampleR_threshold = 0.01, OOS_R_threshold = 0.01, treatment_pval_threshold = 0.05) {
            data_matrix <- synthetic_data$data_matrix
            Y0_coefficients <- synthetic_data$Y0_coefficients
            Y1_coefficients <- synthetic_data$Y1_coefficients
            Y0 <- data_matrix$Y0
            Y1 <- data_matrix$Y1
            obsW <- sample(c(rep(0, length(Y0)/2), rep(1, length(Y0)/2)), replace = FALSE)
            obsY <- Y0 * (1 - obsW) + Y1 * obsW
            X <- data_matrix[,1:ncol(X)]
            X <- as.matrix(X)
            data_matrix <- synthetic_data$data_matrix
            Y0_coefficients <- synthetic_data$Y0_coefficients
            Y1_coefficients <- synthetic_data$Y1_coefficients
            Y0 <- data_matrix$Y0
            Y1 <- data_matrix$Y1
            obsW <- sample(c(rep(0, length(Y0)/2), rep(1, length(Y0)/2)), replace = FALSE)
            obsY <- Y0 * (1 - obsW) + Y1 * obsW
            X <- data_matrix[,1:ncol(X)]
            X <- as.matrix(X)

            lm_model_Y0 <- lm(Y0 ~ X)
            assert_that(summary(lm_model_Y0)$r.squared > InSampleR_threshold, msg = "R-squared for Y0 is not greater than 0.01")
            lm_model_Y1 <- lm(Y1 ~ X)
            assert_that(summary(lm_model_Y1)$r.squared > InSampleR_threshold, msg = "R-squared for Y1 is not greater than 0.01")
            lm_model_obsY <- lm(obsY ~ X)
            assert_that(summary(lm_model_obsY)$r.squared > InSampleR_threshold, msg = "R-squared for obsY is not greater than 0.01")
            
            # Calculate Out of Sample R-squared for Y0 and Y1
            Y0_mean <- mean(Y0)
            Y1_mean <- mean(Y1)
            Y0_pred <- X %*% Y0_coefficients
            Y1_pred <- X %*% Y1_coefficients + 0.5
            Y0_r2 <- 1 - sum((Y0 - Y0_pred)^2) / sum((Y0 - Y0_mean)^2)
            Y1_r2 <- 1 - sum((Y1 - Y1_pred)^2) / sum((Y1 - Y1_mean)^2)
            assert_that(Y0_r2 > OOS_R_threshold, msg = "R-squared for Y0 is not greater than 0.01")
            assert_that(Y1_r2 > OOS_R_threshold, msg = "R-squared for Y1 is not greater than 0.01")

            # Fit linear model and extract p-value for treatment effect
            lm_model_obsY_obsW <- lm(obsY ~ obsW)
            treatment_pval <- summary(lm_model_obsY_obsW)$coefficients["obsW", "Pr(>|t|)"]
            assert_that(!is.na(treatment_pval), msg = "Treatment effect p-value is NA")
            assert_that(treatment_pval < treatment_pval_threshold, msg = "Treatment effect p-value is not less than 0.05")
            return c(lm_model_Y0, lm_model_Y1, lm_model_obsY, lm_model_obsY_obsW)
        }
