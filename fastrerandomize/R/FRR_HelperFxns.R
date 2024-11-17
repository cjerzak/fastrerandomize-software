# Imports
scale_fxn <- function(old_seq, newmin, newmax){(newmax - newmin) * (old_seq - min(old_seq)) / (max(old_seq) - min(old_seq)) + newmin}

mahalanobis_distance <- function(data_matrix_internal){
  treatment_indices <- which(data_matrix_internal$treatment_indicator==1)
  n <- length(data_matrix_internal$treatment_indicator)
  p_w <- sum(data_matrix_internal$treatment_indicator) / n
  X <- as.matrix( data_matrix_internal[,1:k_covars]  )
  cov_X <- t(X) %*% X

  X_c <- as.matrix( data_matrix_internal[-treatment_indices,1:k_covars] )
  X_t <- as.matrix( data_matrix_internal[treatment_indices,1:k_covars] )

  X_c_bar <- as.matrix( colMeans(X_c) )
  X_t_bar <- as.matrix( colMeans(X_t) )

  M <- n * (n-1) * p_w * (1-p_w) * t(X_t_bar - X_c_bar) %*% solve(cov_X) %*% (X_t_bar - X_c_bar)
  return(as.vector(M))
}

approx_log_n_choose_m <- function(n, m){
  #http://math.stackexchange.com/questions/64716/approximating-the-logarithm-of-the-binomial-coefficient
  result <- n * log(n) - m * log(m) - (n - m)*log(n-m) + 0.5 * (log(n) - log(m) - log(n-m) - log(2*pi))
  return(result)
}

generalized_cauchy <- function(x, s, t){
  density_at_x <- 1 / (s*pi*(1 + ((x - t)/s)^2 ) )
  return(density_at_x)
}

jth_derivative_log_min_p_val_OLD <- function(p_a, n_randomizations, derivative){
  if(derivative==1){return( -1/p_a  )}
  if(derivative==2){return(  1/p_a^2  )}
  if(derivative==3){return( -2/p_a^3  )}
  if(derivative==4){return(  6/p_a^4  )}
}

jth_derivative_min_p_val_OLD <- function(p_a, n_randomizations, derivative){
  if(derivative==1){return(  -1/( (n_randomizations+1)*p_a^2) )}
  if(derivative==2){return(   2/( (n_randomizations+1)*p_a^3) )}
  if(derivative==3){return(  -6/( (n_randomizations+1)*p_a^4) )}
  if(derivative==4){return(  24/( (n_randomizations+1)*p_a^5) )}
}

#analytical first derivative function
derivative_v_a <- function(a, k){
  cdf_k <- pchisq(q=a, df=k)
  pdf_k <- dchisq(x=a, df=k)
  cdf_k_plus_2 <- pchisq(q=a, df=k+2)
  pdf_k_plus_2 <- dchisq(x=a, df=k+2)

  value <- (pdf_k_plus_2 * cdf_k - cdf_k_plus_2 * pdf_k) / cdf_k^2
  return(value)
}

v_a_value <- function(k_covars, a){
  value <- pchisq(q=a, df=k_covars + 2) / pchisq(q=a, df=k_covars)
  return( value )
}

expected_min_p_value <- function(n, p_a){
  value <- 1 / (p_a * (n + 1 ) )  - (1 - p_a)^(n+ 1) / (p_a * (n + 1))
  return(value)
}

expected_min_p_value_first_derivative <-  function(n, p_a){
  value <- - 1 /  ((n+1)*p_a^2 ) - (1 - p_a)^(n+1) * (n*p_a + 1) / ((n + 1)*p_a^2)
  return(value)
}

expected_min_p_value_second_derivative <- function(n, p_a){
  value <- 2 / ( (n+1) * p_a^3 ) + 2 * (n*p_a + 1) * (1-p_a)^n / ( (n+1) * p_a^3 ) +
    n * (n * p_a + 1 ) * (1 - p_a)^(n-1) / ( (n+1) * p_a^2 )
  return( value  )
}

v_a_plus_p_a_derivative <- function(p_a, n, k_covars , epsilon=0.0001){
  p_a_new <- p_a + epsilon
  a <- qchisq(p=p_a, df=k_covars)
  a_new <- qchisq(p=p_a_new, df=k_covars)

  value_old <- expected_min_p_value(n=n_randomizations, p_a=p_a) + v_a_value(k_covars=k_covars, a=a)
  value_new <- expected_min_p_value(n=n_randomizations, p_a=p_a_new) + v_a_value(k_covars=k_covars, a=a_new)
  numerator <- value_new - value_old
  denominator <- p_a_new - p_a
  return_value <- numerator / denominator
  return( return_value )
}

#' Print timestamped messages with optional quieting
#'
#' This function prints messages prefixed with the current timestamp in a standardized format.
#' Messages can be suppressed using the quiet parameter.
#'
#' @param text A character string containing the message to be printed
#' @param quiet A logical value indicating whether to suppress output. Default is FALSE
#'
#' @return No return value, called for side effect of printing
#'
#' @examples
#' # Print a basic message with timestamp
#' print2("Processing started")
#'
#' # Suppress output
#' print2("This won't show", quiet = TRUE)
#'
#' # Use in a loop
#' for(i in 1:3) {
#'   print2(sprintf("Processing item %d", i))
#' }
#'
#' @details
#' The function prepends the current timestamp in "YYYY-MM-DD HH:MM:SS" format
#' to the provided message. 
#'
#' @seealso
#' \code{Sys.time} for the underlying timestamp functionality
#'
#' @export
#' @md
print2 <- function(text, quiet = F){
  if(!quiet){
    print( sprintf("[%s] %s" ,format(Sys.time(), "%Y-%m-%d %H:%M:%S"),text) )
  }
}

JaxKey <- function(int_){ jax$random$PRNGKey(int_)}

SoftPlus_r <- function(x){ log(exp(x)+1) }