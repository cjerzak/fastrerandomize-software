#' Fast randomization test
#' 
#' 
#'
#'
#' @param obsW A numeric vector where `0`'s correspond to control units and `1`'s to treated units.
#' @param obsY A required numeric vector of observed outcomes with the same length as
#'   \code{obsW}.
#' @param alpha The significance level for the test. Default is `0.05`.
#' @param candidate_randomizations A numeric matrix of candidate randomizations.
#' @param candidate_randomizations_array An optional 'JAX' array of candidate randomizations. If not provided, the function coerces `candidate_randomizations` into a 'JAX' array.
#' @param n0_array An optional array specifying the number of control units.
#' @param n1_array An optional array specifying the number of treated units.
#' @param findFI A logical value indicating whether to find the fiducial interval. Default is FALSE.
#' @param c_initial A finite positive step-size scale for the fiducial interval
#'   search. The default of \code{2} preserves the standard update scale; smaller
#'   or larger values decrease or increase the stochastic update size.
#' @param conda_env A character string specifying the name of the conda environment to use 
#'   via \code{reticulate}. Default is \code{"fastrerandomize_env"}.
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
#'   candidate_randomizations = RandomizationSet_MC$randomizations
#' )
#' print(results_base)
#'
#' # Perform randomization test with fiducial interval
#' result_fi <- randomization_test(
#'   obsW = W,
#'   obsY = obsY,
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
                               alpha = 0.05,
                               candidate_randomizations = NULL,
                               candidate_randomizations_array = NULL,
                               n0_array = NULL,
                               n1_array = NULL,
                               findFI = FALSE,
                               c_initial = 2,
                               conda_env = "fastrerandomize_env",
                               conda_env_required = TRUE
                               ){
  if (is.null(obsW)) {
    stop("'obsW' is required.", call. = FALSE)
  }
  if (is.null(obsY)) {
    stop("'obsY' is required.", call. = FALSE)
  }
  if (length(alpha) != 1L || !is.numeric(alpha) || !is.finite(alpha) ||
      alpha <= 0 || alpha >= 1) {
    stop("'alpha' must be a finite number strictly between 0 and 1.", call. = FALSE)
  }
  if (length(findFI) != 1L || !is.logical(findFI) || is.na(findFI)) {
    stop("'findFI' must be TRUE or FALSE.", call. = FALSE)
  }
  if(is.null(candidate_randomizations) && is.null(candidate_randomizations_array)){
    stop("Either 'candidate_randomizations' or 'candidate_randomizations_array' must be provided.",
         call. = FALSE)
  }

  obsW <- c(unlist(obsW))
  obsY <- c(unlist(obsY))
  if (!is.numeric(obsW) && !is.logical(obsW)) {
    stop("'obsW' must be a numeric or logical 0/1 vector.", call. = FALSE)
  }
  if (!is.numeric(obsY)) {
    stop("'obsY' must be numeric.", call. = FALSE)
  }
  if (length(obsW) == 0L || length(obsW) != length(obsY)) {
    stop("'obsW' and 'obsY' must be non-empty vectors of equal length.", call. = FALSE)
  }
  if (anyNA(obsW) || any(!is.finite(obsW)) || !all(obsW %in% c(0, 1))) {
    stop("'obsW' must contain only finite 0 and 1 values.", call. = FALSE)
  }
  if (!all(c(0, 1) %in% obsW)) {
    stop("'obsW' must contain at least one treated and one control unit.", call. = FALSE)
  }
  if (anyNA(obsY) || any(!is.finite(obsY))) {
    stop("'obsY' must contain only finite values.", call. = FALSE)
  }
  if (!is.null(candidate_randomizations)) {
    candidate_randomizations <- as.matrix(candidate_randomizations)
    if (nrow(candidate_randomizations) == 0L ||
        ncol(candidate_randomizations) != length(obsW)) {
      stop("'candidate_randomizations' must have one column per observed unit and at least one row.",
           call. = FALSE)
    }
  }
  step_multiplier <- if (findFI) .fi_step_multiplier(c_initial) else 1

  if( is.null(check_jax_availability(conda_env=conda_env)) ) { return(NULL) }

  tau_obs <- FI <- NULL
  if (!"VectorizedFastHotel2T2" %in% ls(envir = fastrr_env)) {
    initialize_jax(conda_env = conda_env, conda_env_required = conda_env_required)
  }

  # Convert between formats as needed
  if(is.null(candidate_randomizations)){
      candidate_randomizations <- as.matrix(fastrr_env$np$array(candidate_randomizations_array))
      if (nrow(candidate_randomizations) == 0L ||
          ncol(candidate_randomizations) != length(obsW)) {
        stop("'candidate_randomizations_array' must have one column per observed unit and at least one row.",
             call. = FALSE)
      }
  }
  if(is.null(candidate_randomizations_array)){
      candidate_randomizations_array <- fastrr_env$jnp$array(candidate_randomizations,
                                                             dtype = fastrr_env$jnp$float32)
  }

  if(is.null(n0_array)){ n0_array <- fastrr_env$jnp$array(sum(obsW == 0)) }
  if(is.null(n1_array)){ n1_array <- fastrr_env$jnp$array(sum(obsW == 1)) }

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
      upperBound_storage_vec <- lowerBound_storage_vec <- rep(NA, n_search_attempts)
      {
        initial_bounds <- .fi_initial_bounds(tau_obs)
        lowerBound_estimate_step_t <- initial_bounds[1]
        upperBound_estimate_step_t <- initial_bounds[2]

        #setting optimal c
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

            c_step_t <- step_multiplier * k * (tau_obs - lowerBound_estimate_step_t)
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

            c_step_t <- step_multiplier * k * (upperBound_estimate_step_t - tau_obs)
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
        tau_pseudo_seq <- .fi_search_grid(FI, tau_obs, length.out = 100L)
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
        FI <- .fi_accepted_range(tau_pseudo_seq, pvals_vec, alpha)
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
