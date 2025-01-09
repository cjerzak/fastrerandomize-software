#' Generate randomizations for a rerandomization-based experimental design
#'
#' This function generates randomizations for experimental design using either exact enumeration
#' or Monte Carlo sampling methods. It provides a unified interface to both approaches while
#' handling memory and computational constraints appropriately.
#'
#' @param n_units An integer specifying the total number of experimental units
#' @param n_treated An integer specifying the number of units to be assigned to treatment
#' @param X A numeric matrix of covariates used for balance checking. Cannot be NULL.
#' @param randomization_accept_prob A numeric value between 0 and 1 specifying the probability threshold for accepting randomizations based on balance
#' @param threshold_func A JAX function that computes a balance measure for each randomization. Only used for Monte Carlo sampling.
#' @param max_draws An integer specifying the maximum number of randomizations to draw in Monte Carlo sampling
#' @param seed An integer seed for random number generation in Monte Carlo sampling
#' @param batch_size An integer specifying batch size for Monte Carlo processing
#' @param randomization_type A string specifying the type of randomization: either "exact" or "monte_carlo"
#' @param file A string specifying where to save candidate randomizations (if saving, not returning)
#' @param approximate_inv A logical value indicating whether to use an approximate inverse 
#'   (diagonal of the covariance matrix) instead of the full matrix inverse when computing 
#'   balance metrics. This can speed up computations for high-dimensional covariates.
#'   Default is `TRUE`.
#' @param return_type A string specifying the format of the returned randomizations and balance 
#'   measures. Allowed values are "R" for base R objects (e.g., \code{matrix}, \code{numeric}) 
#'   or "jax" for JAX/NumPy arrays. Default is "R".
#' @param conda_env A character string specifying the name of the conda environment to use 
#'   via \code{reticulate}. Default is "fastrerandomize".
#' @param conda_env_required A logical indicating whether the specified conda environment 
#'   must be strictly used. If \code{TRUE}, an error is thrown if the environment is not found. 
#'   Default is TRUE.
#' @param verbose A logical value indicating whether to print progress information. Default is TRUE
#'
#' @details
#' The function supports two methods of generating randomizations:
#' 1. Exact enumeration: Generates all possible randomizations (memory intensive but exact)
#' 2. Monte Carlo sampling: Generates randomizations through sampling (more memory efficient)
#'
#' For large problems (e.g., X with >20 columns), Monte Carlo sampling is recommended.
#'
#' @return A JAX array containing the accepted randomizations, where each row represents 
#' one possible treatment assignment vector
#'
#' @examples
#' 
#' \dontrun{
#' # Generate synthetic data 
#' X <- matrix(rnorm(20*5), 20, 5)
#' 
#' # Generate randomizations using exact enumeration
#' RandomizationSet_Exact <- generate_randomizations(
#'                n_units = nrow(X), 
#'                n_treated = round(nrow(X)/2), 
#'                X = X, 
#'                randomization_accept_prob=0.1,
#'                randomization_type="exact")
#'
#' # Generate randomizations using Monte Carlo sampling
#' RandomizationSet_MC <- generate_randomizations(
#'                n_units = nrow(X), 
#'                n_treated = round(nrow(X)/2), 
#'                X = X,
#'                randomization_accept_prob = 0.1,
#'                randomization_type = "monte_carlo",
#'                max_draws = 100000,
#'                batch_size = 1000)
#'  }
#'
#' @seealso
#' \code{\link{generate_randomizations_exact}} for the exact enumeration method
#' \code{\link{generate_randomizations_mc}} for the Monte Carlo sampling method
#'
#' @export
#' @md
generate_randomizations <- function(n_units, 
                                   n_treated, 
                                   X = NULL, 
                                   randomization_accept_prob, 
                                   threshold_func = NULL, 
                                   max_draws = 10^6, 
                                   batch_size = 10^5, 
                                   randomization_type = "monte_carlo", 
                                   approximate_inv = TRUE, 
                                   seed = NULL, 
                                   verbose = TRUE,
                                   file = NULL, 
                                   return_type = "R", 
                                   conda_env = "fastrerandomize", 
                                   conda_env_required = TRUE
                                   ){
  if (is.null(check_jax_availability(conda_env=conda_env))) { return() }
  
  if (!"VectorizedFastHotel2T2" %in% ls(envir = fastrr_env)) {
      initialize_jax(conda_env = conda_env, conda_env_required = conda_env_required) 
  }
  if(is.null(threshold_func)){ threshold_func <- fastrr_env$VectorizedFastHotel2T2 }
  
  if (randomization_type == "exact"){
        if (verbose){
            print("Using exact randomization")
        }
        if (ncol(X) > 20){
            print("Warning: X has more than 20 columns. This may cause memory issues.")
        }
        candidate_randomizations <- fastrerandomize::generate_randomizations_exact(
                                                n_units = n_units,
                                                n_treated = n_treated,
                                                X = X, 
                                                randomization_accept_prob = randomization_accept_prob, 
                                                threshold_func = threshold_func, 
                                                file = file, 
                                                seed = seed)
    } else if (randomization_type == "monte_carlo"){
        if (verbose){
            print("Using monte carlo randomization")
        }
        candidate_randomizations <- fastrerandomize::generate_randomizations_mc(
                                                                    n_units = n_units, 
                                                                    n_treated = n_treated, 
                                                                    X = X, 
                                                                    randomization_accept_prob = randomization_accept_prob, 
                                                                    threshold_func = threshold_func, 
                                                                    max_draws = max_draws, 
                                                                    batch_size = batch_size, 
                                                                    seed = seed, 
                                                                    file = file, 
                                                                    verbose = verbose)
    } else {
        stop("Invalid randomization type")
    }
    
    print2("Returning generate_randomizations...")
    if (is.null(file)) {
      # Wrap in S3 constructor
      return(
        fastrerandomize_class(
          randomizations = output2output(candidate_randomizations$candidate_randomizations, return_type),
          balance = output2output(candidate_randomizations$M_candidate_randomizations, return_type),
          call = match.call()
        )
      )
    } else {
      # existing file-writing logic
      utils::write.csv( fastrr_env$np$array(candidate_randomizations$candidate_randomizations), file = file)
      return(sprintf("File saved at %s", file))
    }
      
}
