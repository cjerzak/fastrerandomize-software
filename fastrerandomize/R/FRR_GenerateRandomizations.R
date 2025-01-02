#' Generate randomizations for experimental design
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
#' # Generate randomizations using exact enumeration
#' X <- matrix(rnorm(100*5), 100, 5)
#' RandomizationSet_Exact <- GenerateRandomizations(100, 50, X, 
#'                randomization_accept_prob=0.1,
#'                randomization_type="exact")
#'
#' # Generate randomizations using Monte Carlo sampling
#' RandomizationSet_MC <- GenerateRandomizations(
#'                n_units = 100, 
#'                n_treated = 50, 
#'                X = X,
#'                randomization_accept_prob=0.1,
#'                randomization_type="monte_carlo",
#'                max_draws=1000)
#'
#' @seealso
#' \code{\link{GenerateRandomizations_Exact}} for the exact enumeration method
#' \code{\link{GenerateRandomizations_MonteCarlo}} for the Monte Carlo sampling method
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
                                   conda_env = "fastrerandomize", conda_env_required = T
                                   ){
  if(!"VectorizedFastHotel2T2" %in% ls(envir = .GlobalEnv)){
      initialize_jax_code <- paste(deparse(initialize_jax),collapse="\n")
      initialize_jax_code <- gsub(initialize_jax_code,pattern="function \\(\\)",replace="")
      eval( parse( text = initialize_jax_code ), envir = environment() )
    }
    if(is.null(threshold_func)){ threshold_func <- VectorizedFastHotel2T2 }
  
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
    # gc();py_gc$collect()
    
    print2("Returning generate_randomizations...")
    if (is.null(file)) {
      # Wrap in S3 constructor
      return(
        fastrerandomize_class(
          candidate_randomizations = candidate_randomizations$candidate_randomizations, 
          randomizations_balance = candidate_randomizations$M_candidate_randomizations,
          call = match.call()
        )
      )
    } else {
      # existing file-writing logic
      write.csv(np$array(candidate_randomizations$candidate_randomizations), file = file)
      return(sprintf("File saved at %s", file))
    }
      
}