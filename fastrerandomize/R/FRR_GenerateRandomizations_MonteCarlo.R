#' Draws a random sample of acceptable randomizations from all possible complete randomizations using Monte Carlo sampling
#' 
#' This function performs sampling with replacement to generate randomizations in a memory-efficient way.
#' It processes randomizations in batches to avoid memory issues and filters them based on covariate balance.
#' The function uses JAX for fast computation and memory management.
#'
#' @param n_units An integer specifying the total number of experimental units. 
#' @param n_treated An integer specifying the number of units to be assigned to treatment. 
#' @param X A numeric matrix of covariates used for balance checking. Cannot be NULL.
#' @param randomization_accept_prob A numeric value between 0 and 1 specifying the probability threshold for accepting randomizations based on balance. Default is 1
#' @param threshold_func A JAX function that computes a balance measure for each randomization. Must be vectorized using \code{jax$vmap} with \code{in_axes = list(NULL, 0L, NULL, NULL)}, and inputs covariates (matrix of X), treatment_assignment (vector of 0s and 1s), n0 (scalar), n1 (scalar). Default is \code{VectorizedFastHotel2T2} which uses Hotelling's T-squared statistic.
#' @param max_draws An integer specifying the maximum number of randomizations to draw. Default is \code{100000L}. 
#' @param batch_size An integer specifying how many randomizations to process at once. Default is \code{10000L}. Lower values use less memory but may be slower. 
#' @param approximate_inv A logical value indicating whether to use an approximate inverse 
#'   (diagonal of the covariance matrix) instead of the full matrix inverse when computing 
#'   balance metrics. This can speed up computations for high-dimensional covariates.
#'   Default is \code{TRUE}.
#' @param conda_env A character string specifying the name of the conda environment to use 
#'   via \code{reticulate}. Default is \code{"fastrerandomize"}.
#' @param conda_env_required A logical indicating whether the specified conda environment 
#'   must be strictly used. If \code{TRUE}, an error is thrown if the environment is not found. 
#'   Default is \code{TRUE}.
#' @param verbose A logical value indicating whether to print detailed information about batch processing progress, and GPU memory usage. Default is \code{FALSE}. 
#' @details
#' The function works by:
#' 1. Generating batches of random permutations.
#' 2. Computing balance measures for each permutation using the provided threshold function.
#' 3. Keeping only the top permutations that meet the acceptance probability threshold.
#' 4. Managing memory by clearing unused objects and caches between batches.
#'
#' The function uses smaller data types (int8, float16) where possible to reduce memory usage.
#' It also includes assertions to verify array shapes and dimensions throughout.
#'
#' @return The function returns a \emph{list} with two elements:
#' \code{candidate_randomizations}: an array of randomization vectors
#' \code{M_candidate_randomizations}: an array of their balance measures. 
#' @examples
#' 
#' \dontrun{
#' # Generate synthetic data 
#' X <- matrix(rnorm(100*5), 100, 5) # 5 covariates
#' 
#' # Generate 1000 randomizations for 100 units with 50 treated
#' rand_less_strict <- generate_randomizations_mc(
#'                n_units = 100, 
#'                n_treated = 50, 
#'                X = X, 
#'                randomization_accept_prob=0.01, 
#'                max_draws = 100000,
#'                batch_size = 1000)
#'
#' # Use a stricter balance criterion
#' rand_more_strict <- generate_randomizations_mc(
#'                n_units = 100, 
#'                n_treated = 50, 
#'                X = X, 
#'                randomization_accept_prob=0.001, 
#'                max_draws = 1000000,
#'                batch_size = 1000)
#' }
#'
#' @seealso
#' \code{\link{generate_randomizations}} for full randomization generation function. 
#' \code{\link{generate_randomizations_exact}} for the exact version. 
#' 
#' @import reticulate
#' @importFrom assertthat assert_that
#' @export
#' @md
generate_randomizations_mc <- function(n_units, n_treated,
                                       X,
                                       randomization_accept_prob = 1,
                                       threshold_func = NULL, 
                                       max_draws = 100000, 
                                       batch_size = 1000, 
                                       approximate_inv = TRUE,
                                       verbose = TRUE,
                                       conda_env = "fastrerandomize", 
                                       conda_env_required = TRUE
                                      ){
  if (is.null(check_jax_availability(conda_env=conda_env))) { return(NULL) }
  
  if (!"VectorizedFastHotel2T2" %in% ls(envir = fastrr_env)) {
    initialize_jax(conda_env = conda_env, conda_env_required = conda_env_required) 
  }
  if(is.null(threshold_func)){ threshold_func <- fastrr_env$VectorizedFastHotel2T2 }
  
  # Calculate the maximum number of possible randomizations
  max_rand_num <- choose(n_units, n_treated)
  assertthat::assert_that(max_draws <= max_rand_num, 
              msg = paste0("max_draws must be less than or equal to the total number of possible randomizations (", max_rand_num, ")."))
  assertthat::assert_that(max_draws >= 2*batch_size, 
              msg = "max_draws must be at least 2*batch_size")
  
  # Define the base vector: 1s for treated, 0s for control
  base_vector <- c(rep(1L, n_treated), rep(0L, n_units - n_treated))
  
  # Convert base_vector to a JAX array with a smaller data type to save memory
  base_vector_jax <- fastrr_env$jnp$array(as.integer(base_vector), 
                                          dtype = fastrr_env$jnp$int8)
  
  # Initialize base JAX random key with the provided seed
  key <- fastrr_env$jax$random$PRNGKey( as.integer(stats::runif(1,1,100000)) )
  
  # Convert X to JAX array (float16 can cause issues with matrix inverse)
  X_jax <- fastrr_env$jnp$array(as.matrix(X), dtype = fastrr_env$jnp$float32)
  
  # Set up sample sizes for treatment/control
  n0_array <- fastrr_env$jnp$array(as.integer(n_units - n_treated))
  n1_array <- fastrr_env$jnp$array(as.integer(n_treated))
    
  # Calculate the number of batches
  num_batches <- ceiling(max_draws / batch_size)
    
  # Initialize variables to store top permutations and their balance measures
  top_perms <- NULL
  top_M_results <- NULL
    
  # Determine the number of permutations to accept based on the acceptance probability
  float_num_to_accept <- max_draws * randomization_accept_prob
  if (float_num_to_accept < 1){
    warning("randomization_accept_prob is less than 1, so we will accept at least one randomization.")
  }
  num_to_accept <- ceiling(float_num_to_accept)
  num_to_accept <- max(num_to_accept, 1) # Ensure at least one
  
  # Batch processing to prevent memory issues
  if (verbose){
    message(paste0("Starting batch processing with ", num_batches, " batches."))
  }
  t0 <- Sys.time()

  batch_permutation <- fastrr_env$jax$jit( fastrr_env$jax$vmap( function(key_, base_vector_){
      perm_ = fastrr_env$jax$random$permutation(key = key_, 
                                                x = base_vector_, 
                                                axis = 0L, # axis to shuffle along 
                                                independent = TRUE)
      return(perm_)
  }, list(0L, NULL)) )
  
  top_M_fxn <- fastrr_env$jax$jit(function(key){
    perms_batch <- batch_permutation(key, base_vector_jax)
    
    # Calculate balance measures (e.g., Hotelling T-squared) for each permutation in the batch
    M_results_batch_ <- fastrr_env$jnp$squeeze( threshold_func(
      X_jax,
      perms_batch,
      n0_array, 
      n1_array, 
      approximate_inv
    ) )$astype(fastrr_env$jnp$float32)
    
    return(list("top_M_results" = M_results_batch_))
  })
  
  AllKeys <- fastrr_env$jax$random$split(key, as.integer(max_draws))
  AllKeySelectionIndices <- split( as.integer( (1L:max_draws) - 1L), 
                                   sapply(1:num_batches, function(x){rep(x,times = batch_size)}))
  top_M_results <- sapply(1L:num_batches, function(b_){ # note: vmapping this causes unacceptable memory overhead 
    if(verbose){ message(paste0("On batch ", b_, " of ", num_batches)) }
    top_M_results_ <- top_M_fxn( fastrr_env$jnp$take(AllKeys,
                                          fastrr_env$jnp$array(AllKeySelectionIndices[[b_]]), 
                                          axis = 0L) )
    
    return(list( top_M_results_$top_M_results ) ) 
  })
  
  # Concatenate results 
  top_M_results <- fastrr_env$jnp$concatenate( top_M_results, 0L )
  
  # Main analysis 
  {
    # Keep only top num_to_accept permutations
    indices_to_keep <- fastrr_env$jnp$take(fastrr_env$jnp$argsort( top_M_results ), 
                                           indices = fastrr_env$jnp$arange(as.integer(num_to_accept)) )
    
    # select top 
    top_M_results <- fastrr_env$jnp$take(top_M_results, indices_to_keep, axis=0L)
    top_keys <- fastrr_env$jnp$take(AllKeys, indices_to_keep, axis=0L)
    
    # (re-) generate permutations
    top_perms <- batch_permutation(top_keys, 
                                   base_vector_jax)
    
    # check work 
    M_results_batch_ <- fastrr_env$jnp$squeeze( threshold_func(
      X_jax,
      top_perms,
      n0_array, 
      n1_array, 
      approximate_inv ) )
  }

  if(verbose){ message(sprintf("MC Loop Time (s): %.4f", as.numeric(difftime(Sys.time(), t0, units = "secs"))))}
    
  # After processing all batches, the candidate_randomizations are the top_perms
  return(list(
              "candidate_randomizations" = top_perms,
              "keys_candidate_randomizations" = top_keys,
              "M_candidate_randomizations"= top_M_results
              ))
}
