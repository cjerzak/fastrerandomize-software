#' Draws a random sample of acceptable randomizations from all possible complete randomizations using Monte Carlo sampling
#' 
#' This function performs sampling with replacement to generate randomizations in a memory-efficient way.
#' It processes randomizations in batches to avoid memory issues and filters them based on covariate balance.
#' The function uses JAX for fast computation and memory management.
#'
#' @param n_units An integer specifying the total number of experimental units
#' @param n_treated An integer specifying the number of units to be assigned to treatment
#' @param X A numeric matrix of covariates used for balance checking. Cannot be NULL.
#' @param randomization_accept_prob A numeric value between 0 and 1 specifying the probability threshold for accepting randomizations based on balance. Default is 1
#' @param threshold_func A JAX function that computes a balance measure for each randomization. Must be vectorized using jax$vmap with in_axes = list(NULL, 0L, NULL, NULL), and inputs covariates (matrix of X), treatment_assignment (vector of 0s and 1s), n0 (scalar), n1 (scalar). Default is VectorizedFastHotel2T2 which uses Hotelling's T^2 statistic
#' @param max_draws An integer specifying the maximum number of randomizations to draw. Default is 100000
#' @param seed An integer seed for random number generation. Default is 42
#' @param batch_size An integer specifying how many randomizations to process at once. Default is 10000. Lower values use less memory but may be slower
#' @param verbose A logical value indicating whether to print detailed information about batch processing progress, and GPU memory usage. Default is FALSE
#' @details
#' The function works by:
#' 1. Generating batches of random permutations using JAX's random permutation functionality
#' 2. Computing balance measures for each permutation using the provided threshold function
#' 3. Keeping only the top permutations that meet the acceptance probability threshold
#' 4. Managing memory by clearing unused objects and JAX caches between batches
#'
#' The function uses smaller data types (int8, float16) where possible to reduce memory usage.
#' It also includes assertions to verify array shapes and dimensions throughout.
#'
#' @return A JAX array containing the accepted randomizations, where each row represents one possible treatment assignment vector
#' @examples
#' # Generate 1000 randomizations for 100 units with 50 treated
#' X <- matrix(rnorm(100*5), 100, 5) # 5 covariates
#' rand <- GenerateRandomizations_MonteCarlo(100, 50, X, max_draws=1000)
#'
#' # Use a stricter balance criterion
#' rand_strict <- GenerateRandomizations_MonteCarlo(
#'                n_units = 100, 
#'                n_treated = 50, 
#'                X = X, 
#'                randomization_accept_prob=0.1, 
#'                max_draws=1000)
#'
#' @seealso
#' \code{\link{GenerateRandomizations}} for the non-Monte Carlo version
#' \code{\link{VectorizedFastHotel2T2}} for the default threshold function
#' 
#' @import reticulate
#' @importFrom assertthat assert_that
#' @export
#' @md
generate_randomizations_mc <- function(n_units, n_treated,
                                       X,
                                       randomization_accept_prob = 1,
                                       threshold_func = VectorizedFastHotel2T2, 
                                       max_draws = 100000, 
                                       batch_size = 1000, 
                                       seed = NULL,
                                       approximate_inv = TRUE,
                                       verbose = FALSE,
                                       file = NULL, 
                                       conda_env = "fastrerandomize", conda_env_required = T
                                      ){
  if(!"VectorizedFastHotel2T2" %in% ls(envir = .GlobalEnv)){
    initialize_jax_code <- paste(deparse(initialize_jax),collapse="\n")
    initialize_jax_code <- gsub(initialize_jax_code, pattern="function \\(\\)",replace="")
    eval( parse( text = initialize_jax_code ), envir = environment() )
  }

  if(is.null(seed)){ seed <- as.integer(runif(1, 0, 100000)) }
  
  # Calculate the maximum number of possible randomizations
  max_rand_num <- choose(n_units, n_treated)
  assert_that(max_draws <= max_rand_num, 
              msg = paste0("max_draws must be less than or equal to the total number of possible randomizations (", max_rand_num, ")."))
  assert_that(max_draws >= 2*batch_size, 
              msg = "max_draws must be at least 2*batch_size")
  
  # Define the base vector: 1s for treated, 0s for control
  base_vector <- c(rep(1L, n_treated), rep(0L, n_units - n_treated))
  
  # Convert base_vector to a JAX array with a smaller data type to save memory
  base_vector_jax <- jnp$array(as.integer(base_vector), dtype = jnp$int8)
  
  # Initialize JAX random key with the provided seed
  key <- jax$random$PRNGKey(as.integer(seed))
  
  # Convert X to JAX array (float16 can cause issues with matrix inverse)
  X_jax <- jnp$array(as.matrix(X), dtype = jnp$float32)
  
  # Set up sample sizes for treatment/control
  n0_array <- jnp$array(as.integer(n_units - n_treated))
  n1_array <- jnp$array(as.integer(n_treated))
    
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
    print(paste0("Starting batch processing with ", num_batches, " batches."))
  }
  t0 <- Sys.time()

  batch_permutation <- jax$jit( jax$vmap( function(key_, base_vector_){
      perm_ = jax$random$permutation(key_, base_vector_)
      return(perm_)
  }, list(0L, NULL)) )
  
  top_M_fxn <- jax$jit(function(key, b_){ 
    key <- jax$random$fold_in(key, b_)
    vkey <- jax$random$split(key, as.integer(batch_size))
    
    # Generate permutations for the current batch - run on CPU? 
    perms_batch <- batch_permutation(vkey, base_vector_jax)
    
    # Calculate balance measures (e.g., Hotelling T-squared) for each permutation in the batch
    M_results_batch_ <- jnp$squeeze( threshold_func(
      X_jax,
      perms_batch,
      n0_array, 
      n1_array, 
      approximate_inv
    ) )$astype(jnp$float16)
    return(list("top_keys"=vkey,
                "top_M_results"=M_results_batch_))
  })
  
  top_M_results <- sapply(1L:num_batches, function(b_){ # note: vmapping this causes unacceptable memory overhead 
    top_M_results_ <- top_M_fxn(key, jnp$array(as.integer(b_)))
  return(list("top_keys"=top_M_results_$top_keys, "top_M_results"=top_M_results_$top_M_results) ) })
  
  # concatenate results 
  top_keys <- jnp$concatenate(top_M_results["top_keys",],0L)
  top_M_results <- jnp$concatenate(top_M_results["top_M_results",],0L)
  
  # Limit the size of combined arrays
  combined_length <- top_M_results$shape[[1]]
  {
    # Get indices of combined_M_results sorted in ascending order
    sorted_indices <- jnp$argsort(top_M_results)
    
    # Keep only top num_to_accept permutations
    indices_to_keep <- sorted_indices[0:num_to_accept]
    top_M_results <- jnp$take(top_M_results, indices_to_keep)
    top_keys <- jnp$take(top_keys, indices_to_keep, axis=0L)
    top_perms <- batch_permutation(top_keys, base_vector_jax)
  }

  #assert_that(top_M_results$shape[[1]] <= num_to_accept, msg = paste0("top_M_results must have dimensions ", num_to_accept, " x 1."))
  # assert_that(top_perms$shape[[1]] <= num_to_accept, msg = paste0("top_perms must have dimensions ", num_to_accept, " x ", n_units, "."))
  # rm( top_keys )
  # gc(); py_gc$collect()
  
  print(sprintf("MC Loop Time (s): %.4f", as.numeric(difftime(Sys.time(), t0, units = "secs"))))
    
  # After processing all batches, the candidate_randomizations are the top_perms
  #assert_that(all(top_perms$shape == c(num_to_accept, n_units)), msg = paste0("candidate_randomizations must have dimensions ",  num_to_accept, " x ", n_units, ".") )
  return(list(
              "candidate_randomizations" = top_perms,
              "keys_candidate_randomizations" = top_keys,
              "M_candidate_randomizations"= top_M_results
              ))
}
