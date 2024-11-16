
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
#' rand_strict <- GenerateRandomizations_MonteCarlo(100, 50, X, 
#'                randomization_accept_prob=0.1, max_draws=1000)
#'
#' @seealso
#' \code{\link{GenerateRandomizations}} for the non-Monte Carlo version
#' \code{\link{VectorizedFastHotel2T2}} for the default threshold function
#'
#' @export
#' @md
GenerateRandomizations_MonteCarlo <- function(n_units, n_treated,
                                             X,
                                             randomization_accept_prob = 1,
                                             threshold_func = VectorizedFastHotel2T2, 
                                             max_draws = 100000, seed = 42,
                                             batch_size = 10000, verbose = FALSE){
  # Initialize JAX via reticulate
  jax <- reticulate::import("jax")
  jnp <- jax$numpy
  
  # Define the batch_permutation function in Python using JAX
  # Uses vmap to vectorize the permutation operation over the batch size
  # Uses jit to compile the function
  jax_code <- "
import jax
import jax.numpy as jnp

def batch_permutation(key, base_vector, num_perms):
    base_vector = jnp.broadcast_to(base_vector, (num_perms, len(base_vector)))
    keys = jax.random.split(key, num_perms)
    perms = jax.vmap(jax.random.permutation)(keys, base_vector)
    assert perms.dtype == jnp.int8, 'perms must be a boolean array'
    return perms

# Apply jit after function definition
batch_permutation = jax.jit(batch_permutation, static_argnums=2)
  "
  py_run_string(jax_code)
  
  # Calculate the maximum number of possible randomizations
  max_rand_num <- choose(n_units, n_treated)
  assert_that(max_draws <= max_rand_num, msg = paste0("max_draws must be less than or equal to the number of possible randomizations, which is ", max_rand_num, "."))
  
  # Define the base vector: 1s for treated, 0s for control
  base_vector <- c(rep(1L, n_treated), rep(0L, n_units - n_treated))
  
  # Convert base_vector to a JAX array with a smaller data type to save memory
  base_vector_jax <- jnp$array(as.integer(base_vector), dtype = jnp$int8)
  
  # Initialize JAX random key with the provided seed
  key <- jax$random$PRNGKey(as.integer(seed))
  
  # Convert X to JAX array
  X_jax <- jnp$array(as.matrix(X), dtype = jnp$float16)
  
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
  for (batch_idx in seq_len(num_batches)){
    if (verbose){
      print(paste0("At batch_idx ", batch_idx, " of ", num_batches, "."))
      # Run nvidia-smi and capture the output
      tryCatch({
        # Try to get GPU info
        gpu_info <- system("nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits", intern = TRUE)
        if (length(gpu_info) > 0) {
          print("Running on GPU:")
          # Parse and format GPU info
          gpu_processes <- lapply(strsplit(gpu_info, ","), trimws)
          gpu_table <- do.call(rbind, lapply(gpu_processes, function(x) {
            sprintf("PID: %s | Process: %s | Memory: %s MB", x[1], x[2], x[3])
          }))
          print(gpu_table)
        }
      }, error = function(e){
        warning("No GPU detected - running on CPU only")
      })
    }
    # Determine the number of permutations in this batch
    perms_in_batch <- min(batch_size, max_draws - (batch_idx - 1) * batch_size)
    
    # Update the random key for the current batch to ensure uniqueness
    batch_key <- jax$random$fold_in(key, batch_idx)
    
    # Generate permutations for the current batch
    perms_batch <- py$batch_permutation(batch_key, base_vector_jax, as.integer(perms_in_batch))
    
    # Calculate balance measures (e.g., Hotelling T²) for each permutation in the batch
    M_results_batch <- threshold_func(
      X_jax,
      perms_batch,
      n0_array, 
      n1_array
    )
    
    # Flatten M_results_batch to 1D array
    M_results_batch <- jnp$squeeze(M_results_batch)
    
    if (is.null(top_M_results)){
        combined_M_results <- M_results_batch
        combined_perms <- perms_batch
    } else {
        combined_M_results <- jnp$concatenate(list(top_M_results, M_results_batch))
        combined_perms <- jnp$concatenate(list(top_perms, perms_batch), axis=0L)
    }

    # Limit the size of combined arrays
    combined_length <- combined_M_results$shape[[1]]
    if (combined_length > num_to_accept){
        # Get indices of combined_M_results sorted in ascending order
        sorted_indices <- jnp$argsort(combined_M_results)
        # Keep only top num_to_accept permutations
        indices_to_keep <- sorted_indices[0:num_to_accept]
        top_M_results <- jnp$take(combined_M_results, indices_to_keep)
        top_perms <- jnp$take(combined_perms, indices_to_keep, axis=0L)
    } else {
        # Keep combined results as top results
        top_M_results <- combined_M_results
        top_perms <- combined_perms
    }

    assert_that(top_M_results$shape[[1]] <= num_to_accept, msg = paste0("top_M_results must have dimensions ", num_to_accept, " x 1."))
    assert_that(top_perms$shape[[1]] <= num_to_accept, msg = paste0("top_perms must have dimensions ", num_to_accept, " x ", n_units, "."))
    rm(perms_batch)
    rm(M_results_batch)
    rm(combined_M_results)
    rm(combined_perms)
    jax$clear_caches()
    gc()  # Force garbage collection
    py_run_string("import gc; gc.collect()")

    # Update the key for the next batch
    key <- jax$random$fold_in(key, batch_idx)
  }
    
  # After processing all batches, the candidate_randomizations are the top_perms
  candidate_randomizations <- top_perms
  assert_that(all(candidate_randomizations$shape == c(num_to_accept, n_units)), 
              msg = paste0("candidate_randomizations must have dimensions ", 
                            num_to_accept, " x ", n_units, "."))
  return(candidate_randomizations)
}