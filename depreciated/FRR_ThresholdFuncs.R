RowBroadcast <- jax$vmap(function(mat, vec){
    jnp$multiply(mat, vec)}, in_axes = list(1L, NULL))

FastHotel2T2 <- ( function(samp_, w_, n0, n1, approximate_inv = FALSE){
    # Assert statements to ensure valid inputs
    assert(n0 > 0, "Number of control units (n0) must be greater than 0.")
    assert(n1 > 0, "Number of treated units (n1) must be greater than 0.")
    assert(samp_.ndim == 2, "samp_ must be a 2D array.")
    assert(w_.ndim == 1, "w_ must be a 1D array.")
    assert(samp_.shape[0] == w_.shape[0], "samp_ and w_ must have the same number of rows.")

    # set up calc
    xbar1 <- jnp$divide(jnp$sum(RowBroadcast(samp_,w_),1L,keepdims = T), n1)
    xbar2 <- jnp$divide(jnp$sum(RowBroadcast(samp_,jnp$subtract(1.,w_)),1L,keepdims = T), n0)
    CovWts <- jnp$add(jnp$reciprocal(n0), jnp$reciprocal(n1))
    if (!approximate_inv){
        CovInv <- jnp$diag( jnp$reciprocal( jnp$multiply(CovPooled,CovWts) )) # for GPU
    } else {
        CovPooled <- jnp$var(samp_,0L); CovInv <- jnp$diag( jnp$reciprocal( jnp$multiply(CovPooled,CovWts) ))
    }
    xbar_diff <- jnp$subtract(xbar1, xbar2)

    Tstat <- jnp$matmul(jnp$matmul(jnp$transpose(xbar_diff), CovInv) , xbar_diff)
})

#' Calculate Vectorized Hotelling's T² Statistic
#'
#' @description
#' Computes a vectorized version of Hotelling's T² statistic to measure multivariate balance 
#' between treatment and control groups. This function utilizes JAX's vectorization capabilities 
#' for efficient computation across multiple samples.
#'
#' @param samp_ A JAX array containing the covariate data matrix
#' @param w_ A JAX array containing the treatment assignment vector (1 for treated, 0 for control)
#' @param n0 Number of control units
#' @param n1 Number of treated units
#' @param approximate_inv Logical. Whether to use an approximate inverse of the covariance matrix. This cannot be used on the metal GPU due to bugs in JAX.
#'
#' @return A JAX array containing Hotelling's T² statistic for each sample
#'
#' @details
#' The function computes Hotelling's T² statistic using the following steps:
#' 1. Calculates means for treatment and control groups
#' 2. Computes pooled covariance matrix
#' 3. Calculates T² = (x̄₁ - x̄₀)ᵀ Σ⁻¹ (x̄₁ - x̄₀)
#' 
#' Uses diagonal covariance matrix for GPU compatibility.
#'  
#' @references
#' Hotelling, H. (1931). The generalization of Student's ratio.
#' The Annals of Mathematical Statistics, 2(3), 360-378.
#' 
#' @export

VectorizedFastHotel2T2 <- jax$jit( jax$vmap(function(samp_, w_, n0, n1, approximate_inv = FALSE){
    FastHotel2T2(samp_, w_, n0, n1, approximate_inv)},
    in_axes = list(NULL, 0L, NULL, NULL, FALSE)) )


if(T == F){ if (verbose){
  print(paste0("At batch_idx ", batch_idx, " of ", num_batches, "."))
  # Run nvidia-smi and capture the output
  tryCatch({
    # Try to get GPU info
    gpu_info <- system("nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits", intern = TRUE)
    if (length(gpu_info) > 0) {
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
}} 