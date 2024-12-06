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

#top_M_results <- jax$vmap( (function(key){
#return(list("top_keys"=vkey, "top_M_results"=M_results_batch_) ) }), in_axes = 0L,out_axes = 1L)(jax$random$split(key,as.integer(num_batches)))
#top_keys <- jnp$reshape(top_M_results$top_keys,list(-1L,top_M_results$top_keys$shape[[3]]))
#top_M_results <- jnp$reshape(top_M_results$top_M_results,list(-1L))

if(simulate == F){
  candidate_randomizations_array <- generate_randomizations(
    n_units = n_units,
    n_treated = n_treated,
    X = X,
    randomization_accept_prob = randomization_accept_prob,
    
    # hyper parameters
    max_draws = max_draws, 
    batch_size = batch_size, 
    randomization_type = randomization_type, 
    approximate_inv = approximate_inv,
    file = file)
}

if( simulate ){
  n_treated <- ( n_units <- nrow(X) ) / 2
  candidate_randomizations_array <- generate_randomizations(
    n_units = n_units, 
    n_treated = n_treated, 
    X = X,
    randomization_accept_prob = randomization_accept_prob,
    
    # hyper parameters
    max_draws = max_draws, 
    batch_size = batch_size, 
    randomization_type = randomization_type, 
    approximate_inv = approximate_inv,
    file = file)$candidate_randomizations
}
if( simulate ){
  obsY1_array <- jnp$array( replicate(nSimulate_obsY, {
    prior_coef_draw <- coef_prior()
    Y_0 <- X %*% prior_coef_draw
    
    #tau_samp <- rnorm(n=n_units, mean=prior_treatment_effect_mean, sd = prior_treatment_effect_SD) # assumes tau_i
    tau_samp <- rnorm(n=1, mean=prior_treatment_effect_mean, sd = prior_treatment_effect_SD) # assumes one tau
    Y_1 <- Y_0 + tau_samp
    
    return(cbind(Y_0, Y_1))
  }) )
  obsY1_array <- jnp$transpose(obsY1_array, axes = c(2L,0L,1L))
  obsY0_array <- jnp$take(obsY1_array, 0L, axis = 2L)
  obsY1_array <- jnp$take(obsY1_array, 1L, axis = 2L)
  
  chi_squared_approx <- F
  M_results <- jnp$squeeze(VectorizedFastHotel2T2(jnp$array( X ),
                                                  candidate_randomizations_array,
                                                  n0_array, n1_array), 1L:2L)
  a_threshold_vec <- jnp$quantile(M_results, jnp$array(prob_accept_randomization_seq))
  M_results <- np$array( M_results )
  #if(chi_squared_approx==T){ a_threshold <- qchisq(p=prob_accept_randomization_seq[ii], df=k_covars) }
  
  GetPvals_vmapped <- jax$jit(jax$vmap( function( sampledIndex, acceptedWs_array ){
    print2("Jitting...")
    obsW_ <- VectorizedTakeAxis0_R(acceptedWs_array, sampledIndex)
    obsY_array <- Potential2Obs_R(obsY0_array, obsY1_array, obsW_)
    tau_obs <- Y_VectorizedFastDiffInMeans_R(obsY_array, obsW_, n0_array, n1_array)
    tau_perm_null_0 <-  YW_VectorizedFastDiffInMeans_R(
      obsY_array,  # y_ =
      acceptedWs_array, # w_ =
      n0_array, # n0 =
      n1_array) # n1 =
    p_value_inner <- GreaterEqualMagCompare_R(tau_perm_null_0, tau_obs) # pvals across yhat
    return( p_value_inner ) #  mean here is over Yhat
  }, in_axes = list(0L,NULL)))
  p_value <- sapply(np$array(a_threshold_vec),   function(a_){
    print(a_ / max(np$array(a_threshold_vec)))
    py_gc$collect()
    
    # select acceptable randomizations based on threshold
    acceptedWs_array <- candidate_randomizations_array[jnp$less_equal( M_results,  a_),]
    sampTheseIndices <- 0L:((AcceptedRandomizations <- acceptedWs_array$shape[[1]])  - 1L)
    sampledIndices <- jnp$expand_dims(jnp$array( as.integer(as.numeric(
      sample(as.character(sampTheseIndices), nSimulate_obsW, replace = T) ))),1L)
    p_value_outer <-  jnp$mean(GetPvals_vmapped(sampledIndices, acceptedWs_array))
  })
  p_value <- np$array( p_value )
  suggested_randomization_accept_prob <- prob_accept_randomization_seq[ which.min(p_value)[1] ]
  plot( p_value )
}

if( simulate ){
  return( list(p_value = p_value,
               suggested_randomization_accept_prob = suggested_randomization_accept_prob,
               FI = FI,
               tau_obs = tau_obs) )
}