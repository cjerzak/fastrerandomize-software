initialize_jax <- function(){ #initialize_jax <- function(conda_env = "fastrerandomize", conda_env_required = T){
  { 
  print2("Loading JAX...")
  {
  library(reticulate)
  if(!is.null(conda_env)){
    try(reticulate::use_condaenv(condaenv = conda_env, required = conda_env_required),T)
  }

  # import Python packages
  if(!"jax" %in% ls()){  jax <<- reticulate::import("jax") }
  if(!"jnp" %in% ls()){  jnp <<- reticulate::import("jax.numpy") }
  if(!"np" %in% ls()){  np <<- reticulate::import("numpy") }
  if(!"tf" %in% ls()){ tf <<- reticulate::import("tensorflow") }
  if(!"py_gc" %in% ls()){  py_gc <<- reticulate::import("gc") }

  # disable 64 bit computations  
  jax$config$update("jax_enable_x64", FALSE); jaxFloatType <<- jnp$float32 
  }
  print2("Success loading JAX!")

  print2("Setup of core JAX functions...")
  {
    expand_grid_JAX <<- function(n_treated, n_control){
      expand_grid_jax_text <- paste(rep("jnp$array(0L:1L)",times = n_units), collapse = ", ")
      eval(parse(text = "grid <- jnp$meshgrid(..., indexing='ij')"))
      grid <- jnp$vstack(lapply(grid,function(zer){jnp$ravel(zer)}))
      grid <- jnp$transpose(grid)
      return( grid )
    }

    InsertOnes <<- jax$jit( function(treat_indices_, zeros_){
      zeros_ <- zeros_$at[treat_indices_]$add(1L)
      return(  zeros_ )
    } )
    InsertOnesVectorized <<- jax$jit( jax$vmap(function(treat_indices_, zeros_){
      InsertOnes(treat_indices_, zeros_)
    }, list(1L,NULL)))

    if(T == F){ # sanity checks
      expand_grid_text <- paste(rep("0L:1L",times = n_units <- 20), collapse = ", ")
      expand_grid_jax_text <- paste(rep("jnp$array(0L:1L)",times = n_units), collapse = ", ")
      system.time( tmp1 <- eval(parse(text = sprintf("expand_grid_JAX(n_units,n_units/2)",expand_grid_jax_text))) )
      system.time( tmp2 <- eval(parse(text = sprintf("expand.grid(%s)",expand_grid_text))) )
    }

    FastDiffInMeans <<- jax$jit( FastDiffInMeans_R <<- function(y_,w_, n0, n1){
      my1 <- jnp$divide(jnp$sum(jnp$multiply(y_, w_)), n1)
      my0 <- jnp$divide(jnp$sum(jnp$multiply(y_, jnp$subtract(1.,w_))), n0)
      return( diff10 <- jnp$subtract(my1, my0) )
    })

    VectorizedTakeAxis0 <<- jax$jit( VectorizedTakeAxis0_R <<- function(A_, I_){
      jnp$expand_dims(jnp$take(A_, I_, axis = 0L), 0L)
    })

    Potential2Obs <<- jax$jit(Potential2Obs_R <<- function(Y0__, Y1__, obsW__){
      jnp$add( jnp$multiply(Y0__, jnp$subtract(1, obsW__)),
               jnp$multiply(Y1__, obsW__))
    })

    W_VectorizedFastDiffInMeans <<- jax$jit( W_VectorizedFastDiffInMeans_R <<- jax$vmap(function(y_, w_, n0, n1){
      FastDiffInMeans_R(y_, w_, n0, n1)},
      in_axes = list(NULL, 0L, NULL, NULL)) )

    Y_VectorizedFastDiffInMeans <<- jax$jit( Y_VectorizedFastDiffInMeans_R <<- jax$vmap(function(y_, w_, n0, n1){
      FastDiffInMeans_R(y_, w_, n0, n1)},
      in_axes = list(0L, NULL, NULL, NULL)) )

    YW_VectorizedFastDiffInMeans <<- jax$jit( YW_VectorizedFastDiffInMeans_R <<- jax$vmap(function(y_, w_, n0, n1){
      W_VectorizedFastDiffInMeans(y_, w_, n0, n1)},
      in_axes = list(0L, NULL, NULL, NULL)) )

    WVectorizedFastDiffInMeans <<- jax$jit( jax$vmap(function(y_, w_, n0, n1){
      FastDiffInMeans(y_, w_, n0, n1)},
      in_axes = list(NULL, 0L, NULL, NULL)) )

    GreaterEqualMagCompare <<- jax$jit(GreaterEqualMagCompare_R <<- function(NULL_, OBS_){
      jnp$mean(jnp$greater_equal(jnp$abs(NULL_),  jnp$expand_dims(OBS_,1L)), 1L)
    })

    if(T == F){ # sanity checks
      y_ <- jnp$array(rnorm(10))
      w_ <-  jnp$array(rbinom(10,size=1, prob = 0.5))
      n0 <- jnp$array(10.); n1 <- jnp$array(3.)
      w_mat <- jnp$array( matrix(rbinom(100*10, size = 1, prob = 0.5),nrow = 100) )
      FastDiffInMeans(y_, w_, n0, n1)
      np$array( VectorizedFastDiffInMeans(y_, w_mat, n0, n1) )
    }

    get_stat_vec_at_tau_pseudo <<- jax$jit( function(treatment_pseudo,
                                                     obsY_array,
                                                     obsW_array,
                                                     tau_pseudo,
                                                     n0_array,
                                                     n1_array){
      #Y0_under_null <- obsY - obsW*tau_pseudo
      Y0_under_null <- jnp$subtract(obsY_array,  jnp$multiply(obsW_array, tau_pseudo))

      #Y1_under_null_pseudo <- Y0_under_null + treatment_pseudo*tau_pseudo
      Y1_under_null_pseudo <- jnp$add(Y0_under_null,  jnp$multiply(treatment_pseudo, tau_pseudo))

      #Yobs_pseudo <- Y1_under_null_pseudo*treatment_pseudo + Y0_under_null * (1-treatment_pseudo)
      Yobs_pseudo <- jnp$add(jnp$multiply(Y1_under_null_pseudo, treatment_pseudo),
                             jnp$multiply(Y0_under_null, jnp$subtract(1., treatment_pseudo)))

      #stat_ <- mean(Yobs_pseudo[treatment_pseudo == 1]) - mean(Yobs_pseudo[treatment_pseudo == 0])
      stat_ <- FastDiffInMeans(Yobs_pseudo, treatment_pseudo, n0_array, n1_array)
    } )

    vec1_get_stat_vec_at_tau_pseudo <<- jax$jit( jax$vmap(function(treatment_pseudo,
                                                                   obsY_array,
                                                                   obsW_array,
                                                                   tau_pseudo,
                                                                   n0_array,
                                                                   n1_array){
      get_stat_vec_at_tau_pseudo(treatment_pseudo,
                                 obsY_array,
                                 obsW_array,
                                 tau_pseudo,
                                 n0_array,
                                 n1_array)
    }, in_axes = list(0L, NULL, NULL, NULL, NULL, NULL)) )
  }
  
  RowBroadcast <<- jax$vmap(function(mat, vec){
    jnp$multiply(mat, vec)}, in_axes = list(1L, NULL))
  
  FastHotel2T2 <<- ( function(samp_, w_, n0, n1, approximate_inv = FALSE){
    # Assert statements to ensure valid inputs
    #assertthat::assert_that(n0 > 0, "Number of control units (n0) must be greater than 0.")
    #assertthat::assert_that(n1 > 0, "Number of treated units (n1) must be greater than 0.")
    #assertthat::assert_that(samp_.ndim == 2, "samp_ must be a 2D array.")
    #assertthat::assert_that(w_.ndim == 1, "w_ must be a 1D array.")
    #assertthat::assert_that(samp_.shape[0] == w_.shape[0], "samp_ and w_ must have the same number of rows.")
    
    # set up calc
    xbar1 <- jnp$divide(jnp$sum(RowBroadcast(samp_,w_),1L,keepdims = T), n1)
    xbar2 <- jnp$divide(jnp$sum(RowBroadcast(samp_,jnp$subtract(1.,w_)),1L,keepdims = T), n0)
    CovWts <- jnp$add(jnp$reciprocal(n0), jnp$reciprocal(n1))
    CovInv <- jax$lax$cond(pred = approximate_inv,
                 true_fun = function(){CovPooled <- jnp$var(samp_, 0L); 
                                        CovInv <- jnp$diag( jnp$reciprocal( jnp$multiply(CovPooled,CovWts) ));
                                        return(CovInv)},
                 false_fun = function(){CovPooled <- jnp$cov(samp_,rowvar = FALSE); 
                                        CovInv <- jnp$reciprocal( jnp$multiply(CovPooled, CovWts) ) ;
                                        return( CovInv )})
    xbar_diff <- jnp$subtract(xbar1, xbar2)
    Tstat <- jnp$matmul(jnp$matmul(jnp$transpose(xbar_diff), CovInv) , xbar_diff)
  })
  
  VectorizedFastHotel2T2 <<- jax$jit(VectorizedFastHotel2T2_R <<- jax$vmap(function(samp_, w_, n0, n1, approximate_inv = FALSE){
    FastHotel2T2(samp_, w_, n0, n1, approximate_inv)},
    in_axes = list(NULL, 0L, NULL, NULL, NULL)) )
  
  BatchedVectorizedFastHotel2T2 <<- function(samp_, w_, n0, n1, NWBatch, approximate_inv = FALSE){
    N_w <- w_$shape[0]  # Total number of w_ vectors
    num_batches <- as.integer( (N_w + NWBatch - 1) / NWBatch )  # Calculate number of batches
    
    # Function to process a single batch
    process_batch <- function(batch_idx, carry){
      start_idx <- batch_idx * NWBatch
      end_idx <- jnp$minimum(start_idx + NWBatch, N_w)
      w_batch <- w_[start_idx:end_idx, ]
      
      # Compute the statistics for the current batch
      Tstat_batch <- VectorizedFastHotel2T2(samp_, w_batch, n0, n1, approximate_inv)
      
      # Accumulate results
      carry <- jnp$concatenate(list(carry, Tstat_batch), axis=0)
      carry
    }
    
    # Initialize carry with an empty array
    carry_init <- jnp$array(NULL, dtype=jnp$float32)
    
    # Loop over batches using lax.fori_loop for efficiency
    Tstats <- jax$lax$fori_loop(
      lower=0,
      upper=num_batches,
      body_fun=function( i, carry ){ process_batch(i, carry)},
      init_val=carry_init
    )
    
    Tstats
  }
  
  print2("Success setting up core JAX functions!")
  }
}
