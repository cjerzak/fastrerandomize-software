#' Initialize JAX Environment for Fast Rerandomization
#'
#' @param conda_env Character string. The conda environment name containing JAX. If NULL, uses default Python environment.
#' @param conda_env_required Logical. Whether to force use of the specified conda environment.
#'
#' @return Initializes JAX environment and defines core JAX functions globally.
#'   Returns invisible(NULL).
#'
#' @details
#' This function must be run before using any other functions in the package.
#' It initializes JAX and defines several core functions used throughout the package:
#' \itemize{
#'   \item expand_grid_JAX: Creates treatment combinations
#'   \item FastDiffInMeans: Computes difference in means
#'   \item Various vectorized versions of these functions
#' }
#'
#' @examples
#' \dontrun{
#' # Basic usage
#' InitializeJAX()
#'
#' # Using specific conda environment
#' InitializeJAX(conda_env = "my_jax_env")
#' }
#'
#' @importFrom reticulate import use_condaenv
#' @export

InitializeJAX <- function(conda_env = NULL, conda_env_required = T){
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
  
  VectorizedFastHotel2T2 <<- jax$jit( jax$vmap(function(samp_, w_, n0, n1, approximate_inv = FALSE){
    FastHotel2T2(samp_, w_, n0, n1, approximate_inv)},
    in_axes = list(NULL, 0L, NULL, NULL, NULL)) )
  print2("Success setting up core JAX functions!")
}
