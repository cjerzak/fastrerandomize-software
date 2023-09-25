#!/usr/bin/env Rscript
#' Initialize JAX functions
#'
#' @usage
#'
#' InitializeJAXFxns()
#'
#' @param ... This function takes no arguments
#'
#' @return This function should be run just after `fastrerandomize::InitializeJax`.This sets up the internal JAX functions for rerandomization analysis.
#'
#'
#' @examples
#' # For tutorials, see
#' # github.com/cjerzak/fastrerandomization-software
#'
#' @export
#' @md

InitializeJAXFxns <- function(){

  expand_grid_jax <- function(...){
    #grid <- jnp$meshgrid(jnp$array(c(0,1L)), jnp$array(c(0,1L)), jnp$array(c(0,1L)), indexing = "ij")
    grid <- jnp$meshgrid(..., indexing='ij')
    grid <- jnp$vstack(lapply(grid,function(zer){jnp$ravel(zer)}))
    grid <- jnp$transpose(grid)
    return( grid )
  }

  if(T == F){
    n_units <- 35
    expand_grid_jax_text <- paste(rep("jnp$array(0L:1L)",times = n_units), collapse = ", ")
    expand_grid_text <- paste(rep("0L:1L",times = n_units), collapse = ", ")
    system.time( tmp1 <- eval(parse(text = sprintf("expand_grid_jax(%s)",expand_grid_jax_text))) )
    system.time( tmp2 <- eval(parse(text = sprintf("expand.grid(%s)",expand_grid_text))) )
    tmp1$shape
    dim( tmp2 )
  }

  FastHotel2T2 <<- jax$jit( function(samp_, w_, n0, n1){
    # set up calc
    RowBroadcast <- jax$vmap(function(mat, vec){
      jnp$multiply(mat, vec)},
      in_axes = list(1L, NULL))
    xbar1 <- jnp$divide(jnp$sum(RowBroadcast(samp_,w_),1L,keepdims = T), n1)
    xbar2 <- jnp$divide(jnp$sum(RowBroadcast(samp_,jnp$subtract(1.,w_)),1L,keepdims = T), n0)
    CovPooled <- jnp$cov(samp_, rowvar = F)
    CovWts <- jnp$add(jnp$reciprocal(n0), jnp$reciprocal(n1))
    xbar_diff <- jnp$subtract(xbar1, xbar2)

    # perform calc
    Tstat <- jnp$matmul(
      jnp$matmul(jnp$transpose(xbar_diff),
                 jnp$linalg$inv( jnp$multiply(CovPooled,CovWts) )),
      xbar_diff)
  })

  VectorizedFastHotel2T2 <<- jax$jit( jax$vmap(function(samp_, w_, n0, n1){
    FastHotel2T2(samp_, w_, n0, n1)},
    in_axes = list(NULL, 0L, NULL, NULL)) )

  if(T == F){
    # tests
    Compositional::hotel2T2()
    samp_ <- jnp$array(matrix(rnorm(10*100),nrow=100))
    w_ <- jnp$array(sample(c(0,1), size = 100, replace =T))
    w_full <- jnp$array(matrix(rbinom(100*34,size = 1, prob = 0.5),nrow = 100))
    FastHotel2T2(samp_ = samp_, w_ = w_, n0 = n0, n1 = n1)
    VectorizedFastHotel2T2(samp_, w_full, n0, n1)
  }

  FastDiffInMeans <<- jax$jit( function(y_,w_, n0, n1){
    my1 <- jnp$divide(jnp$sum(jnp$multiply(y_, w_)), n1)
    my0 <- jnp$divide(jnp$sum(jnp$multiply(y_, jnp$subtract(1.,w_))), n0)
    return( diff10 <- jnp$subtract(my1, my0) )
  })

  VectorizedTakeAxis0 <<- jax$jit( function(A_, I_){
    jnp$expand_dims(jnp$take(A_, I_, axis = 0L), 0L)
  })

  Potentisl2Obs <<- jax$jit(function(Y0__, Y1__, obsW__){
    jnp$add( jnp$multiply(Y0__, jnp$subtract(1, obsW__)),
             jnp$multiply(Y1__, obsW__))
  })

  W_VectorizedFastDiffInMeans <<- jax$jit( jax$vmap(function(y_, w_, n0, n1){
    FastDiffInMeans(y_, w_, n0, n1)},
    in_axes = list(NULL, 0L, NULL, NULL)) )

  Y_VectorizedFastDiffInMeans <<- jax$jit( jax$vmap(function(y_, w_, n0, n1){
    FastDiffInMeans(y_, w_, n0, n1)},
    in_axes = list(0L, NULL, NULL, NULL)) )

  YW_VectorizedFastDiffInMeans <<- jax$jit( jax$vmap(function(y_, w_, n0, n1){
    W_VectorizedFastDiffInMeans(y_, w_, n0, n1)},
    in_axes = list(0L, NULL, NULL, NULL)) )

  WVectorizedFastDiffInMeans <<- jax$jit( jax$vmap(function(y_, w_, n0, n1){
    FastDiffInMeans(y_, w_, n0, n1)},
    in_axes = list(NULL, 0L, NULL, NULL)) )

  GreaterEqualMagCompare <<- jax$jit(function(NULL_, OBS_){
    jnp$mean(jnp$greater_equal(jnp$abs(NULL_),  jnp$expand_dims(OBS_,1L)), 1L)
  })

  if(T == F){
    # tests
    y_ <- jnp$array(rnorm(10))
    w_ <-  jnp$array(rbinom(10,size=1, prob = 0.5))
    n0 <- jnp$array(10.)
    n1 <- jnp$array(3.)
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
    Yobs_pseudo <- jnp$add(jnp$multiply(Y1_under_null_pseudo,treatment_pseudo),
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

